from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import onnx

from model_pipeline.benchmarks.qdq import inspect_benchmark_qdq
from model_pipeline.benchmarks.vlsp import (
    VlspBenchmarkRequest,
    evaluate_vpcd_quality_gate,
    evaluate_zipformer_quality_gate,
)
from model_pipeline.core import Stage, sha256_file, sha256_path, stable_digest
from model_pipeline.datasets.audio import probe_audio
from model_pipeline.datasets.vlsp import (
    iter_vlsp_rows,
    select_vlsp_calibration_evaluation,
    write_vlsp_calibration_evaluation,
)
from model_pipeline.evaluation.reports import write_evaluation_json, write_sample_jsonl
from model_pipeline.evaluation.vlsp100 import (
    evaluate_vpcd_samples,
    evaluate_zipformer_samples,
    load_mono_audio,
    load_vlsp_evaluation_samples,
    summarize_vpcd_parity,
    summarize_zipformer_regression,
)
from model_pipeline.integrations.aihub import (
    CompiledModelContract,
    EvidenceStore,
    HostedInferenceStore,
    QualcommAiHubClient,
    run_hosted_inputs,
    validate_compiled_model,
)
from model_pipeline.models import get_recipe
from model_pipeline.models.aimet_service import AimetServiceClient
from model_pipeline.models.vpcd.adapter import VpcdAdapter
from model_pipeline.models.vpcd.runtime import VpcdLocalRuntime
from model_pipeline.models.zipformer.adapter import ZipformerAdapter
from model_pipeline.models.zipformer.runtime import (
    ZipformerLocalRuntime,
    extract_zipformer_features,
)
from model_pipeline.pipeline import ModelPipeline


class ProductionVlspBenchmarkBackend:
    """Execute the reproducible benchmark through existing production boundaries."""

    def __init__(self, *, request: VlspBenchmarkRequest, repo_root: Path):
        """Initialize benchmark paths and lazy local/cloud integrations.

        Args:
            request: Validated benchmark request.
            repo_root: Repository root containing sources and ignored build output.

        Returns:
            None.

        Raises:
            ValueError: If the build root is outside the repository AIMET mount.
        """
        self.request = request
        self.repo_root = Path(repo_root).resolve()
        self.build_root = (
            (self.repo_root / request.build_root).resolve()
            if not request.build_root.is_absolute()
            else request.build_root.resolve()
        )
        if self.build_root != self.repo_root and self.repo_root not in self.build_root.parents:
            raise ValueError("VLSP benchmark build root must stay inside the repository")
        self.pipeline_root = self.build_root / "model-pipeline"
        self.aimet_service = AimetServiceClient(
            repo_root=self.repo_root,
            url=os.environ.get("AIMET_SERVICE_URL", "http://127.0.0.1:18080"),
        )
        self._aihub_client: QualcommAiHubClient | None = None

    def materialize_dataset(
        self,
        *,
        dataset_root: Path,
        output_dir: Path,
    ) -> Mapping[str, Any]:
        """Materialize the exact 24/100 disjoint VLSP split.

        Args:
            dataset_root: Source parquet shard directory.
            output_dir: Step-owned dataset output directory.

        Returns:
            Portable paths, counts, shards, and manifest checksum.
        """
        split = select_vlsp_calibration_evaluation(
            iter_vlsp_rows(dataset_root),
            calibration_count=24,
            evaluation_count=100,
            probe=probe_audio,
        )
        outputs = write_vlsp_calibration_evaluation(split, output_dir)
        return {
            "manifest": self._portable(outputs["manifest"]),
            "calibration_transcriptions": self._portable(
                outputs["calibration_transcriptions"]
            ),
            "evaluation_transcriptions": self._portable(
                outputs["evaluation_transcriptions"]
            ),
            "dataset_checksum": sha256_file(outputs["manifest"]),
            "calibration_count": len(split.calibration),
            "evaluation_count": len(split.evaluation),
            "calibration_shards": sorted({row.source_shard for row in split.calibration}),
            "evaluation_shards": sorted({row.source_shard for row in split.evaluation}),
        }

    def local_input_digest(
        self,
        *,
        model: str,
        dataset: Mapping[str, Any],
        providers: tuple[str, ...],
    ) -> str:
        """Hash exact source bytes, recipes, dataset, and provider request.

        Args:
            model: Canonical model family.
            dataset: Materialized dataset evidence.
            providers: Requested local provider names.

        Returns:
            Stable digest invalidated by any benchmark input change.
        """
        manifest = self._resolve(dataset["manifest"])
        calibration_text = self._resolve(dataset["calibration_transcriptions"])
        sources: dict[str, str] = {}
        recipes: dict[str, str] = {}
        for configuration in ("fp32-fixed-shape", "aimet-int8-int16-encoder-matmul"):
            recipe = get_recipe(model, configuration)
            recipes[configuration] = recipe.digest
            adapter = self._adapter(
                model,
                calibration_manifest=manifest,
                calibration_text=calibration_text,
            )
            for role, path in sorted(adapter.source_files(recipe).items()):
                sources[f"{configuration}:{role}"] = sha256_file(path)
        return stable_digest(
            {
                "model": model,
                "dataset_checksum": dataset["dataset_checksum"],
                "providers": list(providers),
                "recipes": recipes,
                "sources": sources,
            }
        )

    def run_local(
        self,
        *,
        model: str,
        dataset: Mapping[str, Any],
        providers: tuple[str, ...],
        output_dir: Path,
    ) -> Mapping[str, Any]:
        """Run prepare, AIMET, strict QDQ, and local quality evaluation.

        Args:
            model: Canonical model family.
            dataset: Materialized VLSP dataset evidence.
            providers: CPU and optional CUDA provider requests.
            output_dir: Step-owned local evidence directory.

        Returns:
            Artifact paths, graph evidence, provider truth, metrics, and quality gate.
        """
        manifest = self._resolve(dataset["manifest"])
        calibration_text = self._resolve(dataset["calibration_transcriptions"])
        fp32_recipe = get_recipe(model, "fp32-fixed-shape")
        aimet_recipe = get_recipe(model, "aimet-int8-int16-encoder-matmul")
        fp32_adapter = self._adapter(
            model,
            calibration_manifest=manifest,
            calibration_text=calibration_text,
        )
        aimet_adapter = self._adapter(
            model,
            calibration_manifest=manifest,
            calibration_text=calibration_text,
        )
        pipeline = ModelPipeline(
            build_root=self.pipeline_root,
            evidence_store=EvidenceStore(self.pipeline_root / "aihub-evidence"),
            aihub_client=None,
        )
        pipeline.run(recipe=fp32_recipe, adapter=fp32_adapter, through=Stage.VALIDATE)
        pipeline.run(recipe=aimet_recipe, adapter=aimet_adapter, through=Stage.VALIDATE)
        fp32 = self._stage_outputs(fp32_recipe.artifact.artifact_id, Stage.QUANTIZE)
        aimet = self._stage_outputs(aimet_recipe.artifact.artifact_id, Stage.QUANTIZE)
        model_role = "encoder" if model == "zipformer" else "model"
        policy = json.loads(aimet["quantization_policy"].read_text(encoding="utf-8"))
        qdq_dir = output_dir / "qdq"
        self.aimet_service.healthcheck()
        self.aimet_service.export_qdq(
            fp32_model_path=fp32[model_role],
            encodings_path=aimet["encodings"],
            output_dir=qdq_dir,
            config_path=aimet["aimet_config"],
            policy_path=aimet["quantization_policy"],
        )
        qdq_model = qdq_dir / "model.qdq.onnx"
        if not qdq_model.is_file():
            raise FileNotFoundError(f"AIMET service did not export benchmark QDQ: {qdq_model}")
        graph = inspect_benchmark_qdq(
            qdq_model,
            selected_matmul_names=policy["quantize_op_names"],
            expected_total_matmul=278 if model == "zipformer" else 265,
        )
        samples = load_vlsp_evaluation_samples(manifest)
        evaluation = (
            self._evaluate_zipformer(
                samples=samples,
                fp32=fp32,
                qdq_model=qdq_model,
                providers=providers,
                output_dir=output_dir,
            )
            if model == "zipformer"
            else self._evaluate_vpcd(
                samples=samples,
                fp32=fp32,
                qdq_model=qdq_model,
                providers=providers,
                output_dir=output_dir,
            )
        )
        summary_path = write_evaluation_json(
            output_dir / "local-summary.json",
            {
                "model": model,
                "dataset_checksum": dataset["dataset_checksum"],
                "graph": graph.to_dict(),
                **evaluation,
            },
        )
        return {
            "model": model,
            "dataset": dict(dataset),
            "dataset_checksum": dataset["dataset_checksum"],
            "fp32_artifact_id": fp32_recipe.artifact.artifact_id,
            "quantized_artifact_id": aimet_recipe.artifact.artifact_id,
            "fp32_model": self._portable(fp32[model_role]),
            "aimet_model": self._portable(aimet[model_role]),
            "aimet_encodings": self._portable(aimet["encodings"]),
            "aimet_config": self._portable(aimet["aimet_config"]),
            "quantization_policy": self._portable(aimet["quantization_policy"]),
            "compile_source": self._portable(aimet[model_role].parent),
            "compile_source_checksum": sha256_path(aimet[model_role].parent),
            "qdq_model": self._portable(qdq_model),
            "support": {
                role: self._portable(path)
                for role, path in fp32.items()
                if role != model_role
            },
            "hosted_fixtures": evaluation["hosted_fixtures"],
            "graph": graph.to_dict(),
            "providers": evaluation["providers"],
            "quality_gate": evaluation["quality_gate"],
            "quality_passed": evaluation["quality_gate"]["passed"],
            "evidence": self._portable(summary_path),
            "_resume_artifacts": {
                self._resume_relative(fp32[model_role].parent): sha256_path(
                    fp32[model_role].parent
                ),
                self._resume_relative(aimet[model_role].parent): sha256_path(
                    aimet[model_role].parent
                ),
            },
        }

    def compile(
        self,
        *,
        model: str,
        local: Mapping[str, Any],
        output_dir: Path,
    ) -> Mapping[str, Any]:
        """Compile the exact AIMET package and validate downloaded EPContext.

        Args:
            model: Canonical model family.
            local: Passing local benchmark evidence.
            output_dir: Step-owned compile evidence directory.

        Returns:
            Compile identity, checksums, representation, I/O, and support files.
        """
        recipe = get_recipe(model, "aimet-int8-int16-encoder-matmul")
        dataset = dict(local["dataset"])
        adapter = self._adapter(
            model,
            calibration_manifest=self._resolve(dataset["manifest"]),
            calibration_text=self._resolve(dataset["calibration_transcriptions"]),
        )
        client = self._client()
        evidence_store = EvidenceStore(self.pipeline_root / "aihub-evidence")
        pipeline = ModelPipeline(
            build_root=self.pipeline_root,
            evidence_store=evidence_store,
            aihub_client=client,
        )
        pipeline.run(recipe=recipe, adapter=adapter, through=Stage.COMPILE)
        outputs = self._stage_outputs(recipe.artifact.artifact_id, Stage.COMPILE)
        role = "encoder" if model == "zipformer" else "model"
        compiled_model = outputs[role]
        source_model = self._resolve(local["aimet_model"])
        input_dtypes, output_dtypes = _compiled_dtype_contract(source_model)
        evidence = validate_compiled_model(
            compiled_model,
            CompiledModelContract(
                artifact=recipe.artifact,
                input_dtypes=input_dtypes,
                output_dtypes=output_dtypes,
                requires_int64_to_int32=True,
            ),
        )
        compile_record = evidence_store.resolve(str(local["compile_source_checksum"]))
        if compile_record is None:
            raise RuntimeError("Checksum-keyed AI Hub compile evidence is missing")
        support_files = [
            path
            for name, path in outputs.items()
            if name != role
        ]
        if not any(path.suffix.lower() in {".bin", ".data"} for path in support_files):
            raise ValueError("Compiled EPContext package is missing model.bin or external data")
        payload = {
            "model": model,
            "compile_job_id": compile_record.job_id,
            "compile_source_checksum": local["compile_source_checksum"],
            "compiled_package_checksum": compile_record.output_checksum,
            "compiled_model": self._portable(compiled_model),
            "compiled_model_checksum": sha256_file(compiled_model),
            "support_files": [self._portable(path) for path in support_files],
            "representation": "onnx-epcontext",
            "execution_target": evidence.execution_target,
            "input_dtypes": dict(evidence.input_dtypes),
            "output_dtypes": dict(evidence.output_dtypes),
        }
        evidence_path = write_evaluation_json(output_dir / "compile.json", payload)
        return {
            **payload,
            "evidence": self._portable(evidence_path),
            "_resume_artifacts": {
                self._resume_relative(compiled_model.parent): sha256_path(
                    compiled_model.parent
                )
            },
        }

    def hosted_validate(
        self,
        *,
        model: str,
        local: Mapping[str, Any],
        compiled: Mapping[str, Any],
        input_limit: int,
        output_dir: Path,
    ) -> Mapping[str, Any]:
        """Run exactly five checksum-keyed hosted parity inputs.

        Args:
            model: Canonical model family.
            local: Passing local benchmark evidence and fixtures.
            compiled: Validated compile evidence.
            input_limit: Hard hosted input limit.
            output_dir: Step-owned hosted evidence directory.

        Returns:
            Hosted job IDs, output checksums, parity counts, and pass status.

        Raises:
            ValueError: If fixture count differs from the hard quota.
        """
        fixtures_root = self._resolve(local["hosted_fixtures"])
        fixture_paths = sorted(fixtures_root.glob("input-*.npz"))
        if len(fixture_paths) != input_limit:
            raise ValueError(
                f"Hosted validation requires exactly {input_limit} fixtures; found {len(fixture_paths)}"
            )
        inputs = []
        for path in fixture_paths:
            with np.load(path, allow_pickle=False) as arrays:
                inputs.append({name: np.asarray(arrays[name]) for name in arrays.files})
        recipe = get_recipe(model, "aimet-int8-int16-encoder-matmul")
        results = run_hosted_inputs(
            artifact=recipe.artifact,
            compile_job_id=str(compiled["compile_job_id"]),
            inputs=inputs,
            client=self._client(),
            evidence_store=HostedInferenceStore(output_dir / "records"),
        )
        expected = json.loads((fixtures_root / "expected.json").read_text(encoding="utf-8"))
        matches = (
            self._validate_zipformer_hosted(local, results, expected)
            if model == "zipformer"
            else self._validate_vpcd_hosted(results, expected)
        )
        payload = {
            "model": model,
            "input_count": len(results),
            "parity_matches": matches,
            "quality_passed": matches == input_limit,
            "compile_job_id": compiled["compile_job_id"],
            "hosted_job_ids": [result.evidence.inference_job_id for result in results],
            "input_checksums": [result.evidence.input_checksum for result in results],
            "output_checksums": [result.evidence.output_checksum for result in results],
        }
        evidence_path = write_evaluation_json(output_dir / "hosted.json", payload)
        return {**payload, "evidence": self._portable(evidence_path)}

    def _evaluate_zipformer(
        self,
        *,
        samples: Sequence[Any],
        fp32: Mapping[str, Path],
        qdq_model: Path,
        providers: tuple[str, ...],
        output_dir: Path,
    ) -> dict[str, Any]:
        """Evaluate Zipformer FP32 and QDQ with shared FP32 support components.

        Args:
            samples: Ordered held-out VLSP audio records.
            fp32: FP32 pipeline component paths.
            qdq_model: Benchmark-only explicit-QDQ encoder.
            providers: CPU and optional CUDA requests.
            output_dir: Step-owned local evidence directory.

        Returns:
            Provider summaries, CPU quality gate, and hosted fixture directory.
        """
        provider_results: dict[str, Any] = {}
        cpu_fp32_records: Sequence[Mapping[str, object]] | None = None
        cpu_qdq_records: Sequence[Mapping[str, object]] | None = None
        for provider in providers:
            if provider == "cuda" and not _cuda_available():
                provider_results[provider] = {"status": "unavailable", "cuda_executed": False}
                continue
            prefer_cuda = provider == "cuda"
            fp32_runtime = ZipformerLocalRuntime.from_paths(
                encoder_path=fp32["encoder"],
                decoder_path=fp32["decoder"],
                joiner_path=fp32["joiner"],
                tokens_path=fp32["tokens"],
                prefer_cuda=prefer_cuda,
            )
            qdq_runtime = ZipformerLocalRuntime.from_paths(
                encoder_path=qdq_model,
                decoder_path=fp32["decoder"],
                joiner_path=fp32["joiner"],
                tokens_path=fp32["tokens"],
                prefer_cuda=prefer_cuda,
            )
            fp32_records = evaluate_zipformer_samples(fp32_runtime, samples)
            qdq_records = evaluate_zipformer_samples(qdq_runtime, samples)
            comparison = summarize_zipformer_regression(fp32_records, qdq_records)
            fp32_profile = fp32_runtime.finish_provider_profile()
            qdq_profile = qdq_runtime.finish_provider_profile()
            write_sample_jsonl(output_dir / provider / "fp32.jsonl", fp32_records)
            write_sample_jsonl(output_dir / provider / "qdq.jsonl", qdq_records)
            provider_results[provider] = {
                "status": _provider_status(provider, qdq_profile.node_counts, qdq_profile.cuda_executed),
                "fp32": comparison["fp32"],
                "qdq": comparison["quantized"],
                "exact_transcript_parity": comparison["exact_transcript_parity"],
                "fp32_profile": fp32_profile.to_dict(),
                "qdq_profile": qdq_profile.to_dict(),
            }
            if provider == "cpu":
                cpu_fp32_records = fp32_records
                cpu_qdq_records = qdq_records
        if cpu_fp32_records is None or cpu_qdq_records is None:
            raise RuntimeError("Zipformer CPU benchmark evidence is missing")
        cpu = provider_results["cpu"]
        fp32_metrics = cpu["fp32"]["transcript_metrics"]
        qdq_metrics = cpu["qdq"]["transcript_metrics"]
        gate = evaluate_zipformer_quality_gate(
            fp32_cer=fp32_metrics["character_error_rate"],
            qdq_cer=qdq_metrics["character_error_rate"],
            fp32_wer=fp32_metrics["word_error_rate"],
            qdq_wer=qdq_metrics["word_error_rate"],
            empty_outputs=cpu["qdq"]["empty_outputs"],
            collapse_outputs=cpu["qdq"]["repetition_collapses"],
        )
        fixtures = output_dir / "hosted-inputs"
        _write_zipformer_hosted_fixtures(samples[:5], cpu_fp32_records[:5], fixtures)
        return {
            "providers": provider_results,
            "quality_gate": gate,
            "hosted_fixtures": self._portable(fixtures),
        }

    def _evaluate_vpcd(
        self,
        *,
        samples: Sequence[Any],
        fp32: Mapping[str, Path],
        qdq_model: Path,
        providers: tuple[str, ...],
        output_dir: Path,
    ) -> dict[str, Any]:
        """Evaluate VPCD FP32 and QDQ with the same CPU tokenizer and loop.

        Args:
            samples: Ordered held-out VLSP transcription records.
            fp32: FP32 pipeline component paths.
            qdq_model: Benchmark-only explicit-QDQ VPCD model.
            providers: CPU and optional CUDA requests.
            output_dir: Step-owned local evidence directory.

        Returns:
            Provider summaries, CPU quality gate, and hosted fixture directory.
        """
        tokenizer_directory = self.repo_root / "assets" / "vietnamese-punc-cap-denorm-v1"
        provider_results: dict[str, Any] = {}
        cpu_fp32_records: Sequence[Mapping[str, object]] | None = None
        cpu_qdq_records: Sequence[Mapping[str, object]] | None = None
        cpu_runtime: VpcdLocalRuntime | None = None
        for provider in providers:
            if provider == "cuda" and not _cuda_available():
                provider_results[provider] = {"status": "unavailable", "cuda_executed": False}
                continue
            prefer_cuda = provider == "cuda"
            fp32_runtime = VpcdLocalRuntime.from_paths(
                model_path=fp32["model"],
                tokenizer_directory=tokenizer_directory,
                prefer_cuda=prefer_cuda,
            )
            qdq_runtime = VpcdLocalRuntime.from_paths(
                model_path=qdq_model,
                tokenizer_directory=tokenizer_directory,
                prefer_cuda=prefer_cuda,
            )
            fp32_records = evaluate_vpcd_samples(fp32_runtime, samples)
            qdq_records = evaluate_vpcd_samples(qdq_runtime, samples)
            summary, comparisons = summarize_vpcd_parity(
                fp32_records,
                qdq_records,
                eos_token_id=fp32_runtime.eos_token_id,
            )
            fp32_profile = fp32_runtime.finish_provider_profile()
            qdq_profile = qdq_runtime.finish_provider_profile()
            write_sample_jsonl(output_dir / provider / "fp32.jsonl", fp32_records)
            write_sample_jsonl(output_dir / provider / "qdq.jsonl", qdq_records)
            write_sample_jsonl(output_dir / provider / "comparison.jsonl", comparisons)
            provider_results[provider] = {
                "status": _provider_status(provider, qdq_profile.node_counts, qdq_profile.cuda_executed),
                "summary": summary,
                "fp32_profile": fp32_profile.to_dict(),
                "qdq_profile": qdq_profile.to_dict(),
            }
            if provider == "cpu":
                cpu_fp32_records = fp32_records
                cpu_qdq_records = qdq_records
                cpu_runtime = fp32_runtime
        if cpu_fp32_records is None or cpu_qdq_records is None or cpu_runtime is None:
            raise RuntimeError("VPCD CPU benchmark evidence is missing")
        summary = provider_results["cpu"]["summary"]
        gate = evaluate_vpcd_quality_gate(
            sample_count=summary["sample_count"],
            exact_output_matches=summary["exact_output_matches"],
            first_five_top1_matches=summary["first_five_top1_matches"],
            first_five_step_count=summary["first_five_step_count"],
            early_eos_count=summary["early_eos_count"],
            collapse_count=summary["collapse_count"],
        )
        fixtures = output_dir / "hosted-inputs"
        _write_vpcd_hosted_fixtures(samples[:5], cpu_fp32_records[:5], cpu_runtime, fixtures)
        return {
            "providers": provider_results,
            "quality_gate": gate,
            "hosted_fixtures": self._portable(fixtures),
        }

    def _validate_zipformer_hosted(
        self,
        local: Mapping[str, Any],
        results: Sequence[Any],
        expected: Sequence[Mapping[str, Any]],
    ) -> int:
        """Decode hosted encoder outputs with shared FP32 support components.

        Args:
            local: Local evidence containing FP32 decoder, joiner, and tokens.
            results: Hosted encoder outputs in fixture order.
            expected: Expected FP32 transcripts.

        Returns:
            Exact transcript parity count.
        """
        support = {name: self._resolve(path) for name, path in dict(local["support"]).items()}
        runtime = ZipformerLocalRuntime.from_paths(
            encoder_path=self._resolve(local["fp32_model"]),
            decoder_path=support["decoder"],
            joiner_path=support["joiner"],
            tokens_path=support["tokens"],
            prefer_cuda=False,
        )
        matches = 0
        for result, reference in zip(results, expected):
            arrays = _flatten_hosted_outputs(result.outputs)
            encoded = next(value for value in arrays.values() if value.ndim >= 3)
            length = next(value for value in arrays.values() if value.ndim <= 1 or value.size == 1)
            transcript = runtime.decode_encoder_outputs(
                encoded,
                encoded_length=int(length.reshape(-1)[0]),
            ).transcript
            matches += int(transcript == str(reference["transcript"]))
        runtime.finish_provider_profile()
        return matches

    @staticmethod
    def _validate_vpcd_hosted(
        results: Sequence[Any],
        expected: Sequence[Mapping[str, Any]],
    ) -> int:
        """Compare hosted VPCD logits with expected local first-step tokens.

        Args:
            results: Hosted VPCD outputs in fixture order.
            expected: Expected local first-step top-one tokens.

        Returns:
            Top-one parity count.
        """
        matches = 0
        for result, reference in zip(results, expected):
            arrays = _flatten_hosted_outputs(result.outputs)
            logits = next(value for value in arrays.values() if value.ndim >= 3)
            token_id = int(np.argmax(logits[0, 0, :]))
            matches += int(token_id == int(reference["top1_token_id"]))
        return matches

    def _adapter(
        self,
        model: str,
        *,
        calibration_manifest: Path,
        calibration_text: Path,
    ) -> ZipformerAdapter | VpcdAdapter:
        """Create a model adapter with the exact materialized calibration source.

        Args:
            model: Canonical model family.
            calibration_manifest: Portable shared VLSP split manifest.
            calibration_text: Materialized 24-record calibration transcription file.

        Returns:
            Zipformer or VPCD adapter wired to the shared AIMET service.
        """
        if model == "zipformer":
            return ZipformerAdapter(
                self.repo_root,
                calibration_manifest=calibration_manifest,
                aimet_service=self.aimet_service,
            )
        return VpcdAdapter(
            self.repo_root,
            calibration_text=calibration_text,
            aimet_service=self.aimet_service,
        )

    def _stage_outputs(self, artifact_id: str, stage: Stage) -> dict[str, Path]:
        """Restore verified pipeline output paths from stage state.

        Args:
            artifact_id: Canonical artifact identity.
            stage: Completed pipeline stage.

        Returns:
            Logical component roles mapped to absolute paths.

        Raises:
            FileNotFoundError: If required stage state or output is missing.
        """
        stage_dir = self.pipeline_root / artifact_id / stage.value
        state_path = stage_dir / "stage-state.json"
        if not state_path.is_file():
            raise FileNotFoundError(f"Pipeline stage state is missing: {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        outputs = {name: stage_dir / relative for name, relative in state["outputs"].items()}
        missing = [path for path in outputs.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Pipeline stage outputs are missing: {missing!r}")
        return outputs

    def _client(self) -> QualcommAiHubClient:
        """Authenticate and cache the explicitly configured AI Hub client.

        Returns:
            Authenticated client pinned to request device and QAIRT version.
        """
        if self._aihub_client is None:
            self._aihub_client = QualcommAiHubClient(
                device_name=str(self.request.device),
                api_token=os.environ.get("QAI_HUB_API_TOKEN"),
                qairt_version=self.request.qairt_version,
            )
            self._aihub_client.authenticate()
        return self._aihub_client

    def _portable(self, path: str | Path) -> str:
        """Render a generated path relative to the repository root.

        Args:
            path: Generated file or directory below the repository root.

        Returns:
            Portable POSIX path relative to the repository root.
        """
        return Path(path).resolve().relative_to(self.repo_root).as_posix()

    def _resume_relative(self, path: str | Path) -> str:
        """Render a generated artifact path relative to the benchmark root.

        Args:
            path: Generated file or directory below the benchmark root.

        Returns:
            POSIX path used only by ignored resume state.
        """
        return Path(path).resolve().relative_to(self.build_root).as_posix()

    def _resolve(self, relative: object) -> Path:
        """Resolve one repository-relative generated path.

        Args:
            relative: Portable repository-relative path stored in benchmark evidence.

        Returns:
            Absolute path below the repository root.
        """
        return (self.repo_root / str(relative)).resolve()


def _compiled_dtype_contract(model_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Derive compiled I/O dtypes while applying the 64-to-32-bit transform.

    Args:
        model_path: Canonical AIMET compile-source ONNX model.

    Returns:
        Expected compiled input and output dtype mappings.
    """
    model = onnx.load(model_path.as_posix(), load_external_data=False)

    def collect(values: object) -> dict[str, str]:
        """Normalize graph value dtypes for compiled validation.

        Args:
            values: ONNX graph input or output value information.

        Returns:
            Tensor names mapped to expected compiled NumPy dtype names.
        """
        result: dict[str, str] = {}
        for value in values:
            dtype = np.dtype(
                onnx.helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type)
            ).name
            result[value.name] = "int32" if dtype == "int64" else dtype
        return result

    return collect(model.graph.input), collect(model.graph.output)


def _cuda_available() -> bool:
    """Return whether ONNX Runtime currently registers CUDA execution.

    Returns:
        `True` when `CUDAExecutionProvider` is registered.
    """
    import onnxruntime as ort

    return "CUDAExecutionProvider" in ort.get_available_providers()


def _provider_status(
    requested: str,
    node_counts: Mapping[str, int],
    cuda_executed: bool,
) -> str:
    """Classify provider evidence without claiming unobserved CUDA execution.

    Args:
        requested: Requested local provider name.
        node_counts: Executed node counts by ONNX Runtime provider.
        cuda_executed: Whether any CUDA node event was observed.

    Returns:
        `cpu`, `cuda`, `cuda-mixed`, or `cuda-not-observed`.
    """
    if requested == "cpu":
        return "cpu"
    if not cuda_executed:
        return "cuda-not-observed"
    return "cuda-mixed" if node_counts.get("CPUExecutionProvider", 0) else "cuda"


def _write_zipformer_hosted_fixtures(
    samples: Sequence[Any],
    fp32_records: Sequence[Mapping[str, object]],
    output_dir: Path,
) -> None:
    """Materialize five fixed encoder inputs and expected FP32 transcripts.

    Args:
        samples: First five held-out VLSP audio records.
        fp32_records: Matching FP32 transcript records.
        output_dir: Directory receiving compressed inputs and expected JSON.

    Returns:
        None.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    expected: list[dict[str, Any]] = []
    for index, (sample, record) in enumerate(zip(samples, fp32_records), start=1):
        waveform, sample_rate = load_mono_audio(sample.audio_path)
        features = extract_zipformer_features(waveform, sample_rate)
        padded = np.zeros((1, 2009, 80), dtype=np.float32)
        padded[0, : features.shape[0], :] = features
        np.savez_compressed(
            output_dir / f"input-{index:02d}.npz",
            x=padded,
            x_lens=np.asarray([features.shape[0]], dtype=np.int32),
        )
        expected.append({"sample_id": sample.sample_id, "transcript": record["transcript"]})
    (output_dir / "expected.json").write_text(
        json.dumps(expected, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_vpcd_hosted_fixtures(
    samples: Sequence[Any],
    fp32_records: Sequence[Mapping[str, object]],
    runtime: VpcdLocalRuntime,
    output_dir: Path,
) -> None:
    """Materialize five teacher-forced prefixes and expected FP32 top-one tokens.

    Args:
        samples: First five held-out VLSP text records.
        fp32_records: Matching FP32 autoregressive output records.
        runtime: FP32 runtime exposing tokenizer and special-token contracts.
        output_dir: Directory receiving compressed inputs and expected JSON.

    Returns:
        None.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    expected: list[dict[str, Any]] = []
    for index, (sample, record) in enumerate(zip(samples, fp32_records), start=1):
        source_ids, source_mask = runtime.encode_text(sample.transcription)
        fixed_ids = np.full((1, 384), runtime.pad_token_id, dtype=np.int32)
        fixed_mask = np.zeros((1, 384), dtype=np.int32)
        fixed_ids[0, : len(source_ids)] = np.asarray(source_ids, dtype=np.int32)
        fixed_mask[0, : len(source_mask)] = np.asarray(source_mask, dtype=np.int32)
        decoder_ids = np.full((1, 64), runtime.pad_token_id, dtype=np.int32)
        decoder_mask = np.zeros((1, 64), dtype=np.int32)
        decoder_ids[0, 0] = runtime.decoder_start_token_id
        decoder_mask[0, 0] = 1
        np.savez_compressed(
            output_dir / f"input-{index:02d}.npz",
            input_ids=fixed_ids,
            attention_mask=fixed_mask,
            decoder_input_ids=decoder_ids,
            decoder_attention_mask=decoder_mask,
        )
        top1 = list(record["top1_token_ids"])
        if not top1:
            raise ValueError(f"VPCD FP32 output has no top-one token for {sample.sample_id}")
        expected.append({"sample_id": sample.sample_id, "top1_token_id": int(top1[0])})
    (output_dir / "expected.json").write_text(
        json.dumps(expected, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _flatten_hosted_outputs(outputs: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Normalize AI Hub batched output lists to one array per tensor name.

    Args:
        outputs: Hosted output mapping returned by AI Hub.

    Returns:
        Tensor names mapped to the first hosted batch array.

    Raises:
        ValueError: If a hosted tensor list is empty.
    """
    flattened: dict[str, np.ndarray] = {}
    for name, value in outputs.items():
        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError(f"Hosted output {name!r} is empty")
            value = value[0]
        flattened[str(name)] = np.asarray(value)
    return flattened
