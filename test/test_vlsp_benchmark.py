from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest
import onnx
from onnx import TensorProto, helper

from model_pipeline.benchmarks.vlsp import (
    BenchmarkStepRunner,
    VlspBenchmarkRequest,
    build_benchmark_plan,
    evaluate_vpcd_quality_gate,
    evaluate_zipformer_quality_gate,
    run_vlsp_benchmark,
)
from model_pipeline.benchmarks.qdq import inspect_benchmark_qdq
from model_pipeline.benchmarks.vlsp_runtime import ProductionVlspBenchmarkBackend
from model_pipeline.datasets.selection import AudioInfo
from model_pipeline.datasets.vlsp import VlspRow
from model_pipeline.cli import main


def test_benchmark_request_requires_explicit_cloud_opt_in() -> None:
    """Verify compile and hosted stop points cannot consume cloud implicitly.

    Returns:
        None.
    """
    with pytest.raises(ValueError, match="--submit-cloud"):
        VlspBenchmarkRequest(
            model="all",
            dataset_root=Path("dataset"),
            build_root=Path("build"),
            providers=("cpu", "cuda"),
            through="compile",
        )


def test_benchmark_plan_describes_cumulative_stages_without_io(tmp_path: Path) -> None:
    """Verify dry-run planning is portable and does not require dataset files.

    Args:
        tmp_path: Isolated nonexistent dataset and build paths.

    Returns:
        None.
    """
    request = VlspBenchmarkRequest(
        model="all",
        dataset_root=tmp_path / "missing-vlsp",
        build_root=tmp_path / "build",
        providers=("cpu", "cuda"),
        through="hosted",
        submit_cloud=True,
        device="Samsung Galaxy S23 (Family)",
        qairt_version="2.45",
    )

    plan = build_benchmark_plan(request)

    assert plan["models"] == ["zipformer", "vpcd"]
    assert plan["stages"] == ["dataset", "local", "compile", "hosted"]
    assert plan["dataset"] == {"calibration_count": 24, "evaluation_count": 100}
    assert plan["cloud"]["hosted_input_limit_per_model"] == 5
    assert plan["writes"] is True
    assert plan["cloud_calls"] is True


def test_benchmark_cli_dry_run_has_no_external_side_effects(tmp_path: Path, capsys) -> None:
    """Verify CLI dry-run prints the benchmark plan without touching build output.

    Args:
        tmp_path: Isolated nonexistent dataset and build paths.
        capsys: Pytest capture fixture for emitted JSON.

    Returns:
        None.
    """
    build_root = tmp_path / "build"
    exit_code = main(
        [
            "benchmark-vlsp",
            "--model",
            "all",
            "--dataset-root",
            str(tmp_path / "missing-vlsp"),
            "--build-root",
            str(build_root),
            "--providers",
            "cpu,cuda",
            "--through",
            "compile",
            "--submit-cloud",
            "--device",
            "Samsung Galaxy S23 (Family)",
            "--qairt-version",
            "2.45",
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["stages"] == ["dataset", "local", "compile"]
    assert payload["writes"] is False
    assert payload["cloud_calls"] is False
    assert not build_root.exists()


@pytest.mark.parametrize("through", ["compile", "hosted"])
def test_benchmark_cli_rejects_cloud_stage_without_submit_flag(
    through: str,
    tmp_path: Path,
) -> None:
    """Verify cloud stop points fail before dataset or network access.

    Args:
        through: Cloud-backed terminal benchmark stage.
        tmp_path: Isolated nonexistent dataset and build paths.

    Returns:
        None.
    """
    with pytest.raises(ValueError, match="--submit-cloud"):
        main(
            [
                "benchmark-vlsp",
                "--model",
                "zipformer",
                "--dataset-root",
                str(tmp_path / "missing-vlsp"),
                "--through",
                through,
                "--dry-run",
            ]
        )


def test_qdq_graph_contract_rejects_quantization_outside_policy(tmp_path: Path) -> None:
    """Verify benchmark QDQ inspection rejects an unselected quantized MatMul.

    Args:
        tmp_path: Isolated ONNX and policy paths.

    Returns:
        None.
    """
    nodes = [
        helper.make_node("DequantizeLinear", ["x", "scale", "zero"], ["x-dq"], name="dq"),
        helper.make_node("MatMul", ["x-dq", "weight"], ["selected-output"], name="selected"),
        helper.make_node("DequantizeLinear", ["other", "scale", "zero"], ["other-dq"], name="other-dq"),
        helper.make_node("MatMul", ["other-dq", "weight"], ["other-output"], name="other"),
    ]
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "qdq",
            [
                helper.make_tensor_value_info("x", TensorProto.INT8, [1, 1]),
                helper.make_tensor_value_info("other", TensorProto.INT8, [1, 1]),
                helper.make_tensor_value_info("scale", TensorProto.FLOAT, []),
                helper.make_tensor_value_info("zero", TensorProto.INT8, []),
                helper.make_tensor_value_info("weight", TensorProto.FLOAT, [1, 1]),
            ],
            [helper.make_tensor_value_info("other-output", TensorProto.FLOAT, [1, 1])],
        )
    )
    model_path = tmp_path / "model.qdq.onnx"
    onnx.save(model, model_path)

    with pytest.raises(ValueError, match="outside the policy"):
        inspect_benchmark_qdq(
            model_path,
            selected_matmul_names=("selected",),
            expected_total_matmul=2,
        )


def test_model_quality_gates_match_historical_acceptance_contracts() -> None:
    """Verify local quality gates use the documented Zipformer and VPCD limits.

    Returns:
        None.
    """
    zipformer = evaluate_zipformer_quality_gate(
        fp32_cer=0.10,
        qdq_cer=0.109,
        fp32_wer=0.20,
        qdq_wer=0.219,
        empty_outputs=0,
        collapse_outputs=0,
    )
    vpcd = evaluate_vpcd_quality_gate(
        sample_count=100,
        exact_output_matches=95,
        first_five_top1_matches=500,
        first_five_step_count=500,
        early_eos_count=0,
        collapse_count=0,
    )

    assert zipformer["passed"] is True
    assert zipformer["cer_increase_percentage_points"] == pytest.approx(0.9)
    assert vpcd["passed"] is True


def test_full_benchmark_workflow_resumes_and_limits_hosted_inputs(tmp_path: Path) -> None:
    """Verify fake local-to-hosted workflow is deterministic and checksum-resumable.

    Args:
        tmp_path: Isolated benchmark build root.

    Returns:
        None.
    """
    class FakeBackend:
        def __init__(self, source_version: str = "one") -> None:
            """Initialize a call log for fake benchmark operations.

            Args:
                source_version: Fake model-source identity used for cache invalidation.

            Returns:
                None.
            """
            self.calls: list[tuple[str, object]] = []
            self.source_version = source_version

        def local_input_digest(
            self,
            *,
            model: str,
            dataset: dict,
            providers: tuple[str, ...],
        ) -> str:
            """Return a fake model-source digest.

            Args:
                model: Canonical model family.
                dataset: Materialized dataset evidence.
                providers: Requested local providers.

            Returns:
                Source identity including model, dataset, providers, and version.
            """
            return "|".join(
                (model, dataset["dataset_checksum"], *providers, self.source_version)
            )

        def materialize_dataset(self, *, dataset_root: Path, output_dir: Path):
            """Create deterministic fake VLSP evidence.

            Args:
                dataset_root: Ignored source root.
                output_dir: Step-owned output directory.

            Returns:
                Portable fake dataset evidence.
            """
            self.calls.append(("dataset", dataset_root.as_posix()))
            manifest = output_dir / "manifest.json"
            manifest.write_text('{"calibration_count":24,"evaluation_count":100}\n')
            return {"manifest": manifest.name, "dataset_checksum": "dataset-sha"}

        def run_local(self, *, model: str, dataset: dict, providers: tuple[str, ...], output_dir: Path):
            """Create passing fake local evidence for one model.

            Args:
                model: Model family under evaluation.
                dataset: Materialized dataset evidence.
                providers: Requested local provider names.
                output_dir: Step-owned output directory.

            Returns:
                Passing local evidence and canonical compile-source checksum.
            """
            self.calls.append(("local", model))
            evidence = output_dir / "local.json"
            evidence.write_text("{}\n")
            return {
                "quality_passed": True,
                "compile_source_checksum": f"{model}-source",
                "providers": list(providers),
                "evidence": evidence.name,
                "dataset_checksum": dataset["dataset_checksum"],
            }

        def compile(self, *, model: str, local: dict, output_dir: Path):
            """Create passing fake compile evidence.

            Args:
                model: Model family being compiled.
                local: Passing local evidence.
                output_dir: Step-owned output directory.

            Returns:
                Fake EPContext compile evidence.
            """
            self.calls.append(("compile", model))
            evidence = output_dir / "compile.json"
            evidence.write_text("{}\n")
            return {
                "compile_job_id": f"{model}-compile",
                "compile_source_checksum": local["compile_source_checksum"],
                "representation": "onnx-epcontext",
                "evidence": evidence.name,
            }

        def hosted_validate(
            self,
            *,
            model: str,
            local: dict,
            compiled: dict,
            input_limit: int,
            output_dir: Path,
        ):
            """Create passing fake hosted evidence while recording the quota.

            Args:
                model: Model family under validation.
                local: Passing local evidence.
                compiled: Compiled target evidence.
                input_limit: Hard hosted input quota.
                output_dir: Step-owned output directory.

            Returns:
                Fake five-input parity evidence.
            """
            del local, compiled
            self.calls.append(("hosted", (model, input_limit)))
            evidence = output_dir / "hosted.json"
            evidence.write_text("{}\n")
            return {
                "input_count": input_limit,
                "parity_matches": input_limit,
                "quality_passed": True,
                "evidence": evidence.name,
            }

    request = VlspBenchmarkRequest(
        model="all",
        dataset_root=tmp_path / "vlsp",
        build_root=tmp_path / "benchmark",
        providers=("cpu", "cuda"),
        through="hosted",
        submit_cloud=True,
        device="Samsung Galaxy S23 (Family)",
        qairt_version="2.45",
    )
    first_backend = FakeBackend()

    first = run_vlsp_benchmark(request, repo_root=tmp_path, backend=first_backend)

    assert first["status"] == "passed"
    assert first_backend.calls == [
        ("dataset", (tmp_path / "vlsp").as_posix()),
        ("local", "zipformer"),
        ("compile", "zipformer"),
        ("hosted", ("zipformer", 5)),
        ("local", "vpcd"),
        ("compile", "vpcd"),
        ("hosted", ("vpcd", 5)),
    ]
    assert (request.build_root / "comparison.json").is_file()
    environment = json.loads((request.build_root / "environment.json").read_text(encoding="utf-8"))
    assert "python_version" in environment
    assert "onnx_runtime" in environment
    assert "current_working_directory" not in environment

    resumed_backend = FakeBackend()
    second = run_vlsp_benchmark(request, repo_root=tmp_path, backend=resumed_backend)

    assert second == first
    assert resumed_backend.calls == []

    changed_backend = FakeBackend(source_version="two")
    run_vlsp_benchmark(request, repo_root=tmp_path, backend=changed_backend)
    assert changed_backend.calls == [
        ("local", "zipformer"),
        ("compile", "zipformer"),
        ("hosted", ("zipformer", 5)),
        ("local", "vpcd"),
        ("compile", "vpcd"),
        ("hosted", ("vpcd", 5)),
    ]


def test_production_backend_materializes_canonical_24_100_split(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify production dataset materialization locks canonical counts and shards.

    Args:
        tmp_path: Isolated repository and benchmark roots.
        monkeypatch: Pytest fixture replacing parquet decoding and audio probing.

    Returns:
        None.
    """
    calibration = [
        VlspRow("train-00000.parquet", index, f"c-{index}.wav", b"c", f"calibration {index}")
        for index in range(24)
    ]
    evaluation = [
        VlspRow(
            "train-00001.parquet",
            index,
            f"e-{index}.wav",
            b"e",
            f"mot hai ba bon {index}",
        )
        for index in range(100)
    ]
    monkeypatch.setattr(
        "model_pipeline.benchmarks.vlsp_runtime.iter_vlsp_rows",
        lambda _root: iter((*calibration, *evaluation)),
    )
    monkeypatch.setattr(
        "model_pipeline.benchmarks.vlsp_runtime.probe_audio",
        lambda _row: AudioInfo(3.0, 16_000),
    )
    request = VlspBenchmarkRequest(
        model="zipformer",
        dataset_root=tmp_path / "vlsp",
        build_root=tmp_path / "build",
        providers=("cpu",),
        through="local",
    )
    backend = ProductionVlspBenchmarkBackend(request=request, repo_root=tmp_path)

    result = backend.materialize_dataset(
        dataset_root=request.dataset_root,
        output_dir=request.build_root / ".state" / "dataset",
    )

    manifest = tmp_path / result["manifest"]
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["calibration_count"] == 24
    assert payload["evaluation_count"] == 100
    assert result["calibration_shards"] == ["train-00000.parquet"]
    assert result["evaluation_shards"] == ["train-00001.parquet"]


def test_benchmark_step_cache_invalidates_changed_inputs_and_outputs(tmp_path: Path) -> None:
    """Verify step resume requires both matching input digest and output bytes.

    Args:
        tmp_path: Isolated benchmark step cache.

    Returns:
        None.
    """
    runner = BenchmarkStepRunner(tmp_path)
    executions: list[str] = []

    external = tmp_path / "pipeline" / "model.onnx"

    def execute(output_dir: Path):
        """Write one tracked evidence file and record execution.

        Args:
            output_dir: Step-owned evidence directory.

        Returns:
            Deterministic fake step result.
        """
        executions.append("run")
        (output_dir / "evidence.json").write_text("{}\n", encoding="utf-8")
        external.parent.mkdir(parents=True, exist_ok=True)
        external.write_text(f"model-{len(executions)}\n", encoding="utf-8")
        checksum = hashlib.sha256(external.read_bytes()).hexdigest()
        return {
            "value": len(executions),
            "_resume_artifacts": {"pipeline/model.onnx": checksum},
        }

    first = runner.run(name="model/local", input_digest="input-one", execute=execute)
    resumed = runner.run(name="model/local", input_digest="input-one", execute=execute)
    (tmp_path / "model/local/evidence.json").write_text("changed\n", encoding="utf-8")
    output_changed = runner.run(name="model/local", input_digest="input-one", execute=execute)
    external.write_text("corrupted\n", encoding="utf-8")
    external_changed = runner.run(name="model/local", input_digest="input-one", execute=execute)
    input_changed = runner.run(name="model/local", input_digest="input-two", execute=execute)

    assert first == resumed == {"value": 1}
    assert output_changed == {"value": 2}
    assert external_changed == {"value": 3}
    assert input_changed == {"value": 4}
