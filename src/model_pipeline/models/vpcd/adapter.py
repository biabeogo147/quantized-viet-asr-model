from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Mapping

from model_pipeline.core import RecipeSpec, ValidationResult
from model_pipeline.models.base import CompileInput
from model_pipeline.models.aimet import (
    build_matmul_only_aimet_config,
    write_aimet_calibration_inputs,
)
from model_pipeline.models.onnx_tools import freeze_input_shapes
from model_pipeline.models.vpcd.calibration import build_calibration_batches
from model_pipeline.models.vpcd.graph import (
    inspect_vpcd_matmuls,
    rewrite_encoder_attention_mask_boolean_casts_for_qnn,
)
from model_pipeline.models.vpcd.quantization import (
    build_encoder_matmul_policy,
    inspect_encoder_matmul_aimet_encodings,
)
from model_pipeline.models.vpcd.tokenizer import export_tokenizer
from model_pipeline.models.aimet_service import AimetServiceClient


class VpcdAdapter:
    def __init__(
        self,
        repo_root: str | Path,
        calibration_text: str | Path | None = None,
        aimet_service: AimetServiceClient | None = None,
    ):
        """Initialize VPCD sources, calibration data, and AIMET service access.

        Args:
            repo_root: Root of the Python model repository.
            calibration_text: Optional calibration source overriding repository defaults.
            aimet_service: Optional injected service client for testing or custom execution.

        Returns:
            None.
        """
        self.repo_root = Path(repo_root).resolve()
        self.model_dir = self.repo_root / "assets" / "vietnamese-punc-cap-denorm-v1"
        self.android_model_dir = (
            self.repo_root.parent
            / "BKMeeting"
            / "modelassets"
            / "src"
            / "main"
            / "assets"
            / "models"
            / "punctuation"
            / "vpcd"
            / "fp32"
        )
        local_fp32_model = self.model_dir / "onnx" / "model.fp32.onnx"
        self.fp32_model = (
            local_fp32_model if local_fp32_model.is_file() else self.android_model_dir / "model.mobile.onnx"
        )
        default_calibration = self.repo_root / "build" / "calibration" / "vlsp2020" / "transcriptions.txt"
        if not default_calibration.is_file():
            default_calibration = self.repo_root / "assets" / "punctuation" / "default_golden_samples.jsonl"
        self.calibration_text = (
            Path(calibration_text).resolve()
            if calibration_text is not None
            else default_calibration
        )
        self.aimet_service = aimet_service or AimetServiceClient(
            repo_root=self.repo_root,
            url=os.environ.get("AIMET_SERVICE_URL", "http://127.0.0.1:18080"),
        )

    def source_files(self, recipe: RecipeSpec) -> Mapping[str, Path]:
        """Resolve VPCD model, tokenizer, and optional calibration sources.

        Args:
            recipe: Canonical configuration selecting whether calibration is required.

        Returns:
            Source roles mapped to local or tracked Android fallback files.
        """
        native = {
            "tokenizer_model": self.model_dir / "sentencepiece.bpe.model",
            "tokenizer_config": self.model_dir / "tokenizer_config.json",
            "special_tokens": self.model_dir / "special_tokens_map.json",
            "vocabulary": self.model_dir / "dict.txt",
            "generation_config": self.model_dir / "generation_config.json",
            "model_config": self.model_dir / "config.json",
        }
        sources = {"model": self.fp32_model}
        if all(path.is_file() for path in native.values()):
            sources.update(native)
        else:
            sources.update(
                {
                    "tokenizer_encode": self.android_model_dir / "tokenizer.encode.onnx",
                    "tokenizer_decode": self.android_model_dir / "tokenizer.decode.onnx",
                    "tokenizer_to_model_id_map": self.android_model_dir / "tokenizer.to_model_id_map.json",
                    "model_to_tokenizer_id_map": self.android_model_dir / "tokenizer.from_model_id_map.json",
                }
            )
        if recipe.parameters["quantize_action"] == "aimet":
            sources["calibration_text"] = self.calibration_text
        return sources

    def prepare(self, recipe: RecipeSpec, sources: Mapping[str, Path], output_dir: Path):
        """Freeze model shapes and materialize CPU tokenizer/runtime artifacts.

        Args:
            recipe: VPCD recipe containing the fixed 384-by-64 shape contract.
            sources: Resolved model and tokenizer sources.
            output_dir: Stage directory for prepared artifacts.

        Returns:
            Fixed model, tokenizer bridges, and autoregressive-loop contract.
        """
        fixed_before_qnn_rewrite = freeze_input_shapes(
            sources["model"],
            output_dir / "model.fp32.fixed.before-qnn-boolean-cast-rewrite.onnx",
            recipe.parameters["fixed_input_shapes"],
        )
        model = output_dir / "model.fp32.fixed.onnx"
        rewrite_encoder_attention_mask_boolean_casts_for_qnn(
            fixed_before_qnn_rewrite,
            model,
        )
        fixed_before_qnn_rewrite.unlink()
        tokenizer_dir = output_dir / "tokenizer"
        if "tokenizer_encode" in sources:
            tokenizer_dir.mkdir(parents=True, exist_ok=True)
            tokenizer_paths = {}
            for role in (
                "tokenizer_encode",
                "tokenizer_decode",
                "tokenizer_to_model_id_map",
                "model_to_tokenizer_id_map",
            ):
                destination = tokenizer_dir / sources[role].name
                shutil.copyfile(sources[role], destination)
                tokenizer_paths[role] = destination
        else:
            tokenizer = export_tokenizer(self.model_dir, tokenizer_dir)
            tokenizer_paths = {
                "tokenizer_encode": tokenizer.encode,
                "tokenizer_decode": tokenizer.decode,
                "tokenizer_to_model_id_map": tokenizer.to_model_ids,
                "model_to_tokenizer_id_map": tokenizer.from_model_ids,
            }
        runtime = output_dir / "autoregressive-loop.json"
        runtime.write_text(
            json.dumps(
                {
                    "execution_target": "cpu",
                    "algorithm": "greedy-autoregressive",
                    "source_length": 384,
                    "decoder_length": 64,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "model": model,
            **tokenizer_paths,
            "autoregressive_loop": runtime,
        }

    def quantize(self, recipe: RecipeSpec, prepared: Mapping[str, Path], output_dir: Path):
        """Copy the FP32 control or export the AIMET quantized package.

        Args:
            recipe: VPCD fixed-shape or AIMET quantized recipe.
            prepared: Fixed model and CPU support artifacts.
            output_dir: Stage directory for quantized artifacts and evidence.

        Returns:
            Quantized model, support artifacts, and quantization evidence.

        Raises:
            FileNotFoundError: If AIMET does not materialize required package files.
        """
        support = {
            role: path for role, path in prepared.items() if role != "model"
        }
        copied_support = _copy_components(support, output_dir / "support")
        if recipe.parameters["quantize_action"] == "explicit-skip":
            model = output_dir / "model.fp32.fixed.onnx"
            shutil.copyfile(prepared["model"], model)
            return {"model": model, **copied_support}

        calibration, stats = build_calibration_batches(
            model_dir=self.model_dir,
            fp32_model_path=self.fp32_model,
            text_source=self.calibration_text,
            tokenizer_encode_path=prepared["tokenizer_encode"],
            tokenizer_to_model_ids_path=prepared["tokenizer_to_model_id_map"],
        )
        calibration_manifest = write_aimet_calibration_inputs(
            [batch.inputs for batch in calibration],
            output_dir / "calibration",
        )
        config_path = output_dir / "aimet-config.json"
        config_path.write_text(
            json.dumps(
                build_matmul_only_aimet_config(
                    select_operators_from_policy=True,
                    symmetric_activations=True,
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        policy = build_encoder_matmul_policy(prepared["model"])
        policy_path = output_dir / "quantization-policy.json"
        policy_path.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        stats_path = output_dir / "calibration-summary.json"
        stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        aimet_dir = output_dir / "aimet"
        self.aimet_service.healthcheck()
        self.aimet_service.export(
            fp32_model_path=prepared["model"],
            calibration_dir=calibration_manifest.parent,
            output_dir=aimet_dir,
            config_path=config_path,
            policy_path=policy_path,
        )
        aimet = {"model": aimet_dir / "model.onnx", "encodings": aimet_dir / "model.encodings"}
        if not all(path.is_file() for path in aimet.values()):
            raise FileNotFoundError(f"AIMET service did not materialize the expected package: {aimet!r}")
        external_data = aimet_dir / "model.onnx.data"
        if external_data.is_file():
            aimet["external_data"] = external_data
        return {
            "model": aimet["model"],
            "encodings": aimet["encodings"],
            **({"external_data": aimet["external_data"]} if "external_data" in aimet else {}),
            "aimet_config": config_path,
            "quantization_policy": policy_path,
            "calibration_manifest": calibration_manifest,
            "calibration_summary": stats_path,
            **copied_support,
        }

    def validate(self, recipe: RecipeSpec, quantized_components: Mapping[str, Path]) -> ValidationResult:
        """Validate canonical MatMul inventory and encoder-only policy coverage.

        Args:
            recipe: Recipe containing expected graph and quantization contracts.
            quantized_components: Quantized model and policy files.

        Returns:
            Structured graph, policy, and CPU-host execution evidence.
        """
        inventory = inspect_vpcd_matmuls(quantized_components["model"])
        counts = inventory.counts
        expected = recipe.parameters["matmul_contract"]
        coverage_ok = (counts["encoder"], counts["decoder"], counts["lm_head"], counts["other"]) == (
            expected["encoder"], expected["decoder"], expected["lm_head"], 0
        )
        policy_ok = True
        encoding_ok = True
        encoding_checks: dict[str, object] = {}
        if recipe.parameters["quantize_action"] == "aimet":
            policy = json.loads(
                quantized_components["quantization_policy"].read_text(encoding="utf-8")
            )
            policy_ok = (
                policy["coverage"] == {"quantized": 96, "total_matmul": 265}
                and len(policy["disable_op_names"]) == 169
                and policy["quantizer_selection"] == "operator-name-allowlist"
                and policy["symmetric_activation_encodings"] is True
            )
            encoding_checks = inspect_encoder_matmul_aimet_encodings(
                quantized_components["encodings"]
            )
            encoding_ok = (
                encoding_checks["activation_count"] == 168
                and encoding_checks["parameter_count"] == 72
                and encoding_checks["activation_contract"] is True
                and encoding_checks["parameter_contract"] is True
                and not encoding_checks["non_encoder_names"]
            )
        checks = {
            "encoder_matmul": counts["encoder"],
            "decoder_matmul": counts["decoder"],
            "lm_head_matmul": counts["lm_head"],
            "graph_contract": coverage_ok,
            "encoder_matmul_policy": policy_ok,
            "encoder_matmul_encodings": encoding_ok,
            **{f"encoding_{name}": value for name, value in encoding_checks.items()},
            "tokenizer_execution_target": "cpu",
            "autoregressive_execution_target": "cpu",
        }
        return ValidationResult(
            "passed" if coverage_ok and policy_ok and encoding_ok else "failed",
            checks,
        )

    def compile_inputs(self, recipe: RecipeSpec, validated_components: Mapping[str, Path]):
        """Describe the whole VPCD package required by AI Hub compilation.

        Args:
            recipe: Recipe containing compilation and fixed I/O contracts.
            validated_components: Validated AIMET or FP32 files.

        Returns:
            An empty control list or one package-level compile input.
        """
        if recipe.artifact.compilation.compiler == "none":
            return []
        return [
            CompileInput(
                "model",
                validated_components["model"].parent,
                recipe.parameters["fixed_input_shapes"],
                True,
                {name: "int64" for name in recipe.parameters["fixed_input_shapes"]},
            )
        ]

    def bundle_components(self, recipe, validated_components, compiled_components):
        """Describe model and CPU support artifacts for manifest-v2 packaging.

        Args:
            recipe: Recipe accepted for adapter protocol consistency.
            validated_components: Validated local model and support files.
            compiled_components: Compiled model and optional external data.

        Returns:
            Component files with explicit execution targets and formats.
        """
        del recipe
        model = compiled_components.get("model", validated_components["model"])
        result = {
            "model": (
                model,
                "qnn-htp" if "model" in compiled_components else "cpu",
                "onnx-epcontext" if "model" in compiled_components else "onnx",
            )
        }
        formats = {
            "tokenizer_encode": "onnx",
            "tokenizer_decode": "onnx",
            "tokenizer_to_model_id_map": "json",
            "model_to_tokenizer_id_map": "json",
            "autoregressive_loop": "host-runtime-contract",
        }
        for role, file_format in formats.items():
            result[role] = (validated_components[role], "cpu", file_format)
        if "model_external_data" in compiled_components:
            result["model_external_data"] = (
                compiled_components["model_external_data"], "qnn-htp", "onnx-external-data"
            )
        return result


def _copy_components(components: Mapping[str, Path], output_dir: Path) -> dict[str, Path]:
    """Copy CPU support components into a stage-owned directory.

    Args:
        components: Source support files keyed by role.
        output_dir: Destination directory owned by the current stage.

    Returns:
        Copied support paths keyed by their original roles.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for role, source in components.items():
        destination = output_dir / source.name
        shutil.copyfile(source, destination)
        outputs[role] = destination
    return outputs
