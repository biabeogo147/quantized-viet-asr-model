from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from model_pipeline.core import RecipeSpec, ValidationResult
from model_pipeline.models.onnx_tools import freeze_input_shapes
from model_pipeline.models.zipformer.graph import (
    BOOLEAN_MASK_SLICE_NODES,
    BOOLEAN_MASK_UNSQUEEZE_NODES,
    graph_matmul_count,
    prepare_encoder_for_aihub,
)
from model_pipeline.models.base import CompileInput
from model_pipeline.models.aimet import (
    build_matmul_only_aimet_config,
    write_aimet_calibration_inputs,
)
from model_pipeline.models.zipformer.quantization import (
    build_zipformer_encoder_matmul_policy,
    build_zipformer_calibration_inputs,
    inspect_zipformer_qdq_coverage,
    quantize_zipformer_encoder_ortqnn,
)


class ZipformerAdapter:
    def __init__(
        self,
        repo_root: str | Path,
        *,
        calibration_inputs: Sequence[Mapping[str, np.ndarray]] | None = None,
        calibration_manifest: str | Path | None = None,
        aimet_service=None,
    ):
        """Initialize local and sibling-Android Zipformer source locations.

        Args:
            repo_root: Root of the Python model repository.
            calibration_inputs: Optional injected fixed-shape calibration batches.
            calibration_manifest: Optional portable VLSP manifest overriding the default.
            aimet_service: Optional generic AIMET service client.

        Returns:
            None.
        """
        self.repo_root = Path(repo_root).resolve()
        self.model_dir = self.repo_root / "assets" / "zipformer"
        self.android_model_dir = (
            self.repo_root.parent
            / "BKMeeting"
            / "modelassets"
            / "src"
            / "main"
            / "assets"
            / "models"
            / "asr"
            / "zipformer"
            / "fp32"
        )
        self.calibration_inputs = calibration_inputs
        self.aimet_service = aimet_service
        self.calibration_manifest = (
            Path(calibration_manifest).resolve()
            if calibration_manifest is not None
            else self.repo_root
            / "build"
            / "datasets"
            / "vlsp"
            / "vlsp-calibration-evaluation-manifest.json"
        )

    def source_files(self, recipe: RecipeSpec) -> Mapping[str, Path]:
        """Resolve local Zipformer sources with a tracked Android fallback.

        Args:
            recipe: Recipe selecting whether calibration sources are required.

        Returns:
            Encoder, decoder, joiner, and token paths by component role.
        """
        local = {
            "encoder": self.model_dir / "encoder-epoch-20-avg-1.onnx",
            "decoder": self.model_dir / "decoder-epoch-20-avg-1.onnx",
            "joiner": self.model_dir / "joiner-epoch-20-avg-1.onnx",
            "tokens": self.model_dir / "tokens.txt",
        }
        sources = (
            local
            if all(path.is_file() for path in local.values())
            else {
                "encoder": self.android_model_dir / "encoder.onnx",
                "decoder": self.android_model_dir / "decoder.onnx",
                "joiner": self.android_model_dir / "joiner.onnx",
                "tokens": self.android_model_dir / "tokens.txt",
            }
        )
        if recipe.parameters["quantize_action"] != "explicit-skip" and self.calibration_inputs is None:
            import json

            payload = json.loads(self.calibration_manifest.read_text(encoding="utf-8"))
            sources["calibration_manifest"] = self.calibration_manifest
            for index, sample in enumerate(payload.get("samples", ())):
                if sample.get("partition") == "calibration":
                    sources[f"calibration_audio_{index:03d}"] = (
                        self.calibration_manifest.parent / str(sample["audio_path"])
                    )
        return sources

    def prepare(self, recipe: RecipeSpec, sources: Mapping[str, Path], output_dir: Path):
        """Freeze component shapes and prepare the encoder for Qualcomm HTP.

        Args:
            recipe: Zipformer configuration and fixed-shape contract.
            sources: Resolved model and token source files.
            output_dir: Stage directory for prepared components.

        Returns:
            Prepared encoder, decoder, joiner, and token files.
        """
        shapes = recipe.parameters["fixed_input_shapes"]
        fixed_encoder = freeze_input_shapes(
            sources["encoder"], output_dir / "encoder.fixed.source.onnx", shapes["encoder"]
        )
        encoder = output_dir / "encoder.onnx"
        if recipe.parameters["prepare_scope"] == "encoder":
            prepare_encoder_for_aihub(fixed_encoder, encoder)
        else:
            shutil.copyfile(fixed_encoder, encoder)
        fixed_encoder.unlink(missing_ok=True)
        decoder = freeze_input_shapes(sources["decoder"], output_dir / "decoder.onnx", shapes["decoder"])
        joiner = freeze_input_shapes(sources["joiner"], output_dir / "joiner.onnx", shapes["joiner"])
        tokens = output_dir / "tokens.txt"
        shutil.copyfile(sources["tokens"], tokens)
        return {"encoder": encoder, "decoder": decoder, "joiner": joiner, "tokens": tokens}

    def quantize(self, recipe: RecipeSpec, prepared: Mapping[str, Path], output_dir: Path):
        """Copy FP32 components or quantize only the prepared encoder.

        Args:
            recipe: Zipformer precision and quantization configuration.
            prepared: Prepared component files.
            output_dir: Stage directory for quantized components.

        Returns:
            Copied FP32 components by role.

        Raises:
            ValueError: If required calibration inputs are unavailable.
        """
        action = recipe.parameters["quantize_action"]
        if action == "explicit-skip":
            return _copy_components(prepared, output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = _copy_components(
            {role: path for role, path in prepared.items() if role != "encoder"},
            output_dir,
        )
        if action == "ortqnn":
            calibration_inputs = self.calibration_inputs
            if calibration_inputs is None:
                calibration_inputs = build_zipformer_calibration_inputs(
                    self.calibration_manifest
                )
            outputs["encoder"] = quantize_zipformer_encoder_ortqnn(
                prepared["encoder"],
                output_dir / "encoder.onnx",
                calibration_inputs,
            )
            return outputs
        if action == "aimet":
            if self.aimet_service is None:
                raise ValueError("Zipformer AIMET service is required")
            calibration_inputs = self.calibration_inputs
            if calibration_inputs is None:
                calibration_inputs = build_zipformer_calibration_inputs(
                    self.calibration_manifest
                )
            calibration_manifest = write_aimet_calibration_inputs(
                calibration_inputs,
                output_dir / "calibration",
            )
            config_path = output_dir / "aimet-config.json"
            config_path.write_text(
                json.dumps(
                    build_matmul_only_aimet_config(
                        select_operators_from_policy=True,
                    ),
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            policy_path = output_dir / "quantization-policy.json"
            policy_path.write_text(
                json.dumps(
                    build_zipformer_encoder_matmul_policy(prepared["encoder"]),
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            aimet_dir = output_dir / "aimet"
            self.aimet_service.healthcheck()
            self.aimet_service.export(
                fp32_model_path=prepared["encoder"],
                calibration_dir=calibration_manifest.parent,
                output_dir=aimet_dir,
                config_path=config_path,
                policy_path=policy_path,
            )
            encoder = aimet_dir / "model.onnx"
            encodings = aimet_dir / "model.encodings"
            if not encoder.is_file() or not encodings.is_file():
                raise FileNotFoundError("AIMET service did not export Zipformer encoder package")
            outputs.update(
                {
                    "encoder": encoder,
                    "encodings": encodings,
                    "aimet_config": config_path,
                    "quantization_policy": policy_path,
                    "calibration_manifest": calibration_manifest,
                }
            )
            external_data = aimet_dir / "model.onnx.data"
            if external_data.is_file():
                outputs["external_data"] = external_data
            return outputs
        raise ValueError(f"Unsupported Zipformer quantization action: {action!r}")

    def validate(self, recipe: RecipeSpec, quantized_components: Mapping[str, Path]) -> ValidationResult:
        """Validate Zipformer MatMul inventory and encoder mask rewrites.

        Args:
            recipe: Recipe containing expected graph contracts.
            quantized_components: Quantized component files by role.

        Returns:
            Structured graph and precision validation evidence.
        """
        counts = {
            role: graph_matmul_count(quantized_components[role])
            for role in ("encoder", "decoder", "joiner")
        }
        expected = dict(recipe.parameters["matmul_contract"])
        quantization_engine = recipe.artifact.quantization.engine
        checks: dict[str, bool | int | str] = {
            "encoder_matmul": counts["encoder"],
            "decoder_matmul": counts["decoder"],
            "joiner_matmul": counts["joiner"],
            "matmul_contract": counts == expected,
            "quantization": quantization_engine,
        }
        if quantization_engine == "ortqnn":
            qdq_inventory = inspect_zipformer_qdq_coverage(
                quantized_components["encoder"]
            )
            checks["quantized_encoder_matmul"] = qdq_inventory.quantized_matmul_count
            checks["qdq_contract"] = (
                qdq_inventory.matmul_count == expected["encoder"]
                and qdq_inventory.quantized_matmul_count == expected["encoder"]
                and not qdq_inventory.unquantized_matmul_names
            )
        if quantization_engine == "aimet":
            policy = json.loads(
                quantized_components["quantization_policy"].read_text(encoding="utf-8")
            )
            checks["aimet_policy_contract"] = (
                policy.get("quantization_scope") == "encoder-matmul"
                and policy.get("quantize_op_types") == ["MatMul"]
                and policy.get("disable_op_names") == []
                and policy.get("coverage")
                == {"quantized": 278, "total_matmul": 278}
                and quantized_components["encodings"].is_file()
            )
        if recipe.parameters["prepare_scope"] == "encoder":
            import onnx

            model = onnx.load(quantized_components["encoder"].as_posix(), load_external_data=False)
            names = {node.name for node in model.graph.node}
            checks["boolean_mask_casts"] = (
                "/GreaterOrEqual_output_0_u8_cast" in names
                and all(f"{name}_cast_bool" in names for name in BOOLEAN_MASK_UNSQUEEZE_NODES)
                and all(name in names for name in BOOLEAN_MASK_SLICE_NODES)
            )
        passed = all(value is True for key, value in checks.items() if key.endswith("contract") or key.endswith("casts"))
        return ValidationResult("passed" if passed else "failed", checks)

    def compile_inputs(self, recipe: RecipeSpec, validated_components: Mapping[str, Path]):
        """Select only the validated encoder for AI Hub compilation.

        Args:
            recipe: Recipe containing the compilation contract.
            validated_components: Validated component files.

        Returns:
            An empty list for local controls or one encoder compile input.
        """
        if recipe.artifact.compilation.compiler == "none":
            return []
        source_path = validated_components["encoder"]
        if recipe.artifact.quantization.engine == "aimet":
            source_path = source_path.parent
        return [
            CompileInput(
                "encoder",
                source_path,
                recipe.parameters["fixed_input_shapes"]["encoder"],
                True,
                {"x": "float32", "x_lens": "int64"},
            )
        ]

    def bundle_components(self, recipe, validated_components, compiled_components):
        """Describe Zipformer bundle files and per-component execution truth.

        Args:
            recipe: Recipe accepted for adapter protocol consistency.
            validated_components: Validated local component files.
            compiled_components: Compiled encoder and optional external data.

        Returns:
            Bundle component tuples keyed by runtime role.
        """
        del recipe
        encoder = compiled_components.get("encoder", validated_components["encoder"])
        result = {
            "encoder": (
                encoder,
                "qnn-htp" if "encoder" in compiled_components else "cpu",
                "onnx-epcontext" if "encoder" in compiled_components else "onnx",
            ),
            "decoder": (validated_components["decoder"], "cpu", "onnx"),
            "joiner": (validated_components["joiner"], "cpu", "onnx"),
            "tokens": (validated_components["tokens"], "cpu", "text"),
        }
        if "encoder_external_data" in compiled_components:
            result["encoder_external_data"] = (
                compiled_components["encoder_external_data"], "qnn-htp", "onnx-external-data"
            )
        return result


def _copy_components(components: Mapping[str, Path], output_dir: Path) -> dict[str, Path]:
    """Copy component files into a stage-owned output directory.

    Args:
        components: Source files keyed by logical role.
        output_dir: Destination directory owned by the current stage.

    Returns:
        Copied files keyed by their original roles.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for role, source in components.items():
        destination = output_dir / source.name
        shutil.copyfile(source, destination)
        outputs[role] = destination
    return outputs
