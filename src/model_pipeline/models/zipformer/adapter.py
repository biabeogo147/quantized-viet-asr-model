from __future__ import annotations

import shutil
from pathlib import Path
from typing import Mapping

from model_pipeline.core import RecipeSpec, ValidationResult
from model_pipeline.models.onnx_tools import freeze_input_shapes
from model_pipeline.models.zipformer.graph import (
    BOOLEAN_MASK_SLICE_NODES,
    BOOLEAN_MASK_UNSQUEEZE_NODES,
    graph_matmul_count,
    prepare_encoder_for_aihub,
)
from model_pipeline.models.base import CompileInput


class ZipformerAdapter:
    def __init__(self, repo_root: str | Path):
        """Initialize local and sibling-Android Zipformer source locations.

        Args:
            repo_root: Root of the Python model repository.

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

    def source_files(self, recipe: RecipeSpec) -> Mapping[str, Path]:
        """Resolve local Zipformer sources with a tracked Android fallback.

        Args:
            recipe: Recipe accepted for protocol consistency; it does not alter sources.

        Returns:
            Encoder, decoder, joiner, and token paths by component role.
        """
        del recipe
        local = {
            "encoder": self.model_dir / "encoder-epoch-20-avg-1.onnx",
            "decoder": self.model_dir / "decoder-epoch-20-avg-1.onnx",
            "joiner": self.model_dir / "joiner-epoch-20-avg-1.onnx",
            "tokens": self.model_dir / "tokens.txt",
        }
        if all(path.is_file() for path in local.values()):
            return local
        return {
            "encoder": self.android_model_dir / "encoder.onnx",
            "decoder": self.android_model_dir / "decoder.onnx",
            "joiner": self.android_model_dir / "joiner.onnx",
            "tokens": self.android_model_dir / "tokens.txt",
        }

    def prepare(self, recipe: RecipeSpec, sources: Mapping[str, Path], output_dir: Path):
        """Freeze component shapes and prepare the production encoder for HTP.

        Args:
            recipe: Zipformer profile and fixed-shape contract.
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
        if recipe.profile == "production":
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
        """Copy prepared FP32 components through the explicit quantization skip.

        Args:
            recipe: Zipformer recipe that must declare `explicit-skip`.
            prepared: Prepared component files.
            output_dir: Stage directory for candidate components.

        Returns:
            Copied FP32 candidates by component role.

        Raises:
            ValueError: If a recipe attempts local Zipformer quantization.
        """
        if recipe.parameters["quantize_action"] != "explicit-skip":
            raise ValueError("Zipformer has no local quantization stage")
        return _copy_components(prepared, output_dir)

    def validate(self, recipe: RecipeSpec, candidate: Mapping[str, Path]) -> ValidationResult:
        """Validate Zipformer MatMul inventory and production mask rewrites.

        Args:
            recipe: Recipe containing expected graph contracts.
            candidate: Candidate component files by role.

        Returns:
            Structured graph and precision validation evidence.
        """
        counts = {role: graph_matmul_count(candidate[role]) for role in ("encoder", "decoder", "joiner")}
        expected = dict(recipe.parameters["matmul_contract"])
        checks: dict[str, bool | int | str] = {
            "encoder_matmul": counts["encoder"],
            "decoder_matmul": counts["decoder"],
            "joiner_matmul": counts["joiner"],
            "matmul_contract": counts == expected,
            "quantization": "none",
        }
        if recipe.profile == "production":
            import onnx

            model = onnx.load(candidate["encoder"].as_posix(), load_external_data=False)
            names = {node.name for node in model.graph.node}
            checks["boolean_mask_casts"] = (
                "/GreaterOrEqual_output_0_u8_cast" in names
                and all(f"{name}_cast_bool" in names for name in BOOLEAN_MASK_UNSQUEEZE_NODES)
                and all(name in names for name in BOOLEAN_MASK_SLICE_NODES)
            )
        passed = all(value is True for key, value in checks.items() if key.endswith("contract") or key.endswith("casts"))
        return ValidationResult("passed" if passed else "failed", checks)

    def compile_inputs(self, recipe: RecipeSpec, candidate: Mapping[str, Path]):
        """Select only the prepared encoder for production AI Hub compilation.

        Args:
            recipe: Recipe containing the compilation contract.
            candidate: Validated candidate component files.

        Returns:
            An empty list for controls or one encoder compile input for production.
        """
        if recipe.artifact.compilation.compiler == "none":
            return []
        return [
            CompileInput(
                "encoder",
                candidate["encoder"],
                recipe.parameters["fixed_input_shapes"]["encoder"],
                False,
                {"x": "float32", "x_lens": "int64"},
            )
        ]

    def bundle_components(self, recipe, candidate, compiled):
        """Describe Zipformer bundle files and per-component execution truth.

        Args:
            recipe: Recipe accepted for adapter protocol consistency.
            candidate: Validated local component files.
            compiled: Compiled encoder and optional external data.

        Returns:
            Bundle component tuples keyed by runtime role.
        """
        del recipe
        encoder = compiled.get("encoder", candidate["encoder"])
        result = {
            "encoder": (
                encoder,
                "qnn-htp" if "encoder" in compiled else "cpu",
                "onnx-epcontext" if "encoder" in compiled else "onnx",
            ),
            "decoder": (candidate["decoder"], "cpu", "onnx"),
            "joiner": (candidate["joiner"], "cpu", "onnx"),
            "tokens": (candidate["tokens"], "cpu", "text"),
        }
        if "encoder_external_data" in compiled:
            result["encoder_external_data"] = (
                compiled["encoder_external_data"], "qnn-htp", "onnx-external-data"
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
