from __future__ import annotations

import shutil
from pathlib import Path

from model_pipeline.core import RecipeSpec, ValidationResult
from model_pipeline.integrations.aihub import EvidenceStore, FakeAiHubClient
from model_pipeline.models import get_recipe
from model_pipeline.pipeline import CompileInput, ModelPipeline


class FakeVpcdAdapter:
    def __init__(self, source_root: Path):
        """Initialize a fake adapter backed by deterministic source files.

        Args:
            source_root: Directory containing fake model and runtime sources.

        Returns:
            None.
        """
        self.source_root = source_root

    def source_files(self, recipe: RecipeSpec):
        """Resolve the fake VPCD source inventory.

        Args:
            recipe: Recipe accepted for protocol consistency.

        Returns:
            Fake component roles mapped to source files.
        """
        del recipe
        return {
            "model": self.source_root / "model.onnx",
            "tokenizer_encode": self.source_root / "tokenizer.encode.onnx",
            "tokenizer_decode": self.source_root / "tokenizer.decode.onnx",
            "autoregressive_loop": self.source_root / "runtime.json",
        }

    def prepare(self, recipe, sources, output_dir):
        """Copy fake source files through the preparation stage.

        Args:
            recipe: Recipe accepted for protocol consistency.
            sources: Fake source files by role.
            output_dir: Preparation-stage output directory.

        Returns:
            Copied prepared files by role.
        """
        del recipe
        return self._copy(sources, output_dir)

    def quantize(self, recipe, prepared, output_dir):
        """Assert the AIMET action and copy fake quantized files.

        Args:
            recipe: Quantized configuration expected to request AIMET.
            prepared: Fake prepared files by role.
            output_dir: Quantization-stage output directory.

        Returns:
            Copied quantized files by role.
        """
        assert recipe.parameters["quantize_action"] == "aimet"
        return self._copy(prepared, output_dir)

    def validate(self, recipe, quantized_components):
        """Return passing validation when fake quantized components are present.

        Args:
            recipe: Recipe accepted for protocol consistency.
            quantized_components: Fake quantized files by role.

        Returns:
            Structured fake validation evidence.
        """
        del recipe
        return ValidationResult("passed", {"graph_contract": bool(quantized_components)})

    def compile_inputs(self, recipe, validated_components):
        """Describe one package-level fake compile input.

        Args:
            recipe: Quantized recipe containing fixed shapes.
            validated_components: Fake validated files by role.

        Returns:
            One VPCD model compile input.
        """
        return [
            CompileInput(
                role="model",
                source_path=validated_components["model"],
                input_shapes=recipe.parameters["fixed_input_shapes"],
                truncate_64bit_io=True,
            )
        ]

    def bundle_components(self, recipe, validated_components, compiled_components):
        """Describe compiled model and CPU support bundle components.

        Args:
            recipe: Recipe accepted for protocol consistency.
            validated_components: Fake validated local files.
            compiled_components: Fake compiled model output.

        Returns:
            Component file, target, and format tuples by role.
        """
        del recipe
        result = {
            role: (path, "cpu", "json" if path.suffix == ".json" else "onnx")
            for role, path in validated_components.items()
        }
        result["model"] = (
            compiled_components["model"],
            "qnn-htp",
            "onnx-epcontext",
        )
        return result

    @staticmethod
    def _copy(inputs, output_dir):
        """Copy fake stage inputs into a stage-owned directory.

        Args:
            inputs: Source files keyed by logical role.
            output_dir: Destination stage directory.

        Returns:
            Copied output files keyed by role.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {}
        for role, source in inputs.items():
            destination = output_dir / source.name
            shutil.copyfile(source, destination)
            outputs[role] = destination
        return outputs


def test_full_pipeline_with_fake_aihub_and_deterministic_sync(tmp_path: Path) -> None:
    """Verify the full fake pipeline compiles once and resumes deterministically.

    Args:
        tmp_path: Isolated source, build, evidence, and Android directories.

    Returns:
        None.
    """
    sources = tmp_path / "sources"
    sources.mkdir()
    for name in ("model.onnx", "tokenizer.encode.onnx", "tokenizer.decode.onnx", "runtime.json"):
        (sources / name).write_bytes(name.encode())
    recipe = get_recipe("vpcd", "aimet-int8-int16-encoder-matmul")
    client = FakeAiHubClient(compiled_bytes=b"compiled-model")
    pipeline = ModelPipeline(
        build_root=tmp_path / "build",
        evidence_store=EvidenceStore(tmp_path / "evidence"),
        aihub_client=client,
    )

    first = pipeline.run(
        recipe=recipe,
        adapter=FakeVpcdAdapter(sources),
        through="sync",
        android_destination=tmp_path / "android",
    )
    second = pipeline.run(
        recipe=recipe,
        adapter=FakeVpcdAdapter(sources),
        through="sync",
        android_destination=tmp_path / "android",
    )

    assert first.validation.status == "passed"
    assert first.bundle.manifest_path.is_file()
    assert (tmp_path / "android" / "artifact-manifest.json").is_file()
    assert first.bundle.bundle_checksum == second.bundle.bundle_checksum
    assert client.submit_count == 1
    assert second.resumed_stages == (
        "source",
        "prepare",
        "quantize",
        "validate",
        "compile",
        "package",
        "sync",
    )
