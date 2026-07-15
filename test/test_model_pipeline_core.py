from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_pipeline.core import (
    ArtifactManifest,
    ArtifactSpec,
    CompileSpec,
    ComponentSpec,
    Provenance,
    QuantizationSpec,
    Stage,
    StageRunner,
    ValidationResult,
    sha256_file,
)


ZIPFORMER_ID = (
    "zipformer__q-none-fp32-fp32-none__s-enc1x2009x80-dec1x2-join1x512"
    "__c-aihub-qnn-htp-encoder"
)
VPCD_ID = (
    "vpcd__q-aimet-int8-int16-encoder-matmul__s-src1x384-dec1x64"
    "__c-aihub-qnn-htp-model"
)


@pytest.mark.parametrize("artifact_id", [ZIPFORMER_ID, VPCD_ID])
def test_artifact_id_round_trip(artifact_id: str) -> None:
    """Verify canonical artifact IDs parse and serialize without drift.

    Args:
        artifact_id: Parameterized canonical artifact identifier.

    Returns:
        None.
    """
    parsed = ArtifactSpec.parse(artifact_id)

    assert parsed.artifact_id == artifact_id
    assert ArtifactSpec.parse(parsed.artifact_id) == parsed


@pytest.mark.parametrize(
    "artifact_id",
    [
        "historical_zipformer",
        "vpcd_default_alias",
        "zipformer__q-custom-int8-int16-all__s-dynamic__c-none-cpu-none",
        "zipformer__q-none-fp32-fp32-none__s-dynamic",
        "ZIPFORMER__q-none-fp32-fp32-none__s-dynamic__c-none-cpu-none",
    ],
)
def test_artifact_id_rejects_legacy_or_malformed_names(artifact_id: str) -> None:
    """Verify legacy and malformed artifact aliases are rejected.

    Args:
        artifact_id: Parameterized invalid artifact identifier.

    Returns:
        None.
    """
    with pytest.raises(ValueError):
        ArtifactSpec.parse(artifact_id)


def test_manifest_v2_serializes_component_truth(tmp_path: Path) -> None:
    """Verify manifest v2 preserves component execution and provenance truth.

    Args:
        tmp_path: Isolated directory for temporary component files.

    Returns:
        None.
    """
    encoder = tmp_path / "encoder.onnx"
    encoder.write_bytes(b"encoder")
    artifact = ArtifactSpec(
        model="zipformer",
        quantization=QuantizationSpec("none", "fp32", "fp32", "none"),
        shape="enc1x2009x80-dec1x2-join1x512",
        compilation=CompileSpec("aihub", "qnn-htp", "encoder"),
    )
    manifest = ArtifactManifest(
        artifact=artifact,
        stage=Stage.PACKAGE,
        components=(
            ComponentSpec(
                role="encoder",
                file="encoder.onnx",
                format="onnx-epcontext",
                precision="fp32",
                input_shapes={"x": [1, 2009, 80], "x_lens": [1]},
                quantization_engine="none",
                quantization_scope="none",
                execution_target="qnn-htp",
                checksum=sha256_file(encoder),
            ),
            ComponentSpec(
                role="decoder",
                file="decoder.onnx",
                format="onnx",
                precision="fp32",
                input_shapes={"y": [1, 2]},
                quantization_engine="none",
                quantization_scope="none",
                execution_target="cpu",
                checksum="a" * 64,
            ),
        ),
        provenance=Provenance(source_checksums={"encoder": "b" * 64}, recipe_digest="c" * 64),
        validation=ValidationResult(status="passed", checks={"graph_contract": True}),
    )

    restored = ArtifactManifest.from_json(manifest.to_json())

    assert restored == manifest
    assert restored.schema_version == 2
    assert restored.components[0].execution_target == "qnn-htp"
    assert restored.components[1].execution_target == "cpu"
    assert json.loads(restored.to_json())["artifact_id"] == ZIPFORMER_ID


def test_stage_runner_resumes_only_for_identical_inputs_and_recipe(tmp_path: Path) -> None:
    """Verify stage resume requires identical recipe and input digests.

    Args:
        tmp_path: Isolated stage-runner build root.

    Returns:
        None.
    """
    calls: list[str] = []
    runner = StageRunner(tmp_path)

    def execute(stage_dir: Path) -> dict[str, Path]:
        """Materialize one deterministic stage output for cache testing.

        Args:
            stage_dir: Directory allocated to the test stage.

        Returns:
            Logical output role mapped to its created file.
        """
        calls.append("called")
        output = stage_dir / "model.onnx"
        output.write_bytes(b"stable")
        return {"model": output}

    first = runner.run(
        stage=Stage.PREPARE,
        artifact_id=ZIPFORMER_ID,
        recipe_digest="1" * 64,
        input_digests={"source": "2" * 64},
        execute=execute,
    )
    resumed = runner.run(
        stage=Stage.PREPARE,
        artifact_id=ZIPFORMER_ID,
        recipe_digest="1" * 64,
        input_digests={"source": "2" * 64},
        execute=execute,
    )
    invalidated = runner.run(
        stage=Stage.PREPARE,
        artifact_id=ZIPFORMER_ID,
        recipe_digest="3" * 64,
        input_digests={"source": "2" * 64},
        execute=execute,
    )

    assert first.resumed is False
    assert resumed.resumed is True
    assert invalidated.resumed is False
    assert len(calls) == 2


def test_stage_runner_invalidates_cache_when_output_was_modified(tmp_path: Path) -> None:
    """Verify modified cached bytes force stage re-execution.

    Args:
        tmp_path: Isolated stage-runner build root.

    Returns:
        None.
    """
    runner = StageRunner(tmp_path)
    calls = 0

    def execute(stage_dir: Path) -> dict[str, Path]:
        """Materialize versioned output bytes for invalidation testing.

        Args:
            stage_dir: Directory allocated to the test stage.

        Returns:
            Logical output role mapped to its created file.
        """
        nonlocal calls
        calls += 1
        output = stage_dir / "model.onnx"
        output.write_bytes(f"version-{calls}".encode())
        return {"model": output}

    result = runner.run(
        stage=Stage.QUANTIZE,
        artifact_id=VPCD_ID,
        recipe_digest="1" * 64,
        input_digests={"source": "2" * 64},
        execute=execute,
    )
    result.outputs["model"].write_bytes(b"corrupt")
    rerun = runner.run(
        stage=Stage.QUANTIZE,
        artifact_id=VPCD_ID,
        recipe_digest="1" * 64,
        input_digests={"source": "2" * 64},
        execute=execute,
    )

    assert rerun.resumed is False
    assert calls == 2
