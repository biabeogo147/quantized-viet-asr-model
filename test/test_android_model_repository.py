"""Contracts for the canonical Android model repository."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_pipeline.core import sha256_file
from model_pipeline.integrations.android.repository import (
    AndroidArtifactInput,
    AndroidComponentInput,
    load_model_index,
    materialize_model_repository,
)
from model_pipeline.integrations.android.repository_runtime import (
    resolve_retained_repository_inputs,
)
from model_pipeline.models import get_recipe


def _write(path: Path, content: bytes) -> Path:
    """Write one fake binary used by repository contract tests.

    Args:
        path: Destination file.
        content: Deterministic fake model bytes.

    Returns:
        The written file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _artifact(
    tmp_path: Path,
    *,
    model: str,
    configuration: str,
    representation: str,
    execution_target: str,
    build_surfaces: tuple[str, ...],
) -> AndroidArtifactInput:
    """Build one minimal checksummed Android artifact input.

    Args:
        tmp_path: Isolated source directory.
        model: Canonical model family.
        configuration: Canonical recipe configuration.
        representation: Android representation label.
        execution_target: Runtime execution target.
        build_surfaces: Android surfaces allowed to package the artifact.

    Returns:
        An artifact input with one model component.
    """
    recipe = get_recipe(model, configuration)
    role = "encoder" if model == "zipformer" else "model"
    source = _write(tmp_path / f"{model}-{configuration}.onnx", recipe.artifact.artifact_id.encode())
    return AndroidArtifactInput(
        artifact=recipe.artifact,
        configuration=configuration,
        representation=representation,
        execution_target=execution_target,
        build_surfaces=build_surfaces,
        components=(
            AndroidComponentInput(
                role=role,
                source=source,
                relative_file=f"artifacts/{model}/{configuration}/{execution_target}/{role}.onnx",
                format="onnx",
                precision="fp32" if configuration == "fp32-fixed-shape" else "int8/int16",
                input_shapes={},
                quantization_engine=recipe.artifact.quantization.engine,
                quantization_scope=recipe.artifact.quantization.scope,
                execution_target=execution_target,
            ),
        ),
        fixtures={},
        source_checksums={"source": sha256_file(source)},
        validation_checks={"graph_contract": True},
    )


def test_repository_materializes_deterministic_manifest_v2_index(tmp_path: Path) -> None:
    """Verify repository output is deterministic and fully checksummed.

    Args:
        tmp_path: Isolated source and destination root.

    Returns:
        None.
    """
    artifacts = (
        _artifact(
            tmp_path,
            model="zipformer",
            configuration="fp32-fixed-shape",
            representation="onnx-fp32-fixed-shape",
            execution_target="cpu",
            build_surfaces=("cpuCompat", "benchmark"),
        ),
        _artifact(
            tmp_path,
            model="vpcd",
            configuration="aimet-int8-int16-encoder-matmul",
            representation="onnx-epcontext-external-binary",
            execution_target="qnn-htp",
            build_surfaces=("qnnOfficialArm64", "benchmark"),
        ),
    )
    first = materialize_model_repository(artifacts=artifacts, destination=tmp_path / "one")
    second = materialize_model_repository(artifacts=artifacts, destination=tmp_path / "two")

    assert first.repository_checksum == second.repository_checksum
    assert first.index_path.read_bytes() == second.index_path.read_bytes()
    index = load_model_index(first.index_path)
    assert index.schema_version == 1
    assert [row.model for row in index.artifacts] == ["vpcd", "zipformer"]
    assert all("\\" not in row.manifest for row in index.artifacts)
    for row in index.artifacts:
        manifest = json.loads((first.root / row.manifest).read_text(encoding="utf-8"))
        assert manifest["schema_version"] == 2
        assert manifest["artifact_id"] == row.artifact_id
        for component in manifest["components"]:
            component_path = first.root / component["file"]
            assert component_path.is_file()
            assert sha256_file(component_path) == component["checksum"]


def test_repository_rejects_unsafe_component_path(tmp_path: Path) -> None:
    """Verify repository components cannot escape the canonical root.

    Args:
        tmp_path: Isolated source and destination root.

    Returns:
        None.
    """
    base = _artifact(
        tmp_path,
        model="zipformer",
        configuration="fp32-fixed-shape",
        representation="onnx-fp32-fixed-shape",
        execution_target="cpu",
        build_surfaces=("cpuCompat",),
    )
    component = base.components[0]
    unsafe = AndroidArtifactInput(
        artifact=base.artifact,
        configuration=base.configuration,
        representation=base.representation,
        execution_target=base.execution_target,
        build_surfaces=base.build_surfaces,
        components=(
            AndroidComponentInput(
                role=component.role,
                source=component.source,
                relative_file="../escaped.onnx",
                format=component.format,
                precision=component.precision,
                input_shapes=component.input_shapes,
                quantization_engine=component.quantization_engine,
                quantization_scope=component.quantization_scope,
                execution_target=component.execution_target,
            ),
        ),
        fixtures=base.fixtures,
        source_checksums=base.source_checksums,
        validation_checks=base.validation_checks,
    )

    with pytest.raises(ValueError, match="relative path"):
        materialize_model_repository(artifacts=(unsafe,), destination=tmp_path / "repository")
    assert not (tmp_path / "escaped.onnx").exists()


def test_repository_preserves_existing_destination_when_staging_fails(tmp_path: Path) -> None:
    """Verify failed staging never leaves a partially promoted repository.

    Args:
        tmp_path: Isolated source and destination root.

    Returns:
        None.
    """
    destination = tmp_path / "repository"
    destination.mkdir()
    marker = _write(destination / "existing.marker", b"keep")
    artifact = _artifact(
        tmp_path,
        model="vpcd",
        configuration="fp32-fixed-shape",
        representation="onnx-fp32-fixed-shape",
        execution_target="cpu",
        build_surfaces=("cpuCompat",),
    )
    artifact.components[0].source.unlink()

    with pytest.raises(FileNotFoundError):
        materialize_model_repository(artifacts=(artifact,), destination=destination)

    assert marker.read_bytes() == b"keep"
    assert not (destination / "model-index.json").exists()


def _write_retained_payload(root: Path, model: str) -> tuple[Path, str]:
    """Create a minimal retained benchmark payload for source-resolution tests.

    Args:
        root: Payload root receiving model files.
        model: Canonical model family.

    Returns:
        Compiled model path and its checksum.
    """
    model_root = root / model
    roles = (
        ("fp32_model", "components/fp32/model.onnx"),
        ("compiled_model", "components/compiled/model.onnx"),
        ("compiled_external_data", "components/compiled/model.bin"),
    )
    if model == "zipformer":
        roles += (
            ("decoder", "components/support/decoder.onnx"),
            ("joiner", "components/support/joiner.onnx"),
            ("tokens", "components/support/tokens.txt"),
        )
    else:
        roles += (
            ("tokenizer_encode", "components/support/tokenizer.encode.onnx"),
            ("tokenizer_decode", "components/support/tokenizer.decode.onnx"),
            ("tokenizer_to_model_id_map", "components/support/to-map.json"),
            ("model_to_tokenizer_id_map", "components/support/from-map.json"),
            ("autoregressive_loop", "components/support/loop.json"),
        )
    component_rows = []
    compiled_path = model_root / "components/compiled/model.onnx"
    for role, relative in roles:
        source = _write(model_root / relative, f"{model}:{role}".encode())
        component_rows.append(
            {
                "role": role,
                "file": relative,
                "checksum": sha256_file(source),
                "size_bytes": source.stat().st_size,
            }
        )
    fixture = _write(model_root / "fixtures/fixture-000/input.bin", b"fixture")
    manifest = {
        "artifact_id": get_recipe(
            model,
            "aimet-int8-int16-encoder-matmul",
        ).artifact.artifact_id,
        "components": component_rows,
        "fixtures": [
            {
                "fixture_index": 0,
                "inputs": {
                    "input": {
                        "file": fixture.relative_to(model_root).as_posix(),
                        "checksum": sha256_file(fixture),
                        "dtype": "<f4",
                        "shape": [1],
                    }
                },
                "expected_output": {"value": 1},
            }
        ],
    }
    (model_root / "benchmark-manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return compiled_path, sha256_file(compiled_path)


def test_retained_payload_resolution_builds_four_repository_artifacts(tmp_path: Path) -> None:
    """Verify retained exact sources become two canonical artifacts per model.

    Args:
        tmp_path: Isolated retained payload root.

    Returns:
        None.
    """
    _, zipformer_checksum = _write_retained_payload(tmp_path, "zipformer")
    _, vpcd_checksum = _write_retained_payload(tmp_path, "vpcd")

    artifacts = resolve_retained_repository_inputs(
        payload_root=tmp_path,
        generated_root=tmp_path / "generated",
        expected_compiled_checksums={
            "zipformer": zipformer_checksum,
            "vpcd": vpcd_checksum,
        },
    )

    assert len(artifacts) == 4
    assert {row.configuration for row in artifacts} == {
        "fp32-fixed-shape",
        "aimet-int8-int16-encoder-matmul",
    }
    assert {row.execution_target for row in artifacts} == {"cpu", "qnn-htp"}
    assert all("qdq" not in component.role for row in artifacts for component in row.components)
    fixture_manifest = json.loads(
        (tmp_path / "generated" / "zipformer" / "fixture-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert (
        fixture_manifest["fixtures"][0]["inputs"]["input"]["file"]
        == "fixtures/zipformer/fixture-000/input.bin"
    )


def test_retained_payload_resolution_rejects_compiled_checksum_mismatch(tmp_path: Path) -> None:
    """Verify a retained target cannot be paired with mismatched compiled bytes.

    Args:
        tmp_path: Isolated retained payload root.

    Returns:
        None.
    """
    _write_retained_payload(tmp_path, "zipformer")
    _, vpcd_checksum = _write_retained_payload(tmp_path, "vpcd")

    with pytest.raises(ValueError, match="compiled ONNX checksum"):
        resolve_retained_repository_inputs(
            payload_root=tmp_path,
            generated_root=tmp_path / "generated",
            expected_compiled_checksums={
                "zipformer": "0" * 64,
                "vpcd": vpcd_checksum,
            },
        )
