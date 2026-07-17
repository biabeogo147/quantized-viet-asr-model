"""Portable Android benchmark payload materialization."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from model_pipeline.core import ArtifactSpec, sha256_file


_REQUIRED_COMPONENTS = {
    "fp32_model",
    "qdq_model",
    "compiled_model",
    "compiled_external_data",
}


def materialize_payload(
    *,
    model: str,
    artifact_id: str,
    components: Mapping[str, Path],
    fixtures: Sequence[Mapping[str, np.ndarray]],
    expected_outputs: Sequence[Mapping[str, object]],
    output_dir: str | Path,
) -> Path:
    """Create a checksummed payload consumed by the Android benchmark APK.

    Args:
        model: Canonical model family, either `zipformer` or `vpcd`.
        artifact_id: Canonical artifact identity shared with compile evidence.
        components: Model and support files keyed by benchmark role.
        fixtures: Named input tensors for deterministic inference.
        expected_outputs: Expected quality evidence paired with each fixture.
        output_dir: Directory receiving the portable payload.

    Returns:
        Path to the generated payload manifest.

    Raises:
        ValueError: If model identity, required roles, or fixture counts are invalid.
        FileNotFoundError: If a component file does not exist.
    """
    artifact = ArtifactSpec.parse(artifact_id)
    if model not in {"zipformer", "vpcd"} or artifact.model != model:
        raise ValueError("Payload model must match its canonical artifact ID")
    missing_roles = sorted(_REQUIRED_COMPONENTS - set(components))
    if missing_roles:
        raise ValueError(f"Missing required benchmark components: {missing_roles!r}")
    normalized_fixtures = tuple(fixtures)
    normalized_outputs = tuple(expected_outputs)
    if len(normalized_fixtures) != len(normalized_outputs) or not normalized_fixtures:
        raise ValueError("Fixtures and expected outputs must be non-empty and paired")

    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    component_records = []
    for role, source_value in sorted(components.items()):
        source = Path(source_value).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = _component_destination(root, role, source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        component_records.append(
            {
                "role": role,
                "file": destination.relative_to(root).as_posix(),
                "checksum": sha256_file(destination),
                "size_bytes": destination.stat().st_size,
            }
        )

    fixture_records = []
    for fixture_index, (fixture, expected) in enumerate(
        zip(normalized_fixtures, normalized_outputs, strict=True)
    ):
        inputs = {}
        for name, raw_array in sorted(fixture.items()):
            array = np.asarray(raw_array)
            if array.dtype.kind not in {"f", "i", "u", "b"}:
                raise ValueError(f"Unsupported raw tensor dtype for {name!r}: {array.dtype}")
            little_endian = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<"), copy=False))
            relative = Path("fixtures") / f"fixture-{fixture_index:03d}" / f"{name}.bin"
            tensor_path = root / relative
            tensor_path.parent.mkdir(parents=True, exist_ok=True)
            tensor_path.write_bytes(little_endian.tobytes(order="C"))
            inputs[name] = {
                "file": relative.as_posix(),
                "dtype": little_endian.dtype.str,
                "shape": list(little_endian.shape),
                "checksum": sha256_file(tensor_path),
            }
        fixture_records.append(
            {
                "fixture_index": fixture_index,
                "inputs": inputs,
                "expected_output": dict(expected),
            }
        )

    manifest = root / "benchmark-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": model,
                "artifact_id": artifact_id,
                "components": component_records,
                "fixtures": fixture_records,
                "warmup_iterations": 10,
                "measurement_iterations": 100,
                "repetitions": 3,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def _component_destination(root: Path, role: str, source: Path) -> Path:
    """Resolve a component role to a runtime-safe adjacent file layout.

    Args:
        root: Payload root directory.
        role: Logical component role.
        source: Source file whose suffix is preserved when appropriate.

    Returns:
        Destination path below the payload root.
    """
    if role == "fp32_model":
        return root / "components" / "fp32" / "model.onnx"
    if role == "fp32_external_data":
        return root / "components" / "fp32" / source.name
    if role == "qdq_model":
        return root / "components" / "qdq" / "model.onnx"
    if role == "qdq_external_data":
        return root / "components" / "qdq" / source.name
    if role == "compiled_model":
        return root / "components" / "compiled" / "model.onnx"
    if role == "compiled_external_data":
        return root / "components" / "compiled" / "model.bin"
    return root / "components" / "support" / f"{role}-{source.name}"
