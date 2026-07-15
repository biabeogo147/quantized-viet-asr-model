from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from model_pipeline.core import (
    ArtifactManifest,
    ArtifactSpec,
    ComponentSpec,
    Provenance,
    Stage,
    ValidationResult,
    sha256_file,
    stable_digest,
)


@dataclass(frozen=True)
class BundleResult:
    bundle_dir: Path
    manifest_path: Path
    bundle_checksum: str


def materialize_bundle(
    *,
    artifact: ArtifactSpec,
    components: Mapping[str, tuple[Path, str, str]],
    output_dir: str | Path,
    input_shapes_by_role: Mapping[str, Mapping[str, list[int | str]]] | None = None,
    source_checksums: Mapping[str, str] | None = None,
    recipe_digest: str | None = None,
    validation: ValidationResult | None = None,
    runtime_metadata: Mapping[str, object] | None = None,
    fixtures: Mapping[str, str] | None = None,
) -> BundleResult:
    """Materialize component files and their canonical manifest-v2 truth.

    Args:
        artifact: Canonical artifact identity.
        components: Source files with execution targets and formats by role.
        output_dir: Bundle directory to create or update.
        input_shapes_by_role: Optional component input-shape contracts.
        source_checksums: Optional upstream source provenance.
        recipe_digest: Optional recipe digest; derived when omitted.
        validation: Optional model validation evidence.
        runtime_metadata: Optional app-facing runtime contract fields.
        fixtures: Optional fixture paths included in the manifest.

    Returns:
        Bundle directory, manifest path, and deterministic manifest digest.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[ComponentSpec] = []
    for role, (source, execution_target, file_format) in sorted(components.items()):
        source_path = Path(source)
        suffix = "".join(source_path.suffixes) or ".bin"
        destination = root / (source_path.name if role.endswith("_external_data") else f"{role}{suffix}")
        shutil.copyfile(source_path, destination)
        model_payload = role in {"encoder", "model", "encoder_external_data", "model_external_data"}
        precision = (
            f"{artifact.quantization.weight}/{artifact.quantization.activation}"
            if model_payload and artifact.quantization.engine != "none"
            else "fp32"
        )
        rows.append(
            ComponentSpec(
                role=role,
                file=destination.name,
                format=file_format,
                precision=precision,
                input_shapes=dict((input_shapes_by_role or {}).get(role, {})),
                quantization_engine=(
                    artifact.quantization.engine if model_payload else "none"
                ),
                quantization_scope=(
                    artifact.quantization.scope if model_payload else "none"
                ),
                execution_target=execution_target,
                checksum=sha256_file(destination),
            )
        )
    manifest_object = ArtifactManifest(
        artifact=artifact,
        stage=Stage.PACKAGE,
        components=tuple(rows),
        provenance=Provenance(
            source_checksums=dict(source_checksums or {"materialized": "0" * 64}),
            recipe_digest=recipe_digest or stable_digest(artifact.to_dict()),
        ),
        validation=validation or ValidationResult("not-run", {}),
        runtime_metadata=dict(runtime_metadata or {}),
        fixtures=dict(fixtures or {}),
    )
    payload = manifest_object.to_dict()
    manifest = root / "artifact-manifest.json"
    manifest.write_text(manifest_object.to_json(), encoding="utf-8")
    return BundleResult(root, manifest, stable_digest(payload))
