"""Canonical manifest-v2 repository shared by Android production and benchmarks."""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

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
class AndroidComponentInput:
    """Describe one source file and its repository-relative runtime contract."""

    role: str
    source: Path
    relative_file: str
    format: str
    precision: str
    input_shapes: Mapping[str, list[int | str]]
    quantization_engine: str
    quantization_scope: str
    execution_target: str


@dataclass(frozen=True)
class AndroidArtifactInput:
    """Describe one canonical artifact to place in the Android repository."""

    artifact: ArtifactSpec
    configuration: str
    representation: str
    execution_target: str
    build_surfaces: tuple[str, ...]
    components: tuple[AndroidComponentInput, ...]
    fixtures: Mapping[str, Path]
    source_checksums: Mapping[str, str]
    validation_checks: Mapping[str, bool | int | float | str]
    runtime_metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelIndexArtifact:
    """Describe one manifest entry exposed to Android build surfaces."""

    model: str
    configuration: str
    artifact_id: str
    manifest: str
    representation: str
    execution_target: str
    build_surfaces: tuple[str, ...]

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ModelIndexArtifact":
        """Parse and validate one serialized index artifact.

        Args:
            payload: Serialized artifact fields.

        Returns:
            The validated model-index artifact.
        """
        artifact_id = str(payload["artifact_id"])
        artifact = ArtifactSpec.parse(artifact_id)
        model = str(payload["model"])
        if artifact.model != model:
            raise ValueError("Model-index artifact model does not match artifact ID")
        manifest = _safe_relative_path(str(payload["manifest"]), "manifest")
        return cls(
            model=model,
            configuration=str(payload["configuration"]),
            artifact_id=artifact_id,
            manifest=manifest,
            representation=str(payload["representation"]),
            execution_target=str(payload["execution_target"]),
            build_surfaces=tuple(str(value) for value in payload["build_surfaces"]),
        )


@dataclass(frozen=True)
class ModelIndex:
    """Represent the deterministic root index consumed by Android."""

    artifacts: tuple[ModelIndexArtifact, ...]
    schema_version: int = 1

    def to_dict(self) -> dict[str, object]:
        """Serialize the model index to JSON-compatible fields.

        Returns:
            Deterministically ordered model-index fields.
        """
        return {
            "schema_version": self.schema_version,
            "artifacts": [
                {
                    **asdict(artifact),
                    "build_surfaces": list(artifact.build_surfaces),
                }
                for artifact in self.artifacts
            ],
        }

    def to_json(self) -> str:
        """Serialize the model index as deterministic JSON.

        Returns:
            Human-readable JSON terminated by a newline.
        """
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ModelIndex":
        """Parse and validate serialized model-index fields.

        Args:
            payload: Serialized model index.

        Returns:
            The validated model index.

        Raises:
            ValueError: If the schema, artifact identities, or paths are invalid.
        """
        if int(payload["schema_version"]) != 1:
            raise ValueError("Only Android model-index schema v1 is supported")
        artifacts = tuple(
            ModelIndexArtifact.from_dict(row)
            for row in payload["artifacts"]
        )
        identities = [row.artifact_id for row in artifacts]
        if not artifacts or len(identities) != len(set(identities)):
            raise ValueError("Model index must contain unique artifacts")
        return cls(artifacts=artifacts)


@dataclass(frozen=True)
class ModelRepositoryResult:
    """Report the promoted repository root and deterministic checksum."""

    root: Path
    index_path: Path
    repository_checksum: str


def load_model_index(path: str | Path) -> ModelIndex:
    """Load and validate a canonical Android model index.

    Args:
        path: Model-index JSON path.

    Returns:
        The validated model index.
    """
    return ModelIndex.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def materialize_model_repository(
    *,
    artifacts: Sequence[AndroidArtifactInput],
    destination: str | Path,
) -> ModelRepositoryResult:
    """Stage and atomically promote a canonical Android model repository.

    Args:
        artifacts: Canonical artifact inputs to publish together.
        destination: Repository directory replaced only after staging succeeds.

    Returns:
        Promoted root, model-index path, and repository checksum.

    Raises:
        FileNotFoundError: If a declared source component or fixture is missing.
        ValueError: If identities, paths, roles, or checksums are inconsistent.
    """
    ordered = tuple(
        sorted(
            artifacts,
            key=lambda value: (
                value.artifact.model,
                value.configuration,
                value.execution_target,
            ),
        )
    )
    if not ordered:
        raise ValueError("At least one Android artifact is required")
    artifact_ids = [value.artifact.artifact_id for value in ordered]
    if len(artifact_ids) != len(set(artifact_ids)):
        raise ValueError("Android artifacts must have unique artifact IDs")

    target = Path(destination).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.staging-{uuid.uuid4().hex}"
    backup = target.parent / f".{target.name}.backup-{uuid.uuid4().hex}"
    try:
        staging.mkdir()
        index_rows: list[ModelIndexArtifact] = []
        manifest_checksums: dict[str, str] = {}
        for artifact_input in ordered:
            row, manifest_checksum = _materialize_artifact(staging, artifact_input)
            index_rows.append(row)
            manifest_checksums[row.artifact_id] = manifest_checksum
        index = ModelIndex(tuple(index_rows))
        index_path = staging / "model-index.json"
        index_path.write_text(index.to_json(), encoding="utf-8")
        repository_checksum = stable_digest(
            {
                "index": index.to_dict(),
                "manifests": manifest_checksums,
            }
        )
        if target.exists():
            target.rename(backup)
        try:
            staging.rename(target)
        except BaseException:
            if backup.exists() and not target.exists():
                backup.rename(target)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        return ModelRepositoryResult(
            root=target,
            index_path=target / "model-index.json",
            repository_checksum=repository_checksum,
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if backup.exists() and target.exists():
            shutil.rmtree(backup)


def _materialize_artifact(
    root: Path,
    artifact_input: AndroidArtifactInput,
) -> tuple[ModelIndexArtifact, str]:
    """Materialize one artifact and its manifest inside a staged repository.

    Args:
        root: Staged repository root.
        artifact_input: Canonical source and runtime metadata.

    Returns:
        Model-index row and manifest checksum.
    """
    rows: list[ComponentSpec] = []
    roles: set[str] = set()
    for component in sorted(artifact_input.components, key=lambda value: value.role):
        if component.role in roles:
            raise ValueError(f"Duplicate Android component role: {component.role!r}")
        roles.add(component.role)
        source = Path(component.source).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        relative = _safe_relative_path(component.relative_file, "component")
        destination = root / PurePosixPath(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _copy_consistent(source, destination)
        rows.append(
            ComponentSpec(
                role=component.role,
                file=relative,
                format=component.format,
                precision=component.precision,
                input_shapes=dict(component.input_shapes),
                quantization_engine=component.quantization_engine,
                quantization_scope=component.quantization_scope,
                execution_target=component.execution_target,
                checksum=sha256_file(destination),
            )
        )

    fixture_paths: dict[str, str] = {}
    for fixture_name, fixture_source in sorted(artifact_input.fixtures.items()):
        relative = _safe_relative_path(fixture_name, "fixture")
        source = Path(fixture_source).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = root / PurePosixPath(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _copy_consistent(source, destination)
        fixture_paths[relative] = relative

    manifest = ArtifactManifest(
        artifact=artifact_input.artifact,
        stage=Stage.PACKAGE,
        components=tuple(rows),
        provenance=Provenance(
            source_checksums=dict(artifact_input.source_checksums),
            recipe_digest=stable_digest(
                {
                    "artifact": artifact_input.artifact.to_dict(),
                    "configuration": artifact_input.configuration,
                }
            ),
        ),
        validation=ValidationResult("passed", dict(artifact_input.validation_checks)),
        runtime_metadata={
            "configuration": artifact_input.configuration,
            "representation": artifact_input.representation,
            "execution_target": artifact_input.execution_target,
            "build_surfaces": list(artifact_input.build_surfaces),
            **dict(artifact_input.runtime_metadata),
        },
        fixtures=fixture_paths,
    )
    manifest_relative = (
        f"manifests/{artifact_input.artifact.model}/"
        f"{artifact_input.configuration}/{artifact_input.execution_target}.json"
    )
    manifest_path = root / PurePosixPath(manifest_relative)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest.to_json(), encoding="utf-8")
    return (
        ModelIndexArtifact(
            model=artifact_input.artifact.model,
            configuration=artifact_input.configuration,
            artifact_id=artifact_input.artifact.artifact_id,
            manifest=manifest_relative,
            representation=artifact_input.representation,
            execution_target=artifact_input.execution_target,
            build_surfaces=tuple(artifact_input.build_surfaces),
        ),
        sha256_file(manifest_path),
    )


def _copy_consistent(source: Path, destination: Path) -> None:
    """Copy a file while rejecting conflicting shared repository paths.

    Args:
        source: Existing source file.
        destination: Repository destination.

    Returns:
        None.

    Raises:
        ValueError: If the destination already contains different bytes.
    """
    if destination.exists():
        if sha256_file(source) != sha256_file(destination):
            raise ValueError(f"Conflicting Android repository file: {destination}")
        return
    shutil.copyfile(source, destination)


def _safe_relative_path(value: str, label: str) -> str:
    """Validate and normalize one repository-relative POSIX path.

    Args:
        value: Candidate path.
        label: Field label used in validation errors.

    Returns:
        Normalized relative POSIX path.

    Raises:
        ValueError: If the path is absolute, empty, traversing, or non-POSIX.
    """
    if not value or "\\" in value:
        raise ValueError(f"{label} must be a safe relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} must be a safe relative path")
    return path.as_posix()
