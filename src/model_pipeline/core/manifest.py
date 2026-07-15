from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from model_pipeline.core.specs import ArtifactSpec, Stage


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ComponentSpec:
    role: str
    file: str
    format: str
    precision: str
    input_shapes: Mapping[str, list[int | str]]
    quantization_engine: str
    quantization_scope: str
    execution_target: str
    checksum: str

    def __post_init__(self) -> None:
        """Validate required component identity and checksum fields.

        Returns:
            None.

        Raises:
            ValueError: If the role, file, or checksum is invalid.
        """
        if not self.role or not self.file:
            raise ValueError("Component role and file are required")
        _require_sha256("component checksum", self.checksum)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ComponentSpec":
        """Construct a component specification from serialized manifest fields.

        Args:
            payload: Component mapping with shapes represented as JSON arrays.

        Returns:
            The validated component specification.
        """
        values = dict(payload)
        values["input_shapes"] = {
            str(name): list(shape) for name, shape in dict(values["input_shapes"]).items()
        }
        return cls(**values)


@dataclass(frozen=True)
class Provenance:
    source_checksums: Mapping[str, str]
    recipe_digest: str
    tool_versions: Mapping[str, str] = field(default_factory=dict)
    parent_artifact_id: str | None = None

    def __post_init__(self) -> None:
        """Validate recipe and source checksums in provenance metadata.

        Returns:
            None.

        Raises:
            ValueError: If any required digest is not lowercase SHA-256.
        """
        _require_sha256("recipe digest", self.recipe_digest)
        for role, checksum in self.source_checksums.items():
            _require_sha256(f"source checksum for {role}", checksum)


@dataclass(frozen=True)
class ValidationResult:
    status: str
    checks: Mapping[str, bool | int | float | str]

    def __post_init__(self) -> None:
        """Validate the closed set of manifest validation statuses.

        Returns:
            None.

        Raises:
            ValueError: If the status is not recognized.
        """
        if self.status not in {"not-run", "passed", "failed"}:
            raise ValueError(f"Invalid validation status: {self.status!r}")


@dataclass(frozen=True)
class ArtifactManifest:
    artifact: ArtifactSpec
    stage: Stage
    components: tuple[ComponentSpec, ...]
    provenance: Provenance
    validation: ValidationResult
    runtime_metadata: Mapping[str, Any] = field(default_factory=dict)
    fixtures: Mapping[str, str] = field(default_factory=dict)
    schema_version: int = 2

    def __post_init__(self) -> None:
        """Validate schema version and unique component roles.

        Returns:
            None.

        Raises:
            ValueError: If the schema or component inventory is invalid.
        """
        if self.schema_version != 2:
            raise ValueError("Only manifest schema v2 is supported")
        roles = [component.role for component in self.components]
        if not roles or len(set(roles)) != len(roles):
            raise ValueError("Manifest components must be non-empty and have unique roles")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete manifest to JSON-compatible fields.

        Returns:
            A mapping preserving component truth, provenance, and validation evidence.
        """
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact.artifact_id,
            "artifact": self.artifact.to_dict(),
            "stage": self.stage.value,
            "components": [asdict(component) for component in self.components],
            "provenance": asdict(self.provenance),
            "validation": asdict(self.validation),
            "runtime_metadata": dict(self.runtime_metadata),
            "fixtures": dict(self.fixtures),
        }

    def to_json(self) -> str:
        """Serialize the manifest as deterministic human-readable JSON.

        Returns:
            UTF-8-compatible JSON text terminated by a newline.
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactManifest":
        """Parse and validate a manifest-v2 mapping.

        Args:
            payload: Serialized manifest fields.

        Returns:
            The validated artifact manifest.

        Raises:
            ValueError: If identity, schema, component, or digest fields are invalid.
        """
        artifact = ArtifactSpec.from_dict(dict(payload["artifact"]))
        if payload.get("artifact_id") != artifact.artifact_id:
            raise ValueError("artifact_id does not match artifact fields")
        return cls(
            schema_version=int(payload["schema_version"]),
            artifact=artifact,
            stage=Stage(str(payload["stage"])),
            components=tuple(ComponentSpec.from_dict(row) for row in payload["components"]),
            provenance=Provenance(**dict(payload["provenance"])),
            validation=ValidationResult(**dict(payload["validation"])),
            runtime_metadata=dict(payload.get("runtime_metadata") or {}),
            fixtures={str(key): str(value) for key, value in dict(payload.get("fixtures") or {}).items()},
        )

    @classmethod
    def from_json(cls, content: str) -> "ArtifactManifest":
        """Parse a manifest from JSON text.

        Args:
            content: Manifest-v2 JSON document.

        Returns:
            The validated artifact manifest.
        """
        return cls.from_dict(json.loads(content))


def _require_sha256(label: str, value: str) -> None:
    """Require a lowercase SHA-256 value for a labeled manifest field.

    Args:
        label: Field description included in validation failures.
        value: Digest value to validate.

    Returns:
        None.

    Raises:
        ValueError: If the value is not a lowercase SHA-256 digest.
    """
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
