from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping

from model_pipeline.core.files import stable_digest


_TOKEN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_SHAPE = re.compile(r"^[a-z][a-z0-9x]*(?:-[a-z][a-z0-9x]*)*$")
_ALLOWED_QUANTIZATION = {
    ("none", "fp32", "fp32", "none"),
    ("aimet", "int8", "int16", "encoder-matmul"),
    ("ortqnn", "uint8", "uint16", "encoder-matmul"),
}
_ALLOWED_COMPILATION = {
    ("none", "cpu", "none"),
    ("aihub", "qnn-htp", "encoder"),
    ("aihub", "qnn-htp", "model"),
}


class Stage(str, Enum):
    SOURCE = "source"
    PREPARE = "prepare"
    QUANTIZE = "quantize"
    VALIDATE = "validate"
    COMPILE = "compile"
    PACKAGE = "package"
    SYNC = "sync"

    @classmethod
    def ordered(cls) -> tuple["Stage", ...]:
        """Return pipeline stages in their canonical execution order.

        Returns:
            All stages from source resolution through Android synchronization.
        """
        return tuple(cls)


@dataclass(frozen=True)
class QuantizationSpec:
    engine: str
    weight: str
    activation: str
    scope: str

    def __post_init__(self) -> None:
        """Validate that the quantization fields form a supported contract.

        Returns:
            None.

        Raises:
            ValueError: If a token is malformed or the contract is unsupported.
        """
        for name, value in asdict(self).items():
            _validate_token(name, value)
        if (self.engine, self.weight, self.activation, self.scope) not in _ALLOWED_QUANTIZATION:
            raise ValueError(f"Unsupported quantization contract: {self.slug!r}")

    @property
    def slug(self) -> str:
        """Render the canonical quantization segment used in artifact IDs.

        Returns:
            The normalized engine, precision, and scope slug.
        """
        return f"{self.engine}-{self.weight}-{self.activation}-{self.scope}"


@dataclass(frozen=True)
class CompileSpec:
    compiler: str
    target: str
    scope: str

    def __post_init__(self) -> None:
        """Validate that the compilation fields form a supported contract.

        Returns:
            None.

        Raises:
            ValueError: If a token is malformed or the contract is unsupported.
        """
        for name, value in asdict(self).items():
            _validate_token(name, value)
        if (self.compiler, self.target, self.scope) not in _ALLOWED_COMPILATION:
            raise ValueError(f"Unsupported compilation contract: {self.slug!r}")

    @property
    def slug(self) -> str:
        """Render the canonical compilation segment used in artifact IDs.

        Returns:
            The normalized compiler, target, and scope slug.
        """
        return f"{self.compiler}-{self.target}-{self.scope}"


@dataclass(frozen=True)
class ArtifactSpec:
    model: str
    quantization: QuantizationSpec
    shape: str
    compilation: CompileSpec

    def __post_init__(self) -> None:
        """Validate the model name and fixed-shape identity.

        Returns:
            None.

        Raises:
            ValueError: If the model or shape slug is not canonical.
        """
        _validate_token("model", self.model)
        if not _SHAPE.fullmatch(self.shape):
            raise ValueError(f"Invalid artifact shape slug: {self.shape!r}")
        if self.model not in {"zipformer", "vpcd"}:
            raise ValueError(f"Unsupported model: {self.model!r}")

    @property
    def artifact_id(self) -> str:
        """Render the complete canonical artifact identifier.

        Returns:
            The model, quantization, shape, and compilation identity.
        """
        return (
            f"{self.model}__q-{self.quantization.slug}__s-{self.shape}"
            f"__c-{self.compilation.slug}"
        )

    @classmethod
    def parse(cls, artifact_id: str) -> "ArtifactSpec":
        """Parse and validate a canonical artifact identifier.

        Args:
            artifact_id: Identifier using the public artifact grammar.

        Returns:
            The validated structured artifact specification.

        Raises:
            ValueError: If the identifier is malformed, unsupported, or noncanonical.
        """
        parts = artifact_id.split("__")
        if len(parts) != 4 or not parts[1].startswith("q-") or not parts[2].startswith("s-") or not parts[3].startswith("c-"):
            raise ValueError(f"Invalid artifact ID: {artifact_id!r}")

        quantization = _parse_quantization(parts[1][2:])
        compilation = _parse_compilation(parts[3][2:])
        result = cls(parts[0], quantization, parts[2][2:], compilation)
        if result.artifact_id != artifact_id:
            raise ValueError(f"Artifact ID is not canonical: {artifact_id!r}")
        return result

    def to_dict(self) -> dict[str, Any]:
        """Serialize the artifact specification to manifest-compatible fields.

        Returns:
            A JSON-compatible artifact mapping.
        """
        return {
            "model": self.model,
            "quantization": asdict(self.quantization),
            "shape": self.shape,
            "compilation": asdict(self.compilation),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactSpec":
        """Construct an artifact specification from manifest-compatible fields.

        Args:
            payload: Mapping containing model, quantization, shape, and compilation fields.

        Returns:
            The validated artifact specification.

        Raises:
            KeyError: If a required field is missing.
            ValueError: If a field violates the artifact contract.
        """
        return cls(
            model=str(payload["model"]),
            quantization=QuantizationSpec(**dict(payload["quantization"])),
            shape=str(payload["shape"]),
            compilation=CompileSpec(**dict(payload["compilation"])),
        )


@dataclass(frozen=True)
class RecipeSpec:
    artifact: ArtifactSpec
    configuration: str
    components: tuple[str, ...]
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Validate configuration and component invariants for a recipe.

        Returns:
            None.

        Raises:
            ValueError: If the configuration is unsupported or components are invalid.
        """
        supported_configurations = {
            "fp32-fixed-shape",
            "fp32-fixed-shape-aihub-encoder",
            "ortqnn-uint8-uint16-encoder-matmul",
            "aimet-int8-int16-encoder-matmul",
        }
        if self.configuration not in supported_configurations:
            raise ValueError(f"Unsupported configuration: {self.configuration!r}")
        if not self.components or len(set(self.components)) != len(self.components):
            raise ValueError("Recipe components must be non-empty and unique")

    @property
    def digest(self) -> str:
        """Compute the deterministic digest of all recipe-defining fields.

        Returns:
            The lowercase recipe digest used for cache invalidation.
        """
        return stable_digest(
            {
                "artifact": self.artifact.to_dict(),
                "configuration": self.configuration,
                "components": list(self.components),
                "parameters": dict(self.parameters),
            }
        )


def _validate_token(name: str, value: str) -> None:
    """Validate one lowercase token used by artifact identity fields.

    Args:
        name: Human-readable field name used in validation errors.
        value: Token value to validate.

    Returns:
        None.

    Raises:
        ValueError: If the value is not a canonical token.
    """
    if not isinstance(value, str) or not _TOKEN.fullmatch(value):
        raise ValueError(f"Invalid {name}: {value!r}")


def _parse_quantization(slug: str) -> QuantizationSpec:
    """Parse a quantization slug into its structured contract.

    Args:
        slug: Engine, precision, and scope segment without the `q-` prefix.

    Returns:
        The validated quantization specification.

    Raises:
        ValueError: If the slug is malformed or unsupported.
    """
    tokens = slug.split("-")
    if len(tokens) < 4:
        raise ValueError(f"Invalid quantization slug: {slug!r}")
    return QuantizationSpec(tokens[0], tokens[1], tokens[2], "-".join(tokens[3:]))


def _parse_compilation(slug: str) -> CompileSpec:
    """Parse a compilation slug while preserving hyphenated targets.

    Args:
        slug: Compiler, target, and scope segment without the `c-` prefix.

    Returns:
        The validated compilation specification.

    Raises:
        ValueError: If the slug or scope is malformed or unsupported.
    """
    tokens = slug.split("-")
    if len(tokens) < 3:
        raise ValueError(f"Invalid compilation slug: {slug!r}")
    compiler = tokens[0]
    remainder = tokens[1:]
    known_scopes = {"all", "encoder", "model", "none"}
    if remainder[-1] not in known_scopes:
        raise ValueError(f"Invalid compilation scope: {remainder[-1]!r}")
    return CompileSpec(compiler, "-".join(remainder[:-1]), remainder[-1])
