from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from model_pipeline.core import RecipeSpec, ValidationResult


@dataclass(frozen=True)
class CompileInput:
    role: str
    source_path: Path
    input_shapes: Mapping[str, list[int]]
    truncate_64bit_io: bool
    input_dtypes: Mapping[str, str] | None = None


class ModelAdapter(Protocol):
    def source_files(self, recipe: RecipeSpec) -> Mapping[str, Path]:
        """Resolve every source file required by a recipe.

        Args:
            recipe: Canonical recipe selecting model inputs and configuration.

        Returns:
            Logical source roles mapped to existing paths.
        """
        ...

    def prepare(self, recipe: RecipeSpec, sources: Mapping[str, Path], output_dir: Path) -> Mapping[str, Path]:
        """Apply fixed-shape and graph preparation transforms.

        Args:
            recipe: Canonical recipe controlling preparation.
            sources: Resolved source files by logical role.
            output_dir: Stage directory for prepared artifacts.

        Returns:
            Prepared component roles mapped to output files.
        """
        ...

    def quantize(self, recipe: RecipeSpec, prepared: Mapping[str, Path], output_dir: Path) -> Mapping[str, Path]:
        """Quantize prepared components or record an explicit precision skip.

        Args:
            recipe: Canonical recipe controlling quantization.
            prepared: Prepared component files by role.
            output_dir: Stage directory for quantized artifacts.

        Returns:
            Quantized components and supporting quantization evidence.
        """
        ...

    def validate(self, recipe: RecipeSpec, quantized_components: Mapping[str, Path]) -> ValidationResult:
        """Validate graph and precision contracts for quantized components.

        Args:
            recipe: Canonical recipe containing expected contracts.
            quantized_components: Quantized component files by role.

        Returns:
            Structured pass/fail evidence.
        """
        ...

    def compile_inputs(
        self,
        recipe: RecipeSpec,
        validated_components: Mapping[str, Path],
    ) -> Sequence[CompileInput]:
        """Describe components that require hosted compilation.

        Args:
            recipe: Canonical recipe controlling compilation.
            validated_components: Validated component files.

        Returns:
            Zero or more compile-input contracts.
        """
        ...

    def bundle_components(
        self,
        recipe: RecipeSpec,
        validated_components: Mapping[str, Path],
        compiled_components: Mapping[str, Path],
    ) -> Mapping[str, tuple[Path, str, str]]:
        """Describe files and runtime truth to store in an Android bundle.

        Args:
            recipe: Canonical recipe defining component precision and targets.
            validated_components: Validated local component files.
            compiled_components: Hosted compilation outputs and support files.

        Returns:
            Component roles mapped to file, execution target, and format tuples.
        """
        ...
