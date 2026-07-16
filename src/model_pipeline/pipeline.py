from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from model_pipeline.core import RecipeSpec, Stage, StageRunner, ValidationResult, sha256_file, stable_digest
from model_pipeline.integrations.aihub import AiHubClient, CompileRequest, EvidenceStore, compile_or_reuse
from model_pipeline.integrations.android import BundleResult, materialize_bundle, sync_bundle
from model_pipeline.models.base import CompileInput, ModelAdapter


@dataclass(frozen=True)
class PipelineResult:
    recipe: RecipeSpec
    validation: ValidationResult
    bundle: BundleResult | None
    resumed_stages: tuple[str, ...]


class ModelPipeline:
    def __init__(
        self,
        *,
        build_root: str | Path,
        evidence_store: EvidenceStore,
        aihub_client: AiHubClient | None,
    ):
        """Initialize stage execution and external integration dependencies.

        Args:
            build_root: Directory used for deterministic stage state and outputs.
            evidence_store: Checksum-keyed Qualcomm AI Hub evidence store.
            aihub_client: Compile client, or `None` for recipes that skip hosted stages.

        Returns:
            None.
        """
        self.runner = StageRunner(build_root)
        self.evidence_store = evidence_store
        self.aihub_client = aihub_client

    def run(
        self,
        *,
        recipe: RecipeSpec,
        adapter: ModelAdapter,
        through: str | Stage,
        android_destination: str | Path | None = None,
    ) -> PipelineResult:
        """Run the canonical stage sequence through a requested terminal stage.

        Args:
            recipe: Validated model configuration controlling every stage.
            adapter: Model-specific implementation of preparation and validation.
            through: Final stage to execute, inclusive.
            android_destination: Optional directory receiving the packaged bundle.

        Returns:
            Validation, bundle, and resume evidence accumulated by the pipeline.

        Raises:
            FileNotFoundError: If any source artifact is unavailable.
            RuntimeError: If compilation is required without an AI Hub client.
            ValueError: If model validation or a stage output contract fails.
        """
        final_stage = Stage(through)
        stages = Stage.ordered()[: Stage.ordered().index(final_stage) + 1]
        resumed: list[str] = []
        artifact_id = recipe.artifact.artifact_id
        sources = {role: Path(path).resolve() for role, path in adapter.source_files(recipe).items()}
        missing = [path for path in sources.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing model source files: {missing!r}")

        source_digests = {role: sha256_file(path) for role, path in sorted(sources.items())}

        def source_action(stage_dir: Path):
            """Write a deterministic descriptor for resolved source checksums.

            Args:
                stage_dir: Directory allocated to the source stage.

            Returns:
                The logical source role mapped to its descriptor file.
            """
            descriptor = stage_dir / "source.json"
            descriptor.write_text(
                json.dumps({"components": source_digests}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return {"source": descriptor}

        source_result = self.runner.run(
            stage=Stage.SOURCE,
            artifact_id=artifact_id,
            recipe_digest=recipe.digest,
            input_digests=source_digests,
            execute=source_action,
        )
        _track_resume(source_result, resumed)
        if final_stage == Stage.SOURCE:
            return PipelineResult(recipe, ValidationResult("not-run", {}), None, tuple(resumed))

        prepare_result = self.runner.run(
            stage=Stage.PREPARE,
            artifact_id=artifact_id,
            recipe_digest=recipe.digest,
            input_digests=source_digests,
            execute=lambda directory: adapter.prepare(recipe, sources, directory),
        )
        _track_resume(prepare_result, resumed)
        prepared = dict(prepare_result.outputs)
        if final_stage == Stage.PREPARE:
            return PipelineResult(recipe, ValidationResult("not-run", {}), None, tuple(resumed))

        quantize_result = self.runner.run(
            stage=Stage.QUANTIZE,
            artifact_id=artifact_id,
            recipe_digest=recipe.digest,
            input_digests={**source_digests, **prepare_result.output_digests},
            execute=lambda directory: adapter.quantize(recipe, prepared, directory),
        )
        _track_resume(quantize_result, resumed)
        quantized_components = dict(quantize_result.outputs)
        if final_stage == Stage.QUANTIZE:
            return PipelineResult(recipe, ValidationResult("not-run", {}), None, tuple(resumed))

        validation_holder: dict[str, ValidationResult] = {}

        def validate_action(stage_dir: Path):
            """Run model validation and materialize its structured report.

            Args:
                stage_dir: Directory allocated to the validation stage.

            Returns:
                The validation role mapped to its report file.

            Raises:
                ValueError: If the quantized components fail their model-specific contract.
            """
            validation = adapter.validate(recipe, quantized_components)
            validation_holder["result"] = validation
            report = stage_dir / "validation.json"
            report.write_text(
                json.dumps({"status": validation.status, "checks": dict(validation.checks)}, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            if validation.status != "passed":
                raise ValueError(f"Model validation failed: {validation.checks!r}")
            return {"validation": report}

        validate_result = self.runner.run(
            stage=Stage.VALIDATE,
            artifact_id=artifact_id,
            recipe_digest=recipe.digest,
            input_digests=quantize_result.output_digests,
            execute=validate_action,
        )
        _track_resume(validate_result, resumed)
        validation = validation_holder.get("result") or _read_validation(validate_result.outputs["validation"])
        if final_stage == Stage.VALIDATE:
            return PipelineResult(recipe, validation, None, tuple(resumed))

        compile_specs = tuple(adapter.compile_inputs(recipe, quantized_components))

        def compile_action(stage_dir: Path):
            """Compile requested components or record an explicit compile skip.

            Args:
                stage_dir: Directory allocated to the compile stage.

            Returns:
                Compiled components and support files keyed by logical role.

            Raises:
                RuntimeError: If hosted compilation is required without a client.
            """
            if not compile_specs:
                skipped = stage_dir / "compile-skip.json"
                skipped.write_text('{"action":"explicit-skip"}\n', encoding="utf-8")
                return {"compile_skip": skipped}
            if self.aihub_client is None:
                raise RuntimeError("AI Hub client is required for this recipe")
            outputs: dict[str, Path] = {}
            for item in compile_specs:
                result = compile_or_reuse(
                    CompileRequest(
                        recipe.artifact,
                        item.role,
                        item.source_path,
                        item.input_shapes,
                        item.truncate_64bit_io,
                        item.input_dtypes,
                    ),
                    client=self.aihub_client,
                    evidence_store=self.evidence_store,
                    output_dir=stage_dir,
                )
                outputs[item.role] = result.output_path
                for support in result.support_files:
                    key = (
                        f"{item.role}_external_data"
                        if support.suffix.lower() in {".bin", ".data"}
                        else f"{item.role}_support_{support.name}"
                    )
                    outputs[key] = support
            return outputs

        compile_result = self.runner.run(
            stage=Stage.COMPILE,
            artifact_id=artifact_id,
            recipe_digest=recipe.digest,
            input_digests={**quantize_result.output_digests, **validate_result.output_digests},
            execute=compile_action,
        )
        _track_resume(compile_result, resumed)
        compiled_components = {} if not compile_specs else dict(compile_result.outputs)
        if final_stage == Stage.COMPILE:
            return PipelineResult(recipe, validation, None, tuple(resumed))

        bundle_holder: dict[str, BundleResult] = {}

        def package_action(stage_dir: Path):
            """Materialize a manifest-v2 bundle from validated and compiled components.

            Args:
                stage_dir: Directory allocated to the package stage.

            Returns:
                Every package file keyed by its filename for stage tracking.
            """
            bundle = materialize_bundle(
                artifact=recipe.artifact,
                components=adapter.bundle_components(
                    recipe,
                    quantized_components,
                    compiled_components,
                ),
                output_dir=stage_dir / "bundle",
                input_shapes_by_role=_component_input_shapes(recipe),
                source_checksums=source_digests,
                recipe_digest=recipe.digest,
                validation=validation,
                runtime_metadata=recipe.parameters.get("runtime_metadata", {}),
            )
            bundle_holder["result"] = bundle
            return {
                path.name: path
                for path in sorted(bundle.bundle_dir.iterdir())
                if path.is_file()
            }

        package_result = self.runner.run(
            stage=Stage.PACKAGE,
            artifact_id=artifact_id,
            recipe_digest=recipe.digest,
            input_digests={**quantize_result.output_digests, **compile_result.output_digests},
            execute=package_action,
        )
        _track_resume(package_result, resumed)
        bundle = bundle_holder.get("result") or _restore_bundle(package_result)
        if final_stage == Stage.PACKAGE:
            return PipelineResult(recipe, validation, bundle, tuple(resumed))

        def sync_action(stage_dir: Path):
            """Synchronize a bundle or record an explicit destination skip.

            Args:
                stage_dir: Directory allocated to the synchronization stage.

            Returns:
                The synchronization role mapped to its result record.
            """
            record = stage_dir / "sync.json"
            if android_destination is None:
                payload = {"action": "explicit-skip"}
            else:
                sync_bundle(bundle.bundle_dir, android_destination)
                payload = {"action": "synced", "bundle_checksum": bundle.bundle_checksum}
            record.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return {"sync": record}

        sync_result = self.runner.run(
            stage=Stage.SYNC,
            artifact_id=artifact_id,
            recipe_digest=recipe.digest,
            input_digests={
                **package_result.output_digests,
                "destination": stable_digest(str(Path(android_destination).resolve()) if android_destination else "none"),
            },
            execute=sync_action,
        )
        _track_resume(sync_result, resumed)
        return PipelineResult(recipe, validation, bundle, tuple(resumed))


def _track_resume(result, resumed: list[str]) -> None:
    """Append a stage name when its outputs were restored from verified cache.

    Args:
        result: Stage execution result containing the resume flag.
        resumed: Mutable list accumulating resumed stage names.

    Returns:
        None.
    """
    if result.resumed:
        resumed.append(result.stage.value)


def _read_validation(path: Path) -> ValidationResult:
    """Restore validation evidence from a cached JSON report.

    Args:
        path: Validation report path.

    Returns:
        The restored validation result.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ValidationResult(payload["status"], payload["checks"])


def _restore_bundle(package_result) -> BundleResult:
    """Reconstruct a bundle result from resumed package-stage outputs.

    Args:
        package_result: Cached package-stage result containing the manifest.

    Returns:
        Bundle directory, manifest path, and deterministic manifest digest.
    """
    manifest = package_result.outputs["artifact-manifest.json"]
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return BundleResult(manifest.parent, manifest, stable_digest(payload))


def _component_input_shapes(recipe: RecipeSpec) -> dict[str, Mapping[str, list[int | str]]]:
    """Map recipe fixed shapes to the component roles stored in manifest v2.

    Args:
        recipe: Canonical recipe containing model-specific fixed input shapes.

    Returns:
        Component roles mapped to named input shape arrays.
    """
    shapes = recipe.parameters.get("fixed_input_shapes", {})
    if recipe.artifact.model == "zipformer":
        return {str(role): dict(inputs) for role, inputs in shapes.items()}
    return {"model": {str(name): list(shape) for name, shape in shapes.items()}}
