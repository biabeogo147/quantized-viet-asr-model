from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from model_pipeline.core.files import sha256_file
from model_pipeline.core.specs import ArtifactSpec, Stage


@dataclass(frozen=True)
class StageResult:
    stage: Stage
    stage_dir: Path
    outputs: Mapping[str, Path]
    output_digests: Mapping[str, str]
    resumed: bool


class StageRunner:
    """Runs deterministic stages and resumes only from verified stage state."""

    def __init__(self, build_root: str | Path):
        """Initialize deterministic stage storage below a build root.

        Args:
            build_root: Directory that owns per-artifact stage outputs and state.

        Returns:
            None.
        """
        self.build_root = Path(build_root)

    def run(
        self,
        *,
        stage: Stage,
        artifact_id: str,
        recipe_digest: str,
        input_digests: Mapping[str, str],
        execute: Callable[[Path], Mapping[str, Path]],
    ) -> StageResult:
        """Execute a stage or resume it when all cached evidence still matches.

        Args:
            stage: Canonical pipeline stage being executed.
            artifact_id: Canonical identity that namespaces stage outputs.
            recipe_digest: Digest of every recipe-defining field.
            input_digests: Digests of all inputs consumed by the stage.
            execute: Callback that materializes output files inside the stage directory.

        Returns:
            Verified output paths, digests, and resume status for the stage.

        Raises:
            ValueError: If the artifact ID or produced output set violates the contract.
        """
        ArtifactSpec.parse(artifact_id)
        stage_dir = self.build_root / artifact_id / stage.value
        state_path = stage_dir / "stage-state.json"
        expected_inputs = dict(sorted(input_digests.items()))
        cached = self._read_state(state_path)
        if self._cache_matches(cached, artifact_id, stage, recipe_digest, expected_inputs, stage_dir):
            outputs = {name: stage_dir / relative for name, relative in cached["outputs"].items()}
            return StageResult(stage, stage_dir, outputs, dict(cached["output_digests"]), True)

        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        stage_dir.mkdir(parents=True, exist_ok=True)
        raw_outputs = dict(execute(stage_dir))
        outputs = self._normalize_outputs(stage_dir, raw_outputs)
        digests = {name: sha256_file(path) for name, path in outputs.items()}
        state = {
            "schema_version": 1,
            "artifact_id": artifact_id,
            "stage": stage.value,
            "recipe_digest": recipe_digest,
            "input_digests": expected_inputs,
            "outputs": {name: path.relative_to(stage_dir).as_posix() for name, path in outputs.items()},
            "output_digests": digests,
        }
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return StageResult(stage, stage_dir, outputs, digests, False)

    @staticmethod
    def _read_state(path: Path) -> dict | None:
        """Read cached stage state while treating corruption as a cache miss.

        Args:
            path: Stage-state JSON path.

        Returns:
            The decoded state mapping, or `None` when unavailable or invalid.
        """
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    @staticmethod
    def _cache_matches(
        state: dict | None,
        artifact_id: str,
        stage: Stage,
        recipe_digest: str,
        input_digests: Mapping[str, str],
        stage_dir: Path,
    ) -> bool:
        """Check cached metadata and output bytes against expected stage inputs.

        Args:
            state: Previously decoded stage state, if available.
            artifact_id: Expected canonical artifact identity.
            stage: Expected pipeline stage.
            recipe_digest: Expected recipe digest.
            input_digests: Expected input digest mapping.
            stage_dir: Directory containing cached output files.

        Returns:
            `True` only when metadata and every output checksum match.
        """
        if not state:
            return False
        if (
            state.get("artifact_id") != artifact_id
            or state.get("stage") != stage.value
            or state.get("recipe_digest") != recipe_digest
            or state.get("input_digests") != input_digests
        ):
            return False
        outputs = state.get("outputs") or {}
        digests = state.get("output_digests") or {}
        if not outputs or outputs.keys() != digests.keys():
            return False
        for name, relative in outputs.items():
            path = stage_dir / relative
            if not path.is_file() or sha256_file(path) != digests[name]:
                return False
        return True

    @staticmethod
    def _normalize_outputs(stage_dir: Path, outputs: Mapping[str, Path]) -> dict[str, Path]:
        """Validate and normalize stage outputs to files inside the stage directory.

        Args:
            stage_dir: Directory allocated to the current stage.
            outputs: Logical output roles mapped to produced file paths.

        Returns:
            A role-sorted mapping of resolved output paths.

        Raises:
            ValueError: If no outputs exist or an output escapes the stage directory.
        """
        if not outputs:
            raise ValueError("A pipeline stage must produce at least one output")
        stage_root = stage_dir.resolve()
        normalized: dict[str, Path] = {}
        for name, raw_path in sorted(outputs.items()):
            path = Path(raw_path).resolve()
            if stage_root not in path.parents or not path.is_file():
                raise ValueError(f"Stage output must be a file inside its stage directory: {raw_path}")
            normalized[str(name)] = path
        return normalized
