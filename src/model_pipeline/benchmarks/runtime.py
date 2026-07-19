"""Filesystem orchestration for benchmark result commands."""

from __future__ import annotations

import json
from pathlib import Path

from model_pipeline.benchmarks.report import build_comparison


def aggregate_result_directory(*, results_root: Path, output_dir: Path) -> dict[str, object]:
    """Aggregate raw Android result files for every present model.

    Args:
        results_root: Directory recursively containing result JSON files.
        output_dir: Directory receiving machine-readable comparisons.

    Returns:
        Model names mapped to comparison results.

    Raises:
        FileNotFoundError: If no result JSON files are present.
    """
    paths = sorted(Path(results_root).resolve().rglob("result-*.json"))
    if not paths:
        raise FileNotFoundError("No Android benchmark result JSON files were found")
    by_model: dict[str, list[dict[str, object]]] = {}
    for path in paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        by_model.setdefault(str(row["model"]), []).append(row)
    comparisons = {model: build_comparison(model, rows) for model, rows in sorted(by_model.items())}
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "comparison.json").write_text(
        json.dumps(comparisons, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return comparisons
