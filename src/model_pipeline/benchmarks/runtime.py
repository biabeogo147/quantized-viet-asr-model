"""Filesystem orchestration for benchmark payload and result commands."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from model_pipeline.benchmarks.payload import materialize_payload
from model_pipeline.benchmarks.report import build_comparison
from model_pipeline.benchmarks.graph import validate_benchmark_qdq
from model_pipeline.models import get_recipe
from model_pipeline.models.aimet_service import AimetServiceClient


def materialize_from_pipeline_build(*, model: str, build_root: Path, output_dir: Path) -> Path:
    """Materialize a payload from canonical pipeline stage outputs.

    Args:
        model: Canonical model family.
        build_root: Root containing deterministic pipeline artifact stages.
        output_dir: Directory receiving the Android benchmark payload.

    Returns:
        Generated benchmark manifest path.

    Raises:
        FileNotFoundError: If stage outputs or benchmark fixtures are unavailable.
    """
    recipe = get_recipe(model, "aimet-int8-int16-encoder-matmul")
    artifact_root = Path(build_root).resolve() / recipe.artifact.artifact_id
    prepare = _read_stage_outputs(artifact_root / "prepare")
    quantize = _read_stage_outputs(artifact_root / "quantize")
    compile_outputs = _read_stage_outputs(artifact_root / "compile")
    aimet_service = AimetServiceClient(
        repo_root=Path.cwd().resolve(),
        url=os.environ.get("AIMET_SERVICE_URL", "http://127.0.0.1:18080"),
    )
    qdq_outputs = export_qdq_from_pipeline_build(
        model=model,
        artifact_root=artifact_root,
        aimet_service=aimet_service,
    )
    validate_benchmark_qdq(model, qdq_outputs["model"], quantize["encodings"])
    fixture_path = artifact_root / "benchmark-fixtures.npz"
    expected_path = artifact_root / "benchmark-expected.json"
    if not fixture_path.is_file() or not expected_path.is_file():
        raise FileNotFoundError(
            "Benchmark fixtures must be materialized as benchmark-fixtures.npz and benchmark-expected.json"
        )
    with np.load(fixture_path, allow_pickle=False) as arrays:
        indexes = sorted({int(name.split("__", 1)[0]) for name in arrays.files})
        fixtures = [
            {
                name.split("__", 1)[1]: np.asarray(arrays[name])
                for name in arrays.files
                if int(name.split("__", 1)[0]) == index
            }
            for index in indexes
        ]
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    model_role = "encoder" if model == "zipformer" else "model"
    components = {
        "fp32_model": prepare[model_role],
        "qdq_model": qdq_outputs["model"],
        "compiled_model": compile_outputs[model_role],
        "compiled_external_data": compile_outputs[f"{model_role}_external_data"],
    }
    if "external_data" in qdq_outputs:
        components["qdq_external_data"] = qdq_outputs["external_data"]
    for role in (
        "decoder",
        "joiner",
        "tokens",
        "tokenizer_encode",
        "tokenizer_decode",
        "tokenizer_to_model_id_map",
        "model_to_tokenizer_id_map",
        "autoregressive_loop",
    ):
        if role in quantize:
            components[role] = quantize[role]
        elif role in prepare:
            components[role] = prepare[role]
    return materialize_payload(
        model=model,
        artifact_id=recipe.artifact.artifact_id,
        components=components,
        fixtures=fixtures,
        expected_outputs=expected,
        output_dir=output_dir,
    )


def export_qdq_from_pipeline_build(
    *,
    model: str,
    artifact_root: Path,
    aimet_service,
) -> dict[str, Path]:
    """Export benchmark QDQ from exact prepared and AIMET stage outputs.

    Args:
        model: Canonical model family selecting the prepared model role.
        artifact_root: Deterministic pipeline directory for one artifact ID.
        aimet_service: Healthy service client implementing strict QDQ export.

    Returns:
        QDQ model and optional external-data paths.

    Raises:
        FileNotFoundError: If the service does not materialize its declared files.
    """
    root = Path(artifact_root).resolve()
    prepare = _read_stage_outputs(root / "prepare")
    quantize = _read_stage_outputs(root / "quantize")
    model_role = "encoder" if model == "zipformer" else "model"
    output_dir = root / "benchmark-qdq"
    aimet_service.healthcheck()
    response = aimet_service.export_qdq(
        fp32_model_path=prepare[model_role],
        encodings_path=quantize["encodings"],
        output_dir=output_dir,
        config_path=quantize["aimet_config"],
        policy_path=quantize["quantization_policy"],
    )
    declared = dict(response.get("outputs") or {})
    model_name = str(declared.get("model", "model.qdq.onnx"))
    outputs = {"model": output_dir / model_name}
    external_name = declared.get("external_data")
    if external_name:
        outputs["external_data"] = output_dir / str(external_name)
    missing = [path for path in outputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"AIMET service did not materialize QDQ outputs: {missing!r}")
    return outputs


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


def _read_stage_outputs(stage_dir: Path) -> dict[str, Path]:
    """Resolve verified stage-state output paths.

    Args:
        stage_dir: Deterministic pipeline stage directory.

    Returns:
        Logical roles mapped to existing output files.

    Raises:
        FileNotFoundError: If state or any referenced output is missing.
    """
    state_path = stage_dir / "stage-state.json"
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    outputs = {name: stage_dir / relative for name, relative in state["outputs"].items()}
    missing = [path for path in outputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing pipeline stage outputs: {missing!r}")
    return outputs
