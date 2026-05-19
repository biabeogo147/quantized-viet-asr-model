from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from tools.aihub_option1_pilots import Option1RuntimeConfig

GO = "GO"
WARN = "WARN"
NO_GO = "NO_GO"

EXACT_MATCH = "exact_match"
MINOR_TEXT_DRIFT = "minor_text_drift"
MAJOR_TEXT_DRIFT = "major_text_drift"
CATASTROPHIC_DECODE_FAILURE = "catastrophic_decode_failure"
COMPARISON_UNAVAILABLE = "comparison_unavailable"

PHASE4_RECORD_KIND = "phase4_gate"


@dataclass(frozen=True)
class Option1PilotLayout:
    canonical_name: str
    phase2_compile_pilot_name: str
    phase3_hybrid_pilot_name: str
    phase4_gate_pilot_name: str
    contract_pilot_name: str
    deployment_notes: tuple[str, ...]


@dataclass(frozen=True)
class Phase4GateConfig:
    minor_text_drift_threshold: float = 0.12
    catastrophic_short_generated_id_count: int = 3
    catastrophic_expected_text_min_chars: int = 12
    zipformer_go_max_average_cloud_inference_seconds: float | None = 240.0
    zipformer_warn_max_average_cloud_inference_seconds: float | None = 360.0
    vpcd_go_max_average_cloud_inference_seconds: float | None = 480.0
    vpcd_warn_max_average_cloud_inference_seconds: float | None = 900.0


PILOT_LAYOUTS = {
    "zipformer": Option1PilotLayout(
        canonical_name="zipformer",
        phase2_compile_pilot_name="zipformer_encoder_option1",
        phase3_hybrid_pilot_name="zipformer_hybrid_option1",
        phase4_gate_pilot_name="zipformer_phase4_option1",
        contract_pilot_name="zipformer",
        deployment_notes=(
            "Zipformer is currently validated only as encoder-only on NPU; decoder and joiner remain on CPU.",
        ),
    ),
    "vpcd": Option1PilotLayout(
        canonical_name="vpcd",
        phase2_compile_pilot_name="vpcd_option1_local_aimet",
        phase3_hybrid_pilot_name="vpcd_hybrid_option1",
        phase4_gate_pilot_name="vpcd_phase4_option1",
        contract_pilot_name="vpcd",
        deployment_notes=(
            "VPCD currently keeps tokenizer encode/decode on CPU while the model step runs on NPU.",
        ),
    ),
}


def build_phase4_gate_config(**overrides: Any) -> Phase4GateConfig:
    return replace(Phase4GateConfig(), **overrides)


def resolve_option1_pilot_layout(pilot_name: str) -> Option1PilotLayout:
    normalized = _normalize_optional_string(pilot_name)
    if not normalized:
        raise ValueError("pilot_name must not be empty.")
    for layout in PILOT_LAYOUTS.values():
        if normalized in {
            layout.canonical_name,
            layout.phase2_compile_pilot_name,
            layout.phase3_hybrid_pilot_name,
            layout.phase4_gate_pilot_name,
            layout.contract_pilot_name,
        }:
            return layout
    raise ValueError(f"Unsupported Option 1 pilot name: {pilot_name}")


def run_phase4_benchmark_sweep(
    *,
    hybrid_runner: Callable[..., Mapping[str, Any]],
    iterations: int,
    max_samples: int,
    run_label: str | None = None,
    explicit_target_model_id: str | None = None,
) -> dict[str, Any]:
    if int(iterations) <= 0:
        raise ValueError("iterations must be at least 1.")

    iteration_rows: list[dict[str, Any]] = []
    last_report: Mapping[str, Any] | None = None
    for index in range(int(iterations)):
        report = hybrid_runner(
            run_label=run_label,
            max_samples=int(max_samples),
            explicit_target_model_id=explicit_target_model_id,
        )
        last_report = report
        sample_results = list(report.get("results", []))
        cloud_total = round(sum(float(row.get("cloud_inference_seconds") or 0.0) for row in sample_results), 6)
        decode_total = round(sum(float(row.get("decode_seconds") or 0.0) for row in sample_results), 6)
        total_seconds = round(cloud_total + decode_total, 6)
        record_path = report.get("record_path")
        iteration_rows.append(
            {
                "iteration_index": index + 1,
                "total_seconds": total_seconds,
                "cloud_inference_seconds": cloud_total,
                "decode_seconds": decode_total,
                "record_path": _normalize_path_value(record_path),
            }
        )

    warmup_row = iteration_rows[0]
    steady_state_rows = iteration_rows[1:]
    steady_totals = [row["total_seconds"] for row in steady_state_rows]
    steady_cloud = [row["cloud_inference_seconds"] for row in steady_state_rows]
    steady_decode = [row["decode_seconds"] for row in steady_state_rows]
    latency_summary = _summarize_sample_latencies((last_report or {}).get("results", []))
    return {
        "iterations": iteration_rows,
        "warmup": {
            "total_seconds": warmup_row["total_seconds"],
            "cloud_inference_seconds": warmup_row["cloud_inference_seconds"],
            "decode_seconds": warmup_row["decode_seconds"],
        },
        "steady_state": {
            "count": len(steady_state_rows),
            "total_seconds_mean": _mean_or_none(steady_totals),
            "total_seconds_min": min(steady_totals) if steady_totals else None,
            "total_seconds_max": max(steady_totals) if steady_totals else None,
            "cloud_inference_seconds_mean": _mean_or_none(steady_cloud),
            "decode_seconds_mean": _mean_or_none(steady_decode),
        },
        "latency_summary": latency_summary,
        "last_report": dict(last_report or {}),
    }


def classify_phase4_sample(
    *,
    pilot_name: str,
    sample_result: Mapping[str, Any],
    config: Phase4GateConfig,
) -> dict[str, Any]:
    _ = resolve_option1_pilot_layout(pilot_name)
    sample_key = sample_result.get("sample_id", sample_result.get("sample_index", "sample"))
    actual_text = str(sample_result.get("text", "") or "")
    expected_text = str(sample_result.get("expected_text", "") or "")
    generated_ids = [int(value) for value in sample_result.get("generated_ids", [])]
    comparison_note = _normalize_optional_string(sample_result.get("comparison_note"))
    truncated_by_decode_step_limit = bool(sample_result.get("truncated_by_decode_step_limit"))

    if not expected_text:
        return {
            "sample_key": sample_key,
            "severity": COMPARISON_UNAVAILABLE,
            "normalized_text_distance": None,
            "reasons": ["expected_text_unavailable"],
        }
    if truncated_by_decode_step_limit or comparison_note == "decode_step_limit_reached_before_eos":
        return {
            "sample_key": sample_key,
            "severity": COMPARISON_UNAVAILABLE,
            "normalized_text_distance": None,
            "reasons": ["decode_step_limit_reached_before_eos"],
        }

    if actual_text == expected_text:
        return {
            "sample_key": sample_key,
            "severity": EXACT_MATCH,
            "normalized_text_distance": 0.0,
            "reasons": ["exact_text_match"],
        }

    catastrophic_reasons = _catastrophic_failure_reasons(
        actual_text=actual_text,
        expected_text=expected_text,
        generated_ids=generated_ids,
        config=config,
    )
    if catastrophic_reasons:
        return {
            "sample_key": sample_key,
            "severity": CATASTROPHIC_DECODE_FAILURE,
            "normalized_text_distance": None,
            "reasons": catastrophic_reasons,
        }

    distance = _normalized_text_distance(actual_text, expected_text)
    if distance <= float(config.minor_text_drift_threshold):
        return {
            "sample_key": sample_key,
            "severity": MINOR_TEXT_DRIFT,
            "normalized_text_distance": distance,
            "reasons": ["localized_text_drift"],
        }
    return {
        "sample_key": sample_key,
        "severity": MAJOR_TEXT_DRIFT,
        "normalized_text_distance": distance,
        "reasons": ["broad_text_divergence"],
    }


def build_phase4_correctness_summary(
    *,
    pilot_name: str,
    sample_results: Sequence[Mapping[str, Any]],
    config: Phase4GateConfig,
) -> dict[str, Any]:
    classified_rows = [
        classify_phase4_sample(pilot_name=pilot_name, sample_result=row, config=config)
        for row in sample_results
    ]
    severity_counts: dict[str, int] = {}
    for row in classified_rows:
        severity = str(row["severity"])
        severity_counts[severity] = severity_counts.get(severity, 0) + 1

    comparable_rows = [row for row in classified_rows if row["severity"] != COMPARISON_UNAVAILABLE]
    worst_severity = COMPARISON_UNAVAILABLE
    if comparable_rows:
        worst_severity = max(
            comparable_rows,
            key=lambda item: _severity_rank(str(item["severity"])),
        )["severity"]

    return {
        "sample_count": len(sample_results),
        "comparable_samples": len(comparable_rows),
        "matched_samples": sum(1 for row in classified_rows if row["severity"] == EXACT_MATCH),
        "mismatched_samples": sum(
            1
            for row in classified_rows
            if row["severity"] in {MINOR_TEXT_DRIFT, MAJOR_TEXT_DRIFT, CATASTROPHIC_DECODE_FAILURE}
        ),
        "severity_counts": severity_counts,
        "worst_severity": worst_severity,
        "sample_results": classified_rows,
    }


def build_phase4_footprint_summary(
    *,
    prepared_record: Mapping[str, Any],
    live_run_record: Mapping[str, Any] | None,
    hybrid_run_record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    prepared_model = prepared_record.get("prepared_model", {}) if isinstance(prepared_record, Mapping) else {}
    output_footprint_bytes = 0
    if isinstance(live_run_record, Mapping):
        for tensor_rows in (live_run_record.get("output_tensors") or {}).values():
            for item in tensor_rows or []:
                output_footprint_bytes += _tensor_summary_nbytes(item)

    generated_token_count = 0
    if isinstance(hybrid_run_record, Mapping):
        for row in hybrid_run_record.get("sample_results", []) or []:
            ids = row.get("generated_ids")
            if ids is None:
                ids = row.get("token_ids")
            if ids is None:
                continue
            generated_token_count += len(list(ids))

    return {
        "prepared_model_size_bytes": int(prepared_model.get("size_bytes", 0) or 0),
        "output_tensor_footprint_bytes": int(output_footprint_bytes),
        "generated_token_count": int(generated_token_count),
        "generated_token_footprint_bytes": int(generated_token_count * np.dtype(np.int64).itemsize),
        "host_rss_delta_bytes": None,
        "host_rss_status": "unavailable",
        "host_rss_reason": "host RSS observation is not wired into the shared notebook lane.",
    }


def build_phase4_recommendation(
    *,
    pilot_name: str,
    correctness_summary: Mapping[str, Any],
    benchmark_summary: Mapping[str, Any],
    config: Phase4GateConfig,
) -> dict[str, Any]:
    layout = resolve_option1_pilot_layout(pilot_name)
    average_cloud_seconds = _extract_average_cloud_inference_seconds(benchmark_summary)
    go_threshold, warn_threshold = _resolve_latency_thresholds(layout, config)
    worst_severity = str(correctness_summary.get("worst_severity", COMPARISON_UNAVAILABLE))
    reasons: list[str] = []

    if worst_severity == EXACT_MATCH:
        reasons.append("exact_match_only")
    elif worst_severity == MINOR_TEXT_DRIFT:
        reasons.append("minor_text_drift_present")
    elif worst_severity == MAJOR_TEXT_DRIFT:
        reasons.append("major_text_drift_present")
        return _finalize_recommendation(NO_GO, reasons, average_cloud_seconds, go_threshold, warn_threshold)
    elif worst_severity == CATASTROPHIC_DECODE_FAILURE:
        reasons.append("catastrophic_decode_failure_present")
        return _finalize_recommendation(NO_GO, reasons, average_cloud_seconds, go_threshold, warn_threshold)
    else:
        reasons.append("comparison_unavailable")
        return _finalize_recommendation(NO_GO, reasons, average_cloud_seconds, go_threshold, warn_threshold)

    latency_band = _classify_latency_band(
        average_cloud_seconds=average_cloud_seconds,
        go_threshold=go_threshold,
        warn_threshold=warn_threshold,
    )
    if latency_band == "go" and worst_severity == EXACT_MATCH:
        reasons.append("latency_within_go_threshold")
        return _finalize_recommendation(GO, reasons, average_cloud_seconds, go_threshold, warn_threshold)
    if latency_band in {"go", "warn"} and worst_severity in {EXACT_MATCH, MINOR_TEXT_DRIFT}:
        reasons.append("latency_within_warn_threshold")
        return _finalize_recommendation(WARN, reasons, average_cloud_seconds, go_threshold, warn_threshold)

    reasons.append("latency_above_warn_threshold")
    return _finalize_recommendation(NO_GO, reasons, average_cloud_seconds, go_threshold, warn_threshold)


def summarize_phase4_gate_reports(gate_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    recommendations = [str(report.get("recommendation", {}).get("value", "")) for report in gate_reports]
    overall = GO
    if NO_GO in recommendations:
        overall = NO_GO
    elif WARN in recommendations:
        overall = WARN
    return {
        "pilot_count": len(gate_reports),
        "recommendations": recommendations,
        "overall_recommendation": overall,
        "record_paths": [_normalize_path_value(report.get("record_path")) for report in gate_reports],
    }


def build_phase4_gate_record_payload(
    *,
    pilot_name: str,
    runtime_config: Option1RuntimeConfig,
    run_label: str | None,
    phase2_compile_pilot_name_override: str | None = None,
    target_model_id: str,
    compile_record_path: str | Path | None,
    prepared_record_path: str | Path | None,
    live_run_record_path: str | Path | None,
    hybrid_run_record_path: str | Path | None,
    benchmark_summary: Mapping[str, Any],
    correctness_summary: Mapping[str, Any],
    footprint_summary: Mapping[str, Any],
    recommendation: Mapping[str, Any],
    config: Phase4GateConfig,
) -> dict[str, Any]:
    layout = resolve_option1_pilot_layout(pilot_name)
    phase2_compile_pilot_name = (
        _normalize_optional_string(phase2_compile_pilot_name_override) or layout.phase2_compile_pilot_name
    )
    return {
        "record_kind": PHASE4_RECORD_KIND,
        "pilot_name": layout.phase4_gate_pilot_name,
        "canonical_pilot_name": layout.canonical_name,
        "phase2_compile_pilot_name": phase2_compile_pilot_name,
        "phase3_hybrid_pilot_name": layout.phase3_hybrid_pilot_name,
        "device_name": runtime_config.device_name,
        "qairt_version": runtime_config.qairt_version,
        "compute_unit": runtime_config.compute_unit,
        "run_label": _normalize_optional_string(run_label) or "latest",
        "target_model_id": target_model_id,
        "compile_record_path": _normalize_path_value(compile_record_path),
        "prepared_record_path": _normalize_path_value(prepared_record_path),
        "live_run_record_path": _normalize_path_value(live_run_record_path),
        "hybrid_run_record_path": _normalize_path_value(hybrid_run_record_path),
        "benchmark_summary": _json_safe(benchmark_summary),
        "correctness_summary": _json_safe(correctness_summary),
        "footprint_summary": _json_safe(footprint_summary),
        "recommendation": _json_safe(recommendation),
        "gate_config": asdict(config),
        "created_at_utc": _utc_now_isoformat(),
    }


def write_phase4_gate_record(
    *,
    pilot_name: str,
    runtime_config: Option1RuntimeConfig,
    payload: Mapping[str, Any],
    run_label: str | None = None,
    output_path: str | Path | None = None,
) -> Path:
    layout = resolve_option1_pilot_layout(pilot_name)
    record_path = _resolve_phase4_record_path(
        runtime_config=runtime_config,
        phase4_pilot_name=layout.phase4_gate_pilot_name,
        run_label=run_label,
        output_path=output_path,
    )
    serializable_payload = dict(_json_safe(payload))
    serializable_payload["record_path"] = record_path.as_posix()
    record_path.write_text(json.dumps(serializable_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record_path


def run_phase4_gate(
    *,
    pilot_name: str,
    runtime_config: Option1RuntimeConfig,
    hybrid_runner: Callable[..., Mapping[str, Any]],
    iterations: int,
    max_samples: int,
    run_label: str | None = None,
    explicit_target_model_id: str | None = None,
    phase2_compile_pilot_name_override: str | None = None,
    config: Phase4GateConfig | None = None,
) -> dict[str, Any]:
    gate_config = config or build_phase4_gate_config()
    layout = resolve_option1_pilot_layout(pilot_name)
    run_label = _normalize_optional_string(run_label) or "latest"
    benchmark_summary = run_phase4_benchmark_sweep(
        hybrid_runner=hybrid_runner,
        iterations=iterations,
        max_samples=max_samples,
        run_label=run_label,
        explicit_target_model_id=explicit_target_model_id,
    )
    source_records = resolve_phase4_source_records(
        pilot_name=layout.canonical_name,
        runtime_config=runtime_config,
        run_label=run_label,
        hybrid_run_record_path=benchmark_summary["last_report"].get("record_path"),
        phase2_compile_pilot_name_override=phase2_compile_pilot_name_override,
    )
    correctness_summary = build_phase4_correctness_summary(
        pilot_name=layout.canonical_name,
        sample_results=source_records["hybrid_run_record"].get("sample_results", []),
        config=gate_config,
    )
    footprint_summary = build_phase4_footprint_summary(
        prepared_record=source_records["prepared_artifact_record"],
        live_run_record=source_records["live_run_record"],
        hybrid_run_record=source_records["hybrid_run_record"],
    )
    enriched_benchmark_summary = dict(benchmark_summary)
    enriched_benchmark_summary["latency_summary"] = source_records["hybrid_run_record"].get(
        "latency_summary",
        benchmark_summary.get("latency_summary", {}),
    )
    recommendation = build_phase4_recommendation(
        pilot_name=layout.canonical_name,
        correctness_summary=correctness_summary,
        benchmark_summary=enriched_benchmark_summary,
        config=gate_config,
    )
    payload = build_phase4_gate_record_payload(
        pilot_name=layout.canonical_name,
        runtime_config=runtime_config,
        run_label=run_label,
        phase2_compile_pilot_name_override=source_records["phase2_compile_pilot_name"],
        target_model_id=str(source_records["hybrid_run_record"].get("target_model_id", "")),
        compile_record_path=source_records["paths"]["compile_record_path"],
        prepared_record_path=source_records["paths"]["prepared_record_path"],
        live_run_record_path=source_records["paths"]["live_run_record_path"],
        hybrid_run_record_path=source_records["paths"]["hybrid_run_record_path"],
        benchmark_summary=enriched_benchmark_summary,
        correctness_summary=correctness_summary,
        footprint_summary=footprint_summary,
        recommendation=recommendation,
        config=gate_config,
    )
    record_path = write_phase4_gate_record(
        pilot_name=layout.canonical_name,
        runtime_config=runtime_config,
        payload=payload,
        run_label=run_label,
    )
    return {
        "pilot_name": layout.phase4_gate_pilot_name,
        "record_path": record_path,
        "benchmark_summary": enriched_benchmark_summary,
        "correctness_summary": correctness_summary,
        "footprint_summary": footprint_summary,
        "recommendation": recommendation,
        "payload": payload,
    }


def resolve_phase4_source_records(
    *,
    pilot_name: str,
    runtime_config: Option1RuntimeConfig,
    run_label: str | None,
    hybrid_run_record_path: str | Path | None = None,
    phase2_compile_pilot_name_override: str | None = None,
) -> dict[str, Any]:
    layout = resolve_option1_pilot_layout(pilot_name)
    normalized_label = _normalize_record_label(run_label or "latest")
    phase2_compile_pilot_name = (
        _normalize_optional_string(phase2_compile_pilot_name_override) or layout.phase2_compile_pilot_name
    )
    prepared_record_path = runtime_config.pilot_record_dir(phase2_compile_pilot_name) / f"prepared-artifact-{normalized_label}.json"
    compile_record_path = runtime_config.pilot_record_dir(phase2_compile_pilot_name) / f"compile-run-{normalized_label}.json"
    live_run_record_path = runtime_config.pilot_record_dir(phase2_compile_pilot_name) / f"live-run-{normalized_label}.json"
    hybrid_path = (
        Path(hybrid_run_record_path).resolve()
        if hybrid_run_record_path is not None
        else runtime_config.pilot_record_dir(layout.phase3_hybrid_pilot_name) / f"hybrid-run-{normalized_label}.json"
    )

    return {
        "phase2_compile_pilot_name": phase2_compile_pilot_name,
        "paths": {
            "prepared_record_path": prepared_record_path.resolve(),
            "compile_record_path": compile_record_path.resolve(),
            "live_run_record_path": live_run_record_path.resolve(),
            "hybrid_run_record_path": hybrid_path.resolve(),
        },
        "prepared_artifact_record": _read_json_file(prepared_record_path),
        "compile_run_record": _read_json_file(compile_record_path),
        "live_run_record": _read_json_file(live_run_record_path),
        "hybrid_run_record": _read_json_file(hybrid_path),
    }


def _resolve_phase4_record_path(
    *,
    runtime_config: Option1RuntimeConfig,
    phase4_pilot_name: str,
    run_label: str | None,
    output_path: str | Path | None,
) -> Path:
    if output_path is not None:
        resolved = Path(output_path).resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved
    normalized_label = _normalize_record_label(run_label or "latest")
    record_dir = runtime_config.pilot_record_dir(phase4_pilot_name)
    record_dir.mkdir(parents=True, exist_ok=True)
    return (record_dir / f"phase4-gate-{normalized_label}.json").resolve()


def _tensor_summary_nbytes(item: Mapping[str, Any]) -> int:
    shape = [int(dim) for dim in item.get("shape", [])]
    dtype = str(item.get("dtype", "") or "")
    if not shape or not dtype:
        return 0
    try:
        itemsize = np.dtype(dtype).itemsize
    except TypeError:
        return 0
    return int(np.prod(shape, dtype=np.int64)) * int(itemsize)


def _normalized_text_distance(actual_text: str, expected_text: str) -> float:
    ratio = SequenceMatcher(a=expected_text, b=actual_text).ratio()
    return round(1.0 - float(ratio), 6)


def _catastrophic_failure_reasons(
    *,
    actual_text: str,
    expected_text: str,
    generated_ids: Sequence[int],
    config: Phase4GateConfig,
) -> list[str]:
    reasons: list[str] = []
    stripped_actual = actual_text.strip()
    if not stripped_actual:
        reasons.append("empty_output")
    if stripped_actual and all(character in {"?", "⁇", "�", "."} for character in stripped_actual):
        reasons.append("placeholder_like_output")
    if (
        generated_ids
        and
        len(generated_ids) <= int(config.catastrophic_short_generated_id_count)
        and len(expected_text.strip()) >= int(config.catastrophic_expected_text_min_chars)
    ):
        reasons.append("generated_ids_too_short_for_expected_text")
    return reasons


def _severity_rank(severity: str) -> int:
    return {
        EXACT_MATCH: 0,
        MINOR_TEXT_DRIFT: 1,
        MAJOR_TEXT_DRIFT: 2,
        CATASTROPHIC_DECODE_FAILURE: 3,
        COMPARISON_UNAVAILABLE: 4,
    }.get(severity, 5)


def _extract_average_cloud_inference_seconds(benchmark_summary: Mapping[str, Any]) -> float | None:
    latency_summary = benchmark_summary.get("latency_summary") if isinstance(benchmark_summary, Mapping) else None
    if isinstance(latency_summary, Mapping):
        value = latency_summary.get("average_cloud_inference_seconds")
        if value is not None:
            return float(value)
    return None


def _resolve_latency_thresholds(layout: Option1PilotLayout, config: Phase4GateConfig) -> tuple[float | None, float | None]:
    if layout.canonical_name == "zipformer":
        return (
            config.zipformer_go_max_average_cloud_inference_seconds,
            config.zipformer_warn_max_average_cloud_inference_seconds,
        )
    return (
        config.vpcd_go_max_average_cloud_inference_seconds,
        config.vpcd_warn_max_average_cloud_inference_seconds,
    )


def _classify_latency_band(
    *,
    average_cloud_seconds: float | None,
    go_threshold: float | None,
    warn_threshold: float | None,
) -> str:
    if average_cloud_seconds is None:
        return "warn"
    if go_threshold is None:
        return "go"
    if average_cloud_seconds <= float(go_threshold):
        return "go"
    if warn_threshold is None:
        return "warn"
    if average_cloud_seconds <= float(warn_threshold):
        return "warn"
    return "no_go"


def _finalize_recommendation(
    value: str,
    reasons: list[str],
    average_cloud_seconds: float | None,
    go_threshold: float | None,
    warn_threshold: float | None,
) -> dict[str, Any]:
    return {
        "value": value,
        "reasons": reasons,
        "average_cloud_inference_seconds": average_cloud_seconds,
        "go_max_average_cloud_inference_seconds": go_threshold,
        "warn_max_average_cloud_inference_seconds": warn_threshold,
    }


def _summarize_sample_latencies(sample_results: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    cloud_values = [float(row.get("cloud_inference_seconds")) for row in sample_results if row.get("cloud_inference_seconds") is not None]
    decode_values = [float(row.get("decode_seconds")) for row in sample_results if row.get("decode_seconds") is not None]
    return {
        "average_cloud_inference_seconds": _mean_or_none(cloud_values),
        "average_decode_seconds": _mean_or_none(decode_values),
    }


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return round(sum(float(value) for value in values) / len(values), 6)


def _normalize_optional_string(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _normalize_record_label(value: str) -> str:
    normalized = "".join(character if character.isalnum() or character in {"-", "_"} else "-" for character in str(value).strip())
    collapsed = "-".join(part for part in normalized.split("-") if part)
    return collapsed or "run"


def _normalize_path_value(value: str | Path | None) -> str | None:
    if value is None:
        return None
    return Path(value).resolve().as_posix()


def _read_json_file(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Required Phase 4 input record is missing: {resolved}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _utc_now_isoformat() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
