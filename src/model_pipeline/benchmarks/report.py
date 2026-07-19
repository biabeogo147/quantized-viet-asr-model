"""Validation and aggregation of Android benchmark results."""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter, defaultdict
from typing import Mapping, Sequence

from model_pipeline.benchmarks.contracts import (
    BENCHMARK_CONFIGURATIONS,
    FP32_CPU_CONFIGURATION,
    NPU_CONFIGURATION,
)


def calculate_statistics(values: Sequence[float]) -> dict[str, float]:
    """Calculate deterministic latency statistics for finite observations.

    Args:
        values: Non-empty finite latency values in milliseconds.

    Returns:
        Median, nearest-rank p95, extrema, mean, and population deviation.

    Raises:
        ValueError: If values are empty or contain non-finite observations.
    """
    normalized = [float(value) for value in values]
    if not normalized or not all(math.isfinite(value) for value in normalized):
        raise ValueError("Benchmark observations must be non-empty and finite")
    ordered = sorted(normalized)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[p95_index],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "mean_ms": statistics.fmean(ordered),
        "population_stddev_ms": statistics.pstdev(ordered),
    }


def build_comparison(model: str, runs: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Validate six device runs and aggregate comparable CPU/NPU metrics.

    Args:
        model: Canonical model family represented by the results.
        runs: Raw result mappings from fresh Android processes.

    Returns:
        Validation reasons, per-configuration metrics, and speedups when valid.
    """
    rows = [dict(run) for run in runs]
    reasons: list[str] = []
    counts = Counter(str(row.get("configuration")) for row in rows)
    if len(rows) != 6 or any(counts[name] != 3 for name in BENCHMARK_CONFIGURATIONS):
        reasons.append("three-complete-runs")
    if any(
        {
            row.get("run_index")
            for row in rows
            if row.get("configuration") == configuration
        }
        != {1, 2, 3}
        for configuration in BENCHMARK_CONFIGURATIONS
    ):
        reasons.append("balanced-repetitions")
    finite = all(
        len(row.get("latency_ms", [])) == 100
        and all(
            math.isfinite(float(value)) and float(value) > 0
            for value in row.get("latency_ms", [])
        )
        for row in rows
    )
    if not finite:
        reasons.append("finite-latency")
    if not all(bool(row.get("quality_passed")) for row in rows):
        reasons.append("quality-gate")
    expected_quality_contract = {
        "zipformer": "zipformer-transcript-parity-5-of-5",
        "vpcd": "vpcd-teacher-forced-top1-25-of-25",
    }.get(model)
    if expected_quality_contract is None or not all(
        row.get("quality_contract") == expected_quality_contract for row in rows
    ):
        reasons.append("quality-contract")
    device_fingerprints = {
        str(row.get("device_fingerprint", "")).strip() for row in rows
    }
    if len(device_fingerprints) != 1 or "" in device_fingerprints:
        reasons.append("same-device")
    artifact_ids_by_configuration = {
        configuration: {
            str(row.get("artifact_id", "")).strip()
            for row in rows
            if row.get("configuration") == configuration
        }
        for configuration in BENCHMARK_CONFIGURATIONS
    }
    payload_checksums = {
        str(row.get("payload_manifest_checksum", "")).strip().lower()
        for row in rows
    }
    if (
        any(
            len(artifact_ids) != 1 or "" in artifact_ids
            for artifact_ids in artifact_ids_by_configuration.values()
        )
        or len(payload_checksums) != 1
        or not all(re.fullmatch(r"[0-9a-f]{64}", value) for value in payload_checksums)
    ):
        reasons.append("artifact-provenance")
    npu_rows = [row for row in rows if row.get("configuration") == NPU_CONFIGURATION]
    if len(npu_rows) != 3 or not all(
        row.get("execution_provider") == "qnn-htp" and bool(row.get("strict_npu"))
        for row in npu_rows
    ):
        reasons.append("strict-npu-placement")

    result: dict[str, object] = {
        "schema_version": 1,
        "model": model,
        "valid": not reasons,
        "invalid_reasons": reasons,
        "configurations": {},
        "speedups": {},
    }
    if reasons:
        return result

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["configuration"])].append(row)
    summaries = {}
    for configuration in BENCHMARK_CONFIGURATIONS:
        configuration_rows = grouped[configuration]
        observations = [
            float(value)
            for row in configuration_rows
            for value in row["latency_ms"]
        ]
        summaries[configuration] = {
            **calculate_statistics(observations),
            "artifact_id": next(iter(artifact_ids_by_configuration[configuration])),
            "observations": len(observations),
            "median_session_creation_ms": statistics.median(
                float(row["session_creation_ms"]) for row in configuration_rows
            ),
            "median_pss_after_run_kib": statistics.median(
                int(row["pss_after_run_kib"]) for row in configuration_rows
            ),
        }
    npu_median = summaries[NPU_CONFIGURATION]["median_ms"]
    fp32_median = summaries[FP32_CPU_CONFIGURATION]["median_ms"]
    result["configurations"] = summaries
    result["speedups"] = {
        "fp32_cpu_over_npu": fp32_median / npu_median,
    }
    return result
