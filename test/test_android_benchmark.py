from __future__ import annotations

import math

import pytest

from model_pipeline.benchmarks import (
    BENCHMARK_CONFIGURATIONS,
    balanced_run_schedule,
    build_comparison,
    calculate_statistics,
)


def test_balanced_schedule_rotates_all_configurations() -> None:
    """Verify every configuration occupies every ordinal position once.

    Returns:
        None.
    """
    schedule = balanced_run_schedule()

    assert len(schedule) == 6
    assert [item.configuration for item in schedule[:2]] == list(BENCHMARK_CONFIGURATIONS)
    for position in range(2):
        assert {
            schedule[round_index * 2 + position].configuration
            for round_index in range(3)
        } == set(BENCHMARK_CONFIGURATIONS)


def test_statistics_use_nearest_rank_p95_and_population_deviation() -> None:
    """Verify benchmark statistics use the locked deterministic definitions.

    Returns:
        None.
    """
    result = calculate_statistics([float(value) for value in range(1, 101)])

    assert result["median_ms"] == 50.5
    assert result["p95_ms"] == 95.0
    assert result["min_ms"] == 1.0
    assert result["max_ms"] == 100.0
    assert result["mean_ms"] == 50.5
    assert result["population_stddev_ms"] == pytest.approx(28.8660700477)


def test_comparison_requires_three_complete_runs_and_strict_npu() -> None:
    """Verify incomplete or fallback NPU evidence cannot produce valid speedup.

    Returns:
        None.
    """
    rows = []
    for entry in balanced_run_schedule():
        rows.append(
            {
                "configuration": entry.configuration,
                "run_index": entry.round_index,
                "latency_ms": [2.0] * 100,
                "session_creation_ms": 10.0,
                "pss_after_run_kib": 100,
                "quality_passed": True,
                "quality_contract": "zipformer-transcript-parity-5-of-5",
                "strict_npu": entry.provider != "qnn-htp" or True,
                "execution_provider": entry.provider,
                "device_fingerprint": "samsung/s23/device-build",
                "artifact_id": "zipformer-artifact",
                "payload_manifest_checksum": "a" * 64,
            }
        )

    valid = build_comparison("zipformer", rows)
    assert valid["valid"] is True
    assert valid["speedups"]["fp32_cpu_over_npu"] == 1.0
    assert set(valid["speedups"]) == {"fp32_cpu_over_npu"}

    next(
        row
        for row in rows
        if row["configuration"] == "aimet-int8-int16-encoder-matmul-aihub-qnn-htp"
    )["execution_provider"] = "cpu-fallback"
    invalid = build_comparison("zipformer", rows)
    assert invalid["valid"] is False
    assert "strict-npu-placement" in invalid["invalid_reasons"]


def test_comparison_rejects_cross_device_or_payload_results() -> None:
    """Verify same-device and same-payload provenance are mandatory.

    Returns:
        None.
    """
    rows = []
    for entry in balanced_run_schedule():
        rows.append(
            {
                "configuration": entry.configuration,
                "run_index": entry.round_index,
                "latency_ms": [1.0] * 100,
                "session_creation_ms": 1.0,
                "pss_after_run_kib": 1,
                "quality_passed": True,
                "quality_contract": "vpcd-teacher-forced-top1-25-of-25",
                "strict_npu": True,
                "execution_provider": entry.provider,
                "device_fingerprint": "device-a",
                "artifact_id": "vpcd-artifact",
                "payload_manifest_checksum": "b" * 64,
            }
        )
    rows[-1]["device_fingerprint"] = "device-b"

    comparison = build_comparison("vpcd", rows)

    assert comparison["valid"] is False
    assert "same-device" in comparison["invalid_reasons"]


def test_comparison_rejects_duplicate_repetition_indexes() -> None:
    """Verify three rows with duplicate run indexes are not three repetitions.

    Returns:
        None.
    """
    rows = []
    for entry in balanced_run_schedule():
        rows.append(
            {
                "configuration": entry.configuration,
                "run_index": entry.round_index,
                "latency_ms": [1.0] * 100,
                "session_creation_ms": 1.0,
                "pss_after_run_kib": 1,
                "quality_passed": True,
                "quality_contract": "zipformer-transcript-parity-5-of-5",
                "strict_npu": True,
                "execution_provider": entry.provider,
                "device_fingerprint": "device",
                "artifact_id": "artifact",
                "payload_manifest_checksum": "e" * 64,
            }
        )
    rows[-1]["run_index"] = 2

    comparison = build_comparison("zipformer", rows)

    assert comparison["valid"] is False
    assert "balanced-repetitions" in comparison["invalid_reasons"]


def test_comparison_rejects_non_finite_observations() -> None:
    """Verify NaN latency cannot enter published performance statistics.

    Returns:
        None.
    """
    rows = []
    for entry in balanced_run_schedule():
        values = [1.0] * 100
        if not rows:
            values[0] = math.nan
        rows.append(
            {
                "configuration": entry.configuration,
                "run_index": entry.round_index,
                "latency_ms": values,
                "session_creation_ms": 1.0,
                "pss_after_run_kib": 1,
                "quality_passed": True,
                "quality_contract": "vpcd-teacher-forced-top1-25-of-25",
                "strict_npu": True,
                "execution_provider": entry.provider,
                "device_fingerprint": "device",
                "artifact_id": "vpcd-artifact",
                "payload_manifest_checksum": "c" * 64,
            }
        )

    comparison = build_comparison("vpcd", rows)

    assert comparison["valid"] is False
    assert "finite-latency" in comparison["invalid_reasons"]


def test_comparison_rejects_runtime_smoke_as_model_quality_evidence() -> None:
    """Verify finite-output smoke cannot substitute for model quality parity.

    Returns:
        None.
    """
    rows = []
    for entry in balanced_run_schedule():
        rows.append(
            {
                "configuration": entry.configuration,
                "run_index": entry.round_index,
                "latency_ms": [1.0] * 100,
                "session_creation_ms": 1.0,
                "pss_after_run_kib": 1,
                "quality_passed": True,
                "quality_contract": "finite-model-output",
                "strict_npu": True,
                "execution_provider": entry.provider,
                "device_fingerprint": "device",
                "artifact_id": "zipformer-artifact",
                "payload_manifest_checksum": "d" * 64,
            }
        )

    comparison = build_comparison("zipformer", rows)

    assert comparison["valid"] is False
    assert "quality-contract" in comparison["invalid_reasons"]
