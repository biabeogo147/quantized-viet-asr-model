from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from model_pipeline.benchmarks import (
    BENCHMARK_CONFIGURATIONS,
    balanced_run_schedule,
    build_comparison,
    calculate_statistics,
    materialize_payload,
)
from model_pipeline.benchmarks.runtime import export_qdq_from_pipeline_build
from model_pipeline.benchmarks.graph import validate_benchmark_qdq


def test_balanced_schedule_rotates_all_configurations() -> None:
    """Verify every configuration occupies every ordinal position once.

    Returns:
        None.
    """
    schedule = balanced_run_schedule()

    assert len(schedule) == 9
    assert [item.configuration for item in schedule[:3]] == list(BENCHMARK_CONFIGURATIONS)
    for position in range(3):
        assert {
            schedule[round_index * 3 + position].configuration
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
    rows[-3]["run_index"] = 2

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


def test_payload_materializes_relative_checksummed_files_and_raw_tensors(tmp_path: Path) -> None:
    """Verify payload output is portable, checksummed, and little-endian.

    Args:
        tmp_path: Isolated model input and payload destination root.

    Returns:
        None.
    """
    sources = tmp_path / "sources"
    sources.mkdir()
    components = {}
    for role, content in {
        "fp32_model": b"fp32",
        "qdq_model": b"qdq",
        "compiled_model": b"epcontext",
        "compiled_external_data": b"context",
    }.items():
        path = sources / f"{role}.onnx"
        if role == "compiled_external_data":
            path = sources / "model.bin"
        path.write_bytes(content)
        components[role] = path
    fixture = {
        "x": np.arange(6, dtype=np.float32).reshape(1, 2, 3),
        "x_lens": np.array([2], dtype=np.int32),
    }

    manifest_path = materialize_payload(
        model="zipformer",
        artifact_id="zipformer__q-aimet-int8-int16-encoder-matmul__s-enc1x2009x80-dec1x2-join1x512__c-aihub-qnn-htp-encoder",
        components=components,
        fixtures=[fixture],
        expected_outputs=[{"transcript": "xin chao"}],
        output_dir=tmp_path / "payload",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["model"] == "zipformer"
    assert {item["role"] for item in manifest["components"]} == set(components)
    assert all(not Path(item["file"]).is_absolute() for item in manifest["components"])
    x_record = manifest["fixtures"][0]["inputs"]["x"]
    raw = (manifest_path.parent / x_record["file"]).read_bytes()
    assert np.frombuffer(raw, dtype="<f4").tolist() == pytest.approx(list(range(6)))
    assert x_record["shape"] == [1, 2, 3]


def test_payload_rejects_missing_compiled_external_data(tmp_path: Path) -> None:
    """Verify EPContext payloads require the adjacent Qualcomm context binary.

    Args:
        tmp_path: Isolated payload root.

    Returns:
        None.
    """
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")

    with pytest.raises(ValueError, match="compiled_external_data"):
        materialize_payload(
            model="vpcd",
            artifact_id="vpcd__q-aimet-int8-int16-encoder-matmul__s-src1x384-dec1x64__c-aihub-qnn-htp-model",
            components={
                "fp32_model": model,
                "qdq_model": model,
                "compiled_model": model,
            },
            fixtures=[{"input_ids": np.zeros((1, 384), dtype=np.int32)}],
            expected_outputs=[{"top1": 1}],
            output_dir=tmp_path / "payload",
        )


def test_qdq_export_reuses_fixed_model_encodings_config_and_policy(tmp_path: Path) -> None:
    """Verify benchmark QDQ is derived from one canonical AIMET stage.

    Args:
        tmp_path: Isolated deterministic pipeline stage root.

    Returns:
        None.
    """
    artifact_root = tmp_path / "artifact"
    prepare_dir = artifact_root / "prepare"
    quantize_dir = artifact_root / "quantize"
    prepare_dir.mkdir(parents=True)
    quantize_dir.mkdir(parents=True)
    prepared = prepare_dir / "encoder.onnx"
    encodings = quantize_dir / "model.encodings"
    config = quantize_dir / "aimet-config.json"
    policy = quantize_dir / "quantization-policy.json"
    for path in (prepared, encodings, config, policy):
        path.write_text("{}", encoding="utf-8")
    (prepare_dir / "stage-state.json").write_text(
        json.dumps({"outputs": {"encoder": prepared.name}}),
        encoding="utf-8",
    )
    (quantize_dir / "stage-state.json").write_text(
        json.dumps(
            {
                "outputs": {
                    "encodings": encodings.name,
                    "aimet_config": config.name,
                    "quantization_policy": policy.name,
                }
            }
        ),
        encoding="utf-8",
    )

    class FakeService:
        def __init__(self) -> None:
            """Initialize captured fake AIMET service calls.

            Returns:
                None.
            """
            self.arguments = None

        def healthcheck(self) -> None:
            """Accept the fake service healthcheck.

            Returns:
                None.
            """
            return None

        def export_qdq(self, **kwargs):
            """Capture QDQ inputs and materialize fake model bytes.

            Args:
                kwargs: Exact fixed model, encoding, config, policy, and output paths.

            Returns:
                Fake service response.
            """
            self.arguments = kwargs
            output = Path(kwargs["output_dir"])
            output.mkdir(parents=True, exist_ok=True)
            (output / "model.qdq.onnx").write_bytes(b"qdq")
            return {"outputs": {"model": "model.qdq.onnx"}}

    service = FakeService()
    outputs = export_qdq_from_pipeline_build(
        model="zipformer",
        artifact_root=artifact_root,
        aimet_service=service,
    )

    assert outputs["model"].read_bytes() == b"qdq"
    assert service.arguments == {
        "fp32_model_path": prepared,
        "encodings_path": encodings,
        "output_dir": artifact_root / "benchmark-qdq",
        "config_path": config,
        "policy_path": policy,
    }


@pytest.mark.parametrize(
    ("model", "checks"),
    [
        ("zipformer", {"matmul": 278, "quantized_matmul": 278, "scope": True}),
        (
            "vpcd",
            {
                "encoder_matmul": 96,
                "decoder_matmul": 168,
                "language_model_head_matmul": 1,
                "scope": True,
            },
        ),
    ],
)
def test_qdq_graph_contract_requires_canonical_encoder_scope(
    model: str,
    checks: dict[str, object],
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify benchmark QDQ validation locks each model's canonical graph scope.

    Args:
        model: Canonical model family under test.
        checks: Expected normalized graph evidence.
        tmp_path: Isolated placeholder graph and encoding paths.
        monkeypatch: Pytest fixture replacing heavyweight ONNX inspection.

    Returns:
        None.
    """
    from types import SimpleNamespace
    import model_pipeline.benchmarks.graph as graph

    qdq = tmp_path / "model.qdq.onnx"
    encodings = tmp_path / "model.encodings"
    qdq.write_bytes(b"model")
    encodings.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        graph,
        "inspect_zipformer_qdq_coverage",
        lambda _path: SimpleNamespace(
            matmul_count=278,
            quantized_matmul_count=278,
            unquantized_matmul_names=(),
        ),
    )
    monkeypatch.setattr(
        graph,
        "inspect_vpcd_matmuls",
        lambda _path: SimpleNamespace(
            counts={"encoder": 96, "decoder": 168, "lm_head": 1, "other": 0}
        ),
    )
    monkeypatch.setattr(
        graph,
        "inspect_encoder_matmul_aimet_encodings",
        lambda _path: {
            "activation_count": 168,
            "parameter_count": 72,
            "activation_contract": True,
            "parameter_contract": True,
            "non_encoder_names": [],
        },
    )

    assert validate_benchmark_qdq(model, qdq, encodings) == checks
