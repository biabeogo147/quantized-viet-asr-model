import json
from pathlib import Path

import numpy as np


def test_vpcd_phase4_layout_defaults_to_local_aimet_compile_lane():
    from tools.aihub_option1_phase4_gate import resolve_option1_pilot_layout

    layout = resolve_option1_pilot_layout("vpcd")

    assert layout.phase2_compile_pilot_name == "vpcd_option1_local_aimet"


def _init_repo_root(repo_root: Path) -> None:
    (repo_root / "src").mkdir(parents=True, exist_ok=True)
    (repo_root / "assets").mkdir(parents=True, exist_ok=True)
    (repo_root / "test").mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text("[project]\nname = 'python-model-test'\nversion = '0.0.0'\n", encoding="utf-8")


def _build_runtime_config(repo_root: Path):
    from tools.aihub_option1_pilots import build_option1_runtime_config

    return build_option1_runtime_config(
        device_name="Samsung Galaxy S24 (Family)",
        repo_root=repo_root,
    )


def _write_zipformer_phase2_and_phase3_records(repo_root: Path, *, run_label: str = "phase4"):
    from tools.aihub_option1_hybrid_pipeline import ResolvedCompiledTarget, write_hybrid_run_record
    from tools.aihub_option1_pilots import (
        build_option1_runtime_config,
        write_compile_run_record,
        write_live_run_record,
        write_prepared_artifact_record,
    )

    runtime_config = build_option1_runtime_config(
        device_name="Samsung Galaxy S24 (Family)",
        repo_root=repo_root,
    )
    source_model_path = repo_root / "build" / "quantize" / "zipformer" / "qnn_u16u8" / "fixed_shapes" / "encoder.fixed.onnx"
    source_model_path.parent.mkdir(parents=True, exist_ok=True)
    source_model_path.write_bytes(b"zipformer-source-model")
    prepared_model_path = repo_root / "build" / "aihub" / "zipformer_encoder_option1" / "encoder.aihub.option1.onnx"
    prepared_model_path.parent.mkdir(parents=True, exist_ok=True)
    prepared_model_path.write_bytes(b"zipformer-prepared-model")

    prepared_record_path = write_prepared_artifact_record(
        pilot_name="zipformer_encoder_option1",
        runtime_config=runtime_config,
        source_model_path=source_model_path,
        prepared_model_path=prepared_model_path,
        input_specs={
            "x": ((1, 2009, 80), "float32"),
            "x_lens": ((1,), "int64"),
        },
        compile_options="--target_runtime precompiled_qnn_onnx --truncate_64bit_io",
        run_label=run_label,
    )
    compile_record_path = write_compile_run_record(
        pilot_name="zipformer_encoder_option1",
        runtime_config=runtime_config,
        compile_options="--target_runtime precompiled_qnn_onnx --truncate_64bit_io",
        target_model={
            "model_id": "zipformer-target",
            "url": "https://example/models/zipformer-target",
            "name": "zipformer-target",
        },
        run_label=run_label,
    )
    live_record_path = write_live_run_record(
        pilot_name="zipformer_encoder_option1",
        runtime_config=runtime_config,
        compile_options="--target_runtime precompiled_qnn_onnx --truncate_64bit_io",
        job_options="--compute_unit npu",
        profile_job={"job_id": "profile-zip", "url": "https://example/jobs/profile-zip"},
        inference_job={"job_id": "infer-zip", "url": "https://example/jobs/infer-zip"},
        output_tensors={
            "output_0": [np.zeros((1, 4), dtype=np.float32)],
            "output_1": [np.asarray([4], dtype=np.int32)],
        },
        run_label=run_label,
    )
    hybrid_record_path = write_hybrid_run_record(
        pilot_name="zipformer_hybrid_option1",
        runtime_config=runtime_config,
        target_reference=ResolvedCompiledTarget(
            compile_pilot_name="zipformer_encoder_option1",
            target_model_id="zipformer-target",
            compile_record_path=compile_record_path,
            run_label=run_label,
            explicit_override=False,
        ),
        sample_results=[
            {
                "sample_id": "sample-1",
                "audio_path": "assets/speech/sample-1.wav",
                "text": "xin chao",
                "expected_text": "xin chao",
                "matches_expected": True,
                "token_ids": [1, 2],
                "cloud_inference_seconds": 1.25,
                "decode_seconds": 0.15,
            }
        ],
        run_label=run_label,
    )
    return runtime_config, {
        "prepared_record_path": prepared_record_path,
        "compile_record_path": compile_record_path,
        "live_record_path": live_record_path,
        "hybrid_record_path": hybrid_record_path,
    }


def test_phase4_gate_summary_writes_deterministic_record_with_phase_inputs(tmp_path):
    from tools.aihub_option1_phase4_gate import (
        GO,
        build_phase4_gate_record_payload,
        build_phase4_gate_config,
        write_phase4_gate_record,
    )

    repo_root = tmp_path / "repo"
    _init_repo_root(repo_root)
    runtime_config, record_paths = _write_zipformer_phase2_and_phase3_records(repo_root)

    config = build_phase4_gate_config()
    benchmark_summary = {
        "iterations": [{"iteration_index": 1, "total_seconds": 1.4, "cloud_inference_seconds": 1.25, "decode_seconds": 0.15}],
        "warmup": {"total_seconds": 1.4, "cloud_inference_seconds": 1.25, "decode_seconds": 0.15},
        "steady_state": {"count": 0, "total_seconds_mean": None, "total_seconds_min": None, "total_seconds_max": None},
    }
    correctness_summary = {
        "severity_counts": {"exact_match": 1},
        "sample_results": [{"sample_key": "sample-1", "severity": "exact_match", "reasons": ["exact text match"]}],
        "worst_severity": "exact_match",
        "matched_samples": 1,
        "mismatched_samples": 0,
    }
    footprint_summary = {
        "prepared_model_size_bytes": 123,
        "output_tensor_footprint_bytes": 20,
        "generated_token_footprint_bytes": 16,
        "host_rss_delta_bytes": None,
        "host_rss_status": "unavailable",
    }
    recommendation = {
        "value": GO,
        "reasons": ["exact_match_only", "latency_within_go_threshold"],
    }

    payload = build_phase4_gate_record_payload(
        pilot_name="zipformer",
        runtime_config=runtime_config,
        run_label="latest",
        target_model_id="zipformer-target",
        compile_record_path=record_paths["compile_record_path"],
        prepared_record_path=record_paths["prepared_record_path"],
        live_run_record_path=record_paths["live_record_path"],
        hybrid_run_record_path=record_paths["hybrid_record_path"],
        benchmark_summary=benchmark_summary,
        correctness_summary=correctness_summary,
        footprint_summary=footprint_summary,
        recommendation=recommendation,
        config=config,
    )
    record_path = write_phase4_gate_record(
        pilot_name="zipformer",
        runtime_config=runtime_config,
        payload=payload,
        run_label="latest",
    )

    persisted = json.loads(record_path.read_text(encoding="utf-8"))
    assert persisted["record_kind"] == "phase4_gate"
    assert persisted["pilot_name"] == "zipformer_phase4_option1"
    assert persisted["phase2_compile_pilot_name"] == "zipformer_encoder_option1"
    assert persisted["phase3_hybrid_pilot_name"] == "zipformer_hybrid_option1"
    assert persisted["target_model_id"] == "zipformer-target"
    assert persisted["compile_record_path"] == record_paths["compile_record_path"].as_posix()
    assert persisted["live_run_record_path"] == record_paths["live_record_path"].as_posix()
    assert persisted["hybrid_run_record_path"] == record_paths["hybrid_record_path"].as_posix()
    assert persisted["recommendation"]["value"] == GO
    assert record_path == (
        runtime_config.pilot_record_dir("zipformer_phase4_option1") / "phase4-gate-latest.json"
    ).resolve()


def test_phase4_benchmark_sweep_summarizes_warmup_and_steady_state():
    from tools.aihub_option1_phase4_gate import run_phase4_benchmark_sweep

    reports = [
        {
            "results": [
                {"cloud_inference_seconds": 1.0, "decode_seconds": 0.2},
                {"cloud_inference_seconds": 0.5, "decode_seconds": 0.1},
            ],
            "record_path": Path("D:/records/iter-1.json"),
        },
        {
            "results": [
                {"cloud_inference_seconds": 0.8, "decode_seconds": 0.2},
                {"cloud_inference_seconds": 0.4, "decode_seconds": 0.1},
            ],
            "record_path": Path("D:/records/iter-2.json"),
        },
        {
            "results": [
                {"cloud_inference_seconds": 0.9, "decode_seconds": 0.1},
                {"cloud_inference_seconds": 0.4, "decode_seconds": 0.1},
            ],
            "record_path": Path("D:/records/iter-3.json"),
        },
    ]
    seen = {"calls": 0}

    def fake_runner(*, run_label: str | None, max_samples: int, explicit_target_model_id: str | None = None):
        seen["calls"] += 1
        return reports[seen["calls"] - 1]

    summary = run_phase4_benchmark_sweep(
        hybrid_runner=fake_runner,
        iterations=3,
        max_samples=2,
        run_label="phase4",
        explicit_target_model_id="target-explicit",
    )

    assert seen["calls"] == 3
    assert summary["warmup"]["total_seconds"] == 1.8
    assert summary["warmup"]["cloud_inference_seconds"] == 1.5
    assert summary["warmup"]["decode_seconds"] == 0.3
    assert summary["steady_state"]["count"] == 2
    assert summary["steady_state"]["total_seconds_mean"] == 1.5
    assert summary["steady_state"]["total_seconds_min"] == 1.5
    assert summary["steady_state"]["total_seconds_max"] == 1.5
    assert summary["iterations"][2]["record_path"] == "D:/records/iter-3.json"


def test_phase4_classify_supports_exact_minor_major_and_catastrophic_rows():
    from tools.aihub_option1_phase4_gate import (
        CATASTROPHIC_DECODE_FAILURE,
        COMPARISON_UNAVAILABLE,
        EXACT_MATCH,
        MAJOR_TEXT_DRIFT,
        MINOR_TEXT_DRIFT,
        build_phase4_gate_config,
        classify_phase4_sample,
    )

    config = build_phase4_gate_config(minor_text_drift_threshold=0.15)

    exact = classify_phase4_sample(
        pilot_name="zipformer",
        sample_result={"sample_id": "sample-1", "text": "xin chao", "expected_text": "xin chao"},
        config=config,
    )
    minor = classify_phase4_sample(
        pilot_name="zipformer",
        sample_result={"sample_id": "sample-2", "text": "hello client", "expected_text": "hello cliant"},
        config=config,
    )
    major = classify_phase4_sample(
        pilot_name="zipformer",
        sample_result={"sample_id": "sample-3", "text": "goodbye world", "expected_text": "xin chao ban"},
        config=config,
    )
    catastrophic = classify_phase4_sample(
        pilot_name="vpcd",
        sample_result={
            "sample_index": 0,
            "text": "⁇",
            "expected_text": "Hôm nay là buổi nhậm chức của tôi.",
            "generated_ids": [0, 1, 2],
        },
        config=config,
    )

    bounded = classify_phase4_sample(
        pilot_name="vpcd",
        sample_result={
            "sample_index": 1,
            "text": "Hôm nay trời đẹp",
            "expected_text": "Hôm nay trời đẹp và gió mát.",
            "matches_expected": None,
            "comparison_note": "decode_step_limit_reached_before_eos",
            "truncated_by_decode_step_limit": True,
            "generated_ids": [0, 2232, 177, 9, 847],
        },
        config=config,
    )

    assert exact["severity"] == EXACT_MATCH
    assert minor["severity"] == MINOR_TEXT_DRIFT
    assert major["severity"] == MAJOR_TEXT_DRIFT
    assert catastrophic["severity"] == CATASTROPHIC_DECODE_FAILURE
    assert "placeholder_like_output" in catastrophic["reasons"]
    assert bounded["severity"] == COMPARISON_UNAVAILABLE
    assert bounded["reasons"] == ["decode_step_limit_reached_before_eos"]


def test_phase4_footprint_summary_uses_prepared_live_and_hybrid_records(tmp_path):
    from tools.aihub_option1_phase4_gate import build_phase4_footprint_summary

    repo_root = tmp_path / "repo"
    _init_repo_root(repo_root)
    runtime_config, record_paths = _write_zipformer_phase2_and_phase3_records(repo_root, run_label="footprint")
    _ = runtime_config

    prepared_record = json.loads(record_paths["prepared_record_path"].read_text(encoding="utf-8"))
    live_record = json.loads(record_paths["live_record_path"].read_text(encoding="utf-8"))
    hybrid_record = json.loads(record_paths["hybrid_record_path"].read_text(encoding="utf-8"))

    summary = build_phase4_footprint_summary(
        prepared_record=prepared_record,
        live_run_record=live_record,
        hybrid_run_record=hybrid_record,
    )

    assert summary["prepared_model_size_bytes"] == len(b"zipformer-prepared-model")
    assert summary["output_tensor_footprint_bytes"] == 20
    assert summary["generated_token_count"] == 2
    assert summary["generated_token_footprint_bytes"] == 16
    assert summary["host_rss_status"] == "unavailable"


def test_phase4_recommendation_supports_go_warn_and_no_go():
    from tools.aihub_option1_phase4_gate import GO, NO_GO, WARN, build_phase4_gate_config, build_phase4_recommendation

    config = build_phase4_gate_config(
        zipformer_go_max_average_cloud_inference_seconds=2.0,
        zipformer_warn_max_average_cloud_inference_seconds=3.0,
        vpcd_go_max_average_cloud_inference_seconds=5.0,
        vpcd_warn_max_average_cloud_inference_seconds=10.0,
    )

    go = build_phase4_recommendation(
        pilot_name="zipformer",
        correctness_summary={
            "worst_severity": "exact_match",
            "severity_counts": {"exact_match": 2},
        },
        benchmark_summary={
            "steady_state": {"total_seconds_mean": 1.2, "count": 2},
            "warmup": {"total_seconds": 1.4},
            "latency_summary": {"average_cloud_inference_seconds": 1.5},
        },
        config=config,
    )
    warn = build_phase4_recommendation(
        pilot_name="zipformer",
        correctness_summary={
            "worst_severity": "minor_text_drift",
            "severity_counts": {"minor_text_drift": 1, "exact_match": 1},
        },
        benchmark_summary={
            "steady_state": {"total_seconds_mean": 1.6, "count": 2},
            "warmup": {"total_seconds": 1.9},
            "latency_summary": {"average_cloud_inference_seconds": 1.8},
        },
        config=config,
    )
    no_go = build_phase4_recommendation(
        pilot_name="vpcd",
        correctness_summary={
            "worst_severity": "catastrophic_decode_failure",
            "severity_counts": {"catastrophic_decode_failure": 2},
        },
        benchmark_summary={
            "steady_state": {"total_seconds_mean": 8.1, "count": 1},
            "warmup": {"total_seconds": 8.2},
            "latency_summary": {"average_cloud_inference_seconds": 7.9},
        },
        config=config,
    )

    assert go["value"] == GO
    assert "exact_match_only" in go["reasons"]
    assert warn["value"] == WARN
    assert "minor_text_drift_present" in warn["reasons"]
    assert no_go["value"] == NO_GO
    assert "catastrophic_decode_failure_present" in no_go["reasons"]


def test_phase4_source_records_support_vpcd_compile_pilot_override(tmp_path):
    from tools.aihub_option1_hybrid_pipeline import ResolvedCompiledTarget, write_hybrid_run_record
    from tools.aihub_option1_phase4_gate import (
        build_phase4_gate_config,
        build_phase4_gate_record_payload,
        resolve_phase4_source_records,
    )
    from tools.aihub_option1_pilots import (
        build_option1_runtime_config,
        write_compile_run_record,
        write_live_run_record,
        write_prepared_artifact_record,
    )

    repo_root = tmp_path / "repo"
    _init_repo_root(repo_root)
    runtime_config = build_option1_runtime_config(
        device_name="Samsung Galaxy S24 (Family)",
        repo_root=repo_root,
    )

    source_model_path = repo_root / "build" / "aihub" / "vpcd_option1_local_aimet" / "model.fp32.fixed.onnx"
    source_model_path.parent.mkdir(parents=True, exist_ok=True)
    source_model_path.write_bytes(b"vpcd-fp32-fixed")

    prepared_model_path = repo_root / "build" / "aihub" / "vpcd_option1_local_aimet" / "model.option1.qdq.onnx"
    prepared_model_path.write_bytes(b"vpcd-qdq")

    prepared_record_path = write_prepared_artifact_record(
        pilot_name="vpcd_option1_local_aimet",
        runtime_config=runtime_config,
        source_model_path=source_model_path,
        prepared_model_path=prepared_model_path,
        input_specs={
            "input_ids": ((1, 1024), "int64"),
            "attention_mask": ((1, 1024), "int64"),
        },
        compile_options="--target_runtime precompiled_qnn_onnx --truncate_64bit_io",
        source_strategy="local_aimet_compile_candidate",
        run_label="vpcd-phase4",
    )
    compile_record_path = write_compile_run_record(
        pilot_name="vpcd_option1_local_aimet",
        runtime_config=runtime_config,
        compile_options="--target_runtime precompiled_qnn_onnx --truncate_64bit_io",
        target_model={
            "model_id": "vpcd-aimet-target",
            "url": "https://example/models/vpcd-aimet-target",
            "name": "vpcd-aimet-target",
        },
        source_strategy="local_aimet_compile_candidate",
        quantize_stage="local_aimet",
        run_label="vpcd-phase4",
    )
    live_record_path = write_live_run_record(
        pilot_name="vpcd_option1_local_aimet",
        runtime_config=runtime_config,
        compile_options="--target_runtime precompiled_qnn_onnx --truncate_64bit_io",
        job_options="--compute_unit npu",
        profile_job={"job_id": "profile-vpcd", "url": "https://example/jobs/profile-vpcd"},
        inference_job={"job_id": "infer-vpcd", "url": "https://example/jobs/infer-vpcd"},
        output_tensors={"output_0": [np.zeros((1, 8), dtype=np.float32)]},
        run_label="vpcd-phase4",
    )
    hybrid_record_path = write_hybrid_run_record(
        pilot_name="vpcd_hybrid_option1",
        runtime_config=runtime_config,
        target_reference=ResolvedCompiledTarget(
            compile_pilot_name="vpcd_option1_local_aimet",
            target_model_id="vpcd-aimet-target",
            compile_record_path=compile_record_path,
            run_label="vpcd-phase4",
            explicit_override=False,
        ),
        sample_results=[
            {
                "sample_index": 0,
                "raw_text": "hom nay troi dep",
                "text": "Hôm nay trời đẹp",
                "expected_text": "Hôm nay trời đẹp",
                "matches_expected": True,
                "generated_ids": [0, 2232, 177, 9, 847],
                "cloud_inference_seconds": 1.2,
                "decode_seconds": 0.1,
            }
        ],
        run_label="vpcd-phase4",
    )

    resolved = resolve_phase4_source_records(
        pilot_name="vpcd",
        runtime_config=runtime_config,
        run_label="vpcd-phase4",
        hybrid_run_record_path=hybrid_record_path,
        phase2_compile_pilot_name_override="vpcd_option1_local_aimet",
    )

    assert resolved["phase2_compile_pilot_name"] == "vpcd_option1_local_aimet"
    assert resolved["paths"]["prepared_record_path"] == prepared_record_path
    assert resolved["paths"]["compile_record_path"] == compile_record_path
    assert resolved["paths"]["live_run_record_path"] == live_record_path
    assert resolved["compile_run_record"]["pilot_name"] == "vpcd_option1_local_aimet"

    payload = build_phase4_gate_record_payload(
        pilot_name="vpcd",
        runtime_config=runtime_config,
        run_label="vpcd-phase4",
        phase2_compile_pilot_name_override="vpcd_option1_local_aimet",
        target_model_id="vpcd-aimet-target",
        compile_record_path=compile_record_path,
        prepared_record_path=prepared_record_path,
        live_run_record_path=live_record_path,
        hybrid_run_record_path=hybrid_record_path,
        benchmark_summary={
            "iterations": [],
            "warmup": {"total_seconds": 1.3, "cloud_inference_seconds": 1.2, "decode_seconds": 0.1},
            "steady_state": {"count": 0, "total_seconds_mean": None, "total_seconds_min": None, "total_seconds_max": None},
            "latency_summary": {"average_cloud_inference_seconds": 1.2},
        },
        correctness_summary={
            "severity_counts": {"exact_match": 1},
            "sample_results": [{"sample_key": 0, "severity": "exact_match", "reasons": ["exact_text_match"]}],
            "worst_severity": "exact_match",
            "matched_samples": 1,
            "mismatched_samples": 0,
        },
        footprint_summary={"prepared_model_size_bytes": 8, "output_tensor_footprint_bytes": 32, "generated_token_footprint_bytes": 40},
        recommendation={"value": "GO", "reasons": ["exact_match_only"]},
        config=build_phase4_gate_config(),
    )
    assert payload["phase2_compile_pilot_name"] == "vpcd_option1_local_aimet"


def test_phase4_gate_record_writer_serializes_resolved_target_reference(tmp_path):
    from tools.aihub_option1_hybrid_pipeline import ResolvedCompiledTarget
    from tools.aihub_option1_phase4_gate import build_phase4_gate_config, build_phase4_gate_record_payload, write_phase4_gate_record
    from tools.aihub_option1_pilots import build_option1_runtime_config

    repo_root = tmp_path / "repo"
    _init_repo_root(repo_root)
    runtime_config = build_option1_runtime_config(
        device_name="Samsung Galaxy S24 (Family)",
        repo_root=repo_root,
    )

    payload = build_phase4_gate_record_payload(
        pilot_name="zipformer",
        runtime_config=runtime_config,
        run_label="serialize-target-reference",
        target_model_id="zipformer-target",
        compile_record_path=Path("D:/compile-run.json"),
        prepared_record_path=Path("D:/prepared-artifact.json"),
        live_run_record_path=Path("D:/live-run.json"),
        hybrid_run_record_path=Path("D:/hybrid-run.json"),
        benchmark_summary={
            "iterations": [],
            "warmup": {"total_seconds": 1.0, "cloud_inference_seconds": 0.8, "decode_seconds": 0.2},
            "steady_state": {"count": 0, "total_seconds_mean": None, "total_seconds_min": None, "total_seconds_max": None},
            "last_report": {
                "target_reference": ResolvedCompiledTarget(
                    compile_pilot_name="zipformer_encoder_option1",
                    target_model_id="zipformer-target",
                    compile_record_path=Path("D:/compile-run.json"),
                    run_label="serialize-target-reference",
                    explicit_override=False,
                ),
            },
        },
        correctness_summary={
            "severity_counts": {"exact_match": 1},
            "sample_results": [{"sample_key": "sample-1", "severity": "exact_match", "reasons": ["exact_text_match"]}],
            "worst_severity": "exact_match",
            "matched_samples": 1,
            "mismatched_samples": 0,
        },
        footprint_summary={"prepared_model_size_bytes": 12, "output_tensor_footprint_bytes": 20, "generated_token_footprint_bytes": 16},
        recommendation={"value": "GO", "reasons": ["exact_match_only"]},
        config=build_phase4_gate_config(),
    )

    record_path = write_phase4_gate_record(
        pilot_name="zipformer",
        runtime_config=runtime_config,
        payload=payload,
        run_label="serialize-target-reference",
    )

    persisted = json.loads(record_path.read_text(encoding="utf-8"))
    assert persisted["benchmark_summary"]["last_report"]["target_reference"]["compile_pilot_name"] == "zipformer_encoder_option1"
