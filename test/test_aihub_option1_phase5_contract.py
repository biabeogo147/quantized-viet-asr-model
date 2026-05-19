import json
from pathlib import Path

import numpy as np


def _init_repo_root(repo_root: Path) -> None:
    (repo_root / "src").mkdir(parents=True, exist_ok=True)
    (repo_root / "assets").mkdir(parents=True, exist_ok=True)
    (repo_root / "test").mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text("[project]\nname = 'python-model-test'\nversion = '0.0.0'\n", encoding="utf-8")


def _write_phase5_inputs(repo_root: Path, *, run_label: str = "phase5"):
    from tools.aihub_option1_hybrid_pipeline import ResolvedCompiledTarget, write_hybrid_run_record
    from tools.aihub_option1_phase4_gate import build_phase4_gate_config, build_phase4_gate_record_payload, write_phase4_gate_record
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
    gate_payload = build_phase4_gate_record_payload(
        pilot_name="zipformer",
        runtime_config=runtime_config,
        run_label=run_label,
        target_model_id="zipformer-target",
        compile_record_path=compile_record_path,
        prepared_record_path=prepared_record_path,
        live_run_record_path=live_record_path,
        hybrid_run_record_path=hybrid_record_path,
        benchmark_summary={
            "iterations": [{"iteration_index": 1, "total_seconds": 1.4, "cloud_inference_seconds": 1.25, "decode_seconds": 0.15}],
            "warmup": {"total_seconds": 1.4, "cloud_inference_seconds": 1.25, "decode_seconds": 0.15},
            "steady_state": {"count": 0, "total_seconds_mean": None, "total_seconds_min": None, "total_seconds_max": None},
            "latency_summary": {"average_cloud_inference_seconds": 1.25, "average_decode_seconds": 0.15},
        },
        correctness_summary={
            "severity_counts": {"exact_match": 1},
            "sample_results": [{"sample_key": "sample-1", "severity": "exact_match", "reasons": ["exact text match"]}],
            "worst_severity": "exact_match",
            "matched_samples": 1,
            "mismatched_samples": 0,
        },
        footprint_summary={
            "prepared_model_size_bytes": len(b"zipformer-prepared-model"),
            "output_tensor_footprint_bytes": 20,
            "generated_token_footprint_bytes": 16,
            "generated_token_count": 2,
            "host_rss_delta_bytes": None,
            "host_rss_status": "unavailable",
        },
        recommendation={"value": "GO", "reasons": ["exact_match_only"]},
        config=build_phase4_gate_config(),
    )
    phase4_record_path = write_phase4_gate_record(
        pilot_name="zipformer",
        runtime_config=runtime_config,
        payload=gate_payload,
        run_label=run_label,
    )
    return runtime_config, {
        "prepared_record_path": prepared_record_path,
        "compile_record_path": compile_record_path,
        "live_record_path": live_record_path,
        "hybrid_record_path": hybrid_record_path,
        "phase4_record_path": phase4_record_path,
    }


def test_phase5_manifest_builder_includes_promotion_and_io_summary(tmp_path):
    from tools.aihub_option1_phase5_contract import build_phase5_contract_manifest

    repo_root = tmp_path / "repo"
    _init_repo_root(repo_root)

    manifest = build_phase5_contract_manifest(
        pilot_name="zipformer",
        run_label="latest",
        package_label="research-pass-1",
        target_model={
            "model_id": "zipformer-target",
            "url": "https://example/models/zipformer-target",
            "name": "zipformer-target",
        },
        runtime={
            "device_name": "Samsung Galaxy S24 (Family)",
            "qairt_version": None,
            "compute_unit": "npu",
        },
        promotion_status="deployment_candidate",
        phase4_recommendation={
            "value": "GO",
            "reasons": ["exact_match_only"],
        },
        source_artifacts={
            "source_model": {"path": "D:/source.onnx", "size_bytes": 10, "sha256": "abc"},
            "prepared_model": {"path": "D:/prepared.onnx", "size_bytes": 12, "sha256": "def"},
        },
        evidence={
            "prepared_artifact_record": {"source_path": "D:/prepared-record.json", "size_bytes": 30, "sha256": "ghi"},
            "compile_run_record": {"source_path": "D:/compile-record.json", "size_bytes": 31, "sha256": "jkl"},
        },
        io_contract={
            "inputs": {"x": {"shape": [1, 2009, 80], "dtype": "float32"}},
            "outputs": {"output_0": {"shape": [1, 501, 512], "dtype": "float32"}},
            "notes": ["truncate_64bit_io requires int64 host inputs to be cast to int32 for compiled execution."],
        },
        warnings=["profile_artifact missing from live run record"],
    )

    assert manifest["record_kind"] == "phase5_contract_manifest"
    assert manifest["pilot_name"] == "zipformer"
    assert manifest["run_label"] == "latest"
    assert manifest["package_label"] == "research-pass-1"
    assert manifest["promotion_status"] == "deployment_candidate"
    assert manifest["phase4_recommendation"]["value"] == "GO"
    assert manifest["target_model"]["model_id"] == "zipformer-target"
    assert manifest["io_contract_summary"]["inputs"]["x"]["dtype"] == "float32"
    assert manifest["warnings"] == ["profile_artifact missing from live run record"]


def test_phase5_resolve_inputs_requires_required_records_and_preserves_optional_warnings(tmp_path):
    from tools.aihub_option1_phase5_contract import resolve_phase5_evidence_inputs

    repo_root = tmp_path / "repo"
    _init_repo_root(repo_root)
    runtime_config, paths = _write_phase5_inputs(repo_root, run_label="resolve")

    evidence = resolve_phase5_evidence_inputs(
        pilot_name="zipformer",
        runtime_config=runtime_config,
        run_label="resolve",
    )

    assert evidence["required_record_paths"]["prepared_artifact_record"] == paths["prepared_record_path"]
    assert evidence["required_record_paths"]["phase4_gate_record"] == paths["phase4_record_path"]
    assert evidence["prepared_artifact_record"]["pilot_name"] == "zipformer_encoder_option1"
    assert evidence["phase4_gate_record"]["recommendation"]["value"] == "GO"
    assert any("profile_artifact" in warning for warning in evidence["warnings"])


def test_phase5_materialize_package_writes_manifest_io_summary_and_evidence_copies(tmp_path):
    from tools.aihub_option1_phase5_contract import materialize_phase5_contract_package

    repo_root = tmp_path / "repo"
    _init_repo_root(repo_root)
    runtime_config, _paths = _write_phase5_inputs(repo_root, run_label="materialize")

    report = materialize_phase5_contract_package(
        pilot_name="zipformer",
        runtime_config=runtime_config,
        run_label="materialize",
    )

    package_path = report["package_path"]
    assert package_path == (
        runtime_config.artifact_root / "contracts" / "option1" / "zipformer" / "materialize"
    ).resolve()
    assert (package_path / "contract_manifest.json").exists()
    assert (package_path / "io_contract.json").exists()
    assert (package_path / "contract_summary.md").exists()
    assert (package_path / "evidence" / "prepared-artifact-record.json").exists()
    assert (package_path / "evidence" / "phase4-gate-record.json").exists()

    manifest = json.loads((package_path / "contract_manifest.json").read_text(encoding="utf-8"))
    io_contract = json.loads((package_path / "io_contract.json").read_text(encoding="utf-8"))
    assert manifest["promotion_status"] == "deployment_candidate"
    assert manifest["phase4_recommendation"]["value"] == "GO"
    assert io_contract["inputs"]["x_lens"]["runtime_dtype"] == "int32"


def test_phase5_promotion_status_mapping_respects_phase4_verdicts():
    from tools.aihub_option1_phase5_contract import (
        DEPLOYMENT_CANDIDATE,
        RESEARCH_ONLY,
        map_phase4_recommendation_to_promotion_status,
    )

    assert map_phase4_recommendation_to_promotion_status({"value": "GO", "reasons": []}) == DEPLOYMENT_CANDIDATE
    assert map_phase4_recommendation_to_promotion_status({"value": "WARN", "reasons": ["minor_text_drift_present"]}) == DEPLOYMENT_CANDIDATE
    assert map_phase4_recommendation_to_promotion_status({"value": "NO_GO", "reasons": ["catastrophic_decode_failure_present"]}) == RESEARCH_ONLY


def test_phase5_io_contract_export_includes_shapes_dtypes_and_deployment_notes(tmp_path):
    from tools.aihub_option1_phase5_contract import build_phase5_io_contract

    repo_root = tmp_path / "repo"
    _init_repo_root(repo_root)
    runtime_config, paths = _write_phase5_inputs(repo_root, run_label="io")
    _ = runtime_config

    prepared_record = json.loads(paths["prepared_record_path"].read_text(encoding="utf-8"))
    compile_record = json.loads(paths["compile_record_path"].read_text(encoding="utf-8"))
    live_record = json.loads(paths["live_record_path"].read_text(encoding="utf-8"))

    io_contract = build_phase5_io_contract(
        pilot_name="zipformer",
        prepared_artifact_record=prepared_record,
        compile_run_record=compile_record,
        live_run_record=live_record,
    )

    assert io_contract["inputs"]["x"]["shape"] == [1, 2009, 80]
    assert io_contract["inputs"]["x"]["dtype"] == "float32"
    assert io_contract["inputs"]["x_lens"]["dtype"] == "int64"
    assert io_contract["inputs"]["x_lens"]["runtime_dtype"] == "int32"
    assert io_contract["outputs"]["output_0"]["shape"] == [1, 4]
    assert any("encoder-only on NPU" in note for note in io_contract["notes"])
    assert any("truncate_64bit_io" in note for note in io_contract["notes"])


def test_phase5_supports_vpcd_compile_pilot_override(tmp_path):
    from tools.aihub_option1_hybrid_pipeline import ResolvedCompiledTarget, write_hybrid_run_record
    from tools.aihub_option1_phase4_gate import build_phase4_gate_config, build_phase4_gate_record_payload, write_phase4_gate_record
    from tools.aihub_option1_phase5_contract import materialize_phase5_contract_package, resolve_phase5_evidence_inputs
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
        source_kind="local_quantized_candidate",
        packaging_kind="aimet",
        packaging_path=repo_root / "build" / "aihub" / "vpcd_option1_local_aimet" / "wint8_aint16_min_max_local_quality_parity" / "model.option1.aimet",
        run_label="vpcd-phase5",
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
        run_label="vpcd-phase5",
    )
    live_record_path = write_live_run_record(
        pilot_name="vpcd_option1_local_aimet",
        runtime_config=runtime_config,
        compile_options="--target_runtime precompiled_qnn_onnx --truncate_64bit_io",
        job_options="--compute_unit npu",
        profile_job={"job_id": "profile-vpcd", "url": "https://example/jobs/profile-vpcd"},
        inference_job={"job_id": "infer-vpcd", "url": "https://example/jobs/infer-vpcd"},
        output_tensors={"output_0": [np.zeros((1, 8), dtype=np.float32)]},
        run_label="vpcd-phase5",
    )
    hybrid_record_path = write_hybrid_run_record(
        pilot_name="vpcd_hybrid_option1",
        runtime_config=runtime_config,
        target_reference=ResolvedCompiledTarget(
            compile_pilot_name="vpcd_option1_local_aimet",
            target_model_id="vpcd-aimet-target",
            compile_record_path=compile_record_path,
            run_label="vpcd-phase5",
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
        run_label="vpcd-phase5",
    )
    gate_payload = build_phase4_gate_record_payload(
        pilot_name="vpcd",
        runtime_config=runtime_config,
        run_label="vpcd-phase5",
        phase2_compile_pilot_name_override="vpcd_option1_local_aimet",
        target_model_id="vpcd-aimet-target",
        compile_record_path=compile_record_path,
        prepared_record_path=prepared_record_path,
        live_run_record_path=live_record_path,
        hybrid_run_record_path=hybrid_record_path,
        benchmark_summary={
            "iterations": [{"iteration_index": 1, "total_seconds": 1.3, "cloud_inference_seconds": 1.2, "decode_seconds": 0.1}],
            "warmup": {"total_seconds": 1.3, "cloud_inference_seconds": 1.2, "decode_seconds": 0.1},
            "steady_state": {"count": 0, "total_seconds_mean": None, "total_seconds_min": None, "total_seconds_max": None},
            "latency_summary": {"average_cloud_inference_seconds": 1.2, "average_decode_seconds": 0.1},
        },
        correctness_summary={
            "severity_counts": {"exact_match": 1},
            "sample_results": [{"sample_key": 0, "severity": "exact_match", "reasons": ["exact_text_match"]}],
            "worst_severity": "exact_match",
            "matched_samples": 1,
            "mismatched_samples": 0,
        },
        footprint_summary={
            "prepared_model_size_bytes": len(b"vpcd-qdq"),
            "output_tensor_footprint_bytes": 32,
            "generated_token_footprint_bytes": 40,
            "generated_token_count": 5,
            "host_rss_delta_bytes": None,
            "host_rss_status": "unavailable",
        },
        recommendation={"value": "GO", "reasons": ["exact_match_only"]},
        config=build_phase4_gate_config(),
    )
    phase4_record_path = write_phase4_gate_record(
        pilot_name="vpcd",
        runtime_config=runtime_config,
        payload=gate_payload,
        run_label="vpcd-phase5",
    )

    evidence = resolve_phase5_evidence_inputs(
        pilot_name="vpcd",
        runtime_config=runtime_config,
        run_label="vpcd-phase5",
        phase2_compile_pilot_name_override="vpcd_option1_local_aimet",
    )
    assert evidence["phase2_compile_pilot_name"] == "vpcd_option1_local_aimet"
    assert evidence["required_record_paths"]["prepared_artifact_record"] == prepared_record_path
    assert evidence["required_record_paths"]["compile_run_record"] == compile_record_path
    assert evidence["required_record_paths"]["live_run_record"] == live_record_path
    assert evidence["required_record_paths"]["hybrid_run_record"] == hybrid_record_path
    assert evidence["required_record_paths"]["phase4_gate_record"] == phase4_record_path

    report = materialize_phase5_contract_package(
        pilot_name="vpcd",
        runtime_config=runtime_config,
        run_label="vpcd-phase5",
        phase2_compile_pilot_name_override="vpcd_option1_local_aimet",
    )

    manifest = json.loads((report["package_path"] / "contract_manifest.json").read_text(encoding="utf-8"))
    assert manifest["promotion_status"] == "deployment_candidate"
    assert manifest["phase4_recommendation"]["value"] == "GO"
    assert manifest["source_artifacts"]["prepared_model"]["path"] == prepared_model_path.resolve().as_posix()
