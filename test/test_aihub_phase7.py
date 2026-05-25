from __future__ import annotations

import json
from pathlib import Path

from model_bundle.manifest import ModelBundleManifest


def _write_vpcd_bundle(bundle_dir: Path, *, samples: list[dict]) -> Path:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    manifest = ModelBundleManifest(
        bundle_version=1,
        project="vpcd",
        model_family="bartpho-seq2seq",
        model_name="tourmii/vietnamese-punc-cap-denorm-v1",
        model_variant="precompiled_qnn_onnx",
        asset_namespace="models/punctuation/vpcd/precompiled_qnn_onnx",
        runtime_kind="onnx",
        artifacts={
            "model": "model.mobile.onnx",
            "tokenizer_encode": "tokenizer.encode.onnx",
            "tokenizer_decode": "tokenizer.decode.onnx",
            "tokenizer_to_model_id_map": "tokenizer.to_model_id_map.json",
            "model_to_tokenizer_id_map": "model.to_tokenizer_id_map.json",
        },
        fixtures={"golden_samples": "golden_samples.jsonl"},
        metadata={"max_decode_length": 32},
    )
    manifest.write_json(bundle_dir / "bundle_manifest.json")
    (bundle_dir / "golden_samples.jsonl").write_text(
        "".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in samples),
        encoding="utf-8",
    )
    return bundle_dir


def _write_zipformer_bundle(bundle_dir: Path, *, sample_rows: list[dict], expected_rows: list[dict]) -> Path:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    manifest = ModelBundleManifest(
        bundle_version=1,
        project="zipformer",
        model_family="zipformer-rnnt",
        model_name="zipformer/precompiled_qnn_onnx",
        model_variant="precompiled_qnn_onnx",
        asset_namespace="models/asr/zipformer/precompiled_qnn_onnx",
        runtime_kind="rnnt_greedy",
        artifacts={
            "encoder": "encoder.onnx",
            "decoder": "decoder.onnx",
            "joiner": "joiner.onnx",
            "tokens": "tokens.txt",
        },
        fixtures={
            "sample_manifest": "sample_manifest.jsonl",
            "expected_outputs": "expected_outputs.jsonl",
        },
        metadata={"sample_rate": 16000, "feature_dim": 80, "blank_id": 0, "context_size": 2},
    )
    manifest.write_json(bundle_dir / "bundle_manifest.json")
    (bundle_dir / "sample_manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in sample_rows),
        encoding="utf-8",
    )
    (bundle_dir / "expected_outputs.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in expected_rows),
        encoding="utf-8",
    )
    for row in expected_rows:
        audio_path = bundle_dir.parents[5] / row["audio_path"]
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"audio")
    return bundle_dir


def test_normalize_zipformer_text_matches_android_contract():
    from aihub.phase7 import normalize_zipformer_text

    assert normalize_zipformer_text("â–XINâ–CHAO") == "XIN CHAO"
    assert normalize_zipformer_text("Ã¢â€“ÂXIN  Ã¢â€“ÂCHAO") == "XIN CHAO"
    assert normalize_zipformer_text("  ▁XIN   ▁CHAO  ") == "XIN CHAO"


def test_evaluate_vpcd_golden_passes_exact_control(monkeypatch, tmp_path):
    from aihub import phase7

    bundle_dir = _write_vpcd_bundle(
        tmp_path / "vpcd",
        samples=[
            {"raw_text": "xin chao", "expected_output": "Xin chao."},
            {"raw_text": "chào các bạn hôm nay", "expected_output": "Chào các bạn, hôm nay."},
        ],
    )

    outputs = {
        "xin chao": "Xin chao.",
        "chào các bạn hôm nay": "Chào các bạn, hôm nay.",
    }

    class FakeRuntime:
        @classmethod
        def from_manifest_path(cls, _manifest_path: Path, provider: str = "CPUExecutionProvider"):
            assert provider == "CPUExecutionProvider"
            return cls()

        def restore(self, raw_text: str, max_length: int = 32) -> str:
            assert max_length == 32
            return outputs[raw_text]

    monkeypatch.setattr(phase7, "BundleOnnxRuntime", FakeRuntime)

    report = phase7.evaluate_vpcd_golden(bundle_dir, candidate_label="control")

    assert report["passed"] is True
    assert report["exact_match_count"] == 2
    assert report["exact_match_rate"] == 1.0
    assert report["normalized_cer"] == 0.0
    assert report["critical_regression_count"] == 0


def test_evaluate_vpcd_golden_flags_critical_regressions(monkeypatch, tmp_path):
    from aihub import phase7

    bundle_dir = _write_vpcd_bundle(
        tmp_path / "vpcd",
        samples=[
            {"raw_text": "xin chao", "expected_output": "Xin chao."},
            {
                "raw_text": "chào các bạn hôm nay 22 2 2026",
                "expected_output": "Chào các bạn, hôm nay 22/2/2026 - Phước Thành.",
            },
        ],
    )

    outputs = {
        "xin chao": "Xin chao",
        "chào các bạn hôm nay 22 2 2026": "Chào các bạn hôm nay 2026 22 2 phước thành",
    }

    class FakeRuntime:
        @classmethod
        def from_manifest_path(cls, _manifest_path: Path, provider: str = "CPUExecutionProvider"):
            return cls()

        def restore(self, raw_text: str, max_length: int = 32) -> str:
            return outputs[raw_text]

    monkeypatch.setattr(phase7, "BundleOnnxRuntime", FakeRuntime)

    report = phase7.evaluate_vpcd_golden(bundle_dir, candidate_label="bad-lane")

    assert report["passed"] is False
    assert report["critical_regression_count"] >= 1
    mismatch = report["mismatches"][1]
    assert "sentence_final_punctuation" in mismatch["critical_regressions"]
    assert "date_number_formatting" in mismatch["critical_regressions"]
    assert "proper_name_capitalization" in mismatch["critical_regressions"]


def test_evaluate_vpcd_latency_reports_session_and_process_metrics(monkeypatch, tmp_path):
    from aihub import phase7

    bundle_dir = _write_vpcd_bundle(
        tmp_path / "vpcd",
        samples=[
            {"raw_text": "xin chao", "expected_output": "Xin chao."},
            {"raw_text": "chao cac ban", "expected_output": "Chao cac ban."},
        ],
    )

    outputs = {
        "xin chao": "Xin chao.",
        "chao cac ban": "Chao cac ban.",
    }

    class FakeRuntime:
        @classmethod
        def from_manifest_path(cls, _manifest_path: Path, provider: str = "CPUExecutionProvider"):
            assert provider == "CPUExecutionProvider"
            return cls()

        def restore(self, raw_text: str, max_length: int = 32) -> str:
            assert max_length == 32
            return outputs[raw_text]

    perf_ticks = iter((1.0, 1.25, 2.0, 2.02, 3.0, 3.08))
    monkeypatch.setattr(phase7, "BundleOnnxRuntime", FakeRuntime)
    monkeypatch.setattr(phase7.time, "perf_counter", lambda: next(perf_ticks))

    report = phase7.evaluate_vpcd_latency(bundle_dir, candidate_label="control")

    assert report["project"] == "vpcd"
    assert report["candidate_label"] == "control"
    assert report["sample_count"] == 2
    assert report["session_init_ms"] == 250.0
    assert report["median_process_ms"] == 50.0
    assert report["p95_process_ms"] == 80.0
    assert report["total_process_ms"] == 100.0
    assert report["exact_match_rate"] == 1.0
    assert report["normalized_cer"] == 0.0
    assert [row["process_ms"] for row in report["reports"]] == [20.0, 80.0]


def test_evaluate_zipformer_golden_normalizes_wordpiece_separators(monkeypatch, tmp_path):
    from aihub import phase7

    repo_root = tmp_path / "repo"
    bundle_dir = _write_zipformer_bundle(
        repo_root / "modelassets" / "zipformer" / "precompiled_qnn_onnx",
        sample_rows=[
            {"sample_id": "sample-1", "audio_path": "assets/speech/sample-1.wav"},
        ],
        expected_rows=[
            {"sample_id": "sample-1", "audio_path": "assets/speech/sample-1.wav", "text": "â–XINâ–CHAO"},
        ],
    )

    class FakeRuntime:
        @classmethod
        def from_manifest_path(cls, _manifest_path: Path, provider: str = "CPUExecutionProvider"):
            return cls()

        def transcribe(self, audio_path: Path) -> dict[str, str]:
            assert audio_path.exists()
            return {"text": " ▁XIN   ▁CHAO "}

    monkeypatch.setattr(phase7, "BundleAcousticRuntime", FakeRuntime)

    report = phase7.evaluate_zipformer_golden(bundle_dir, candidate_label="control", repo_root=repo_root)

    assert report["passed"] is True
    assert report["exact_match_count"] == 1
    assert report["normalized_cer"] == 0.0


def test_collect_phase7_candidate_metadata_reads_bundle_and_compile_records(tmp_path):
    from aihub.phase7 import collect_phase7_candidate_metadata

    bundle_dir = _write_vpcd_bundle(
        tmp_path / "vpcd",
        samples=[{"raw_text": "xin chao", "expected_output": "Xin chao."}],
    )
    (bundle_dir / "model.mobile.onnx").write_bytes(b"onnx")
    (bundle_dir / "model.bin").write_bytes(b"binary")
    (bundle_dir / "io_contract.json").write_text('{"target_runtime":"precompiled_qnn_onnx"}', encoding="utf-8")

    compile_record = tmp_path / "compile-run.json"
    compile_record.write_text(
        json.dumps(
            {
                "device_name": "Samsung Galaxy S23 (Family)",
                "qairt_version": "2.45",
                "compile_options": "--target_runtime precompiled_qnn_onnx --truncate_64bit_io",
                "target_model": {"model_id": "model-123", "url": "https://example/model-123"},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    metadata = collect_phase7_candidate_metadata(
        project="vpcd",
        candidate_label="VPCD-L0-control",
        model_root=bundle_dir,
        compile_record_path=compile_record,
    )

    assert metadata["project"] == "vpcd"
    assert metadata["candidate_label"] == "VPCD-L0-control"
    assert metadata["bundle"]["artifact_count"] >= 2
    assert metadata["bundle"]["total_bytes"] >= 4
    assert metadata["compile"]["target_model_id"] == "model-123"
    assert metadata["compile"]["qairt_version"] == "2.45"


def test_collect_phase7_candidate_metadata_includes_hybrid_record_when_present(tmp_path):
    from aihub.phase7 import collect_phase7_candidate_metadata

    bundle_dir = _write_vpcd_bundle(
        tmp_path / "vpcd",
        samples=[{"raw_text": "xin chao", "expected_output": "Xin chao."}],
    )
    (bundle_dir / "model.mobile.onnx").write_bytes(b"onnx")

    hybrid_record = tmp_path / "hybrid-run.json"
    hybrid_record.write_text(
        json.dumps(
            {
                "device_name": "Samsung Galaxy S23 (Family)",
                "qairt_version": "2.45",
                "compile_options": "--target_runtime precompiled_qnn_onnx --truncate_64bit_io",
                "target_model_id": "hybrid-model-1",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    metadata = collect_phase7_candidate_metadata(
        project="vpcd",
        candidate_label="VPCD-L2-decoder-expanded",
        model_root=bundle_dir,
        hybrid_record_path=hybrid_record,
    )

    assert metadata["hybrid"]["target_model_id"] == "hybrid-model-1"
    assert metadata["hybrid"]["qairt_version"] == "2.45"


def test_evaluate_vpcd_golden_explains_precompiled_qnn_cpu_incompatibility(monkeypatch, tmp_path):
    from aihub import phase7

    bundle_dir = _write_vpcd_bundle(
        tmp_path / "vpcd",
        samples=[{"raw_text": "xin chao", "expected_output": "Xin chao."}],
    )

    class FakeRuntime:
        @classmethod
        def from_manifest_path(cls, _manifest_path: Path, provider: str = "CPUExecutionProvider"):
            raise RuntimeError(
                "EPContext node generated by 'QNN' is not compatible with any execution provider added to the session."
            )

    monkeypatch.setattr(phase7, "BundleOnnxRuntime", FakeRuntime)

    try:
        phase7.evaluate_vpcd_golden(bundle_dir, candidate_label="shipping-precompiled")
    except RuntimeError as exc:
        assert "precompiled_qnn_onnx" in str(exc)
        assert "source bundle" in str(exc)
    else:
        raise AssertionError("Expected a RuntimeError for precompiled CPU-incompatible bundles.")


def test_materialize_vpcd_local_aimet_candidate_bundle_copies_qdq_bundle_shape(tmp_path):
    from aihub.phase7 import materialize_vpcd_local_aimet_candidate_bundle

    control_bundle_dir = _write_vpcd_bundle(
        tmp_path / "control",
        samples=[{"raw_text": "xin chao", "expected_output": "Xin chao."}],
    )
    control_manifest = ModelBundleManifest.from_path(control_bundle_dir / "bundle_manifest.json")
    (control_bundle_dir / control_manifest.artifacts["model"]).write_bytes(b"control-model")
    (control_bundle_dir / control_manifest.artifacts["tokenizer_encode"]).write_bytes(b"encode")
    (control_bundle_dir / control_manifest.artifacts["tokenizer_decode"]).write_bytes(b"decode")
    (control_bundle_dir / control_manifest.artifacts["tokenizer_to_model_id_map"]).write_text("[0,1,2]\n", encoding="utf-8")
    (control_bundle_dir / control_manifest.artifacts["model_to_tokenizer_id_map"]).write_text("[0,1,2]\n", encoding="utf-8")

    quantize_root = tmp_path / "quantize" / "wint8_aint16_min_max_local_quality_parity"
    qdq_model_path = quantize_root / "model.option1.qdq.onnx"
    qdq_data_path = quantize_root / "model.option1.qdq.onnx.data"
    package_dir = quantize_root / "model.option1.aimet"
    package_dir.mkdir(parents=True, exist_ok=True)
    qdq_model_path.write_bytes(b"qdq-model")
    qdq_data_path.write_bytes(b"qdq-data")

    quantize_report_path = quantize_root / "quantize_report.json"
    quantize_report_path.write_text(
        json.dumps(
            {
                "source_strategy": "local_aimet_compile_candidate",
                "variant_name": "wint8_aint16_min_max_local_quality_parity",
                "package_dir": package_dir.resolve().as_posix(),
                "packaging_path": package_dir.resolve().as_posix(),
                "qdq_reference_model_path": qdq_model_path.resolve().as_posix(),
                "aimet": {
                    "param_type": "int8",
                    "activation_type": "int16",
                    "quant_scheme": "min_max",
                    "policy_mode": "local_quality_parity",
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    candidate_dir = materialize_vpcd_local_aimet_candidate_bundle(
        candidate_label="VPCD-L1-parity-matmul-only",
        control_bundle_root=control_bundle_dir,
        quantize_report_path=quantize_report_path,
        output_root=tmp_path / "phase7-candidates",
    )

    manifest = ModelBundleManifest.from_path(candidate_dir / "bundle_manifest.json")
    assert candidate_dir.name == "vpcd-l1-parity-matmul-only"
    assert manifest.artifacts["model"] == "model.option1.qdq.onnx"
    assert manifest.fixtures["golden_samples"] == "golden_samples.jsonl"
    assert manifest.metadata["phase7_candidate"]["candidate_label"] == "VPCD-L1-parity-matmul-only"
    assert manifest.metadata["phase7_candidate"]["source_strategy"] == "local_aimet_compile_candidate"
    assert manifest.metadata["phase7_candidate"]["variant_name"] == "wint8_aint16_min_max_local_quality_parity"
    assert manifest.metadata["phase7_candidate"]["qdq_reference_model_path"].endswith("model.option1.qdq.onnx")
    assert manifest.metadata["quantization"]["phase7_lane"] == "VPCD-L1-parity-matmul-only"
    assert (candidate_dir / "model.option1.qdq.onnx").read_bytes() == b"qdq-model"
    assert (candidate_dir / "model.option1.qdq.onnx.data").read_bytes() == b"qdq-data"
    assert (candidate_dir / "tokenizer.encode.onnx").read_bytes() == b"encode"
    assert (candidate_dir / "tokenizer.decode.onnx").read_bytes() == b"decode"
    assert (candidate_dir / "tokenizer.to_model_id_map.json").read_text(encoding="utf-8") == "[0,1,2]\n"
    assert (candidate_dir / "model.to_tokenizer_id_map.json").read_text(encoding="utf-8") == "[0,1,2]\n"


def test_materialize_zipformer_component_candidate_bundle_mixes_control_and_quantized_components(tmp_path):
    from aihub.phase7 import materialize_zipformer_component_candidate_bundle

    repo_root = tmp_path / "repo"
    control_bundle_dir = _write_zipformer_bundle(
        repo_root / "build" / "model_bundle" / "zipformer" / "fp32",
        sample_rows=[{"sample_id": "sample-1", "audio_path": "assets/speech/sample-1.wav"}],
        expected_rows=[{"sample_id": "sample-1", "audio_path": "assets/speech/sample-1.wav", "text": "XIN CHAO"}],
    )
    quantized_bundle_dir = _write_zipformer_bundle(
        repo_root / "build" / "model_bundle" / "zipformer" / "qnn_u16u8",
        sample_rows=[{"sample_id": "sample-1", "audio_path": "assets/speech/sample-1.wav"}],
        expected_rows=[{"sample_id": "sample-1", "audio_path": "assets/speech/sample-1.wav", "text": "XIN CHAO"}],
    )

    for bundle_dir, prefix in ((control_bundle_dir, b"control"), (quantized_bundle_dir, b"quantized")):
        (bundle_dir / "encoder.onnx").write_bytes(prefix + b"-encoder")
        (bundle_dir / "decoder.onnx").write_bytes(prefix + b"-decoder")
        (bundle_dir / "joiner.onnx").write_bytes(prefix + b"-joiner")
        (bundle_dir / "tokens.txt").write_text("tok\n", encoding="utf-8")

    candidate_dir = materialize_zipformer_component_candidate_bundle(
        candidate_label="ZIP-L2-decoder-joiner-qnn",
        control_bundle_root=control_bundle_dir,
        quantized_bundle_root=quantized_bundle_dir,
        output_root=repo_root / "build" / "phase7" / "candidates",
        component_sources={
            "encoder": "control",
            "decoder": "quantized",
            "joiner": "quantized",
        },
    )

    manifest = ModelBundleManifest.from_path(candidate_dir / "bundle_manifest.json")
    assert manifest.model_variant == "zip-l2-decoder-joiner-qnn"
    assert manifest.metadata["phase7_candidate"]["candidate_label"] == "ZIP-L2-decoder-joiner-qnn"
    assert manifest.metadata["phase7_candidate"]["component_sources"] == {
        "encoder": "control",
        "decoder": "quantized",
        "joiner": "quantized",
    }
    assert manifest.metadata["quantization"]["phase7_lane"] == "ZIP-L2-decoder-joiner-qnn"
    assert (candidate_dir / "encoder.onnx").read_bytes() == b"control-encoder"
    assert (candidate_dir / "decoder.onnx").read_bytes() == b"quantized-decoder"
    assert (candidate_dir / "joiner.onnx").read_bytes() == b"quantized-joiner"
    assert (candidate_dir / "tokens.txt").read_text(encoding="utf-8") == "tok\n"
