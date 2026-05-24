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
