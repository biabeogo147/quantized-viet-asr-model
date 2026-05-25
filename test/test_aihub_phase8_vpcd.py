from __future__ import annotations

from pathlib import Path

from model_bundle.manifest import ModelBundleManifest


def _write_control_manifest(manifest_path: Path) -> Path:
    manifest = ModelBundleManifest(
        bundle_version=1,
        project="vpcd",
        model_family="bartpho-seq2seq",
        model_name="tourmii/vietnamese-punc-cap-denorm-v1",
        model_variant="qnn_fixed_1024x128",
        asset_namespace="models/punctuation/vpcd/qnn_fixed_1024x128",
        runtime_kind="onnx",
        artifacts={"model": "model.mobile.onnx"},
        fixtures={"golden_samples": "golden_samples.jsonl"},
        metadata={
            "pad_token_id": 1,
            "max_source_length": 1024,
            "max_decode_length": 128,
            "fixed_input_shapes": {
                "model": {
                    "input_ids": [1, 1024],
                    "attention_mask": [1, 1024],
                    "decoder_input_ids": [1, 128],
                    "decoder_attention_mask": [1, 128],
                }
            },
        },
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_json(manifest_path)
    return manifest_path


def test_phase8_vpcd_lane_specs_include_expected_shape_matrix():
    from aihub.phase8_vpcd import list_phase8_vpcd_lane_specs

    specs = {spec.candidate_label: spec for spec in list_phase8_vpcd_lane_specs()}

    assert specs["VPCD-A0-control-1024x128-L2"].encoder_sequence == 1024
    assert specs["VPCD-A0-control-1024x128-L2"].decoder_sequence == 128
    assert specs["VPCD-A1-512x64-L2"].encoder_sequence == 512
    assert specs["VPCD-A1-512x64-L2"].decoder_sequence == 64
    assert specs["VPCD-A2-384x64-L2"].encoder_sequence == 384
    assert specs["VPCD-A2-384x64-L2"].decoder_sequence == 64
    assert specs["VPCD-A3-256x48-L2"].encoder_sequence == 256
    assert specs["VPCD-A3-256x48-L2"].decoder_sequence == 48
    assert specs["VPCD-A4-384x64-L1"].policy_mode == "local_quality_parity"


def test_write_phase8_vpcd_override_manifest_rewrites_lengths(tmp_path):
    from aihub.phase8_vpcd import resolve_phase8_vpcd_lane_spec, write_phase8_vpcd_override_manifest

    control_manifest_path = _write_control_manifest(tmp_path / "control" / "bundle_manifest.json")
    spec = resolve_phase8_vpcd_lane_spec("VPCD-A1-512x64-L2")

    output_path = write_phase8_vpcd_override_manifest(
        control_manifest_path=control_manifest_path,
        lane_spec=spec,
        output_dir=tmp_path / "phase8-manifests",
    )

    manifest = ModelBundleManifest.from_path(output_path)
    assert manifest.metadata["max_source_length"] == 512
    assert manifest.metadata["max_decode_length"] == 64
    assert manifest.metadata["fixed_input_shapes"]["model"]["input_ids"] == [1, 512]
    assert manifest.metadata["fixed_input_shapes"]["model"]["attention_mask"] == [1, 512]
    assert manifest.metadata["fixed_input_shapes"]["model"]["decoder_input_ids"] == [1, 64]
    assert manifest.metadata["fixed_input_shapes"]["model"]["decoder_attention_mask"] == [1, 64]
    assert manifest.metadata["phase8_candidate"]["candidate_label"] == "VPCD-A1-512x64-L2"
    assert manifest.metadata["phase8_candidate"]["policy_mode"] == "decoder_expanded"


def test_materialize_phase8_vpcd_candidate_bundle_uses_phase7_bundle_shape(tmp_path):
    from aihub.phase8_vpcd import materialize_phase8_vpcd_candidate_bundle, resolve_phase8_vpcd_lane_spec

    control_dir = tmp_path / "control"
    manifest_path = _write_control_manifest(control_dir / "bundle_manifest.json")
    (control_dir / "model.mobile.onnx").write_bytes(b"control-model")
    (control_dir / "golden_samples.jsonl").write_text('{"raw_text":"xin chao","expected_output":"Xin chao."}\n', encoding="utf-8")

    quantize_dir = tmp_path / "quantize"
    qdq_model_path = quantize_dir / "model.option1.qdq.onnx"
    qdq_data_path = quantize_dir / "model.option1.qdq.onnx.data"
    qdq_model_path.parent.mkdir(parents=True, exist_ok=True)
    qdq_model_path.write_bytes(b"qdq-model")
    qdq_data_path.write_bytes(b"qdq-data")

    quantize_report_path = quantize_dir / "quantize_report.json"
    quantize_report_path.write_text(
        """{
  "source_strategy": "local_aimet_compile_candidate",
  "variant_name": "wint8_aint16_min_max_decoder_expanded",
  "qdq_reference_model_path": "%s",
  "packaging_path": "%s",
  "packaging_kind": "aimet_dir",
  "source_kind": "local_aimet",
  "transformation_kind": "aimet_service_export",
  "aimet": {
    "policy_mode": "decoder_expanded"
  }
}
""" % (qdq_model_path.as_posix(), quantize_dir.as_posix()),
        encoding="utf-8",
    )

    output_dir = materialize_phase8_vpcd_candidate_bundle(
        lane_spec=resolve_phase8_vpcd_lane_spec("VPCD-A1-512x64-L2"),
        control_bundle_root=control_dir,
        quantize_report_path=quantize_report_path,
        output_root=tmp_path / "phase8-candidates",
    )

    manifest = ModelBundleManifest.from_path(output_dir / "bundle_manifest.json")
    assert manifest.metadata["phase7_candidate"]["candidate_label"] == "VPCD-A1-512x64-L2"
    assert (output_dir / "model.option1.qdq.onnx").exists()
    assert (output_dir / "model.option1.qdq.onnx.data").exists()
