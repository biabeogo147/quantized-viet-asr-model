import json
import os
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from model_bundle.manifest import ModelBundleManifest
from model_bundle.fixtures import AudioSampleFixture, serialize_jsonl
from quantize.types import AimetQuantizeRecipe, CalibrationSample


def _init_repo_root(repo_root: Path) -> None:
    (repo_root / "src").mkdir(parents=True, exist_ok=True)
    (repo_root / "assets").mkdir(parents=True, exist_ok=True)
    (repo_root / "test").mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text("[project]\nname = 'python-model-test'\nversion = '0.0.0'\n", encoding="utf-8")


def _write_zipformer_bundle(bundle_dir: Path, *, fixed_encoder_frames: int = 128, feature_dim: int = 80) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    manifest = ModelBundleManifest(
        bundle_version=1,
        project="zipformer",
        model_family="zipformer-rnnt",
        model_name="zipformer/qnn_u16u8",
        model_variant="qnn_u16u8",
        asset_namespace="models/asr/zipformer/qnn_u16u8",
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
        metadata={
            "sample_rate": 16000,
            "feature_dim": feature_dim,
            "blank_id": 0,
            "context_size": 2,
            "fixed_input_shapes": {
                "encoder": {
                    "x": [1, fixed_encoder_frames, feature_dim],
                    "x_lens": [1],
                },
                "decoder": {"y": [1, 2]},
                "joiner": {"encoder_out": [1, 512], "decoder_out": [1, 512]},
            },
            "fixed_encoder_frames": fixed_encoder_frames,
            "quantization": {
                "format": "QDQ",
                "activation_type": "quint16",
                "weight_type": "quint8",
                "fixed_shapes": True,
            },
        },
    )
    manifest.write_json(bundle_dir / "bundle_manifest.json")
    (bundle_dir / "sample_manifest.jsonl").write_text(
        serialize_jsonl([AudioSampleFixture(sample_id="sample-1", audio_path="assets/speech/sample-1.wav")]),
        encoding="utf-8",
    )
    (bundle_dir / "expected_outputs.jsonl").write_text(
        json.dumps({"sample_id": "sample-1", "audio_path": "assets/speech/sample-1.wav", "text": "xin chao"}) + "\n",
        encoding="utf-8",
    )


def _write_vpcd_bundle(bundle_dir: Path, *, encoder_sequence: int = 8, decoder_sequence: int = 4) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    manifest = ModelBundleManifest(
        bundle_version=1,
        project="vpcd",
        model_family="bartpho-seq2seq",
        model_name="tourmii/vietnamese-punc-cap-denorm-v1",
        model_variant="vpcd_balanced_fixed_8x4",
        asset_namespace="models/punctuation/vpcd/qnn_fixed_8x4",
        runtime_kind="text_seq2seq",
        artifacts={
            "model": "model.mobile.onnx",
            "tokenizer_encode": "tokenizer.encode.onnx",
            "tokenizer_decode": "tokenizer.decode.onnx",
            "tokenizer_to_model_id_map": "tokenizer.to_model_id_map.json",
            "model_to_tokenizer_id_map": "tokenizer.from_model_id_map.json",
        },
        fixtures={"golden_samples": "golden_samples.jsonl"},
        metadata={
            "pad_token_id": 1,
            "eos_token_id": 2,
            "decoder_start_token_id": 2,
            "max_source_length": encoder_sequence,
            "max_decode_length": decoder_sequence,
            "input_text_case": "lower",
            "quantization": {
                "format": "QDQ",
                "activation_type": "quint16",
                "weight_type": "quint8",
                "fixed_shapes": True,
            },
            "qnn_readiness": {
                "target_backend": "qnn_htp",
                "model_session_candidate": True,
                "tokenizer_policy": "cpu_only_first_slice",
                "requires_fixed_shapes": True,
                "fixed_shapes_ready": True,
            },
            "fixed_input_shapes": {
                "model": {
                    "input_ids": [1, encoder_sequence],
                    "attention_mask": [1, encoder_sequence],
                    "decoder_input_ids": [1, decoder_sequence],
                    "decoder_attention_mask": [1, decoder_sequence],
                }
            },
        },
    )
    manifest.write_json(bundle_dir / "bundle_manifest.json")
    (bundle_dir / "golden_samples.jsonl").write_text(
        json.dumps(
            {
                "raw_text": "xin chao",
                "input_ids": [0, 11, 12, 2],
                "expected_output": "Xin chao.",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_minimal_vpcd_fp32_model(model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                "Cast",
                inputs=["decoder_input_ids"],
                outputs=["logits"],
                to=TensorProto.FLOAT,
            )
        ],
        name="vpcd-fp32-minimal",
        inputs=[
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "encoder_sequence"]),
            helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "encoder_sequence"]),
            helper.make_tensor_value_info("decoder_input_ids", TensorProto.INT64, ["batch", "decoder_sequence"]),
            helper.make_tensor_value_info("decoder_attention_mask", TensorProto.INT64, ["batch", "decoder_sequence"]),
        ],
        outputs=[
            helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", "decoder_sequence"]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, model_path.as_posix())


def _build_zipformer_bool_slice_model() -> onnx.ModelProto:
    bool_mask = helper.make_tensor_value_info("/GreaterOrEqual_output_0", TensorProto.BOOL, [1, 8])
    graph_output = helper.make_tensor_value_info("masked", TensorProto.FLOAT, [1, 1, 4])

    starts = helper.make_tensor("starts", TensorProto.INT64, [1], [0])
    ends = helper.make_tensor("ends", TensorProto.INT64, [1], [8])
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [1])
    steps = helper.make_tensor("steps", TensorProto.INT64, [1], [2])
    unsqueeze_axes = helper.make_tensor("unsqueeze_axes", TensorProto.INT64, [1], [1])
    where_true = helper.make_tensor("where_true", TensorProto.FLOAT, [1], [1.0])
    where_false = helper.make_tensor("where_false", TensorProto.FLOAT, [1], [0.0])

    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                "Slice",
                ["/GreaterOrEqual_output_0", "starts", "ends", "axes", "steps"],
                ["/encoder/Slice_1_output_0"],
                name="/encoder/Slice_1",
            ),
            helper.make_node(
                "Unsqueeze",
                ["/encoder/Slice_1_output_0", "unsqueeze_axes"],
                ["/encoder/1/encoder/0/self_attn_weights/Unsqueeze_15_output_0"],
                name="/encoder/1/encoder/0/self_attn_weights/Unsqueeze_15",
            ),
            helper.make_node(
                "Where",
                [
                    "/encoder/1/encoder/0/self_attn_weights/Unsqueeze_15_output_0",
                    "where_true",
                    "where_false",
                ],
                ["masked"],
                name="/encoder/1/encoder/0/self_attn_weights/Where",
            ),
        ],
        name="zipformer-bool-slice",
        inputs=[bool_mask],
        outputs=[graph_output],
        initializer=[starts, ends, axes, steps, unsqueeze_axes, where_true, where_false],
        value_info=[
            helper.make_tensor_value_info("/encoder/Slice_1_output_0", TensorProto.BOOL, [1, 4]),
            helper.make_tensor_value_info(
                "/encoder/1/encoder/0/self_attn_weights/Unsqueeze_15_output_0",
                TensorProto.BOOL,
                [1, 1, 4],
            ),
            helper.make_tensor_value_info("masked", TensorProto.FLOAT, [1, 1, 4]),
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def test_resolve_zipformer_encoder_source_prefers_fixed_shape_encoder_artifact(tmp_path):
    from aihub.session import resolve_zipformer_encoder_source

    repo_root = tmp_path / "repo"
    _init_repo_root(repo_root)
    bundle_dir = repo_root / "build" / "model_bundle" / "zipformer" / "qnn_u16u8"
    _write_zipformer_bundle(bundle_dir, fixed_encoder_frames=144, feature_dim=64)
    fixed_encoder = repo_root / "build" / "quantize" / "zipformer" / "qnn_u16u8" / "fixed_shapes" / "encoder.fixed.onnx"
    fixed_encoder.parent.mkdir(parents=True, exist_ok=True)
    fixed_encoder.write_bytes(b"encoder")

    source = resolve_zipformer_encoder_source(repo_root)

    assert source.source_model_path == fixed_encoder
    assert source.bundle_manifest_path == bundle_dir / "bundle_manifest.json"
    assert source.sample_manifest_path == bundle_dir / "sample_manifest.jsonl"
    assert source.fixed_encoder_frames == 144
    assert source.feature_dim == 64


def test_build_zipformer_encoder_calibration_entries_uses_feature_loader_and_fixed_padding(tmp_path):
    from aihub.session import (
        ZipformerEncoderSource,
        build_zipformer_encoder_calibration_entries,
    )

    repo_root = tmp_path / "repo"
    _init_repo_root(repo_root)
    sample_manifest_path = repo_root / "build" / "zipformer" / "sample_manifest.jsonl"
    sample_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    sample_manifest_path.write_text(
        serialize_jsonl([AudioSampleFixture(sample_id="sample-1", audio_path="assets/speech/sample-1.wav")]),
        encoding="utf-8",
    )

    seen = {}

    def fake_feature_loader(audio_path: Path, *, sample_rate: int, feature_dim: int) -> np.ndarray:
        seen["audio_path"] = audio_path
        seen["sample_rate"] = sample_rate
        seen["feature_dim"] = feature_dim
        return np.arange(6, dtype=np.float32).reshape(3, 2)

    source = ZipformerEncoderSource(
        repo_root=repo_root,
        source_model_path=repo_root / "build" / "zipformer" / "artifacts" / "fixed_shapes" / "encoder.fixed.onnx",
        bundle_manifest_path=repo_root / "build" / "zipformer" / "bundle_manifest.json",
        sample_manifest_path=sample_manifest_path,
        fixed_encoder_frames=5,
        sample_rate=16000,
        feature_dim=2,
    )

    dataset = build_zipformer_encoder_calibration_entries(source, max_samples=1, feature_loader=fake_feature_loader)

    assert seen["audio_path"] == repo_root / "assets" / "speech" / "sample-1.wav"
    assert seen["sample_rate"] == 16000
    assert seen["feature_dim"] == 2
    assert list(dataset.keys()) == ["x", "x_lens"]
    assert len(dataset["x"]) == 1
    assert len(dataset["x_lens"]) == 1
    assert dataset["x"][0].shape == (1, 5, 2)
    assert dataset["x_lens"][0].tolist() == [3]
    np.testing.assert_array_equal(dataset["x"][0][0, :3, :], np.arange(6, dtype=np.float32).reshape(3, 2))


def test_resolve_vpcd_source_reads_fixed_shape_candidate(tmp_path):
    from aihub.session import resolve_vpcd_source

    repo_root = tmp_path / "repo"
    _init_repo_root(repo_root)
    bundle_dir = repo_root / "build" / "model_bundle" / "vpcd" / "qnn_fixed_1024x128"
    _write_vpcd_bundle(bundle_dir, encoder_sequence=1024, decoder_sequence=128)

    source = resolve_vpcd_source(repo_root)

    assert source.bundle_manifest_path == bundle_dir / "bundle_manifest.json"
    assert source.model_path == bundle_dir / "model.mobile.onnx"
    assert source.golden_samples_path == bundle_dir / "golden_samples.jsonl"
    assert source.encoder_sequence == 1024
    assert source.decoder_sequence == 128
    assert source.pad_token_id == 1
    assert source.decoder_start_token_id == 2


def test_prepare_vpcd_source_model_rejects_retired_fp32_fixed_strategy(tmp_path):
    from aihub.session import prepare_vpcd_source_model, resolve_vpcd_source

    repo_root = tmp_path / "repo"
    _init_repo_root(repo_root)
    bundle_dir = repo_root / "build" / "model_bundle" / "vpcd" / "qnn_fixed_1024x128"
    _write_vpcd_bundle(bundle_dir, encoder_sequence=1024, decoder_sequence=128)
    fp32_model_path = repo_root / "assets" / "vietnamese-punc-cap-denorm-v1" / "onnx" / "model.fp32.onnx"
    _write_minimal_vpcd_fp32_model(fp32_model_path)

    source = resolve_vpcd_source(repo_root)
    with pytest.raises(ValueError, match="Unsupported VPCD Option 1 source strategy"):
        prepare_vpcd_source_model(
            source,
            strategy="prefer_fp32_fixed",
        )


def test_prepare_vpcd_source_model_defaults_to_local_aimet_compile_candidate(tmp_path, monkeypatch):
    from aihub.session import prepare_vpcd_source_model, resolve_vpcd_source

    repo_root = tmp_path / "repo"
    _init_repo_root(repo_root)
    bundle_dir = repo_root / "build" / "model_bundle" / "vpcd" / "qnn_fixed_1024x128"
    _write_vpcd_bundle(bundle_dir, encoder_sequence=1024, decoder_sequence=128)
    fp32_model_path = repo_root / "assets" / "vietnamese-punc-cap-denorm-v1" / "onnx" / "model.fp32.onnx"
    _write_minimal_vpcd_fp32_model(fp32_model_path)
    quantize_root = repo_root / "build" / "quantize" / "vpcd" / "local_aimet" / "wint8_aint16_min_max_local_quality_parity"
    package_dir = quantize_root / "model.option1.aimet"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "model.option1.onnx").write_bytes(b"onnx")
    (package_dir / "model.option1.encodings").write_text("{}", encoding="utf-8")
    qdq_path = quantize_root / "model.option1.qdq.onnx"
    qdq_path.write_bytes(b"qdq")
    fixed_model_path = quantize_root / "model.fp32.fixed.onnx"
    _write_minimal_vpcd_fp32_model(fixed_model_path)
    quantize_report_path = quantize_root / "quantize_report.json"
    quantize_report_path.write_text(
        json.dumps(
            {
                "source_strategy": "local_aimet_compile_candidate",
                "variant_name": "wint8_aint16_min_max_local_quality_parity",
                "fixed_model_path": fixed_model_path.resolve().as_posix(),
                "package_dir": package_dir.resolve().as_posix(),
                "qdq_reference_model_path": qdq_path.resolve().as_posix(),
                "aimet_service_url": "http://127.0.0.1:18080",
            }
        ),
        encoding="utf-8",
    )

    source = resolve_vpcd_source(repo_root)
    prepared = prepare_vpcd_source_model(source)

    prepared_model = onnx.load(prepared.prepared_model_path.as_posix())
    assert prepared.source_strategy == "local_aimet_compile_candidate"
    assert prepared.is_quantized_source is True
    assert prepared.prepared_model_path == fixed_model_path.resolve()
    assert prepared.packaging_path == package_dir.resolve()
    assert prepared.diagnostic_model_path == qdq_path.resolve()
    input_dims = {
        value.name: [dim.dim_value if dim.HasField("dim_value") else dim.dim_param for dim in value.type.tensor_type.shape.dim]
        for value in prepared_model.graph.input
    }
    assert input_dims["input_ids"] == ["batch", "encoder_sequence"]
    assert input_dims["attention_mask"] == ["batch", "encoder_sequence"]
    assert input_dims["decoder_input_ids"] == ["batch", "decoder_sequence"]
    assert input_dims["decoder_attention_mask"] == ["batch", "decoder_sequence"]


def test_prepare_vpcd_source_model_builds_local_aimet_compile_candidate(tmp_path, monkeypatch):
    from aihub.session import prepare_vpcd_source_model, resolve_vpcd_source

    repo_root = tmp_path / "repo"
    _init_repo_root(repo_root)
    bundle_dir = repo_root / "build" / "model_bundle" / "vpcd" / "qnn_fixed_1024x128"
    _write_vpcd_bundle(bundle_dir, encoder_sequence=1024, decoder_sequence=128)
    fp32_model_path = repo_root / "assets" / "vietnamese-punc-cap-denorm-v1" / "onnx" / "model.fp32.onnx"
    _write_minimal_vpcd_fp32_model(fp32_model_path)
    quantize_root = repo_root / "build" / "quantize" / "vpcd" / "local_aimet" / "wint8_aint16_min_max_local_quality_parity"
    package_dir = quantize_root / "model.option1.aimet"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "model.option1.onnx").write_bytes(b"onnx")
    (package_dir / "model.option1.encodings").write_text("{}", encoding="utf-8")
    qdq_path = quantize_root / "model.option1.qdq.onnx"
    qdq_path.write_bytes(b"qdq")
    fixed_model_path = quantize_root / "model.fp32.fixed.onnx"
    _write_minimal_vpcd_fp32_model(fixed_model_path)
    quantize_report_path = quantize_root / "quantize_report.json"
    quantize_report_path.write_text(
        json.dumps(
            {
                "source_strategy": "local_aimet_compile_candidate",
                "variant_name": "wint8_aint16_min_max_local_quality_parity",
                "fixed_model_path": fixed_model_path.resolve().as_posix(),
                "package_dir": package_dir.resolve().as_posix(),
                "qdq_reference_model_path": qdq_path.resolve().as_posix(),
                "aimet": {
                    "param_type": "int8",
                    "activation_type": "int16",
                    "variant_name": "wint8_aint16_min_max_local_quality_parity",
                    "policy_mode": "local_quality_parity",
                },
                "aimet_service_url": "http://127.0.0.1:18080",
            }
        ),
        encoding="utf-8",
    )

    source = resolve_vpcd_source(repo_root)
    prepared = prepare_vpcd_source_model(
        source,
        strategy="local_aimet_compile_candidate",
    )

    assert prepared.source_strategy == "local_aimet_compile_candidate"
    assert prepared.source_kind == "local_aimet"
    assert prepared.packaging_kind == "aimet_dir"
    assert prepared.packaging_path.name == "model.option1.aimet"
    assert prepared.packaging_path.parent.name == "wint8_aint16_min_max_local_quality_parity"
    assert prepared.diagnostic_model_path.name == "model.option1.qdq.onnx"
    assert prepared.prepared_model_path == fixed_model_path.resolve()
    assert prepared.report["aimet_service_url"] == "http://127.0.0.1:18080"
    assert prepared.report["aimet"]["param_type"] == "int8"
    assert prepared.report["aimet"]["activation_type"] == "int16"
    assert prepared.report["aimet"]["variant_name"] == "wint8_aint16_min_max_local_quality_parity"
    assert prepared.report["aimet"]["policy_mode"] == "local_quality_parity"
    assert prepared.report["packaging_path"] == prepared.packaging_path.resolve().as_posix()
    assert prepared.report["qdq_reference_model_path"] == prepared.diagnostic_model_path.resolve().as_posix()
    assert prepared.packaging_path.exists()
    assert prepared.diagnostic_model_path.exists()
    assert quantize_report_path.exists()


def test_build_vpcd_single_step_inputs_pads_to_fixed_shapes(tmp_path):
    from aihub.session import VpcdSource, build_vpcd_single_step_inputs

    repo_root = tmp_path / "repo"
    _init_repo_root(repo_root)
    golden_samples = repo_root / "build" / "model_bundle" / "vpcd" / "qnn_fixed_8x4" / "golden_samples.jsonl"
    golden_samples.parent.mkdir(parents=True, exist_ok=True)
    golden_samples.write_text(
        json.dumps(
            {
                "raw_text": "xin chao",
                "input_ids": [0, 11, 12, 2],
                "expected_output": "Xin chao.",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    source = VpcdSource(
        repo_root=repo_root,
        bundle_manifest_path=repo_root / "build" / "model_bundle" / "vpcd" / "qnn_fixed_8x4" / "bundle_manifest.json",
        model_path=repo_root / "build" / "model_bundle" / "vpcd" / "qnn_fixed_8x4" / "model.mobile.onnx",
        golden_samples_path=golden_samples,
        encoder_sequence=8,
        decoder_sequence=4,
        pad_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=2,
        input_text_case="lower",
        is_quantized_source=True,
    )

    inputs = build_vpcd_single_step_inputs(source, sample_index=0)

    assert list(inputs.keys()) == [
        "input_ids",
        "attention_mask",
        "decoder_input_ids",
        "decoder_attention_mask",
    ]
    assert inputs["input_ids"].shape == (1, 8)
    assert inputs["attention_mask"].shape == (1, 8)
    assert inputs["decoder_input_ids"].shape == (1, 4)
    assert inputs["decoder_attention_mask"].shape == (1, 4)
    assert inputs["input_ids"][0, :4].tolist() == [0, 11, 12, 2]
    assert inputs["input_ids"][0, 4:].tolist() == [1, 1, 1, 1]
    assert inputs["attention_mask"][0, :4].tolist() == [1, 1, 1, 1]
    assert inputs["attention_mask"][0, 4:].tolist() == [0, 0, 0, 0]
    assert inputs["decoder_input_ids"][0].tolist() == [2, 1, 1, 1]
    assert inputs["decoder_attention_mask"][0].tolist() == [1, 0, 0, 0]


def test_build_vpcd_autoregressive_calibration_entries_expands_decoder_prefixes(tmp_path):
    from aihub.session import VpcdSource, build_vpcd_autoregressive_calibration_entries

    repo_root = tmp_path / "repo"
    _init_repo_root(repo_root)
    golden_samples = repo_root / "build" / "model_bundle" / "vpcd" / "qnn_fixed_8x4" / "golden_samples.jsonl"
    golden_samples.parent.mkdir(parents=True, exist_ok=True)
    golden_samples.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "raw_text": "xin chao",
                        "input_ids": [0, 11, 12, 2],
                        "expected_output": "Xin chao.",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "raw_text": "xin chao ban",
                        "input_ids": [0, 11, 12, 13, 2],
                        "expected_output": "Xin chao ban.",
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    source = VpcdSource(
        repo_root=repo_root,
        bundle_manifest_path=repo_root / "build" / "model_bundle" / "vpcd" / "qnn_fixed_8x4" / "bundle_manifest.json",
        model_path=repo_root / "build" / "model_bundle" / "vpcd" / "qnn_fixed_8x4" / "model.mobile.onnx",
        golden_samples_path=golden_samples,
        encoder_sequence=8,
        decoder_sequence=4,
        pad_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=2,
        input_text_case="lower",
        is_quantized_source=False,
    )

    decoded_by_text = {
        "xin chao": (
            {
                "input_ids": np.asarray([[0, 11, 12, 2]], dtype=np.int64),
                "attention_mask": np.asarray([[1, 1, 1, 1]], dtype=np.int64),
            },
            [2, 101, 102, 2],
        ),
        "xin chao ban": (
            {
                "input_ids": np.asarray([[0, 11, 12, 13, 2]], dtype=np.int64),
                "attention_mask": np.asarray([[1, 1, 1, 1, 1]], dtype=np.int64),
            },
            [2, 201, 202],
        ),
    }

    dataset, stats = build_vpcd_autoregressive_calibration_entries(
        source,
        decode_ids_fn=lambda text: decoded_by_text[text],
        max_samples=2,
    )

    assert stats["strategy"] == "autoregressive_fp32"
    assert stats["text_samples"] == 2
    assert stats["records"] == 5
    assert stats["max_encoder_len"] == 5
    assert stats["max_decoder_prefix_len"] == 3
    assert stats["session_providers"] == "injected"
    assert dataset["input_ids"][0].shape == (1, 8)
    assert dataset["attention_mask"][0][0, :4].tolist() == [1, 1, 1, 1]
    assert dataset["decoder_input_ids"][0][0].tolist() == [2, 1, 1, 1]
    assert dataset["decoder_attention_mask"][0][0].tolist() == [1, 0, 0, 0]
    assert dataset["decoder_input_ids"][1][0].tolist() == [2, 101, 1, 1]
    assert dataset["decoder_attention_mask"][1][0].tolist() == [1, 1, 0, 0]
    assert dataset["decoder_input_ids"][2][0].tolist() == [2, 101, 102, 1]
    assert dataset["decoder_attention_mask"][2][0].tolist() == [1, 1, 1, 0]
    assert dataset["decoder_input_ids"][4][0].tolist() == [2, 201, 1, 1]
    assert dataset["decoder_attention_mask"][4][0].tolist() == [1, 1, 0, 0]


def test_build_vpcd_autoregressive_calibration_entries_prefers_text_file(tmp_path):
    from aihub.session import VpcdSource, build_vpcd_autoregressive_calibration_entries

    repo_root = tmp_path / "repo"
    _init_repo_root(repo_root)
    calibration_path = repo_root / "build" / "calibration" / "vlsp2020" / "vpcd_transcriptions.txt"
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_path.write_text("dong mot\ndong hai\n", encoding="utf-8")

    golden_samples = repo_root / "build" / "model_bundle" / "vpcd" / "qnn_fixed_8x4" / "golden_samples.jsonl"
    golden_samples.parent.mkdir(parents=True, exist_ok=True)
    golden_samples.write_text(
        json.dumps(
            {
                "raw_text": "khong duoc dung",
                "input_ids": [0, 11, 12, 2],
                "expected_output": "Khong duoc dung.",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    source = VpcdSource(
        repo_root=repo_root,
        bundle_manifest_path=repo_root / "build" / "model_bundle" / "vpcd" / "qnn_fixed_8x4" / "bundle_manifest.json",
        model_path=repo_root / "build" / "model_bundle" / "vpcd" / "qnn_fixed_8x4" / "model.mobile.onnx",
        golden_samples_path=golden_samples,
        encoder_sequence=8,
        decoder_sequence=4,
        pad_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=2,
        input_text_case="lower",
        is_quantized_source=False,
    )

    seen_texts: list[str] = []

    def _decode_ids(text: str) -> tuple[dict[str, np.ndarray], list[int]]:
        seen_texts.append(text)
        return (
            {
                "input_ids": np.asarray([[0, 11, 12, 2]], dtype=np.int64),
                "attention_mask": np.asarray([[1, 1, 1, 1]], dtype=np.int64),
            },
            [2, 10, 2],
        )

    _, stats = build_vpcd_autoregressive_calibration_entries(
        source,
        decode_ids_fn=_decode_ids,
        max_samples=2,
    )

    assert seen_texts == ["dong mot", "dong hai"]
    assert stats["calibration_source_path"] == calibration_path.resolve().as_posix()
    assert stats["text_samples"] == 2
    assert stats["records"] == 4


def test_option_helpers_build_precompiled_and_npu_flags():
    from aihub.session import (
        build_compile_options,
        coerce_inputs_for_compiled_model,
        build_job_options,
        build_zipformer_encoder_input_specs,
        requires_truncate_64bit_io,
        ZipformerEncoderSource,
    )

    assert build_compile_options() == "--target_runtime precompiled_qnn_onnx"
    assert build_compile_options(qairt_version="2.46.0") == "--target_runtime precompiled_qnn_onnx --qairt_version 2.46.0"
    assert build_job_options() == "--compute_unit npu"

    source = ZipformerEncoderSource(
        repo_root=Path("D:/repo"),
        source_model_path=Path("D:/repo/build/encoder.fixed.onnx"),
        bundle_manifest_path=Path("D:/repo/build/bundle_manifest.json"),
        sample_manifest_path=Path("D:/repo/build/sample_manifest.jsonl"),
        fixed_encoder_frames=2009,
        sample_rate=16000,
        feature_dim=80,
    )
    input_specs = build_zipformer_encoder_input_specs(source)
    assert requires_truncate_64bit_io(input_specs) is True
    assert build_compile_options(input_specs=input_specs) == "--target_runtime precompiled_qnn_onnx --truncate_64bit_io"

    compiled_inputs = coerce_inputs_for_compiled_model(
        {
            "x": [np.zeros((1, 2009, 80), dtype=np.float32)],
            "x_lens": [np.asarray([2009], dtype=np.int64)],
        },
        input_specs=input_specs,
    )
    assert compiled_inputs["x"][0].dtype == np.float32
    assert compiled_inputs["x_lens"][0].dtype == np.int32


def test_build_runtime_config_normalizes_defaults(tmp_path):
    from aihub.session import build_runtime_config

    repo_root = tmp_path / "repo"
    _init_repo_root(repo_root)

    config = build_runtime_config(
        device_name="  Samsung Galaxy S24 (Family)  ",
        qairt_version=" 2.46.0 ",
        repo_root=repo_root,
    )

    assert config.repo_root == repo_root.resolve()
    assert config.device_name == "Samsung Galaxy S24 (Family)"
    assert config.qairt_version == "2.46.0"
    assert config.compute_unit == "npu"
    assert config.artifact_root == (repo_root / "build" / "aihub").resolve()
    assert config.record_root == (repo_root / "build" / "aihub" / "records").resolve()
    assert config.pilot_artifact_dir("zipformer_encoder_option1") == (
        repo_root / "build" / "aihub" / "zipformer_encoder_option1"
    ).resolve()
    assert config.pilot_record_dir("zipformer_encoder_option1") == (
        repo_root / "build" / "aihub" / "records" / "zipformer_encoder_option1"
    ).resolve()


def test_write_prepared_artifact_record_captures_hashes_and_input_specs(tmp_path):
    from aihub.session import (
        build_runtime_config,
        write_prepared_artifact_record,
    )

    repo_root = tmp_path / "repo"
    _init_repo_root(repo_root)
    source_model_path = repo_root / "build" / "quantize" / "zipformer" / "qnn_u16u8" / "fixed_shapes" / "encoder.fixed.onnx"
    source_model_path.parent.mkdir(parents=True, exist_ok=True)
    source_model_path.write_bytes(b"source-model")

    prepared_model_path = repo_root / "build" / "quantize" / "zipformer" / "qnn_u16u8" / "aihub_compile" / "encoder.aihub.option1.onnx"
    prepared_model_path.parent.mkdir(parents=True, exist_ok=True)
    prepared_model_path.write_bytes(b"prepared-model")

    config = build_runtime_config(
        device_name="Samsung Galaxy S24 (Family)",
        qairt_version="2.46.0",
        repo_root=repo_root,
    )
    record_path = write_prepared_artifact_record(
        pilot_name="zipformer_encoder_option1",
        runtime_config=config,
        source_model_path=source_model_path,
        prepared_model_path=prepared_model_path,
        input_specs={
            "x": ((1, 2009, 80), "float32"),
            "x_lens": ((1,), "int64"),
        },
        compile_options="--target_runtime precompiled_qnn_onnx --truncate_64bit_io --qairt_version 2.46.0",
        run_label="unit-test",
    )

    payload = json.loads(record_path.read_text(encoding="utf-8"))
    assert payload["record_kind"] == "prepared_artifact"
    assert payload["pilot_name"] == "zipformer_encoder_option1"
    assert payload["device_name"] == "Samsung Galaxy S24 (Family)"
    assert payload["qairt_version"] == "2.46.0"
    assert payload["compile_options"] == "--target_runtime precompiled_qnn_onnx --truncate_64bit_io --qairt_version 2.46.0"
    assert payload["source_model"]["path"].endswith("build/quantize/zipformer/qnn_u16u8/fixed_shapes/encoder.fixed.onnx")
    assert payload["prepared_model"]["path"].endswith("build/quantize/zipformer/qnn_u16u8/aihub_compile/encoder.aihub.option1.onnx")
    assert payload["source_model"]["size_bytes"] == len(b"source-model")
    assert payload["prepared_model"]["size_bytes"] == len(b"prepared-model")
    assert payload["input_specs"]["x"]["shape"] == [1, 2009, 80]
    assert payload["input_specs"]["x"]["dtype"] == "float32"
    assert payload["input_specs"]["x_lens"]["shape"] == [1]
    assert payload["input_specs"]["x_lens"]["dtype"] == "int64"


def test_write_prepared_artifact_record_captures_local_aimet_compile_metadata(tmp_path):
    from aihub.session import (
        build_runtime_config,
        write_prepared_artifact_record,
    )

    repo_root = tmp_path / "repo"
    _init_repo_root(repo_root)
    source_model_path = repo_root / "assets" / "vietnamese-punc-cap-denorm-v1" / "onnx" / "model.fp32.onnx"
    source_model_path.parent.mkdir(parents=True, exist_ok=True)
    source_model_path.write_bytes(b"source-model")

    prepared_model_path = repo_root / "build" / "quantize" / "vpcd" / "local_aimet" / "wint8_aint16_min_max_local_quality_parity" / "model.fp32.fixed.onnx"
    prepared_model_path.parent.mkdir(parents=True, exist_ok=True)
    prepared_model_path.write_bytes(b"prepared-model")
    packaging_path = repo_root / "build" / "quantize" / "vpcd" / "local_aimet" / "wint8_aint16_min_max_local_quality_parity" / "model.option1.aimet"
    packaging_path.mkdir(parents=True, exist_ok=True)

    config = build_runtime_config(
        device_name="Samsung Galaxy S24 (Family)",
        qairt_version="2.46.0",
        repo_root=repo_root,
    )
    record_path = write_prepared_artifact_record(
        pilot_name="vpcd_option1_local_aimet",
        runtime_config=config,
        source_model_path=source_model_path,
        prepared_model_path=prepared_model_path,
        input_specs=None,
        compile_options="--target_runtime precompiled_qnn_onnx --truncate_64bit_io --qairt_version 2.46.0",
        source_strategy="local_aimet_compile_candidate",
        source_kind="local_aimet",
        packaging_kind="aimet_dir",
        packaging_path=packaging_path,
        compatibility={
            "aihub_compile_readiness": "experimental",
            "package_ready": True,
        },
        run_label="unit-test",
    )

    payload = json.loads(record_path.read_text(encoding="utf-8"))
    assert payload["record_kind"] == "prepared_artifact"
    assert payload["source_strategy"] == "local_aimet_compile_candidate"
    assert payload["source_kind"] == "local_aimet"
    assert payload["packaging_kind"] == "aimet_dir"
    assert payload["packaging_path"].endswith("build/quantize/vpcd/local_aimet/wint8_aint16_min_max_local_quality_parity/model.option1.aimet")
    assert payload["compatibility"]["aihub_compile_readiness"] == "experimental"


def test_write_live_run_record_summarizes_jobs_and_outputs(tmp_path):
    from aihub.session import (
        build_runtime_config,
        write_live_run_record,
    )

    class FakeJob:
        def __init__(self, job_id: str, url: str, status: str) -> None:
            self.job_id = job_id
            self.url = url
            self.status = status

    repo_root = tmp_path / "repo"
    _init_repo_root(repo_root)
    config = build_runtime_config(
        device_name="Samsung Galaxy S24 (Family)",
        qairt_version=None,
        repo_root=repo_root,
    )

    profile_path = repo_root / "build" / "aihub" / "profiles" / "zipformer_profile.json"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text('{"latency_ms": 12.34}', encoding="utf-8")

    record_path = write_live_run_record(
        pilot_name="zipformer_encoder_option1",
        runtime_config=config,
        compile_options="--target_runtime precompiled_qnn_onnx --truncate_64bit_io",
        job_options="--compute_unit npu",
        compile_job=FakeJob("compile-1", "https://aihub/jobs/compile-1", "SUCCESS"),
        profile_job=FakeJob("profile-1", "https://aihub/jobs/profile-1", "SUCCESS"),
        inference_job=FakeJob("infer-1", "https://aihub/jobs/infer-1", "SUCCESS"),
        output_tensors={
            "output_0": [np.zeros((1, 501, 512), dtype=np.float32)],
            "output_1": [np.asarray([501], dtype=np.int32)],
        },
        profile_path=profile_path,
        run_label="unit-test",
    )

    payload = json.loads(record_path.read_text(encoding="utf-8"))
    assert payload["record_kind"] == "live_run"
    assert payload["pilot_name"] == "zipformer_encoder_option1"
    assert payload["device_name"] == "Samsung Galaxy S24 (Family)"
    assert payload["job_options"] == "--compute_unit npu"
    assert payload["compile_options"] == "--target_runtime precompiled_qnn_onnx --truncate_64bit_io"
    assert payload["jobs"]["compile"]["job_id"] == "compile-1"
    assert payload["jobs"]["compile"]["url"] == "https://aihub/jobs/compile-1"
    assert payload["jobs"]["profile"]["status"] == "SUCCESS"
    assert payload["jobs"]["inference"]["job_id"] == "infer-1"
    assert payload["profile_artifact"]["path"].endswith("build/aihub/profiles/zipformer_profile.json")
    assert payload["output_tensors"]["output_0"][0]["shape"] == [1, 501, 512]
    assert payload["output_tensors"]["output_0"][0]["dtype"] == "float32"
    assert payload["output_tensors"]["output_1"][0]["shape"] == [1]
    assert payload["output_tensors"]["output_1"][0]["dtype"] == "int32"


def test_write_compile_run_record_captures_target_model_metadata(tmp_path):
    from aihub.session import (
        build_runtime_config,
        write_compile_run_record,
    )

    class FakeJob:
        def __init__(self, job_id: str, url: str, status: str) -> None:
            self.job_id = job_id
            self.url = url
            self.status = status

    class FakeModel:
        def __init__(self, model_id: str, url: str, name: str) -> None:
            self.model_id = model_id
            self.url = url
            self.name = name

    repo_root = tmp_path / "repo"
    _init_repo_root(repo_root)
    config = build_runtime_config(
        device_name="Samsung Galaxy S24 (Family)",
        qairt_version="2.46.0",
        repo_root=repo_root,
    )

    record_path = write_compile_run_record(
        pilot_name="zipformer_encoder_option1",
        runtime_config=config,
        compile_options="--target_runtime precompiled_qnn_onnx --truncate_64bit_io --qairt_version 2.46.0",
        compile_job=FakeJob("compile-1", "https://aihub/jobs/compile-1", "SUCCESS"),
        target_model=FakeModel("model-1", "https://aihub/models/model-1", "zipformer-target"),
        run_label="unit-test",
    )

    payload = json.loads(record_path.read_text(encoding="utf-8"))
    assert payload["record_kind"] == "compile_run"
    assert payload["pilot_name"] == "zipformer_encoder_option1"
    assert payload["device_name"] == "Samsung Galaxy S24 (Family)"
    assert payload["compile_options"] == "--target_runtime precompiled_qnn_onnx --truncate_64bit_io --qairt_version 2.46.0"
    assert payload["jobs"]["compile"]["job_id"] == "compile-1"
    assert payload["target_model"]["model_id"] == "model-1"
    assert payload["target_model"]["url"] == "https://aihub/models/model-1"
    assert payload["target_model"]["name"] == "zipformer-target"


def test_write_compile_run_record_marks_local_aimet_lane_as_quantize_disabled(tmp_path):
    from aihub.session import (
        build_runtime_config,
        write_compile_run_record,
    )

    class FakeJob:
        def __init__(self, job_id: str, url: str, status: str) -> None:
            self.job_id = job_id
            self.url = url
            self.status = status

    class FakeModel:
        def __init__(self, model_id: str, url: str, name: str) -> None:
            self.model_id = model_id
            self.url = url
            self.name = name

    repo_root = tmp_path / "repo"
    _init_repo_root(repo_root)
    config = build_runtime_config(
        device_name="Samsung Galaxy S24 (Family)",
        qairt_version="2.46.0",
        repo_root=repo_root,
    )

    record_path = write_compile_run_record(
        pilot_name="vpcd_option1_local_aimet",
        runtime_config=config,
        compile_options="--target_runtime precompiled_qnn_onnx --truncate_64bit_io --qairt_version 2.46.0",
        compile_job=FakeJob("compile-1", "https://aihub/jobs/compile-1", "SUCCESS"),
        target_model=FakeModel("model-1", "https://aihub/models/model-1", "vpcd-local-aimet-target"),
        source_strategy="local_aimet_compile_candidate",
        quantize_stage="disabled",
        compatibility={"aihub_compile_readiness": "experimental"},
        run_label="unit-test",
    )

    payload = json.loads(record_path.read_text(encoding="utf-8"))
    assert payload["record_kind"] == "compile_run"
    assert payload["pilot_name"] == "vpcd_option1_local_aimet"
    assert payload["source_strategy"] == "local_aimet_compile_candidate"
    assert payload["quantize_stage"] == "disabled"
    assert payload["compatibility"]["aihub_compile_readiness"] == "experimental"


def test_write_quantize_run_record_captures_downloaded_quantized_artifact(tmp_path):
    from aihub.session import (
        build_runtime_config,
        write_quantize_run_record,
    )

    class FakeJob:
        def __init__(self, job_id: str, url: str, status: str) -> None:
            self.job_id = job_id
            self.url = url
            self.status = status

    class FakeModel:
        def __init__(self, model_id: str, url: str, name: str) -> None:
            self.model_id = model_id
            self.url = url
            self.name = name

    repo_root = tmp_path / "repo"
    _init_repo_root(repo_root)
    config = build_runtime_config(
        device_name="Samsung Galaxy S24 (Family)",
        qairt_version=None,
        repo_root=repo_root,
    )
    quantized_model_path = repo_root / "build" / "aihub" / "vpcd_option1" / "model.quantized.unit-test.onnx"
    quantized_model_path.parent.mkdir(parents=True, exist_ok=True)
    quantized_model_path.write_bytes(b"quantized-model")

    record_path = write_quantize_run_record(
        pilot_name="vpcd_option1",
        runtime_config=config,
        quantize_job=FakeJob("quantize-1", "https://aihub/jobs/quantize-1", "SUCCESS"),
        target_model=FakeModel("model-q1", "https://aihub/models/model-q1", "vpcd-quantized"),
        quantized_model_path=quantized_model_path,
        weights_dtype_name="INT8",
        activations_dtype_name="INT16",
        quantize_options="--range_scheme min_max",
        calibration_stats={
            "records": 8,
            "dataset_fingerprint": "abc123",
            "input_order": [
                "input_ids",
                "attention_mask",
                "decoder_input_ids",
                "decoder_attention_mask",
            ],
        },
        run_label="unit-test",
    )

    payload = json.loads(record_path.read_text(encoding="utf-8"))
    assert payload["record_kind"] == "quantize_run"
    assert payload["pilot_name"] == "vpcd_option1"
    assert payload["jobs"]["quantize"]["job_id"] == "quantize-1"
    assert payload["target_model"]["model_id"] == "model-q1"
    assert payload["weights_dtype_name"] == "INT8"
    assert payload["activations_dtype_name"] == "INT16"
    assert payload["quantize_options"] == "--range_scheme min_max"
    assert payload["quantized_model"]["path"].endswith("build/aihub/vpcd_option1/model.quantized.unit-test.onnx")
    assert payload["quantized_model"]["size_bytes"] == len(b"quantized-model")
    assert payload["calibration"]["dataset_fingerprint"] == "abc123"
    assert payload["calibration"]["records"] == 8


def test_download_compiled_target_model_creates_parent_dir_and_returns_downloaded_path(tmp_path):
    from aihub.session import download_compiled_target_model

    class FakeModel:
        def __init__(self) -> None:
            self.download_calls: list[str] = []

        def download(self, filename: str) -> str:
            self.download_calls.append(filename)
            Path(filename).write_bytes(b"compiled-target")
            return filename

    output_path = tmp_path / "build" / "aihub" / "deploy" / "zipformer" / "encoder.precompiled.onnx"
    model = FakeModel()

    downloaded_path = download_compiled_target_model(
        target_model=model,
        output_path=output_path,
    )

    assert downloaded_path == output_path.resolve()
    assert downloaded_path.read_bytes() == b"compiled-target"
    assert model.download_calls == [output_path.resolve().as_posix()]


def test_download_compiled_target_model_fails_when_downloaded_file_missing(tmp_path):
    from aihub.session import download_compiled_target_model

    class FakeModel:
        def download(self, filename: str) -> None:
            return None

    output_path = tmp_path / "build" / "aihub" / "deploy" / "zipformer" / "encoder.precompiled.onnx"

    with pytest.raises(FileNotFoundError, match="Compiled target model was not downloaded"):
        download_compiled_target_model(
            target_model=FakeModel(),
            output_path=output_path,
        )


def test_write_deployment_download_record_captures_downloaded_artifact_metadata(tmp_path):
    from aihub.session import (
        build_runtime_config,
        write_deployment_download_record,
    )

    class FakeModel:
        def __init__(self, model_id: str, url: str, name: str) -> None:
            self.model_id = model_id
            self.url = url
            self.name = name

    repo_root = tmp_path / "repo"
    _init_repo_root(repo_root)
    config = build_runtime_config(
        device_name="Samsung Galaxy S24 (Family)",
        qairt_version="2.46.0",
        repo_root=repo_root,
    )

    compile_record_path = config.pilot_record_dir("zipformer_encoder_option1") / "compile-run-unit-test.json"
    compile_record_path.parent.mkdir(parents=True, exist_ok=True)
    compile_record_path.write_text('{"record_kind": "compile_run"}', encoding="utf-8")

    artifact_path = repo_root / "build" / "aihub" / "deploy" / "zipformer" / "download" / "encoder.precompiled.onnx"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(b"compiled-target")

    record_path = write_deployment_download_record(
        pilot_name="zipformer_encoder_option1",
        runtime_config=config,
        compile_record_path=compile_record_path,
        target_model=FakeModel("model-1", "https://aihub/models/model-1", "zipformer-target"),
        downloaded_artifact_path=artifact_path,
        run_label="unit-test",
    )

    payload = json.loads(record_path.read_text(encoding="utf-8"))
    assert payload["record_kind"] == "deployment_download"
    assert payload["pilot_name"] == "zipformer_encoder_option1"
    assert payload["device_name"] == "Samsung Galaxy S24 (Family)"
    assert payload["qairt_version"] == "2.46.0"
    assert payload["compile_record_path"] == compile_record_path.resolve().as_posix()
    assert payload["target_model"]["model_id"] == "model-1"
    assert payload["target_model"]["url"] == "https://aihub/models/model-1"
    assert payload["target_model"]["name"] == "zipformer-target"
    assert payload["downloaded_artifact"]["path"] == artifact_path.resolve().as_posix()
    assert payload["downloaded_artifact"]["size_bytes"] == len(b"compiled-target")


def test_resolve_downloaded_quantized_model_path_uses_explicit_value_or_quantize_record(tmp_path):
    from aihub.session import (
        build_runtime_config,
        resolve_downloaded_quantized_model_path,
        write_quantize_run_record,
    )

    class FakeJob:
        def __init__(self, job_id: str, url: str, status: str) -> None:
            self.job_id = job_id
            self.url = url
            self.status = status

    class FakeModel:
        def __init__(self, model_id: str, url: str, name: str) -> None:
            self.model_id = model_id
            self.url = url
            self.name = name

    repo_root = tmp_path / "repo"
    _init_repo_root(repo_root)
    config = build_runtime_config(
        device_name="Samsung Galaxy S24 (Family)",
        qairt_version=None,
        repo_root=repo_root,
    )
    quantized_model_path = repo_root / "build" / "aihub" / "vpcd_option1" / "model.quantized.latest.onnx"
    quantized_model_path.parent.mkdir(parents=True, exist_ok=True)
    quantized_model_path.write_bytes(b"quantized-model")

    write_quantize_run_record(
        pilot_name="vpcd_option1",
        runtime_config=config,
        quantize_job=FakeJob("quantize-1", "https://aihub/jobs/quantize-1", "SUCCESS"),
        target_model=FakeModel("model-q1", "https://aihub/models/model-q1", "vpcd-quantized"),
        quantized_model_path=quantized_model_path,
        weights_dtype_name="INT8",
        activations_dtype_name="INT16",
        quantize_options="",
        calibration_stats={"dataset_fingerprint": "abc123"},
        run_label="latest",
    )

    explicit_path = repo_root / "manual" / "quantized.override.onnx"
    explicit_path.parent.mkdir(parents=True, exist_ok=True)
    explicit_path.write_bytes(b"override")

    assert resolve_downloaded_quantized_model_path(
        pilot_name="vpcd_option1",
        runtime_config=config,
        explicit_quantized_model_path=explicit_path,
    ) == explicit_path.resolve()
    assert resolve_downloaded_quantized_model_path(
        pilot_name="vpcd_option1",
        runtime_config=config,
        explicit_quantized_model_path=None,
    ) == quantized_model_path.resolve()


def test_resolve_target_model_id_uses_explicit_value_or_compile_record(tmp_path):
    from aihub.session import (
        build_runtime_config,
        resolve_target_model_id,
        write_compile_run_record,
    )

    class FakeJob:
        def __init__(self, job_id: str, url: str, status: str) -> None:
            self.job_id = job_id
            self.url = url
            self.status = status

    class FakeModel:
        def __init__(self, model_id: str, url: str, name: str) -> None:
            self.model_id = model_id
            self.url = url
            self.name = name

    repo_root = tmp_path / "repo"
    _init_repo_root(repo_root)
    config = build_runtime_config(
        device_name="Samsung Galaxy S24 (Family)",
        qairt_version=None,
        repo_root=repo_root,
    )
    write_compile_run_record(
        pilot_name="zipformer_encoder_option1",
        runtime_config=config,
        compile_options="--target_runtime precompiled_qnn_onnx --truncate_64bit_io",
        compile_job=FakeJob("compile-1", "https://aihub/jobs/compile-1", "SUCCESS"),
        target_model=FakeModel("model-1", "https://aihub/models/model-1", "zipformer-target"),
        run_label="latest",
    )

    assert resolve_target_model_id(
        pilot_name="zipformer_encoder_option1",
        runtime_config=config,
        explicit_target_model_id="model-explicit",
    ) == "model-explicit"
    assert resolve_target_model_id(
        pilot_name="zipformer_encoder_option1",
        runtime_config=config,
        explicit_target_model_id=None,
    ) == "model-1"


def test_load_env_file_populates_missing_values_without_overriding_existing_env(tmp_path, monkeypatch):
    from aihub.session import load_env_file

    env_path = tmp_path / ".env"
    env_path.write_text(
        'QAI_HUB_API_TOKEN="token-from-env"\nOTHER_VALUE=abc123\n',
        encoding="utf-8",
    )

    monkeypatch.delenv("QAI_HUB_API_TOKEN", raising=False)
    monkeypatch.setenv("OTHER_VALUE", "keep-existing")

    loaded = load_env_file(env_path)

    assert loaded["QAI_HUB_API_TOKEN"] == "token-from-env"
    assert loaded["OTHER_VALUE"] == "abc123"
    assert os.environ["QAI_HUB_API_TOKEN"] == "token-from-env"
    assert os.environ["OTHER_VALUE"] == "keep-existing"


def test_resolve_qai_hub_api_token_reads_repo_env_file(tmp_path, monkeypatch):
    from aihub.session import resolve_qai_hub_api_token

    repo_root = tmp_path / "repo"
    _init_repo_root(repo_root)
    (repo_root / ".env").write_text("QAI_HUB_API_TOKEN=token-from-repo-env\n", encoding="utf-8")
    monkeypatch.delenv("QAI_HUB_API_TOKEN", raising=False)

    token = resolve_qai_hub_api_token(repo_root=repo_root)

    assert token == "token-from-repo-env"


def test_compare_output_tensors_reports_diff_stats():
    from aihub.session import compare_output_tensors

    reference = {
        "output_0": [np.asarray([[1.0, 2.0]], dtype=np.float32)],
        "output_1": [np.asarray([3], dtype=np.int32)],
    }
    candidate = {
        "output_0": [np.asarray([[1.0, 2.25]], dtype=np.float32)],
        "output_1": [np.asarray([3], dtype=np.int32)],
    }

    summary = compare_output_tensors(reference, candidate, atol=1e-5, rtol=1e-5)

    assert summary["output_0"]["shape_match"] is True
    assert summary["output_0"]["allclose"] is False
    assert summary["output_0"]["max_abs_diff"] == 0.25
    assert summary["output_0"]["mean_abs_diff"] == 0.125
    assert summary["output_1"]["allclose"] is True
    assert summary["output_1"]["max_abs_diff"] == 0.0


def test_summarize_vpcd_step_logits_uses_active_decoder_position():
    from aihub.session import summarize_vpcd_step_logits

    logits = np.zeros((1, 1, 4, 6), dtype=np.float32)
    logits[0, 0, 1, 5] = 9.0
    logits[0, 0, 1, 2] = 4.0
    logits[0, 0, 3, 1] = 99.0
    decoder_attention_mask = np.asarray([[1, 1, 0, 0]], dtype=np.int64)

    summary = summarize_vpcd_step_logits(logits, decoder_attention_mask, top_k=2)

    assert summary["active_index"] == 1
    assert summary["top_tokens"][0]["token_id"] == 5
    assert summary["top_tokens"][0]["score"] == 9.0
    assert summary["top_tokens"][1]["token_id"] == 2
    assert summary["top_tokens"][1]["score"] == 4.0


