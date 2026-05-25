import json
import threading
from pathlib import Path
from types import SimpleNamespace
from urllib import request as urllib_request

import numpy as np
import onnx
import pytest

from quantize.types import CalibrationSample


def _write_minimal_vpcd_fp32_model(model_path: Path) -> None:
    from onnx import TensorProto, helper, numpy_helper

    model_path.parent.mkdir(parents=True, exist_ok=True)
    lm_head_weight = numpy_helper.from_array(np.asarray([[1.0], [1.0]], dtype=np.float32), name="lm_head_weight")
    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                "Cast",
                inputs=["decoder_input_ids"],
                outputs=["decoder_hidden"],
                to=TensorProto.FLOAT,
                name="/model/decoder/Cast",
            ),
            helper.make_node(
                "MatMul",
                inputs=["decoder_hidden", "lm_head_weight"],
                outputs=["logits"],
                name="/lm_head/MatMul",
            ),
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
        initializer=[lm_head_weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, model_path.as_posix())


def _write_decoder_matmul_policy_model(model_path: Path) -> None:
    from onnx import TensorProto, helper, numpy_helper

    model_path.parent.mkdir(parents=True, exist_ok=True)
    decoder_weight = numpy_helper.from_array(np.asarray([[1.0], [2.0]], dtype=np.float32), name="decoder_weight")
    lm_head_weight = numpy_helper.from_array(np.asarray([[1.0], [1.0]], dtype=np.float32), name="lm_head_weight")
    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                "Cast",
                inputs=["decoder_input_ids"],
                outputs=["decoder_hidden"],
                to=TensorProto.FLOAT,
                name="/model/decoder/Cast",
            ),
            helper.make_node(
                "MatMul",
                inputs=["decoder_hidden", "decoder_weight"],
                outputs=["decoder_projected"],
                name="/model/decoder/attn/MatMul",
            ),
            helper.make_node(
                "MatMul",
                inputs=["decoder_projected", "lm_head_weight"],
                outputs=["logits"],
                name="/lm_head/MatMul",
            ),
        ],
        name="vpcd-decoder-policy",
        inputs=[
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "encoder_sequence"]),
            helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "encoder_sequence"]),
            helper.make_tensor_value_info("decoder_input_ids", TensorProto.INT64, ["batch", "decoder_sequence"]),
            helper.make_tensor_value_info("decoder_attention_mask", TensorProto.INT64, ["batch", "decoder_sequence"]),
        ],
        outputs=[
            helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", "decoder_sequence"]),
        ],
        initializer=[decoder_weight, lm_head_weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, model_path.as_posix())


def _write_decoder_broader_policy_model(model_path: Path) -> None:
    from onnx import TensorProto, helper, numpy_helper

    model_path.parent.mkdir(parents=True, exist_ok=True)
    decoder_weight = numpy_helper.from_array(np.asarray([[1.0], [2.0]], dtype=np.float32), name="decoder_weight")
    lm_head_weight = numpy_helper.from_array(np.asarray([[1.0], [1.0]], dtype=np.float32), name="lm_head_weight")
    add_bias = numpy_helper.from_array(np.asarray([[0.5]], dtype=np.float32), name="decoder_add_bias")
    mul_scale = numpy_helper.from_array(np.asarray([[0.25]], dtype=np.float32), name="decoder_mul_scale")
    div_scale = numpy_helper.from_array(np.asarray([[2.0]], dtype=np.float32), name="decoder_div_scale")
    layernorm_scale = numpy_helper.from_array(np.asarray([1.0], dtype=np.float32), name="decoder_ln_scale")
    layernorm_bias = numpy_helper.from_array(np.asarray([0.0], dtype=np.float32), name="decoder_ln_bias")
    encoder_bias = numpy_helper.from_array(np.asarray([[1.0]], dtype=np.float32), name="encoder_add_bias")
    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                "Cast",
                inputs=["decoder_input_ids"],
                outputs=["decoder_hidden"],
                to=TensorProto.FLOAT,
                name="/model/decoder/Cast",
            ),
            helper.make_node(
                "MatMul",
                inputs=["decoder_hidden", "decoder_weight"],
                outputs=["decoder_projected"],
                name="/model/decoder/attn/MatMul",
            ),
            helper.make_node(
                "Add",
                inputs=["decoder_projected", "add_bias"],
                outputs=["decoder_added"],
                name="/model/decoder/ffn/Add",
            ),
            helper.make_node(
                "Mul",
                inputs=["decoder_added", "mul_scale"],
                outputs=["decoder_scaled"],
                name="/model/decoder/ffn/Mul",
            ),
            helper.make_node(
                "Div",
                inputs=["decoder_scaled", "div_scale"],
                outputs=["decoder_divided"],
                name="/model/decoder/ffn/Div",
            ),
            helper.make_node(
                "LayerNormalization",
                inputs=["decoder_divided", "layernorm_scale", "layernorm_bias"],
                outputs=["decoder_norm"],
                name="/model/decoder/ffn/LayerNormalization",
                axis=-1,
                epsilon=1e-05,
            ),
            helper.make_node(
                "Add",
                inputs=["decoder_norm", "encoder_add_bias"],
                outputs=["encoder_like_hidden"],
                name="/model/encoder/ffn/Add",
            ),
            helper.make_node(
                "MatMul",
                inputs=["encoder_like_hidden", "lm_head_weight"],
                outputs=["logits"],
                name="/lm_head/MatMul",
            ),
        ],
        name="vpcd-decoder-broader-policy",
        inputs=[
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "encoder_sequence"]),
            helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "encoder_sequence"]),
            helper.make_tensor_value_info("decoder_input_ids", TensorProto.INT64, ["batch", "decoder_sequence"]),
            helper.make_tensor_value_info("decoder_attention_mask", TensorProto.INT64, ["batch", "decoder_sequence"]),
        ],
        outputs=[
            helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", "decoder_sequence"]),
        ],
        initializer=[
            decoder_weight,
            lm_head_weight,
            add_bias,
            mul_scale,
            div_scale,
            layernorm_scale,
            layernorm_bias,
            encoder_bias,
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, model_path.as_posix())


def test_calibration_records_to_fixed_input_dataset_preserves_input_order():
    from quantize.projects.vpcd import calibration_records_to_fixed_input_dataset

    records = [
        CalibrationSample(
            inputs={
                "input_ids": np.asarray([[1, 2, 3]], dtype=np.int64),
                "attention_mask": np.asarray([[1, 1, 1]], dtype=np.int64),
            }
        ),
        CalibrationSample(
            inputs={
                "input_ids": np.asarray([[4, 5, 6]], dtype=np.int64),
                "attention_mask": np.asarray([[1, 1, 0]], dtype=np.int64),
            }
        ),
    ]

    dataset = calibration_records_to_fixed_input_dataset(records)

    assert list(dataset.keys()) == ["input_ids", "attention_mask"]
    assert len(dataset["input_ids"]) == 2
    np.testing.assert_array_equal(dataset["input_ids"][1], np.asarray([[4, 5, 6]], dtype=np.int64))
    np.testing.assert_array_equal(dataset["attention_mask"][1], np.asarray([[1, 1, 0]], dtype=np.int64))


def test_summarize_fixed_input_calibration_dataset_fingerprint_is_stable():
    from quantize.projects.vpcd import summarize_fixed_input_calibration_dataset

    dataset = {
        "input_ids": [
            np.asarray([[7, 8, 2, 1]], dtype=np.int64),
            np.asarray([[7, 8, 1, 1]], dtype=np.int64),
        ],
        "attention_mask": [
            np.asarray([[1, 1, 1, 0]], dtype=np.int64),
            np.asarray([[1, 1, 0, 0]], dtype=np.int64),
        ],
    }

    summary_a = summarize_fixed_input_calibration_dataset(dataset)
    summary_b = summarize_fixed_input_calibration_dataset(dataset)

    assert summary_a["input_order"] == ["input_ids", "attention_mask"]
    assert summary_a["input_sample_counts"] == {"input_ids": 2, "attention_mask": 2}
    assert summary_a["input_dtypes"] == {"input_ids": "int64", "attention_mask": "int64"}
    assert summary_a["dataset_fingerprint"] == summary_b["dataset_fingerprint"]


def test_build_vpcd_aimet_quantize_recipe_uses_autoregressive_records(monkeypatch, tmp_path):
    from quantize.projects.vpcd import build_vpcd_aimet_quantize_recipe

    seen: dict[str, object] = {}

    def fake_build_calibration_records(**kwargs):
        seen.update(kwargs)
        return (
            [
                CalibrationSample(
                    inputs={
                        "input_ids": np.asarray([[7, 8, 2]], dtype=np.int64),
                        "attention_mask": np.asarray([[1, 1, 1]], dtype=np.int64),
                        "decoder_input_ids": np.asarray([[2, 9]], dtype=np.int64),
                        "decoder_attention_mask": np.asarray([[1, 1]], dtype=np.int64),
                    }
                )
            ],
            {
                "requested_provider": "cpu",
                "session_providers": "CPUExecutionProvider",
                "source_files": 1,
                "text_samples": 1,
                "records": 1,
                "max_encoder_len": 3,
                "max_decoder_len": 2,
            },
        )

    monkeypatch.setattr("quantize.projects.vpcd.build_calibration_records", fake_build_calibration_records)

    _write_minimal_vpcd_fp32_model(tmp_path / "assets" / "vpcd" / "onnx" / "model.fp32.onnx")

    recipe = build_vpcd_aimet_quantize_recipe(
        model_dir=tmp_path / "assets" / "vpcd",
        fp32_onnx_path=tmp_path / "assets" / "vpcd" / "onnx" / "model.fp32.onnx",
        calibration_source_path=tmp_path / "build" / "calibration" / "vpcd_transcriptions.txt",
        max_calibration_samples=16,
        max_generation_length=32,
        ort_provider="cpu",
        fixed_input_shapes={
            "input_ids": (1, 8),
            "attention_mask": (1, 8),
            "decoder_input_ids": (1, 4),
            "decoder_attention_mask": (1, 4),
        },
        pad_token_id=1,
    )

    assert Path(seen["model_dir"]) == tmp_path / "assets" / "vpcd"
    assert Path(seen["fp32_onnx_path"]) == tmp_path / "assets" / "vpcd" / "onnx" / "model.fp32.onnx"
    assert Path(seen["calibration_source_path"]) == tmp_path / "build" / "calibration" / "vpcd_transcriptions.txt"
    assert seen["max_calibration_samples"] == 16
    assert seen["max_generation_length"] == 32
    assert seen["ort_provider"] == "cpu"
    assert recipe.param_type == "int8"
    assert recipe.activation_type == "int16"
    assert recipe.quant_scheme == "min_max"
    assert recipe.config_file == "vpcd_matmul_only"
    assert recipe.policy_mode == "local_quality_parity"
    assert recipe.calibration_stats["input_order"] == [
        "input_ids",
        "attention_mask",
        "decoder_input_ids",
        "decoder_attention_mask",
    ]
    assert recipe.calibration_stats["input_sample_counts"] == {
        "input_ids": 1,
        "attention_mask": 1,
        "decoder_input_ids": 1,
        "decoder_attention_mask": 1,
    }
    assert recipe.calibration_stats["input_dtypes"] == {
        "input_ids": "int64",
        "attention_mask": "int64",
        "decoder_input_ids": "int64",
        "decoder_attention_mask": "int64",
    }
    assert recipe.calibration_stats["dataset_fingerprint"]
    assert len(recipe.calibration_inputs) == 1
    assert recipe.calibration_inputs[0].inputs["input_ids"].shape == (1, 8)
    assert recipe.calibration_inputs[0].inputs["attention_mask"].shape == (1, 8)
    assert recipe.calibration_inputs[0].inputs["decoder_input_ids"].shape == (1, 4)
    assert recipe.calibration_inputs[0].inputs["decoder_attention_mask"].shape == (1, 4)
    np.testing.assert_array_equal(
        recipe.calibration_inputs[0].inputs["input_ids"][0, :3],
        np.asarray([7, 8, 2], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        recipe.calibration_inputs[0].inputs["input_ids"][0, 3:],
        np.asarray([1, 1, 1, 1, 1], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        recipe.calibration_inputs[0].inputs["attention_mask"][0, 3:],
        np.asarray([0, 0, 0, 0, 0], dtype=np.int64),
    )


def test_build_vpcd_aimet_quantize_recipe_fingerprint_is_stable(monkeypatch, tmp_path):
    from quantize.projects.vpcd import build_vpcd_aimet_quantize_recipe

    records = [
        CalibrationSample(
            inputs={
                "input_ids": np.asarray([[7, 8, 2]], dtype=np.int64),
                "attention_mask": np.asarray([[1, 1, 1]], dtype=np.int64),
                "decoder_input_ids": np.asarray([[2, 9]], dtype=np.int64),
                "decoder_attention_mask": np.asarray([[1, 1]], dtype=np.int64),
            }
        ),
        CalibrationSample(
            inputs={
                "input_ids": np.asarray([[7, 8, 1]], dtype=np.int64),
                "attention_mask": np.asarray([[1, 1, 0]], dtype=np.int64),
                "decoder_input_ids": np.asarray([[2, 5]], dtype=np.int64),
                "decoder_attention_mask": np.asarray([[1, 1]], dtype=np.int64),
            }
        ),
    ]

    def fake_build_calibration_records(**_kwargs):
        return (
            records,
            {
                "requested_provider": "cpu",
                "session_providers": "CPUExecutionProvider",
                "source_files": 1,
                "text_samples": 2,
                "records": 2,
                "max_encoder_len": 3,
                "max_decoder_len": 2,
            },
        )

    monkeypatch.setattr("quantize.projects.vpcd.build_calibration_records", fake_build_calibration_records)

    common_kwargs = dict(
        model_dir=tmp_path / "assets" / "vpcd",
        fp32_onnx_path=tmp_path / "assets" / "vpcd" / "onnx" / "model.fp32.onnx",
        calibration_source_path=tmp_path / "build" / "calibration" / "vpcd_transcriptions.txt",
        max_calibration_samples=16,
        max_generation_length=32,
        ort_provider="cpu",
        fixed_input_shapes={
            "input_ids": (1, 8),
            "attention_mask": (1, 8),
            "decoder_input_ids": (1, 4),
            "decoder_attention_mask": (1, 4),
        },
        pad_token_id=1,
    )

    _write_minimal_vpcd_fp32_model(tmp_path / "assets" / "vpcd" / "onnx" / "model.fp32.onnx")

    recipe_a = build_vpcd_aimet_quantize_recipe(**common_kwargs)
    recipe_b = build_vpcd_aimet_quantize_recipe(**common_kwargs)

    assert recipe_a.calibration_stats["dataset_fingerprint"] == recipe_b.calibration_stats["dataset_fingerprint"]


def test_write_and_load_aimet_calibration_batches_round_trip(tmp_path):
    from quantize.aimet import load_calibration_batches, write_calibration_batches

    calibration_inputs = (
        CalibrationSample(
            inputs={
                "input_ids": np.asarray([[1, 2, 3]], dtype=np.int64),
                "attention_mask": np.asarray([[1, 1, 1]], dtype=np.int64),
            }
        ),
        CalibrationSample(
            inputs={
                "input_ids": np.asarray([[4, 5, 1]], dtype=np.int64),
                "attention_mask": np.asarray([[1, 1, 0]], dtype=np.int64),
            }
        ),
    )

    manifest_path = write_calibration_batches(calibration_inputs, tmp_path / "calibration")
    loaded = load_calibration_batches(manifest_path.parent)

    assert manifest_path.exists()
    assert len(loaded) == 2
    assert list(loaded[0].keys()) == ["input_ids", "attention_mask"]
    np.testing.assert_array_equal(loaded[1]["input_ids"], np.asarray([[4, 5, 1]], dtype=np.int64))


def test_inspect_aimet_package_reports_expected_structure(tmp_path):
    from quantize.aimet import inspect_aimet_package

    package_dir = tmp_path / "model.option1.aimet"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "model.option1.onnx").write_bytes(b"onnx")
    (package_dir / "model.option1.encodings").write_text("{}", encoding="utf-8")
    (package_dir / "model.option1.onnx.data").write_bytes(b"data")
    qdq_path = tmp_path / "model.option1.qdq.onnx"
    qdq_path.write_bytes(b"qdq")

    report = inspect_aimet_package(package_dir, qdq_reference_model_path=qdq_path)

    assert report["package_ready"] is True
    assert report["onnx_files"] == ["model.option1.onnx"]
    assert report["encodings_files"] == ["model.option1.encodings"]
    assert report["data_files"] == ["model.option1.onnx.data"]
    assert report["qdq_reference_model_path"] == qdq_path.resolve().as_posix()


def test_build_matmul_only_aimet_config_disables_defaults_but_enables_matmul_weights():
    from quantize.aimet import build_matmul_only_aimet_config

    config = build_matmul_only_aimet_config()

    assert config["defaults"]["ops"] == {}
    assert config["defaults"]["params"] == {}
    assert config["params"]["bias"]["is_quantized"] == "False"
    assert config["op_type"]["MatMul"]["is_input_quantized"] == "True"
    assert config["op_type"]["MatMul"]["is_output_quantized"] == "True"
    assert config["op_type"]["MatMul"]["params"]["weight"]["is_quantized"] == "True"
    assert config["model_input"]["is_input_quantized"] == "True"
    assert config["model_output"]["is_output_quantized"] == "True"


def test_build_attention_ffn_aimet_config_enables_broader_decoder_friendly_op_types():
    from quantize.aimet import build_attention_ffn_aimet_config

    config = build_attention_ffn_aimet_config()

    assert config["op_type"]["MatMul"]["params"]["weight"]["is_quantized"] == "True"
    assert config["op_type"]["Add"]["is_input_quantized"] == "True"
    assert config["op_type"]["Mul"]["is_output_quantized"] == "True"
    assert config["op_type"]["Div"]["is_output_quantized"] == "True"
    assert config["op_type"]["LayerNormalization"]["is_output_quantized"] == "True"


def test_summarize_vpcd_local_quality_policy_reports_decoder_heavy_exclusions(tmp_path):
    from quantize.projects.vpcd import summarize_vpcd_local_quality_policy

    model_path = tmp_path / "model.fp32.onnx"
    _write_minimal_vpcd_fp32_model(model_path)

    summary = summarize_vpcd_local_quality_policy(model_path)

    assert summary.preset == "local_quality_parity"
    assert summary.total_named_nodes > 0
    assert summary.excluded_node_count == 2
    assert summary.excluded_decoder_node_count == 1
    assert summary.excluded_lm_head_node_count == 1
    assert summary.op_types_to_quantize == ("MatMul",)
    assert summary.quantizable_matmul_node_count == 0


def test_summarize_vpcd_aimet_policy_decoder_expanded_reenables_decoder_matmul(tmp_path):
    from quantize.projects.vpcd import summarize_vpcd_aimet_policy

    model_path = tmp_path / "model.fp32.onnx"
    _write_decoder_matmul_policy_model(model_path)

    summary = summarize_vpcd_aimet_policy(model_path, policy_mode="decoder_expanded")

    assert summary.preset == "decoder_expanded"
    assert summary.excluded_lm_head_node_count == 1
    assert summary.excluded_decoder_node_count == 0
    assert summary.quantizable_matmul_node_count == 1
    assert "/model/decoder/attn/MatMul" in summary.quantizable_matmul_node_names


def test_summarize_vpcd_aimet_policy_broader_attention_ffn_keeps_decoder_ops_only(tmp_path):
    from quantize.projects.vpcd import summarize_vpcd_aimet_policy

    model_path = tmp_path / "model.fp32.onnx"
    _write_decoder_broader_policy_model(model_path)

    summary = summarize_vpcd_aimet_policy(model_path, policy_mode="broader_attention_ffn")

    assert summary.preset == "broader_attention_ffn"
    assert summary.excluded_lm_head_node_count == 1
    assert summary.excluded_decoder_node_count == 0
    assert summary.excluded_node_count >= 2
    assert set(summary.op_types_to_quantize) == {"MatMul", "Add", "Mul", "Div", "LayerNormalization"}
    assert summary.quantizable_matmul_node_count == 1
    assert summary.quantizable_node_count_by_op_type["Add"] == 1
    assert summary.quantizable_node_count_by_op_type["LayerNormalization"] == 1
    assert "/model/decoder/ffn/Add" in summary.quantizable_node_names
    assert "/model/encoder/ffn/Add" not in summary.quantizable_node_names


def test_should_write_vpcd_aimet_policy_manifest_accepts_decoder_expanded():
    from quantize.projects.vpcd import should_write_vpcd_aimet_policy_manifest

    assert should_write_vpcd_aimet_policy_manifest("local_quality_parity") is True
    assert should_write_vpcd_aimet_policy_manifest("decoder_expanded") is True
    assert should_write_vpcd_aimet_policy_manifest("broader_attention_ffn") is True
    assert should_write_vpcd_aimet_policy_manifest("aggressive_int8") is True
    assert should_write_vpcd_aimet_policy_manifest("none") is False


def test_build_vpcd_aimet_quantize_recipe_reuses_fixed_shape_calibration(monkeypatch, tmp_path):
    from quantize.projects.vpcd import build_vpcd_aimet_quantize_recipe

    records = [
        CalibrationSample(
            inputs={
                "input_ids": np.asarray([[7, 8, 2]], dtype=np.int64),
                "attention_mask": np.asarray([[1, 1, 1]], dtype=np.int64),
                "decoder_input_ids": np.asarray([[2, 9]], dtype=np.int64),
                "decoder_attention_mask": np.asarray([[1, 1]], dtype=np.int64),
            }
        ),
        CalibrationSample(
            inputs={
                "input_ids": np.asarray([[7, 8, 1]], dtype=np.int64),
                "attention_mask": np.asarray([[1, 1, 0]], dtype=np.int64),
                "decoder_input_ids": np.asarray([[2, 5]], dtype=np.int64),
                "decoder_attention_mask": np.asarray([[1, 1]], dtype=np.int64),
            }
        ),
    ]

    def fake_build_calibration_records(**_kwargs):
        return (
            records,
            {
                "requested_provider": "cpu",
                "session_providers": "CPUExecutionProvider",
                "source_files": 1,
                "text_samples": 2,
                "records": 2,
                "max_encoder_len": 3,
                "max_decoder_len": 2,
            },
        )

    monkeypatch.setattr("quantize.projects.vpcd.build_calibration_records", fake_build_calibration_records)
    _write_minimal_vpcd_fp32_model(tmp_path / "assets" / "vpcd" / "onnx" / "model.fp32.onnx")

    recipe = build_vpcd_aimet_quantize_recipe(
        model_dir=tmp_path / "assets" / "vpcd",
        fp32_onnx_path=tmp_path / "assets" / "vpcd" / "onnx" / "model.fp32.onnx",
        calibration_source_path=tmp_path / "build" / "calibration" / "vpcd_transcriptions.txt",
        max_calibration_samples=16,
        max_generation_length=32,
        ort_provider="cpu",
        fixed_input_shapes={
            "input_ids": (1, 8),
            "attention_mask": (1, 8),
            "decoder_input_ids": (1, 4),
            "decoder_attention_mask": (1, 4),
        },
        pad_token_id=1,
        policy_mode="local_quality_parity",
        activation_type="int16",
        config_file="vpcd_matmul_only",
    )

    assert recipe.param_type == "int8"
    assert recipe.activation_type == "int16"
    assert recipe.quant_scheme == "min_max"
    assert recipe.config_file == "vpcd_matmul_only"
    assert recipe.policy_mode == "local_quality_parity"
    assert recipe.variant_name == "wint8_aint16_min_max_local_quality_parity"
    assert recipe.calibration_stats["dataset_fingerprint"]
    assert recipe.calibration_stats["quantize_backend"] == "aimet"
    assert recipe.calibration_stats["policy_mode"] == "local_quality_parity"
    assert recipe.calibration_stats["local_quality_policy"]["preset"] == "local_quality_parity"
    assert "excluded_node_count" in recipe.local_quality_policy
    assert "quantizable_matmul_node_names" in recipe.local_quality_policy
    assert len(recipe.calibration_inputs) == 2
    assert recipe.calibration_inputs[0].inputs["input_ids"].shape == (1, 8)
    assert recipe.calibration_inputs[0].inputs["decoder_input_ids"].shape == (1, 4)
    np.testing.assert_array_equal(
        recipe.calibration_inputs[1].inputs["decoder_attention_mask"][0],
        np.asarray([1, 1, 0, 0], dtype=np.int64),
    )


def test_build_vpcd_aimet_quantize_recipe_aggressive_int8_uses_distinct_variant(monkeypatch, tmp_path):
    from quantize.projects.vpcd import build_vpcd_aimet_quantize_recipe

    records = [
        CalibrationSample(
            inputs={
                "input_ids": np.asarray([[7, 8, 2]], dtype=np.int64),
                "attention_mask": np.asarray([[1, 1, 1]], dtype=np.int64),
                "decoder_input_ids": np.asarray([[2, 9]], dtype=np.int64),
                "decoder_attention_mask": np.asarray([[1, 1]], dtype=np.int64),
            }
        ),
    ]

    def fake_build_calibration_records(**_kwargs):
        return (
            records,
            {
                "requested_provider": "cpu",
                "session_providers": "CPUExecutionProvider",
                "source_files": 1,
                "text_samples": 1,
                "records": 1,
                "max_encoder_len": 3,
                "max_decoder_len": 2,
            },
        )

    monkeypatch.setattr("quantize.projects.vpcd.build_calibration_records", fake_build_calibration_records)
    _write_decoder_broader_policy_model(tmp_path / "assets" / "vpcd" / "onnx" / "model.fp32.onnx")

    recipe = build_vpcd_aimet_quantize_recipe(
        model_dir=tmp_path / "assets" / "vpcd",
        fp32_onnx_path=tmp_path / "assets" / "vpcd" / "onnx" / "model.fp32.onnx",
        calibration_source_path=tmp_path / "build" / "calibration" / "vpcd_transcriptions.txt",
        max_calibration_samples=16,
        max_generation_length=32,
        ort_provider="cpu",
        fixed_input_shapes={
            "input_ids": (1, 8),
            "attention_mask": (1, 8),
            "decoder_input_ids": (1, 4),
            "decoder_attention_mask": (1, 4),
        },
        pad_token_id=1,
        policy_mode="aggressive_int8",
        activation_type="int8",
        config_file="vpcd_attention_ffn",
    )

    assert recipe.activation_type == "int8"
    assert recipe.policy_mode == "aggressive_int8"
    assert recipe.variant_name == "wint8_aint8_min_max_aggressive_int8"
    assert recipe.calibration_stats["local_quality_policy"]["preset"] == "aggressive_int8"
    assert recipe.local_quality_policy["quantizable_node_count_by_op_type"]["Add"] >= 1


def test_quantize_cli_defaults_vpcd_to_retained_aimet_only():
    from quantize.cli import parse_args

    args = parse_args(["--project", "vpcd"])

    assert args.output_root == str(Path("build") / "quantize" / "vpcd" / "local_aimet")
    assert args.aimet_service_url == "http://127.0.0.1:18080"
    assert not hasattr(args, "pipeline")
    assert not hasattr(args, "preset")


def test_run_retained_aimet_pipeline_resolves_repo_relative_inputs(monkeypatch, tmp_path):
    from quantize.projects import vpcd

    repo_root = tmp_path / "python-model-test"
    captured: dict[str, Path] = {}

    class StopPipeline(RuntimeError):
        pass

    def fake_build_recipe(**kwargs):
        captured["model_dir"] = kwargs["model_dir"]
        captured["fp32_onnx_path"] = kwargs["fp32_onnx_path"]
        captured["calibration_source_path"] = kwargs["calibration_source_path"]
        raise StopPipeline("captured")

    monkeypatch.setattr(vpcd, "_resolve_repo_root", lambda: repo_root)
    monkeypatch.setattr(vpcd, "_resolve_fixed_bundle_manifest_path", lambda _path: repo_root / "build" / "model_bundle" / "bundle_manifest.json")
    monkeypatch.setattr(
        vpcd,
        "_resolve_vpcd_fixed_input_shapes_from_bundle",
        lambda _path: (
            {
                "input_ids": (1, 1024),
                "attention_mask": (1, 1024),
                "decoder_input_ids": (1, 128),
                "decoder_attention_mask": (1, 128),
            },
            1,
        ),
    )
    monkeypatch.setattr(vpcd, "_resolve_output_root", lambda _args, repo_root=None: (repo_root or tmp_path) / "build" / "quantize" / "vpcd")
    monkeypatch.setattr(vpcd, "build_vpcd_aimet_quantize_recipe", fake_build_recipe)

    args = SimpleNamespace(
        model_dir=str(Path("assets") / "vietnamese-punc-cap-denorm-v1"),
        fp32_onnx=str(Path("assets") / "vietnamese-punc-cap-denorm-v1" / "onnx" / "model.fp32.onnx"),
        calibration_text=str(Path("build") / "calibration" / "vlsp2020" / "vpcd_transcriptions.txt"),
        fixed_bundle_manifest=str(Path("build") / "model_bundle" / "vpcd" / "qnn_fixed_1024x128" / "bundle_manifest.json"),
        output_root=str(Path("build") / "quantize" / "vpcd" / "local_aimet"),
        max_calibration_samples=24,
        max_generation_length=64,
        ort_provider="cpu",
        aimet_param_type="int8",
        aimet_activation_type="int16",
        aimet_quant_scheme="min_max",
        aimet_config_file="vpcd_matmul_only",
        aimet_policy_mode="decoder_expanded",
        aimet_service_url="http://127.0.0.1:18080",
        aimet_service_workspace_root="/workspace",
        aimet_health_timeout_seconds=10.0,
        dry_run=False,
    )

    with pytest.raises(StopPipeline, match="captured"):
        vpcd._run_retained_aimet_pipeline(args)

    assert captured["model_dir"] == repo_root / "assets" / "vietnamese-punc-cap-denorm-v1"
    assert captured["fp32_onnx_path"] == repo_root / "assets" / "vietnamese-punc-cap-denorm-v1" / "onnx" / "model.fp32.onnx"
    assert captured["calibration_source_path"] == repo_root / "build" / "calibration" / "vlsp2020" / "vpcd_transcriptions.txt"


def test_aimet_service_http_contract_supports_health_and_export(tmp_path):
    from http.server import ThreadingHTTPServer

    from quantize.aimet_service import build_handler_class

    seen: dict[str, object] = {}

    def fake_export_callback(payload: dict[str, object]) -> dict[str, object]:
        seen["payload"] = payload
        return {
            "package_ready": True,
            "package_dir": str(tmp_path / "model.option1.aimet"),
            "qdq_reference_model_path": str(tmp_path / "model.option1.qdq.onnx"),
        }

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        build_handler_class(export_callback=fake_export_callback, version_payload={"service": "aimet-test"}),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}"

        with urllib_request.urlopen(f"{base_url}/healthz", timeout=5) as response:
            health_payload = json.loads(response.read().decode("utf-8"))
        assert health_payload["status"] == "ok"

        export_request = {
            "fp32_onnx_path": "/workspace/assets/model.fp32.fixed.onnx",
            "calibration_dir": "/workspace/build/quantize/vpcd/local_aimet/calibration",
            "package_dir": "/workspace/build/quantize/vpcd/local_aimet/model.option1.aimet",
            "qdq_reference_model_path": "/workspace/build/quantize/vpcd/local_aimet/model.option1.qdq.onnx",
            "config_file": "/workspace/build/quantize/vpcd/local_aimet/aimet.config.json",
            "policy_manifest_path": "/workspace/build/quantize/vpcd/local_aimet/aimet.policy.json",
            "param_type": "int8",
            "activation_type": "int16",
            "quant_scheme": "min_max",
            "model_prefix": "model.option1",
        }
        request = urllib_request.Request(
            f"{base_url}/export",
            data=json.dumps(export_request).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib_request.urlopen(request, timeout=5) as response:
            export_payload = json.loads(response.read().decode("utf-8"))

        assert export_payload["package_ready"] is True
        assert seen["payload"] == export_request
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
