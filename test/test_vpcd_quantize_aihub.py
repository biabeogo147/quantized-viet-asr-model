from pathlib import Path

import numpy as np
import onnx

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


def _write_vpcd_compile_candidate_model(
    model_path: Path,
    *,
    use_ms_domain: bool = True,
    main_opset: int = 17,
    quantized_weight_dtype: int = 0,
) -> None:
    from onnx import TensorProto, helper, numpy_helper

    model_path.parent.mkdir(parents=True, exist_ok=True)
    scale = numpy_helper.from_array(np.asarray([0.125], dtype=np.float32), name="weight_scale")
    zero_point_dtype = np.uint16 if quantized_weight_dtype == TensorProto.UINT16 else np.uint8
    zero_point_type = TensorProto.UINT16 if quantized_weight_dtype == TensorProto.UINT16 else TensorProto.UINT8
    quantized_weight = numpy_helper.from_array(
        np.asarray([[1, 2, 3, 4]], dtype=zero_point_dtype),
        name="quantized_weight",
    )
    zero_point = numpy_helper.from_array(np.asarray([0], dtype=zero_point_dtype), name="weight_zero_point")
    float_bias = numpy_helper.from_array(np.asarray([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32), name="float_bias")
    qdq_domain = "com.microsoft" if use_ms_domain else ""
    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                "DequantizeLinear",
                ["quantized_weight", "weight_scale", "weight_zero_point"],
                ["weight_float"],
                name="weight_dequantize",
                domain=qdq_domain,
            ),
            helper.make_node(
                "Add",
                ["weight_float", "float_bias"],
                ["logits"],
                name="add_bias",
            ),
        ],
        name="vpcd-compile-candidate-test",
        inputs=[helper.make_tensor_value_info("decoder_input_ids", TensorProto.FLOAT, [1, 4])],
        outputs=[helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 4])],
        initializer=[scale, zero_point, quantized_weight, float_bias],
    )
    opsets = [helper.make_opsetid("", main_opset)]
    if use_ms_domain:
        opsets.append(helper.make_opsetid("com.microsoft", 1))
    model = helper.make_model(graph, opset_imports=opsets)
    onnx.save(model, model_path.as_posix())


def test_resolve_vpcd_aihub_quantize_dtype_names_follows_preset_policy():
    from quantize.projects.vpcd import resolve_vpcd_aihub_quantize_dtype_names

    assert resolve_vpcd_aihub_quantize_dtype_names(preset="sd8g2_quality") == {
        "weights_dtype_name": "INT8",
        "activations_dtype_name": "INT16",
    }
    assert resolve_vpcd_aihub_quantize_dtype_names(preset="baseline_dynamic_int8") == {
        "weights_dtype_name": "INT8",
        "activations_dtype_name": "INT8",
    }


def test_calibration_records_to_aihub_dataset_preserves_input_order():
    from quantize.projects.vpcd import calibration_records_to_aihub_dataset

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

    dataset = calibration_records_to_aihub_dataset(records)

    assert list(dataset.keys()) == ["input_ids", "attention_mask"]
    assert len(dataset["input_ids"]) == 2
    np.testing.assert_array_equal(dataset["input_ids"][1], np.asarray([[4, 5, 6]], dtype=np.int64))
    np.testing.assert_array_equal(dataset["attention_mask"][1], np.asarray([[1, 1, 0]], dtype=np.int64))


def test_build_vpcd_aihub_quantize_recipe_uses_autoregressive_records(monkeypatch, tmp_path):
    from quantize.projects.vpcd import build_vpcd_aihub_quantize_recipe

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

    recipe = build_vpcd_aihub_quantize_recipe(
        model_dir=tmp_path / "assets" / "vpcd",
        fp32_onnx_path=tmp_path / "assets" / "vpcd" / "onnx" / "model.fp32.onnx",
        calibration_source_path=tmp_path / "build" / "calibration" / "vpcd_transcriptions.txt",
        preset="sd8g2_quality",
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
    assert recipe.preset == "sd8g2_quality"
    assert recipe.activations_dtype_name == "INT16"
    assert recipe.weights_dtype_name == "INT8"
    assert recipe.calibration_stats["quantize_preset"] == "sd8g2_quality"
    assert recipe.calibration_stats["activation_type"] == "quint16"
    assert recipe.calibration_stats["weight_type"] == "quint8"
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
    assert list(recipe.calibration_dataset.keys()) == [
        "input_ids",
        "attention_mask",
        "decoder_input_ids",
        "decoder_attention_mask",
    ]
    assert recipe.calibration_dataset["input_ids"][0].shape == (1, 8)
    assert recipe.calibration_dataset["attention_mask"][0].shape == (1, 8)
    assert recipe.calibration_dataset["decoder_input_ids"][0].shape == (1, 4)
    assert recipe.calibration_dataset["decoder_attention_mask"][0].shape == (1, 4)
    np.testing.assert_array_equal(
        recipe.calibration_dataset["input_ids"][0][0, :3],
        np.asarray([7, 8, 2], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        recipe.calibration_dataset["input_ids"][0][0, 3:],
        np.asarray([1, 1, 1, 1, 1], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        recipe.calibration_dataset["attention_mask"][0][0, 3:],
        np.asarray([0, 0, 0, 0, 0], dtype=np.int64),
    )


def test_build_vpcd_aihub_quantize_recipe_fingerprint_is_stable(monkeypatch, tmp_path):
    from quantize.projects.vpcd import build_vpcd_aihub_quantize_recipe

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
        preset="sd8g2_quality",
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

    recipe_a = build_vpcd_aihub_quantize_recipe(**common_kwargs)
    recipe_b = build_vpcd_aihub_quantize_recipe(**common_kwargs)

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


def test_summarize_vpcd_local_quality_policy_reports_decoder_heavy_exclusions(tmp_path):
    from quantize.projects.vpcd import summarize_vpcd_local_quality_policy

    model_path = tmp_path / "model.fp32.onnx"
    _write_minimal_vpcd_fp32_model(model_path)

    summary = summarize_vpcd_local_quality_policy(model_path)

    assert summary.preset == "sd8g2_quality"
    assert summary.total_named_nodes > 0
    assert summary.excluded_node_count == 2
    assert summary.excluded_decoder_node_count == 1
    assert summary.excluded_lm_head_node_count == 1
    assert summary.op_types_to_quantize == ("MatMul",)
    assert summary.quantizable_matmul_node_count == 0


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
    assert recipe.calibration_stats["local_quality_policy"]["preset"] == "sd8g2_quality"
    assert "excluded_node_count" in recipe.local_quality_policy
    assert "quantizable_matmul_node_names" in recipe.local_quality_policy
    assert len(recipe.calibration_inputs) == 2
    assert recipe.calibration_inputs[0].inputs["input_ids"].shape == (1, 8)
    assert recipe.calibration_inputs[0].inputs["decoder_input_ids"].shape == (1, 4)
    np.testing.assert_array_equal(
        recipe.calibration_inputs[1].inputs["decoder_attention_mask"][0],
        np.asarray([1, 1, 0, 0], dtype=np.int64),
    )


def test_inspect_vpcd_qdq_compile_candidate_reports_conservative_aihub_readiness(tmp_path):
    from quantize.projects.vpcd import inspect_vpcd_qdq_compile_candidate

    model_path = tmp_path / "model.mobile.onnx"
    _write_vpcd_compile_candidate_model(
        model_path,
        use_ms_domain=True,
        main_opset=17,
        quantized_weight_dtype=onnx.TensorProto.UINT16,
    )

    report = inspect_vpcd_qdq_compile_candidate(model_path)

    assert report["opsets"]["main"] == 17
    assert report["ms_qdq_node_count"] > 0
    assert report["uses_uint16_qdq"] is True
    assert report["uses_quantized_weight_initializers"] is True
    assert report["aihub_compile_readiness"] == "unsafe"
    assert "com.microsoft_qdq" in report["readiness_flags"]
