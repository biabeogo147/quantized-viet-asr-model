from pathlib import Path

import numpy as np

from quantize.types import CalibrationSample


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
    assert list(recipe.calibration_dataset.keys()) == [
        "input_ids",
        "attention_mask",
        "decoder_input_ids",
        "decoder_attention_mask",
    ]
