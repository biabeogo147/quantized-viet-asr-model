from __future__ import annotations

from pathlib import Path

from model_pipeline.models import get_recipe
from model_pipeline.models.vpcd.adapter import VpcdAdapter
from model_pipeline.models.zipformer.adapter import ZipformerAdapter


def test_adapters_resolve_sources_from_sibling_android_repo_on_clean_python_clone(tmp_path: Path) -> None:
    """Verify clean Python clones can resolve tracked sibling Android model sources.

    Args:
        tmp_path: Isolated workspace used to simulate sibling repository layout.

    Returns:
        None.
    """
    python_repo = tmp_path / "python-model-test"
    python_repo.mkdir()
    bk_assets = tmp_path / "BKMeeting" / "modelassets" / "src" / "main" / "assets" / "models"
    zip_dir = bk_assets / "asr" / "zipformer" / "fp32"
    vpcd_dir = bk_assets / "punctuation" / "vpcd" / "fp32"
    zip_dir.mkdir(parents=True)
    vpcd_dir.mkdir(parents=True)
    for name in ("encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"):
        (zip_dir / name).write_bytes(name.encode())
    for name in (
        "model.mobile.onnx",
        "tokenizer.encode.onnx",
        "tokenizer.decode.onnx",
        "tokenizer.to_model_id_map.json",
        "tokenizer.from_model_id_map.json",
    ):
        (vpcd_dir / name).write_bytes(name.encode())
    calibration = python_repo / "assets" / "punctuation" / "default_golden_samples.jsonl"
    calibration.parent.mkdir(parents=True)
    calibration.write_text('{"raw_text":"xin chào"}\n', encoding="utf-8")

    zip_sources = ZipformerAdapter(python_repo).source_files(
        get_recipe("zipformer", "fp32-fixed-shape")
    )
    vpcd_sources = VpcdAdapter(python_repo).source_files(
        get_recipe("vpcd", "aimet-int8-int16-encoder-matmul")
    )

    assert zip_sources["encoder"] == zip_dir / "encoder.onnx"
    assert vpcd_sources["model"] == vpcd_dir / "model.mobile.onnx"
    assert vpcd_sources["tokenizer_encode"] == vpcd_dir / "tokenizer.encode.onnx"
    assert vpcd_sources["calibration_text"] == calibration

    external = tmp_path / "model.bin"
    external.write_bytes(b"external")
    validated_components = {
        "encoder": zip_dir / "encoder.onnx",
        "decoder": zip_dir / "decoder.onnx",
        "joiner": zip_dir / "joiner.onnx",
        "tokens": zip_dir / "tokens.txt",
    }
    bundle = ZipformerAdapter(python_repo).bundle_components(
        get_recipe("zipformer", "fp32-fixed-shape-aihub-encoder"),
        validated_components,
        {"encoder": zip_dir / "encoder.onnx", "encoder_external_data": external},
    )
    assert bundle["encoder_external_data"] == (external, "qnn-htp", "onnx-external-data")
