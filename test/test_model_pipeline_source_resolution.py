from __future__ import annotations

from pathlib import Path

from model_pipeline.models import get_recipe
from model_pipeline.models.vpcd.adapter import VpcdAdapter
from model_pipeline.models.zipformer.adapter import ZipformerAdapter


def test_adapters_resolve_sources_from_sibling_canonical_model_repository(tmp_path: Path) -> None:
    """Verify clean model-pipeline clones resolve sibling canonical artifacts.

    Args:
        tmp_path: Isolated workspace used to simulate sibling repository layout.

    Returns:
        None.
    """
    model_repo = tmp_path / "quantized-viet-asr-model"
    model_repo.mkdir()
    bk_repository = (
        tmp_path
        / "BKMeeting"
        / "modelassets"
        / "src"
        / "main"
        / "assets"
        / "model-repository"
        / "artifacts"
    )
    zip_primary = bk_repository / "zipformer" / "fp32-fixed-shape" / "cpu"
    zip_support = bk_repository / "zipformer" / "shared-fp32-cpu"
    vpcd_primary = bk_repository / "vpcd" / "fp32-fixed-shape" / "cpu"
    vpcd_support = bk_repository / "vpcd" / "shared-fp32-cpu"
    for directory in (zip_primary, zip_support, vpcd_primary, vpcd_support):
        directory.mkdir(parents=True)
    for name in ("encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"):
        destination = zip_primary if name == "encoder.onnx" else zip_support
        (destination / name).write_bytes(name.encode())
    (vpcd_primary / "model.onnx").write_bytes(b"model")
    for name in (
        "tokenizer.encode.onnx",
        "tokenizer.decode.onnx",
        "tokenizer.to_model_id_map.json",
        "tokenizer.from_model_id_map.json",
    ):
        (vpcd_support / name).write_bytes(name.encode())
    calibration = model_repo / "assets" / "punctuation" / "default_golden_samples.jsonl"
    calibration.parent.mkdir(parents=True)
    calibration.write_text('{"raw_text":"xin chào"}\n', encoding="utf-8")

    zip_sources = ZipformerAdapter(model_repo).source_files(
        get_recipe("zipformer", "fp32-fixed-shape")
    )
    vpcd_sources = VpcdAdapter(model_repo).source_files(
        get_recipe("vpcd", "aimet-int8-int16-encoder-matmul")
    )

    assert zip_sources["encoder"] == zip_primary / "encoder.onnx"
    assert vpcd_sources["model"] == vpcd_primary / "model.onnx"
    assert vpcd_sources["tokenizer_encode"] == vpcd_support / "tokenizer.encode.onnx"
    assert vpcd_sources["calibration_text"] == calibration

    external = tmp_path / "model.bin"
    external.write_bytes(b"external")
    validated_components = {
        "encoder": zip_primary / "encoder.onnx",
        "decoder": zip_support / "decoder.onnx",
        "joiner": zip_support / "joiner.onnx",
        "tokens": zip_support / "tokens.txt",
    }
    bundle = ZipformerAdapter(model_repo).bundle_components(
        get_recipe("zipformer", "fp32-fixed-shape-aihub-encoder"),
        validated_components,
        {
            "encoder": zip_primary / "encoder.onnx",
            "encoder_external_data": external,
        },
    )
    assert bundle["encoder_external_data"] == (external, "qnn-htp", "onnx-external-data")
