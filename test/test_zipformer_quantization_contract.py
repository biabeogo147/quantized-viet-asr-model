from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType

from model_pipeline.models.zipformer.quantization import (
    TranscriptQualitySummary,
    assess_zipformer_quality,
    inspect_zipformer_qdq_coverage,
    quantize_zipformer_encoder_ortqnn,
    select_zipformer_quantization_engine,
)
from model_pipeline.models import get_recipe
from model_pipeline.models.zipformer.adapter import ZipformerAdapter
from model_pipeline.models.zipformer.graph import ORT_DISABLED_OPTIMIZERS


def test_zipformer_prepare_preserves_matmul_operator_scope() -> None:
    """Verify ORT optimization cannot fuse targeted MatMul nodes into Gemm.

    Returns:
        None.
    """
    assert ORT_DISABLED_OPTIMIZERS == ("MatMulAddFusion",)


def test_ortqnn_static_quantization_uses_unsigned_8bit_16bit_matmul_only(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify ORT-QNN quantizer options match the Zipformer contract.

    Args:
        tmp_path: Isolated source and output paths.
        monkeypatch: Pytest fixture replacing ONNX Runtime quantization.

    Returns:
        None.
    """
    source = tmp_path / "encoder.onnx"
    source.write_bytes(b"source")
    destination = tmp_path / "encoder.quantized.onnx"
    calls: list[dict[str, object]] = []

    def fake_quantize_static(**kwargs) -> None:
        """Capture quantizer options and materialize a fake output.

        Args:
            kwargs: ONNX Runtime static quantization keyword arguments.

        Returns:
            None.
        """
        calls.append(kwargs)
        Path(kwargs["model_output"]).write_bytes(b"quantized")

    monkeypatch.setattr(
        "model_pipeline.models.zipformer.quantization.quantize_static",
        fake_quantize_static,
    )
    calibration_inputs = [
        {
            "x": np.zeros((1, 2009, 80), dtype=np.float32),
            "x_lens": np.asarray([100], dtype=np.int64),
        }
    ]

    result = quantize_zipformer_encoder_ortqnn(
        source,
        destination,
        calibration_inputs,
    )

    assert result == destination.resolve()
    assert calls[0]["quant_format"] == QuantFormat.QDQ
    assert calls[0]["calibrate_method"] == CalibrationMethod.MinMax
    assert calls[0]["weight_type"] == QuantType.QUInt8
    assert calls[0]["activation_type"] == QuantType.QUInt16
    assert calls[0]["per_channel"] is False
    assert calls[0]["op_types_to_quantize"] == ["MatMul"]
    assert calls[0]["calibration_data_reader"].get_next()["x"].shape == (1, 2009, 80)
    assert calls[0]["calibration_data_reader"].get_next() is None


def test_zipformer_qdq_inventory_counts_only_quantized_matmul(tmp_path: Path) -> None:
    """Verify Q/DQ inventory distinguishes quantized and untouched MatMul nodes.

    Args:
        tmp_path: Isolated ONNX graph output directory.

    Returns:
        None.
    """
    scale = helper.make_tensor("scale", TensorProto.FLOAT, [], [0.1])
    zero = helper.make_tensor("zero", TensorProto.UINT8, [], [0])
    weight = helper.make_tensor("weight", TensorProto.FLOAT, [2, 2], [1.0, 0.0, 0.0, 1.0])
    nodes = [
        helper.make_node("QuantizeLinear", ["x", "scale", "zero"], ["x_q"], name="input_quantize"),
        helper.make_node("DequantizeLinear", ["x_q", "scale", "zero"], ["x_dq"], name="input_dequantize"),
        helper.make_node("MatMul", ["x_dq", "weight"], ["quantized_out"], name="quantized_matmul"),
        helper.make_node("MatMul", ["x", "weight"], ["fp32_out"], name="fp32_matmul"),
    ]
    graph = helper.make_graph(
        nodes,
        "qdq-inventory",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
        [
            helper.make_tensor_value_info("quantized_out", TensorProto.FLOAT, [1, 2]),
            helper.make_tensor_value_info("fp32_out", TensorProto.FLOAT, [1, 2]),
        ],
        [scale, zero, weight],
    )
    path = tmp_path / "model.onnx"
    onnx.save(helper.make_model(graph), path)

    inventory = inspect_zipformer_qdq_coverage(path)

    assert inventory.matmul_count == 2
    assert inventory.quantized_matmul_count == 1
    assert inventory.unquantized_matmul_names == ("fp32_matmul",)


def test_zipformer_quality_gate_and_quantizer_fallback_are_explicit() -> None:
    """Verify quality and compile failures select AIMET without changing thresholds.

    Returns:
        None.
    """
    fp32 = TranscriptQualitySummary(
        character_error_rate=0.10,
        word_error_rate=0.20,
        empty_output_count=0,
        repetition_collapse_count=0,
    )
    accepted = TranscriptQualitySummary(
        character_error_rate=0.11,
        word_error_rate=0.22,
        empty_output_count=0,
        repetition_collapse_count=0,
    )
    rejected = TranscriptQualitySummary(
        character_error_rate=0.111,
        word_error_rate=0.22,
        empty_output_count=0,
        repetition_collapse_count=0,
    )

    assert assess_zipformer_quality(fp32, accepted).passed is True
    assert assess_zipformer_quality(fp32, rejected).passed is False
    assert select_zipformer_quantization_engine(fp32, accepted, compile_accepted=True) == "ortqnn"
    assert select_zipformer_quantization_engine(fp32, accepted, compile_accepted=False) == "aimet"
    assert select_zipformer_quantization_engine(fp32, rejected, compile_accepted=True) == "aimet"


def test_zipformer_adapter_quantizes_only_encoder_and_preserves_cpu_components(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify adapter replaces only encoder bytes for ORT-QNN configuration.

    Args:
        tmp_path: Isolated prepared and quantized component directories.
        monkeypatch: Pytest fixture replacing the expensive quantizer call.

    Returns:
        None.
    """
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    prepared = {
        "encoder": prepared_dir / "encoder.onnx",
        "decoder": prepared_dir / "decoder.onnx",
        "joiner": prepared_dir / "joiner.onnx",
        "tokens": prepared_dir / "tokens.txt",
    }
    for role, path in prepared.items():
        path.write_bytes(role.encode())
    calibration_inputs = [
        {
            "x": np.zeros((1, 2009, 80), dtype=np.float32),
            "x_lens": np.asarray([100], dtype=np.int64),
        }
    ]

    def fake_quantize(_source, output, batches):
        """Materialize a fake quantized encoder and verify calibration forwarding.

        Args:
            _source: Unused prepared encoder path.
            output: Quantized encoder destination.
            batches: Forwarded fixed-shape calibration batches.

        Returns:
            Quantized encoder path.
        """
        assert batches is calibration_inputs
        destination = Path(output)
        destination.write_bytes(b"quantized-encoder")
        return destination

    monkeypatch.setattr(
        "model_pipeline.models.zipformer.adapter.quantize_zipformer_encoder_ortqnn",
        fake_quantize,
    )
    adapter = ZipformerAdapter(tmp_path, calibration_inputs=calibration_inputs)

    outputs = adapter.quantize(
        get_recipe("zipformer", "ortqnn-uint8-uint16-encoder-matmul"),
        prepared,
        tmp_path / "quantized",
    )

    assert outputs["encoder"].read_bytes() == b"quantized-encoder"
    assert outputs["decoder"].read_bytes() == b"decoder"
    assert outputs["joiner"].read_bytes() == b"joiner"
    assert outputs["tokens"].read_bytes() == b"tokens"


def test_zipformer_adapter_executes_aimet_fallback_with_same_calibration(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify AIMET fallback exports encoder-only package with shared calibration.

    Args:
        tmp_path: Isolated prepared and AIMET package directories.
        monkeypatch: Pytest fixture replacing canonical graph inventory.

    Returns:
        None.
    """
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    prepared = {
        "encoder": prepared_dir / "encoder.onnx",
        "decoder": prepared_dir / "decoder.onnx",
        "joiner": prepared_dir / "joiner.onnx",
        "tokens": prepared_dir / "tokens.txt",
    }
    for role, path in prepared.items():
        path.write_bytes(role.encode())
    calibration_inputs = [
        {
            "x": np.zeros((1, 2009, 80), dtype=np.float32),
            "x_lens": np.asarray([100], dtype=np.int64),
        }
    ]

    class FakeAimetService:
        def __init__(self):
            """Initialize service call evidence.

            Returns:
                None.
            """
            self.export_calls = 0

        def healthcheck(self) -> None:
            """Accept the fake service healthcheck.

            Returns:
                None.
            """
            return None

        def export(self, **kwargs):
            """Materialize a fake AIMET encoder package.

            Args:
                kwargs: Generic AIMET export paths.

            Returns:
                Fake service output metadata.
            """
            self.export_calls += 1
            output_dir = Path(kwargs["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "model.onnx").write_bytes(b"aimet-encoder")
            (output_dir / "model.encodings").write_text("{}", encoding="utf-8")
            return {"outputs": {"model": "model.onnx", "encodings": "model.encodings"}}

    service = FakeAimetService()
    monkeypatch.setattr(
        "model_pipeline.models.zipformer.adapter.build_zipformer_encoder_matmul_policy",
        lambda _path: {
            "quantization_scope": "encoder-matmul",
            "quantize_op_types": ["MatMul"],
            "quantize_op_names": ["matmul"] * 278,
            "disable_op_names": [],
            "coverage": {"quantized": 278, "total_matmul": 278},
        },
        raising=False,
    )
    adapter = ZipformerAdapter(
        tmp_path,
        calibration_inputs=calibration_inputs,
        aimet_service=service,
    )

    outputs = adapter.quantize(
        get_recipe("zipformer", "aimet-int8-int16-encoder-matmul"),
        prepared,
        tmp_path / "aimet-quantized",
    )

    assert service.export_calls == 1
    assert outputs["encoder"].read_bytes() == b"aimet-encoder"
    assert outputs["encodings"].name == "model.encodings"
    assert outputs["decoder"].read_bytes() == b"decoder"
    assert outputs["joiner"].read_bytes() == b"joiner"


def test_zipformer_compile_input_matches_quantization_package_format(tmp_path: Path) -> None:
    """Verify ORT-QNN compiles ONNX while AIMET compiles its package directory.

    Args:
        tmp_path: Isolated quantization package directory.

    Returns:
        None.
    """
    aimet_dir = tmp_path / "aimet"
    aimet_dir.mkdir()
    encoder = aimet_dir / "model.onnx"
    encoder.write_bytes(b"encoder")
    validated_components = {"encoder": encoder}
    adapter = ZipformerAdapter(tmp_path)

    ortqnn_input = adapter.compile_inputs(
        get_recipe("zipformer", "ortqnn-uint8-uint16-encoder-matmul"),
        validated_components,
    )[0]
    aimet_input = adapter.compile_inputs(
        get_recipe("zipformer", "aimet-int8-int16-encoder-matmul"),
        validated_components,
    )[0]

    assert ortqnn_input.source_path == encoder
    assert aimet_input.source_path == aimet_dir
    assert ortqnn_input.input_dtypes == {"x": "float32", "x_lens": "int64"}
    assert ortqnn_input.truncate_64bit_io is True
    assert aimet_input.truncate_64bit_io is True
