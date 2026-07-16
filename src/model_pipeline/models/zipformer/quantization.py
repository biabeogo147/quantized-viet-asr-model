from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

@dataclass(frozen=True)
class ZipformerQdqInventory:
    matmul_count: int
    quantized_matmul_count: int
    unquantized_matmul_names: tuple[str, ...]


@dataclass(frozen=True)
class TranscriptQualitySummary:
    character_error_rate: float
    word_error_rate: float
    empty_output_count: int
    repetition_collapse_count: int


@dataclass(frozen=True)
class ZipformerQualityGate:
    character_error_rate_increase: float
    word_error_rate_increase: float
    empty_output_count: int
    repetition_collapse_count: int
    passed: bool


class _SequenceCalibrationReader(CalibrationDataReader):
    def __init__(self, batches: Sequence[Mapping[str, np.ndarray]]):
        """Initialize an ordered ONNX Runtime calibration reader.

        Args:
            batches: Non-empty fixed-shape encoder input mappings.

        Returns:
            None.

        Raises:
            ValueError: If no calibration batches are supplied.
        """
        if not batches:
            raise ValueError("Zipformer calibration inputs must not be empty")
        self._iterator = iter(
            tuple(
                {str(name): np.asarray(value) for name, value in batch.items()}
                for batch in batches
            )
        )

    def get_next(self) -> dict[str, np.ndarray] | None:
        """Return the next calibration input mapping.

        Returns:
            The next fixed-shape batch, or `None` after exhaustion.
        """
        return next(self._iterator, None)


def quantize_zipformer_encoder_ortqnn(
    source_path: str | Path,
    output_path: str | Path,
    calibration_inputs: Sequence[Mapping[str, np.ndarray]],
) -> Path:
    """Quantize Zipformer encoder MatMul operations with ORT-QNN static PTQ.

    Args:
        source_path: Prepared fixed-shape FP32 encoder.
        output_path: Destination Q/DQ ONNX encoder.
        calibration_inputs: Ordered fixed-shape encoder input batches.

    Returns:
        Resolved quantized encoder path.
    """
    source = Path(source_path).resolve()
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        model_input=source.as_posix(),
        model_output=destination.as_posix(),
        calibration_data_reader=_SequenceCalibrationReader(calibration_inputs),
        quant_format=QuantFormat.QDQ,
        calibrate_method=CalibrationMethod.MinMax,
        weight_type=QuantType.QUInt8,
        activation_type=QuantType.QUInt16,
        per_channel=False,
        op_types_to_quantize=["MatMul"],
    )
    return destination


def build_zipformer_calibration_inputs(
    manifest_path: str | Path,
    *,
    fixed_encoder_frames: int = 2009,
) -> list[dict[str, np.ndarray]]:
    """Build padded Zipformer encoder calibration inputs from a VLSP manifest.

    Args:
        manifest_path: Portable VLSP calibration/evaluation manifest.
        fixed_encoder_frames: Required padded encoder time dimension.

    Returns:
        Fixed-shape feature and valid-length mappings for calibration records.

    Raises:
        ValueError: If no calibration records exist or one exceeds fixed shape.
    """
    import torchaudio

    from model_pipeline.models.zipformer.runtime import extract_zipformer_features

    manifest = Path(manifest_path).resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    batches: list[dict[str, np.ndarray]] = []
    for sample in payload.get("samples", ()):
        if sample.get("partition") != "calibration":
            continue
        waveform, sample_rate = torchaudio.load(
            str(manifest.parent / str(sample["audio_path"]))
        )
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if int(sample_rate) != 16_000:
            waveform = torchaudio.functional.resample(waveform, int(sample_rate), 16_000)
            sample_rate = 16_000
        features = extract_zipformer_features(
            waveform.squeeze(0).detach().cpu().numpy(),
            int(sample_rate),
        )
        if features.shape[0] > fixed_encoder_frames:
            raise ValueError(
                f"Calibration feature length {features.shape[0]} exceeds "
                f"{fixed_encoder_frames}"
            )
        padded = np.zeros((1, fixed_encoder_frames, 80), dtype=np.float32)
        padded[0, : features.shape[0], :] = features
        batches.append(
            {
                "x": padded,
                "x_lens": np.asarray([features.shape[0]], dtype=np.int64),
            }
        )
    if not batches:
        raise ValueError("VLSP manifest contains no Zipformer calibration records")
    return batches


def inspect_zipformer_qdq_coverage(model_path: str | Path) -> ZipformerQdqInventory:
    """Inspect MatMul Q/DQ coverage in a Zipformer encoder graph.

    Args:
        model_path: FP32 or Q/DQ ONNX encoder.

    Returns:
        Total MatMul count, Q/DQ-covered count, and uncovered node names.
    """
    import onnx

    model = onnx.load(Path(model_path).resolve().as_posix(), load_external_data=False)
    producer_by_output = {
        output: node
        for node in model.graph.node
        for output in node.output
        if output
    }
    consumers_by_input: dict[str, list[object]] = {}
    for node in model.graph.node:
        for input_name in node.input:
            consumers_by_input.setdefault(input_name, []).append(node)
    matmuls = [node for node in model.graph.node if node.op_type == "MatMul"]
    quantized_names: set[str] = set()
    for node in matmuls:
        has_dequantized_input = any(
            producer_by_output.get(input_name) is not None
            and producer_by_output[input_name].op_type == "DequantizeLinear"
            for input_name in node.input
        )
        has_quantized_output = any(
            consumer.op_type == "QuantizeLinear"
            for output_name in node.output
            for consumer in consumers_by_input.get(output_name, ())
        )
        if has_dequantized_input or has_quantized_output:
            quantized_names.add(node.name)
    unquantized = tuple(node.name for node in matmuls if node.name not in quantized_names)
    return ZipformerQdqInventory(
        matmul_count=len(matmuls),
        quantized_matmul_count=len(quantized_names),
        unquantized_matmul_names=unquantized,
    )


def build_zipformer_encoder_matmul_policy(model_path: str | Path) -> dict[str, object]:
    """Build the AIMET policy covering every canonical encoder MatMul.

    Args:
        model_path: Prepared fixed-shape Zipformer encoder.

    Returns:
        Encoder MatMul allowlist, empty disable list, and coverage evidence.

    Raises:
        ValueError: If the encoder does not contain exactly 278 MatMul nodes.
    """
    import onnx

    model = onnx.load(Path(model_path).resolve().as_posix(), load_external_data=False)
    matmul_names = tuple(node.name for node in model.graph.node if node.op_type == "MatMul")
    if len(matmul_names) != 278:
        raise ValueError(
            f"Zipformer encoder requires 278 MatMul nodes; observed {len(matmul_names)}"
        )
    return {
        "schema_version": 1,
        "quantization_scope": "encoder-matmul",
        "quantizer_selection": "operator-name-allowlist",
        "quantize_op_types": ["MatMul"],
        "quantize_op_names": list(matmul_names),
        "disable_op_names": [],
        "coverage": {"quantized": 278, "total_matmul": 278},
    }


def assess_zipformer_quality(
    fp32: TranscriptQualitySummary,
    quantized: TranscriptQualitySummary,
) -> ZipformerQualityGate:
    """Apply Zipformer CER, WER, empty-output, and collapse thresholds.

    Args:
        fp32: Aggregate FP32 control quality metrics.
        quantized: Aggregate quantized-model quality metrics.

    Returns:
        Metric deltas and whether the quantized output passes all gates.
    """
    character_increase = quantized.character_error_rate - fp32.character_error_rate
    word_increase = quantized.word_error_rate - fp32.word_error_rate
    passed = (
        character_increase <= 0.0100000001
        and word_increase <= 0.0200000001
        and quantized.empty_output_count == 0
        and quantized.repetition_collapse_count == 0
    )
    return ZipformerQualityGate(
        character_error_rate_increase=character_increase,
        word_error_rate_increase=word_increase,
        empty_output_count=quantized.empty_output_count,
        repetition_collapse_count=quantized.repetition_collapse_count,
        passed=passed,
    )


def select_zipformer_quantization_engine(
    fp32: TranscriptQualitySummary,
    ortqnn: TranscriptQualitySummary,
    *,
    compile_accepted: bool,
) -> str:
    """Select ORT-QNN or AIMET from local quality and compile compatibility.

    Args:
        fp32: Aggregate FP32 control quality metrics.
        ortqnn: Aggregate ORT-QNN quality metrics.
        compile_accepted: Whether AI Hub accepted the ORT-QNN artifact.

    Returns:
        `ortqnn` when every gate passes; otherwise `aimet`.
    """
    quality_gate = assess_zipformer_quality(fp32, ortqnn)
    return "ortqnn" if quality_gate.passed and compile_accepted else "aimet"
