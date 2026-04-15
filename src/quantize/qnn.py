import contextlib
import uuid
from pathlib import Path
from unittest import mock

from onnxruntime.quantization import CalibrationMethod, QuantType, quantize
from onnxruntime.quantization.execution_providers.qnn.preprocess import qnn_preprocess_model
from onnxruntime.quantization.execution_providers.qnn.quant_config import get_qnn_qdq_config

from quantize.calibration import CalibrationSample, ListCalibrationDataReader
from quantize.config import DEFAULT_TEMP_ROOT
from quantize.runtime import ManualTemporaryDirectory, isolated_model_input, temporary_workspace_tempdir
from quantize.types import QuantizationPlan


def resolve_quant_type(name: str) -> QuantType:
    mapping = {
        "qint8": QuantType.QInt8,
        "quint8": QuantType.QUInt8,
        "qint16": QuantType.QInt16,
        "quint16": QuantType.QUInt16,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported quant type: {name}") from exc


def resolve_safe_stride(total_records: int, requested_chunk_size: int | None) -> int | None:
    if total_records <= 0:
        return None
    if requested_chunk_size is None or requested_chunk_size < 1:
        return total_records

    upper_bound = min(total_records, requested_chunk_size)
    for candidate in range(upper_bound, 0, -1):
        if total_records % candidate == 0:
            return candidate
    return 1


def run_qnn_static_quantization(
    fp32_onnx_path: Path,
    output_path: Path,
    plan: QuantizationPlan,
    records: list[CalibrationSample],
    calibration_method: CalibrationMethod,
    calibration_chunk_size: int | None,
) -> None:
    reader = ListCalibrationDataReader(records)
    activation_type = resolve_quant_type(plan.activation_type)
    weight_type = resolve_quant_type(plan.weight_type)

    with temporary_workspace_tempdir(DEFAULT_TEMP_ROOT):
        with isolated_model_input(fp32_onnx_path, DEFAULT_TEMP_ROOT) as staged_input:
            preprocessed_input = DEFAULT_TEMP_ROOT / f"qnn_preprocessed.{uuid.uuid4().hex}.onnx"
            try:
                modified = qnn_preprocess_model(
                    staged_input,
                    preprocessed_input,
                    save_as_external_data=False,
                )
                model_input = preprocessed_input if modified and preprocessed_input.exists() else staged_input
                stride = resolve_safe_stride(len(reader), calibration_chunk_size)
                quant_config = get_qnn_qdq_config(
                    model_input,
                    calibration_data_reader=reader,
                    calibrate_method=calibration_method,
                    activation_type=activation_type,
                    weight_type=weight_type,
                    per_channel=plan.per_channel,
                    stride=stride,
                    calibration_providers=["CPUExecutionProvider"],
                    op_types_to_quantize=list(plan.op_types_to_quantize),
                    nodes_to_exclude=list(plan.nodes_to_exclude),
                )

                with mock.patch("tempfile.TemporaryDirectory", ManualTemporaryDirectory):
                    quantize(
                        model_input,
                        output_path,
                        quant_config,
                    )
            finally:
                preprocessed_data = preprocessed_input.with_suffix(f"{preprocessed_input.suffix}.data")
                for cleanup_path in (preprocessed_input, preprocessed_data):
                    with contextlib.suppress(FileNotFoundError, PermissionError):
                        cleanup_path.unlink()
