import contextlib
import uuid
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest import mock

import onnx
from onnx import TensorProto
from onnxruntime.quantization import CalibrationMethod, QuantType, quantize
from onnxruntime.quantization.execution_providers.qnn.preprocess import qnn_preprocess_model
from onnxruntime.quantization.execution_providers.qnn.quant_config import get_qnn_qdq_config

from quantize.calibration import CalibrationSample, ListCalibrationDataReader
from quantize.config import DEFAULT_TEMP_ROOT
from quantize.runtime import ManualTemporaryDirectory, isolated_model_input, temporary_workspace_tempdir
from quantize.types import LocalQdqCompatibilityReport, QuantizationPlan


QDQ_OP_TYPES = {"QuantizeLinear", "DequantizeLinear"}
QUANTIZED_TENSOR_TYPES = {
    TensorProto.UINT8,
    TensorProto.INT8,
    TensorProto.UINT16,
    TensorProto.INT16,
}


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


def _tensor_dtype_name(data_type: int) -> str:
    return TensorProto.DataType.Name(data_type)


def _inspect_aihub_onnx_packaging(model_path: Path) -> tuple[str, bool, tuple[str, ...]]:
    if model_path.is_dir():
        notes: list[str] = []
        packaging_kind = "onnx_dir"
        if model_path.suffix != ".onnx":
            notes.append("onnx_dir_missing_dot_onnx_suffix")
        children = [child for child in model_path.iterdir() if child.is_file()]
        onnx_files = [child for child in children if child.suffix == ".onnx"]
        data_files = [child for child in children if child.suffix == ".data"]
        if len(onnx_files) != 1:
            notes.append("onnx_dir_requires_single_onnx_file")
        if len(data_files) != 1:
            notes.append("onnx_dir_requires_single_data_file")
        return packaging_kind, not notes, tuple(notes)

    adjacent_data_path = model_path.with_suffix(f"{model_path.suffix}.data")
    if adjacent_data_path.exists():
        return "onnx_file", False, ("external_data_requires_onnx_dir_packaging",)
    return "onnx_file", True, ()


def inspect_qdq_compile_candidate(model_path: str | Path) -> dict[str, Any]:
    resolved_model_path = Path(model_path).resolve()
    model = onnx.load(resolved_model_path.as_posix(), load_external_data=False)
    initializer_by_name = {initializer.name: initializer for initializer in model.graph.initializer}
    initializer_dtypes = Counter(_tensor_dtype_name(initializer.data_type) for initializer in model.graph.initializer)
    opsets = {
        ("main" if not str(opset.domain or "").strip() else str(opset.domain)): int(opset.version)
        for opset in model.opset_import
    }

    qdq_nodes = [node for node in model.graph.node if node.op_type in QDQ_OP_TYPES]
    qdq_domains = Counter("main" if not str(node.domain or "").strip() else str(node.domain) for node in qdq_nodes)
    uses_uint16_qdq = False
    uses_int16_qdq = False
    quantized_weight_initializer_names: set[str] = set()

    for node in qdq_nodes:
        for input_name in node.input:
            initializer = initializer_by_name.get(str(input_name))
            if initializer is None:
                continue
            if initializer.data_type == TensorProto.UINT16:
                uses_uint16_qdq = True
            if initializer.data_type == TensorProto.INT16:
                uses_int16_qdq = True

        if node.op_type != "DequantizeLinear" or not node.input:
            continue
        quantized_value = initializer_by_name.get(str(node.input[0]))
        if quantized_value is None:
            continue
        if quantized_value.data_type in QUANTIZED_TENSOR_TYPES:
            quantized_weight_initializer_names.add(quantized_value.name)
            if quantized_value.data_type == TensorProto.UINT16:
                uses_uint16_qdq = True
            if quantized_value.data_type == TensorProto.INT16:
                uses_int16_qdq = True

    packaging_kind, packaging_ready, packaging_notes = _inspect_aihub_onnx_packaging(resolved_model_path)
    readiness_flags: list[str] = []
    if int(qdq_domains.get("com.microsoft", 0)) > 0:
        readiness_flags.append("com.microsoft_qdq")
    if (uses_uint16_qdq or uses_int16_qdq) and int(opsets.get("main", 0)) < 21:
        readiness_flags.append("main_opset_lt_21_for_16bit_qdq")
    if quantized_weight_initializer_names:
        readiness_flags.append("quantized_weight_initializers")
    readiness_flags.extend(packaging_notes)

    if any(flag in {"com.microsoft_qdq", "main_opset_lt_21_for_16bit_qdq"} for flag in readiness_flags):
        readiness = "unsafe"
    elif readiness_flags:
        readiness = "experimental"
    else:
        readiness = "ready"

    report = LocalQdqCompatibilityReport(
        model_path=resolved_model_path.as_posix(),
        opsets=opsets,
        qdq_domains=dict(qdq_domains),
        ms_qdq_node_count=int(qdq_domains.get("com.microsoft", 0)),
        main_qdq_node_count=int(qdq_domains.get("main", 0)),
        uses_uint16_qdq=uses_uint16_qdq,
        uses_int16_qdq=uses_int16_qdq,
        uses_quantized_weight_initializers=bool(quantized_weight_initializer_names),
        quantized_weight_initializer_count=len(quantized_weight_initializer_names),
        initializer_dtypes=dict(initializer_dtypes),
        packaging_kind=packaging_kind,
        packaging_ready=packaging_ready,
        packaging_notes=packaging_notes,
        aihub_compile_readiness=readiness,
        readiness_flags=tuple(readiness_flags),
    )
    return asdict(report)


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
