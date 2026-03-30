import contextlib
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Sequence
from unittest import mock

from onnxruntime.quantization.calibrate import TensorsData, create_calibrator
from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_dynamic, quantize_static
from onnxruntime.quantization.qdq_quantizer import QDQQuantizer
from onnxruntime.quantization.quant_utils import load_model_with_shape_infer, model_has_pre_process_metadata

from quantize.calibration import CalibrationSample, ListCalibrationDataReader
from quantize.config import DEFAULT_TEMP_ROOT
from quantize.runtime import ManualTemporaryDirectory, temporary_workspace_tempdir
from quantize.types import QuantizationPlan


def resolve_calibration_method(name: str) -> CalibrationMethod:
    mapping = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
        "distribution": CalibrationMethod.Distribution,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported calibration method: {name}") from exc


def build_static_extra_options(percentile: float) -> dict[str, float]:
    return {"CalibPercentile": percentile}


@contextlib.contextmanager
def isolated_model_input(fp32_onnx_path: Path) -> Path:
    temp_root = DEFAULT_TEMP_ROOT / "model_inputs"
    temp_root.mkdir(parents=True, exist_ok=True)
    staged_input = temp_root / f"{fp32_onnx_path.stem}.{uuid.uuid4().hex}{fp32_onnx_path.suffix}"
    try:
        os.link(fp32_onnx_path, staged_input)
    except OSError:
        shutil.copy2(fp32_onnx_path, staged_input)

    try:
        yield staged_input
    finally:
        inferred_path = staged_input.with_name(f"{staged_input.stem}-inferred{staged_input.suffix}")
        for cleanup_path in (inferred_path, staged_input):
            with contextlib.suppress(FileNotFoundError, PermissionError):
                cleanup_path.unlink()


def run_static_quantization(
    fp32_onnx_path: Path,
    output_path: Path,
    plan: QuantizationPlan,
    records: Sequence[CalibrationSample],
    calibration_method: CalibrationMethod | None = None,
    percentile: float | None = None,
    per_channel: bool | None = None,
    calibration_chunk_size: int | None = None,
) -> None:
    reader = ListCalibrationDataReader(records)
    resolved_method = calibration_method or resolve_calibration_method(plan.calibration_method)
    resolved_per_channel = plan.per_channel if per_channel is None else per_channel
    extra_options = build_static_extra_options(percentile if percentile is not None else plan.percentile)
    with temporary_workspace_tempdir(DEFAULT_TEMP_ROOT):
        with isolated_model_input(fp32_onnx_path) as staged_input:
            with mock.patch("tempfile.TemporaryDirectory", ManualTemporaryDirectory):
                if calibration_chunk_size and calibration_chunk_size > 0:
                    _run_static_quantization_chunked(
                        fp32_onnx_path=staged_input,
                        output_path=output_path,
                        plan=plan,
                        reader=reader,
                        calibration_method=resolved_method,
                        percentile=extra_options["CalibPercentile"],
                        per_channel=resolved_per_channel,
                        calibration_chunk_size=calibration_chunk_size,
                    )
                else:
                    quantize_static(
                        model_input=os.fspath(staged_input),
                        model_output=os.fspath(output_path),
                        calibration_data_reader=reader,
                        quant_format=QuantFormat.QDQ,
                        op_types_to_quantize=list(plan.op_types_to_quantize),
                        per_channel=resolved_per_channel,
                        activation_type=QuantType.QInt8,
                        weight_type=QuantType.QInt8,
                        nodes_to_exclude=list(plan.nodes_to_exclude),
                        calibrate_method=resolved_method,
                        calibration_providers=["CPUExecutionProvider"],
                        extra_options=extra_options,
                    )


def _run_static_quantization_chunked(
    fp32_onnx_path: Path,
    output_path: Path,
    plan: QuantizationPlan,
    reader: ListCalibrationDataReader,
    calibration_method: CalibrationMethod,
    percentile: float,
    per_channel: bool,
    calibration_chunk_size: int,
) -> None:
    model = load_model_with_shape_infer(fp32_onnx_path)
    pre_processed = model_has_pre_process_metadata(model)
    augmented_model_path = DEFAULT_TEMP_ROOT / f"augmented_model.{uuid.uuid4().hex}.onnx"
    calibrator = create_calibrator(
        fp32_onnx_path,
        list(plan.op_types_to_quantize),
        augmented_model_path=augmented_model_path.as_posix(),
        calibrate_method=calibration_method,
        use_external_data_format=False,
        providers=["CPUExecutionProvider"],
        extra_options={"percentile": percentile},
    )
    try:
        total_records = len(reader)
        chunk_size = max(1, calibration_chunk_size)
        for start_index in range(0, total_records, chunk_size):
            end_index = min(total_records, start_index + chunk_size)
            reader.set_range(start_index=start_index, end_index=end_index)
            calibrator.collect_data(reader)

        tensors_range = calibrator.compute_data()
        if not isinstance(tensors_range, TensorsData):
            raise TypeError(f"Unexpected type {type(tensors_range)} for tensors_range.")

        quantizer = QDQQuantizer(
            model,
            per_channel,
            False,
            QuantType.QInt8,
            QuantType.QInt8,
            tensors_range,
            None,
            list(plan.nodes_to_exclude),
            list(plan.op_types_to_quantize),
            build_static_extra_options(percentile),
        )
        quantizer.quantize_model()
        quantizer.model.save_model_to_file(output_path, False)
    finally:
        external_data_path = augmented_model_path.with_suffix(f"{augmented_model_path.suffix}.data")
        for cleanup_path in (augmented_model_path, external_data_path):
            with contextlib.suppress(FileNotFoundError, PermissionError):
                cleanup_path.unlink()

    if not pre_processed:
        logging.warning(
            "Please consider to run pre-processing before quantization. Refer to example: "
            "https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification"
            "/cpu/ReadMe.md "
        )


def run_dynamic_quantization(
    fp32_onnx_path: Path,
    output_path: Path,
    plan: QuantizationPlan,
) -> None:
    with temporary_workspace_tempdir(DEFAULT_TEMP_ROOT):
        with isolated_model_input(fp32_onnx_path) as staged_input:
            with mock.patch("tempfile.TemporaryDirectory", ManualTemporaryDirectory):
                quantize_dynamic(
                    model_input=os.fspath(staged_input),
                    model_output=os.fspath(output_path),
                    op_types_to_quantize=list(plan.op_types_to_quantize),
                    per_channel=plan.per_channel,
                    weight_type=QuantType.QInt8,
                    nodes_to_exclude=list(plan.nodes_to_exclude),
                )


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def build_size_budget_message(size_mb: float, size_budget_mb: float) -> str:
    if size_mb <= size_budget_mb:
        return f"Size budget: PASS ({size_mb:.2f} <= {size_budget_mb:.2f} MB)"
    return f"Size budget: FAIL ({size_mb:.2f} > {size_budget_mb:.2f} MB)"


def recommend_next_steps(plan: QuantizationPlan, size_mb: float, size_budget_mb: float) -> list[str]:
    recommendations: list[str] = []
    if size_mb > size_budget_mb and plan.preset == "sd8g2_quality":
        recommendations.append("sd8g2_quality dang uu tien giu decoder o FP32; chi thu balanced/aggressive neu ban chap nhan rui ro giam quality.")
    if plan.runner_kind == "static":
        recommendations.append("Neu quality chua tot, tang do da dang cua calibration text hoac giam pham vi quantize thay vi mo rong them.")
    return recommendations
