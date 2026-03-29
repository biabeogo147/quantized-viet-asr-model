import contextlib
import os
import shutil
import uuid
from pathlib import Path
from typing import Sequence
from unittest import mock

from onnxruntime.quantization import (
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)

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
    return mapping[name]


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
) -> None:
    reader = ListCalibrationDataReader(records)
    extra_options = {
        "CalibPercentile": percentile if percentile is not None else plan.percentile,
    }
    with temporary_workspace_tempdir(DEFAULT_TEMP_ROOT):
        with isolated_model_input(fp32_onnx_path) as staged_input:
            with mock.patch("tempfile.TemporaryDirectory", ManualTemporaryDirectory):
                quantize_static(
                    model_input=os.fspath(staged_input),
                    model_output=os.fspath(output_path),
                    calibration_data_reader=reader,
                    quant_format=QuantFormat.QDQ,
                    op_types_to_quantize=list(plan.op_types_to_quantize),
                    per_channel=plan.per_channel if per_channel is None else per_channel,
                    activation_type=QuantType.QInt8,
                    weight_type=QuantType.QInt8,
                    nodes_to_exclude=list(plan.nodes_to_exclude),
                    calibrate_method=calibration_method or resolve_calibration_method(plan.calibration_method),
                    extra_options=extra_options,
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
        recommendations.append("Thu preset sd8g2_balanced truoc khi quantize them cac node nhay cam.")
    if plan.runner_kind == "static":
        recommendations.append("Neu quality chua tot, tang do da dang cua calibration text truoc khi mo rong pham vi quantize.")
    return recommendations
