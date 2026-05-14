from __future__ import annotations

from types import SimpleNamespace
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np

from quantize.calibration import build_calibration_records
from quantize.config import (
    DEFAULT_BALANCED_OUTPUT_ONNX,
    DEFAULT_CALIBRATION_CHUNK_SIZE,
    DEFAULT_CALIBRATION_SOURCE,
    DEFAULT_DYNAMIC_OUTPUT_ONNX,
    DEFAULT_FP32_ONNX,
    DEFAULT_MAX_CALIBRATION_SAMPLES,
    DEFAULT_MAX_GENERATION_LENGTH,
    DEFAULT_MODEL_DIR,
    DEFAULT_ORT_PROVIDER,
    DEFAULT_OUTPUT_ONNX,
    DEFAULT_PERCENTILE,
    DEFAULT_SIZE_BUDGET_MB,
)
from quantize.model_introspection import load_model_node_names, summarize_quantization_plan
from quantize.presets import build_quantization_plan, get_preset_spec, list_supported_presets
from quantize.qnn import run_qnn_static_quantization
from quantize.runner import (
    build_size_budget_message,
    file_size_mb,
    recommend_next_steps,
    resolve_calibration_method,
    run_dynamic_quantization,
    run_static_quantization,
)
from quantize.types import AiHubQuantizeRecipe, CalibrationSample

NAME = 'vpcd'
DEFAULT_PRESET = 'sd8g2_quality'
AIHUB_DTYPE_NAME_BY_QUANT_TYPE = {
    "quint8": "INT8",
    "qint8": "INT8",
    "quint16": "INT16",
    "qint16": "INT16",
}


def apply_default_arguments(parser) -> None:
    parser.add_argument('--model-dir', default=str(DEFAULT_MODEL_DIR))
    parser.add_argument('--fp32-onnx', default=str(DEFAULT_FP32_ONNX))
    parser.add_argument('--output')
    parser.add_argument('--calibration-text', '--calibration-source', dest='calibration_text', default=str(DEFAULT_CALIBRATION_SOURCE), help='Duong dan toi file txt hoac thu muc chua nhieu file txt calibration.')
    parser.add_argument('--preset', default=DEFAULT_PRESET)
    parser.add_argument('--max-calibration-samples', type=int, default=DEFAULT_MAX_CALIBRATION_SAMPLES)
    parser.add_argument('--max-generation-length', type=int, default=DEFAULT_MAX_GENERATION_LENGTH)
    parser.add_argument('--calibration-chunk-size', type=int, default=DEFAULT_CALIBRATION_CHUNK_SIZE)
    parser.add_argument('--ort-provider', choices=('cuda', 'cpu'), default=DEFAULT_ORT_PROVIDER)
    parser.add_argument('--size-budget-mb', type=float, default=DEFAULT_SIZE_BUDGET_MB)
    parser.add_argument('--percentile', type=float, default=DEFAULT_PERCENTILE)
    parser.add_argument('--calibration-method', choices=('minmax', 'entropy', 'percentile', 'distribution'))
    parser.add_argument('--per-channel', action='store_true', default=None)
    parser.add_argument('--no-per-channel', dest='per_channel', action='store_false')
    parser.add_argument('--extra-exclude-pattern', action='append', default=[])
    parser.add_argument('--dry-run', action='store_true')


def validate_args(args) -> None:
    if args.preset not in list_supported_presets():
        raise ValueError(f'Unsupported vpcd preset: {args.preset}')
    if args.calibration_chunk_size is not None and args.calibration_chunk_size < 1:
        raise ValueError('--calibration-chunk-size phai >= 1.')


def _resolve_output_path(args) -> Path:
    if args.output:
        return Path(args.output)
    if args.preset == 'sd8g2_balanced':
        return DEFAULT_BALANCED_OUTPUT_ONNX
    if args.preset == 'baseline_dynamic_int8':
        return DEFAULT_DYNAMIC_OUTPUT_ONNX
    return DEFAULT_OUTPUT_ONNX


def resolve_vpcd_aihub_quantize_dtype_names(*, preset: str = DEFAULT_PRESET) -> dict[str, str]:
    spec = get_preset_spec(preset)
    activation_type = str(spec.activation_type).strip().lower()
    weight_type = str(spec.weight_type).strip().lower()
    resolved_activation_dtype = AIHUB_DTYPE_NAME_BY_QUANT_TYPE.get(activation_type)
    resolved_weight_dtype = AIHUB_DTYPE_NAME_BY_QUANT_TYPE.get(weight_type)
    if resolved_activation_dtype is None or resolved_weight_dtype is None:
        raise ValueError(
            "Unsupported VPCD quantization types for AI Hub mapping: "
            f"activation_type={activation_type!r}, weight_type={weight_type!r}"
        )
    return {
        "weights_dtype_name": resolved_weight_dtype,
        "activations_dtype_name": resolved_activation_dtype,
    }


def _pad_array_to_target_shape(
    values: np.ndarray,
    target_shape: Sequence[int],
    pad_value: int,
) -> np.ndarray:
    array = np.asarray(values)
    normalized_target_shape = tuple(int(dimension) for dimension in target_shape)
    if array.ndim != len(normalized_target_shape):
        raise ValueError(
            f"Expected array with rank {len(normalized_target_shape)}, got shape {tuple(array.shape)}."
        )
    if any(current > target for current, target in zip(array.shape, normalized_target_shape)):
        raise ValueError(
            f"Input shape {tuple(array.shape)} exceeds fixed target shape {normalized_target_shape}."
        )
    if tuple(array.shape) == normalized_target_shape:
        return array

    padded = np.full(normalized_target_shape, pad_value, dtype=array.dtype)
    slices = tuple(slice(0, int(dimension)) for dimension in array.shape)
    padded[slices] = array
    return padded


def calibration_records_to_aihub_dataset(
    records: Sequence[CalibrationSample],
    *,
    fixed_input_shapes: dict[str, Sequence[int]] | None = None,
    pad_values: dict[str, int] | None = None,
) -> dict[str, list[np.ndarray]]:
    normalized_records = list(records)
    if not normalized_records:
        raise ValueError("records must not be empty.")

    input_names = tuple(normalized_records[0].inputs.keys())
    dataset = {name: [] for name in input_names}
    for record in normalized_records:
        current_input_names = tuple(record.inputs.keys())
        if current_input_names != input_names:
            raise ValueError(
                "All calibration records must expose the same input ordering. "
                f"Expected {input_names}, got {current_input_names}."
            )
        for name in input_names:
            value = np.asarray(record.inputs[name])
            if fixed_input_shapes and name in fixed_input_shapes:
                value = _pad_array_to_target_shape(
                    value,
                    fixed_input_shapes[name],
                    0 if pad_values is None else int(pad_values.get(name, 0)),
                )
            dataset[name].append(value)
    return dataset


def summarize_aihub_calibration_dataset(
    dataset: dict[str, list[np.ndarray]],
) -> dict[str, object]:
    input_order = list(dataset.keys())
    fingerprint = hashlib.sha256()
    fingerprint.update(len(input_order).to_bytes(8, "little", signed=False))
    input_sample_counts: dict[str, int] = {}
    input_dtypes: dict[str, str] = {}
    input_shapes: dict[str, list[list[int]]] = {}

    for input_name in input_order:
        encoded_name = input_name.encode("utf-8")
        fingerprint.update(len(encoded_name).to_bytes(8, "little", signed=False))
        fingerprint.update(encoded_name)

        arrays = [np.asarray(value) for value in dataset[input_name]]
        input_sample_counts[input_name] = len(arrays)
        input_dtypes[input_name] = str(arrays[0].dtype) if arrays else ""
        input_shapes[input_name] = [list(array.shape) for array in arrays]
        fingerprint.update(len(arrays).to_bytes(8, "little", signed=False))

        for array in arrays:
            normalized = np.ascontiguousarray(array)
            dtype_text = str(normalized.dtype)
            encoded_dtype = dtype_text.encode("utf-8")
            fingerprint.update(len(encoded_dtype).to_bytes(8, "little", signed=False))
            fingerprint.update(encoded_dtype)
            fingerprint.update(normalized.ndim.to_bytes(8, "little", signed=False))
            for dimension in normalized.shape:
                fingerprint.update(int(dimension).to_bytes(8, "little", signed=True))
            raw_bytes = normalized.tobytes()
            fingerprint.update(len(raw_bytes).to_bytes(8, "little", signed=False))
            fingerprint.update(raw_bytes)

    return {
        "input_order": input_order,
        "input_sample_counts": input_sample_counts,
        "input_dtypes": input_dtypes,
        "input_shapes": input_shapes,
        "dataset_fingerprint": fingerprint.hexdigest(),
    }


def build_vpcd_aihub_quantize_recipe(
    *,
    model_dir: str | Path,
    fp32_onnx_path: str | Path,
    calibration_source_path: str | Path,
    preset: str = DEFAULT_PRESET,
    max_calibration_samples: int = DEFAULT_MAX_CALIBRATION_SAMPLES,
    max_generation_length: int = DEFAULT_MAX_GENERATION_LENGTH,
    ort_provider: str = DEFAULT_ORT_PROVIDER,
    fixed_input_shapes: dict[str, Sequence[int]] | None = None,
    pad_token_id: int = 1,
) -> AiHubQuantizeRecipe:
    records, stats = build_calibration_records(
        model_dir=Path(model_dir),
        fp32_onnx_path=Path(fp32_onnx_path),
        calibration_source_path=Path(calibration_source_path),
        max_calibration_samples=max_calibration_samples,
        max_generation_length=max_generation_length,
        ort_provider=ort_provider,
    )
    if not records:
        raise ValueError('Khong tao duoc calibration records tu file dau vao.')

    spec = get_preset_spec(preset)
    dtype_names = resolve_vpcd_aihub_quantize_dtype_names(preset=spec.name)
    calibration_dataset = calibration_records_to_aihub_dataset(
        records,
        fixed_input_shapes=fixed_input_shapes,
        pad_values={
            "input_ids": int(pad_token_id),
            "attention_mask": 0,
            "decoder_input_ids": int(pad_token_id),
            "decoder_attention_mask": 0,
        },
    )
    recipe_stats = dict(stats)
    recipe_stats["quantize_preset"] = spec.name
    recipe_stats["activation_type"] = spec.activation_type
    recipe_stats["weight_type"] = spec.weight_type
    recipe_stats.update(summarize_aihub_calibration_dataset(calibration_dataset))
    return AiHubQuantizeRecipe(
        preset=spec.name,
        activation_type=spec.activation_type,
        weight_type=spec.weight_type,
        activations_dtype_name=dtype_names["activations_dtype_name"],
        weights_dtype_name=dtype_names["weights_dtype_name"],
        calibration_dataset=calibration_dataset,
        calibration_stats=recipe_stats,
    )


def run(args) -> int:
    validate_args(args)
    fp32_onnx_path = Path(args.fp32_onnx)
    output_path = _resolve_output_path(args)
    model_dir = Path(args.model_dir)

    if not fp32_onnx_path.exists():
        raise FileNotFoundError(f'Khong tim thay FP32 ONNX: {fp32_onnx_path}')

    node_names = load_model_node_names(fp32_onnx_path)
    plan = build_quantization_plan(node_names=node_names, preset=args.preset, extra_exclude_patterns=args.extra_exclude_pattern)

    if args.dry_run:
        print(summarize_quantization_plan(plan, node_names))
        print(f'Project: {NAME}')
        print(f'FP32 ONNX: {fp32_onnx_path}')
        print(f'Output: {output_path}')
        return 0

    if not model_dir.exists():
        raise FileNotFoundError(f'Khong tim thay model dir: {model_dir}')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(summarize_quantization_plan(plan, node_names))

    if plan.runner_kind == 'dynamic':
        run_dynamic_quantization(fp32_onnx_path=fp32_onnx_path, output_path=output_path, plan=plan)
    else:
        records, stats = build_calibration_records(
            model_dir=model_dir,
            fp32_onnx_path=fp32_onnx_path,
            calibration_source_path=Path(args.calibration_text),
            max_calibration_samples=args.max_calibration_samples,
            max_generation_length=args.max_generation_length,
            ort_provider=args.ort_provider,
        )
        if not records:
            raise ValueError('Khong tao duoc calibration records tu file dau vao.')
        print(
            'Calibration stats: '
            f"requested_provider={stats['requested_provider']}, "
            f"session_providers={stats['session_providers']}, "
            f"source_files={stats['source_files']}, "
            f"text_samples={stats['text_samples']}, "
            f"records={stats['records']}, "
            f"max_encoder_len={stats['max_encoder_len']}, "
            f"max_decoder_len={stats['max_decoder_len']}"
        )
        resolved_calibration_method = resolve_calibration_method(args.calibration_method or plan.calibration_method)
        if plan.runner_kind == 'qnn_static':
            run_qnn_static_quantization(
                fp32_onnx_path=fp32_onnx_path,
                output_path=output_path,
                plan=plan,
                records=records,
                calibration_method=resolved_calibration_method,
                calibration_chunk_size=args.calibration_chunk_size,
            )
        else:
            run_static_quantization(
                fp32_onnx_path=fp32_onnx_path,
                output_path=output_path,
                plan=plan,
                records=records,
                calibration_method=resolved_calibration_method,
                percentile=args.percentile,
                per_channel=args.per_channel,
                calibration_chunk_size=args.calibration_chunk_size,
            )

    size_mb = file_size_mb(output_path)
    print(f'Project: {NAME}')
    print(f'Quantized ONNX: {output_path}')
    print(f'Output size: {size_mb:.2f} MB')
    print(build_size_budget_message(size_mb, args.size_budget_mb))
    for recommendation in recommend_next_steps(plan, size_mb, args.size_budget_mb):
        print(f'Goi y: {recommendation}')
    return 0
