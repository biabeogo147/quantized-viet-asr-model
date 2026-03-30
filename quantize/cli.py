import argparse
from pathlib import Path
from typing import Sequence

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
from quantize.presets import build_quantization_plan, list_supported_presets
from quantize.qnn import run_qnn_static_quantization
from quantize.runner import (
    build_size_budget_message,
    file_size_mb,
    recommend_next_steps,
    resolve_calibration_method,
    run_dynamic_quantization,
    run_static_quantization,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize vietnamese-punc-cap-denorm-v1 ONNX for Snapdragon 8 Gen 2.",
    )
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--fp32-onnx", default=str(DEFAULT_FP32_ONNX))
    parser.add_argument("--output")
    parser.add_argument(
        "--calibration-text",
        "--calibration-source",
        dest="calibration_text",
        default=str(DEFAULT_CALIBRATION_SOURCE),
        help="Duong dan toi file txt hoac thu muc chua nhieu file txt calibration.",
    )
    parser.add_argument("--preset", choices=list_supported_presets(), default="sd8g2_quality")
    parser.add_argument("--max-calibration-samples", type=int, default=DEFAULT_MAX_CALIBRATION_SAMPLES)
    parser.add_argument("--max-generation-length", type=int, default=DEFAULT_MAX_GENERATION_LENGTH)
    parser.add_argument(
        "--calibration-chunk-size",
        type=int,
        default=DEFAULT_CALIBRATION_CHUNK_SIZE,
        help="So calibration records moi chunk khi ORT collect activation stats. Giam RAM peak cho static quantization.",
    )
    parser.add_argument(
        "--ort-provider",
        choices=("cuda", "cpu"),
        default=DEFAULT_ORT_PROVIDER,
        help="Provider alias cho calibration inference. 'cuda' se thu CUDA truoc va fallback ve CPU neu can.",
    )
    parser.add_argument("--size-budget-mb", type=float, default=DEFAULT_SIZE_BUDGET_MB)
    parser.add_argument("--percentile", type=float, default=DEFAULT_PERCENTILE)
    parser.add_argument(
        "--calibration-method",
        choices=("minmax", "entropy", "percentile", "distribution"),
    )
    parser.add_argument(
        "--per-channel",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--extra-exclude-pattern", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output:
        return Path(args.output)
    if args.preset == "sd8g2_balanced":
        return DEFAULT_BALANCED_OUTPUT_ONNX
    if args.preset == "baseline_dynamic_int8":
        return DEFAULT_DYNAMIC_OUTPUT_ONNX
    return DEFAULT_OUTPUT_ONNX


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    fp32_onnx_path = Path(args.fp32_onnx)
    output_path = _resolve_output_path(args)
    model_dir = Path(args.model_dir)

    if args.calibration_chunk_size is not None and args.calibration_chunk_size < 1:
        raise ValueError("--calibration-chunk-size phai >= 1.")

    if not fp32_onnx_path.exists():
        raise FileNotFoundError(f"Khong tim thay FP32 ONNX: {fp32_onnx_path}")

    node_names = load_model_node_names(fp32_onnx_path)
    plan = build_quantization_plan(
        node_names=node_names,
        preset=args.preset,
        extra_exclude_patterns=args.extra_exclude_pattern,
    )

    if args.dry_run:
        print(summarize_quantization_plan(plan, node_names))
        print(f"FP32 ONNX: {fp32_onnx_path}")
        print(f"Output: {output_path}")
        return 0

    if not model_dir.exists():
        raise FileNotFoundError(f"Khong tim thay model dir: {model_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(summarize_quantization_plan(plan, node_names))

    if plan.runner_kind == "dynamic":
        run_dynamic_quantization(
            fp32_onnx_path=fp32_onnx_path,
            output_path=output_path,
            plan=plan,
        )
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
            raise ValueError("Khong tao duoc calibration records tu file dau vao.")

        print(
            "Calibration stats: "
            f"requested_provider={stats['requested_provider']}, "
            f"session_providers={stats['session_providers']}, "
            f"source_files={stats['source_files']}, "
            f"text_samples={stats['text_samples']}, "
            f"records={stats['records']}, "
            f"max_encoder_len={stats['max_encoder_len']}, "
            f"max_decoder_len={stats['max_decoder_len']}"
        )
        resolved_calibration_method = resolve_calibration_method(args.calibration_method or plan.calibration_method)
        if plan.runner_kind == "qnn_static":
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
    print(f"Quantized ONNX: {output_path}")
    print(f"Output size: {size_mb:.2f} MB")
    print(build_size_budget_message(size_mb, args.size_budget_mb))
    for recommendation in recommend_next_steps(plan, size_mb, args.size_budget_mb):
        print(f"Goi y: {recommendation}")
    return 0
