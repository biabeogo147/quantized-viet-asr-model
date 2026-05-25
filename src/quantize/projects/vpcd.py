from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import onnx

from model_bundle.manifest import ModelBundleManifest
from model_bundle.vpcd_shapes import resolve_vpcd_model_input_shapes
from quantize.aimet import (
    DEFAULT_AIMET_HEALTH_TIMEOUT_SECONDS,
    DEFAULT_AIMET_SERVICE_URL,
    DEFAULT_AIMET_SERVICE_WORKSPACE_ROOT,
    build_attention_ffn_aimet_config,
    build_matmul_only_aimet_config,
    build_vpcd_local_quality_policy_manifest,
    healthcheck_aimet_service,
    map_local_path_to_service_workspace,
    request_aimet_service_export,
    write_aimet_config,
    write_aimet_policy_manifest,
    write_calibration_batches,
)
from quantize.calibration import build_calibration_records
from quantize.fixed_shapes import freeze_model_inputs
from quantize.model_introspection import load_model_node_names
from quantize.types import AimetQuantizeRecipe, CalibrationSample, VpcdLocalQualityPolicySummary
from tools.paths import find_repo_root

NAME = "vpcd"
DEFAULT_MODEL_DIR = Path("assets") / "vietnamese-punc-cap-denorm-v1"
DEFAULT_FP32_ONNX = DEFAULT_MODEL_DIR / "onnx" / "model.fp32.onnx"
DEFAULT_CALIBRATION_SOURCE = Path("build") / "calibration" / "vlsp2020" / "vpcd_transcriptions.txt"
DEFAULT_ORT_PROVIDER = "cpu"
DEFAULT_MAX_CALIBRATION_SAMPLES = 24
DEFAULT_MAX_GENERATION_LENGTH = 32
DEFAULT_AIMET_PARAM_TYPE = "int8"
DEFAULT_AIMET_ACTIVATION_TYPE = "int16"
DEFAULT_AIMET_QUANT_SCHEME = "min_max"
DEFAULT_AIMET_CONFIG_FILE = "vpcd_matmul_only"
DEFAULT_AIMET_POLICY_MODE = "local_quality_parity"
DEFAULT_AIMET_OUTPUT_ROOT = Path("build") / "quantize" / "vpcd" / "local_aimet"
DEFAULT_FIXED_BUNDLE_MANIFEST = Path("build") / "model_bundle" / "vpcd" / "qnn_fixed_1024x128" / "bundle_manifest.json"
LOCAL_QUALITY_POLICY_REFERENCE = "local_quality_parity"
DECODER_EXPANDED_POLICY_REFERENCE = "decoder_expanded"
BROADER_ATTENTION_FFN_POLICY_REFERENCE = "broader_attention_ffn"
AGGRESSIVE_INT8_POLICY_REFERENCE = "aggressive_int8"
LOCAL_QUALITY_QUANTIZABLE_OP_TYPES = ("MatMul",)
BROADER_ATTENTION_FFN_QUANTIZABLE_OP_TYPES = ("MatMul", "Add", "Mul", "Div", "LayerNormalization")


def apply_default_arguments(parser) -> None:
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--fp32-onnx", default=str(DEFAULT_FP32_ONNX))
    parser.add_argument("--output-root", default=str(DEFAULT_AIMET_OUTPUT_ROOT))
    parser.add_argument(
        "--calibration-text",
        "--calibration-source",
        dest="calibration_text",
        default=str(DEFAULT_CALIBRATION_SOURCE),
        help="Duong dan toi file txt hoac thu muc chua nhieu file txt calibration.",
    )
    parser.add_argument("--max-calibration-samples", type=int, default=DEFAULT_MAX_CALIBRATION_SAMPLES)
    parser.add_argument("--max-generation-length", type=int, default=DEFAULT_MAX_GENERATION_LENGTH)
    parser.add_argument("--ort-provider", choices=("cuda", "cpu"), default=DEFAULT_ORT_PROVIDER)
    parser.add_argument("--fixed-bundle-manifest", default=str(DEFAULT_FIXED_BUNDLE_MANIFEST))
    parser.add_argument("--aimet-param-type", default=DEFAULT_AIMET_PARAM_TYPE)
    parser.add_argument("--aimet-activation-type", default=DEFAULT_AIMET_ACTIVATION_TYPE)
    parser.add_argument("--aimet-quant-scheme", default=DEFAULT_AIMET_QUANT_SCHEME)
    parser.add_argument("--aimet-config-file", default=DEFAULT_AIMET_CONFIG_FILE)
    parser.add_argument("--aimet-policy-mode", default=DEFAULT_AIMET_POLICY_MODE)
    parser.add_argument("--aimet-service-url", default=DEFAULT_AIMET_SERVICE_URL)
    parser.add_argument("--aimet-service-workspace-root", default=DEFAULT_AIMET_SERVICE_WORKSPACE_ROOT)
    parser.add_argument("--aimet-health-timeout-seconds", type=float, default=DEFAULT_AIMET_HEALTH_TIMEOUT_SECONDS)
    parser.add_argument("--dry-run", action="store_true")


def validate_args(args) -> None:
    if int(args.max_calibration_samples) < 1:
        raise ValueError("--max-calibration-samples phai >= 1.")
    if int(args.max_generation_length) < 1:
        raise ValueError("--max-generation-length phai >= 1.")
    if str(args.aimet_param_type).strip().lower() not in {"int8", "int16"}:
        raise ValueError(f"Unsupported AIMET param type: {args.aimet_param_type!r}")
    if str(args.aimet_activation_type).strip().lower() not in {"int8", "int16"}:
        raise ValueError(f"Unsupported AIMET activation type: {args.aimet_activation_type!r}")
    if not str(args.aimet_service_url).strip():
        raise ValueError("--aimet-service-url must not be empty.")


def _normalize_variant_fragment(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("post_training_", "")
        .replace(" ", "_")
        .replace("-", "_")
    )


def build_vpcd_aimet_variant_name(
    *,
    param_type: str,
    activation_type: str,
    quant_scheme: str,
    policy_mode: str,
) -> str:
    return "_".join(
        (
            f"w{_normalize_variant_fragment(param_type)}",
            f"a{_normalize_variant_fragment(activation_type)}",
            _normalize_variant_fragment(quant_scheme),
            _normalize_variant_fragment(policy_mode),
        )
    )


def _is_excluded_from_local_quality_policy(node_name: str) -> bool:
    normalized_name = str(node_name)
    return "/decoder/" in normalized_name or normalized_name == "/lm_head/MatMul"


def _is_excluded_from_decoder_expanded_policy(node_name: str) -> bool:
    return str(node_name) == "/lm_head/MatMul"


def _is_excluded_from_broader_attention_ffn_policy(node_name: str) -> bool:
    normalized_name = str(node_name)
    if normalized_name == "/lm_head/MatMul":
        return True
    return "/decoder/" not in normalized_name


def should_write_vpcd_aimet_policy_manifest(policy_mode: str) -> bool:
    normalized_policy_mode = str(policy_mode).strip().lower()
    return normalized_policy_mode not in {"", "none", "off", "disabled"}


def summarize_vpcd_aimet_policy(
    fp32_onnx_path: str | Path,
    *,
    policy_mode: str,
) -> VpcdLocalQualityPolicySummary:
    normalized_policy_mode = str(policy_mode).strip().lower()
    if normalized_policy_mode == LOCAL_QUALITY_POLICY_REFERENCE:
        preset = LOCAL_QUALITY_POLICY_REFERENCE
        exclusion_predicate = _is_excluded_from_local_quality_policy
        op_types_to_quantize = LOCAL_QUALITY_QUANTIZABLE_OP_TYPES
    elif normalized_policy_mode == DECODER_EXPANDED_POLICY_REFERENCE:
        preset = DECODER_EXPANDED_POLICY_REFERENCE
        exclusion_predicate = _is_excluded_from_decoder_expanded_policy
        op_types_to_quantize = LOCAL_QUALITY_QUANTIZABLE_OP_TYPES
    elif normalized_policy_mode == BROADER_ATTENTION_FFN_POLICY_REFERENCE:
        preset = BROADER_ATTENTION_FFN_POLICY_REFERENCE
        exclusion_predicate = _is_excluded_from_broader_attention_ffn_policy
        op_types_to_quantize = BROADER_ATTENTION_FFN_QUANTIZABLE_OP_TYPES
    elif normalized_policy_mode == AGGRESSIVE_INT8_POLICY_REFERENCE:
        preset = AGGRESSIVE_INT8_POLICY_REFERENCE
        exclusion_predicate = _is_excluded_from_broader_attention_ffn_policy
        op_types_to_quantize = BROADER_ATTENTION_FFN_QUANTIZABLE_OP_TYPES
    else:
        raise ValueError(f"Unsupported VPCD AIMET policy mode: {policy_mode!r}")

    resolved_model_path = Path(fp32_onnx_path).resolve()
    node_names = load_model_node_names(resolved_model_path)
    excluded_node_names = tuple(node_name for node_name in node_names if exclusion_predicate(node_name))
    excluded_node_set = set(excluded_node_names)

    model = onnx.load(resolved_model_path.as_posix(), load_external_data=False)
    quantizable_node_names = tuple(
        str(node.name)
        for node in model.graph.node
        if node.name and node.op_type in op_types_to_quantize and node.name not in excluded_node_set
    )
    quantizable_matmul_node_names = tuple(
        node_name
        for node_name in quantizable_node_names
        if node_name in {
            str(node.name)
            for node in model.graph.node
            if node.name and node.op_type == "MatMul"
        }
    )
    quantizable_node_count_by_op_type: dict[str, int] = {}
    for node in model.graph.node:
        if not node.name or node.name in excluded_node_set or node.op_type not in op_types_to_quantize:
            continue
        quantizable_node_count_by_op_type[node.op_type] = int(quantizable_node_count_by_op_type.get(node.op_type, 0)) + 1
    return VpcdLocalQualityPolicySummary(
        preset=preset,
        total_named_nodes=len(node_names),
        excluded_node_count=len(excluded_node_names),
        excluded_decoder_node_count=sum(1 for node_name in excluded_node_names if "/decoder/" in node_name),
        excluded_lm_head_node_count=sum(1 for node_name in excluded_node_names if node_name == "/lm_head/MatMul"),
        quantizable_matmul_node_count=len(quantizable_matmul_node_names),
        quantizable_node_count=len(quantizable_node_names),
        op_types_to_quantize=op_types_to_quantize,
        excluded_node_names=excluded_node_names,
        quantizable_matmul_node_names=quantizable_matmul_node_names,
        quantizable_node_names=quantizable_node_names,
        quantizable_node_count_by_op_type=quantizable_node_count_by_op_type,
    )


def summarize_vpcd_local_quality_policy(
    fp32_onnx_path: str | Path,
) -> VpcdLocalQualityPolicySummary:
    return summarize_vpcd_aimet_policy(
        fp32_onnx_path,
        policy_mode=LOCAL_QUALITY_POLICY_REFERENCE,
    )


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
        raise ValueError(f"Input shape {tuple(array.shape)} exceeds fixed target shape {normalized_target_shape}.")
    if tuple(array.shape) == normalized_target_shape:
        return array

    padded = np.full(normalized_target_shape, pad_value, dtype=array.dtype)
    slices = tuple(slice(0, int(dimension)) for dimension in array.shape)
    padded[slices] = array
    return padded


def calibration_records_to_fixed_input_dataset(
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


def calibration_records_to_fixed_input_batches(
    records: Sequence[CalibrationSample],
    *,
    fixed_input_shapes: dict[str, Sequence[int]] | None = None,
    pad_values: dict[str, int] | None = None,
) -> tuple[CalibrationSample, ...]:
    dataset = calibration_records_to_fixed_input_dataset(
        records,
        fixed_input_shapes=fixed_input_shapes,
        pad_values=pad_values,
    )
    input_order = list(dataset.keys())
    batch_count = len(dataset[input_order[0]]) if input_order else 0
    batches: list[CalibrationSample] = []
    for batch_index in range(batch_count):
        batches.append(
            CalibrationSample(
                inputs={
                    input_name: np.asarray(dataset[input_name][batch_index])
                    for input_name in input_order
                }
            )
        )
    return tuple(batches)


def summarize_fixed_input_calibration_dataset(dataset: dict[str, list[np.ndarray]]) -> dict[str, object]:
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


def build_vpcd_aimet_quantize_recipe(
    *,
    model_dir: str | Path,
    fp32_onnx_path: str | Path,
    calibration_source_path: str | Path,
    max_calibration_samples: int = DEFAULT_MAX_CALIBRATION_SAMPLES,
    max_generation_length: int = DEFAULT_MAX_GENERATION_LENGTH,
    ort_provider: str = DEFAULT_ORT_PROVIDER,
    fixed_input_shapes: dict[str, Sequence[int]] | None = None,
    pad_token_id: int = 1,
    param_type: str = DEFAULT_AIMET_PARAM_TYPE,
    activation_type: str = DEFAULT_AIMET_ACTIVATION_TYPE,
    quant_scheme: str = DEFAULT_AIMET_QUANT_SCHEME,
    config_file: str = DEFAULT_AIMET_CONFIG_FILE,
    policy_mode: str = DEFAULT_AIMET_POLICY_MODE,
) -> AimetQuantizeRecipe:
    records, stats = build_calibration_records(
        model_dir=Path(model_dir),
        fp32_onnx_path=Path(fp32_onnx_path),
        calibration_source_path=Path(calibration_source_path),
        max_calibration_samples=max_calibration_samples,
        max_generation_length=max_generation_length,
        ort_provider=ort_provider,
    )
    if not records:
        raise ValueError("Khong tao duoc calibration records tu file dau vao.")

    calibration_inputs = calibration_records_to_fixed_input_batches(
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
    recipe_stats["quantize_backend"] = "aimet"
    recipe_stats["param_type"] = str(param_type)
    recipe_stats["activation_type"] = str(activation_type)
    recipe_stats["quant_scheme"] = str(quant_scheme)
    recipe_stats["config_file"] = str(config_file)
    recipe_stats["policy_mode"] = str(policy_mode)
    local_quality_policy = summarize_vpcd_aimet_policy(
        fp32_onnx_path,
        policy_mode=str(policy_mode),
    )
    recipe_stats["local_quality_policy"] = {
        "preset": local_quality_policy.preset,
        "total_named_nodes": int(local_quality_policy.total_named_nodes),
        "excluded_node_count": int(local_quality_policy.excluded_node_count),
        "excluded_decoder_node_count": int(local_quality_policy.excluded_decoder_node_count),
        "excluded_lm_head_node_count": int(local_quality_policy.excluded_lm_head_node_count),
        "quantizable_matmul_node_count": int(local_quality_policy.quantizable_matmul_node_count),
        "quantizable_node_count": int(local_quality_policy.quantizable_node_count),
        "op_types_to_quantize": list(local_quality_policy.op_types_to_quantize),
        "quantizable_node_count_by_op_type": dict(local_quality_policy.quantizable_node_count_by_op_type),
    }
    recipe_stats.update(
        summarize_fixed_input_calibration_dataset(
            calibration_records_to_fixed_input_dataset(calibration_inputs)
        )
    )
    return AimetQuantizeRecipe(
        param_type=str(param_type),
        activation_type=str(activation_type),
        quant_scheme=str(quant_scheme),
        config_file=str(config_file),
        calibration_inputs=calibration_inputs,
        calibration_stats=recipe_stats,
        variant_name=build_vpcd_aimet_variant_name(
            param_type=str(param_type),
            activation_type=str(activation_type),
            quant_scheme=str(quant_scheme),
            policy_mode=str(policy_mode),
        ),
        policy_mode=str(policy_mode),
        local_quality_policy={
            "preset": local_quality_policy.preset,
            "total_named_nodes": int(local_quality_policy.total_named_nodes),
            "excluded_node_count": int(local_quality_policy.excluded_node_count),
            "excluded_decoder_node_count": int(local_quality_policy.excluded_decoder_node_count),
            "excluded_lm_head_node_count": int(local_quality_policy.excluded_lm_head_node_count),
            "quantizable_matmul_node_count": int(local_quality_policy.quantizable_matmul_node_count),
            "quantizable_node_count": int(local_quality_policy.quantizable_node_count),
            "op_types_to_quantize": list(local_quality_policy.op_types_to_quantize),
            "excluded_node_names": list(local_quality_policy.excluded_node_names),
            "quantizable_matmul_node_names": list(local_quality_policy.quantizable_matmul_node_names),
            "quantizable_node_names": list(local_quality_policy.quantizable_node_names),
            "quantizable_node_count_by_op_type": dict(local_quality_policy.quantizable_node_count_by_op_type),
        },
    )


def _resolve_repo_root() -> Path:
    return find_repo_root(__file__)


def _resolve_fixed_bundle_manifest_path(path_like: str | Path) -> Path:
    candidate = Path(path_like)
    if candidate.is_absolute():
        return candidate.resolve()
    return (_resolve_repo_root() / candidate).resolve()


def _resolve_repo_relative_path(path_like: str | Path, *, repo_root: Path | None = None) -> Path:
    candidate = Path(path_like)
    if candidate.is_absolute():
        return candidate.resolve()
    resolved_root = repo_root.resolve() if repo_root is not None else _resolve_repo_root()
    return (resolved_root / candidate).resolve()


def _resolve_vpcd_fixed_input_shapes_from_bundle(manifest_path: Path) -> tuple[dict[str, tuple[int, int]], int]:
    manifest = ModelBundleManifest.from_path(manifest_path)
    if manifest.project != "vpcd":
        raise ValueError(f"Expected a vpcd bundle manifest, got: {manifest.project}")
    shapes = resolve_vpcd_model_input_shapes(manifest.metadata)
    if shapes is None:
        raise ValueError("VPCD fixed bundle manifest does not expose fixed input shapes.")
    pad_token_id = int(manifest.metadata.get("pad_token_id", 1))
    return {
        "input_ids": tuple(int(value) for value in shapes.input_ids),
        "attention_mask": tuple(int(value) for value in shapes.attention_mask),
        "decoder_input_ids": tuple(int(value) for value in shapes.decoder_input_ids),
        "decoder_attention_mask": tuple(int(value) for value in shapes.decoder_attention_mask),
    }, pad_token_id


def _resolve_output_root(args, *, repo_root: Path | None = None) -> Path:
    return _resolve_repo_relative_path(args.output_root, repo_root=repo_root)


def _write_vpcd_aimet_quantize_report(
    *,
    report_path: Path,
    fixed_model_path: Path,
    package_dir: Path,
    qdq_reference_model_path: Path,
    variant_root: Path,
    recipe: AimetQuantizeRecipe,
    package_report: dict[str, Any],
    service_url: str,
    config_file_value: str,
    policy_manifest_path: Path | None,
) -> dict[str, Any]:
    payload = {
        "project": NAME,
        "source_strategy": "local_aimet_compile_candidate",
        "source_kind": "local_aimet",
        "packaging_kind": "aimet_dir",
        "transformation_kind": "aimet_service_export",
        "variant_name": recipe.variant_name,
        "fixed_model_path": fixed_model_path.resolve().as_posix(),
        "package_dir": package_dir.resolve().as_posix(),
        "packaging_path": package_dir.resolve().as_posix(),
        "qdq_reference_model_path": qdq_reference_model_path.resolve().as_posix(),
        "quantize_root": variant_root.resolve().as_posix(),
        "quantize_report_path": report_path.resolve().as_posix(),
        "aimet_service_url": str(service_url),
        "aimet": {
            "param_type": recipe.param_type,
            "activation_type": recipe.activation_type,
            "quant_scheme": recipe.quant_scheme,
            "config_file": str(config_file_value),
            "variant_name": recipe.variant_name,
            "policy_mode": recipe.policy_mode,
        },
        "calibration": dict(recipe.calibration_stats),
        "local_quality_policy": dict(recipe.local_quality_policy),
        "package_report": dict(package_report),
    }
    if policy_manifest_path is not None:
        payload["policy_manifest_path"] = policy_manifest_path.resolve().as_posix()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def _run_retained_aimet_pipeline(args) -> int:
    repo_root = _resolve_repo_root()
    fixed_bundle_manifest_path = _resolve_fixed_bundle_manifest_path(args.fixed_bundle_manifest)
    fixed_input_shapes, pad_token_id = _resolve_vpcd_fixed_input_shapes_from_bundle(fixed_bundle_manifest_path)
    output_root = _resolve_output_root(args, repo_root=repo_root)
    output_root.mkdir(parents=True, exist_ok=True)

    recipe = build_vpcd_aimet_quantize_recipe(
        model_dir=_resolve_repo_relative_path(args.model_dir, repo_root=repo_root),
        fp32_onnx_path=_resolve_repo_relative_path(args.fp32_onnx, repo_root=repo_root),
        calibration_source_path=_resolve_repo_relative_path(args.calibration_text, repo_root=repo_root),
        max_calibration_samples=args.max_calibration_samples,
        max_generation_length=args.max_generation_length,
        ort_provider=args.ort_provider,
        fixed_input_shapes=fixed_input_shapes,
        pad_token_id=pad_token_id,
        param_type=args.aimet_param_type,
        activation_type=args.aimet_activation_type,
        quant_scheme=args.aimet_quant_scheme,
        config_file=args.aimet_config_file,
        policy_mode=args.aimet_policy_mode,
    )

    variant_root = (output_root / str(recipe.variant_name)).resolve()
    variant_root.mkdir(parents=True, exist_ok=True)
    fixed_model_path = (variant_root / "model.fp32.fixed.onnx").resolve()
    calibration_dir = (variant_root / "calibration").resolve()
    package_dir = (variant_root / "model.option1.aimet").resolve()
    qdq_reference_model_path = (variant_root / "model.option1.qdq.onnx").resolve()
    report_path = (variant_root / "model.option1.aimet.report.json").resolve()
    quantize_report_path = (variant_root / "quantize_report.json").resolve()

    freeze_model_inputs(Path(args.fp32_onnx), fixed_model_path, fixed_input_shapes)
    write_calibration_batches(recipe.calibration_inputs, calibration_dir)

    config_file_value = str(recipe.config_file)
    policy_manifest_path: Path | None = None
    if should_write_vpcd_aimet_policy_manifest(recipe.policy_mode):
        config_payload = (
            build_attention_ffn_aimet_config()
            if recipe.policy_mode in {BROADER_ATTENTION_FFN_POLICY_REFERENCE, AGGRESSIVE_INT8_POLICY_REFERENCE}
            else build_matmul_only_aimet_config()
        )
        config_path = write_aimet_config(
            config_payload,
            variant_root / "aimet.config.json",
        )
        policy_manifest_path = write_aimet_policy_manifest(
            build_vpcd_local_quality_policy_manifest(
                variant_name=recipe.variant_name,
                policy_mode=recipe.policy_mode,
                local_quality_policy=recipe.local_quality_policy,
            ),
            variant_root / "aimet.policy.json",
        )
        config_file_value = config_path.resolve().as_posix()

    if args.dry_run:
        print(f"Project: {NAME}")
        print(f"Fixed bundle manifest: {fixed_bundle_manifest_path}")
        print(f"Output root: {output_root}")
        print(f"Variant root: {variant_root}")
        print(f"AIMET service URL: {args.aimet_service_url}")
        print(f"Variant: {recipe.variant_name}")
        return 0

    healthcheck_aimet_service(
        args.aimet_service_url,
        timeout_seconds=float(args.aimet_health_timeout_seconds),
    )

    export_payload = {
        "fp32_onnx_path": map_local_path_to_service_workspace(
            fixed_model_path,
            repo_root=repo_root,
            service_workspace_root=args.aimet_service_workspace_root,
        ),
        "calibration_dir": map_local_path_to_service_workspace(
            calibration_dir,
            repo_root=repo_root,
            service_workspace_root=args.aimet_service_workspace_root,
        ),
        "package_dir": map_local_path_to_service_workspace(
            package_dir,
            repo_root=repo_root,
            service_workspace_root=args.aimet_service_workspace_root,
        ),
        "qdq_reference_model_path": map_local_path_to_service_workspace(
            qdq_reference_model_path,
            repo_root=repo_root,
            service_workspace_root=args.aimet_service_workspace_root,
        ),
        "model_prefix": "model.option1",
        "param_type": recipe.param_type,
        "activation_type": recipe.activation_type,
        "quant_scheme": recipe.quant_scheme,
        "config_file": (
            map_local_path_to_service_workspace(
                config_file_value,
                repo_root=repo_root,
                service_workspace_root=args.aimet_service_workspace_root,
            )
            if Path(config_file_value).exists()
            else str(config_file_value)
        ),
        "policy_manifest_path": (
            map_local_path_to_service_workspace(
                policy_manifest_path,
                repo_root=repo_root,
                service_workspace_root=args.aimet_service_workspace_root,
            )
            if policy_manifest_path is not None
            else None
        ),
    }
    package_report = request_aimet_service_export(
        service_url=args.aimet_service_url,
        export_payload=export_payload,
    )
    report_path.write_text(json.dumps(package_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    quantize_report = _write_vpcd_aimet_quantize_report(
        report_path=quantize_report_path,
        fixed_model_path=fixed_model_path,
        package_dir=package_dir,
        qdq_reference_model_path=qdq_reference_model_path,
        variant_root=variant_root,
        recipe=recipe,
        package_report=package_report,
        service_url=args.aimet_service_url,
        config_file_value=config_file_value,
        policy_manifest_path=policy_manifest_path,
    )

    print(f"Project: {NAME}")
    print(f"Variant root: {variant_root}")
    print(f"Quantize report: {quantize_report_path}")
    print(f"Package ready: {bool(package_report.get('package_ready', False))}")
    print(f"Dataset fingerprint: {quantize_report['calibration'].get('dataset_fingerprint', '')}")
    return 0


def run(args) -> int:
    validate_args(args)
    return _run_retained_aimet_pipeline(args)
