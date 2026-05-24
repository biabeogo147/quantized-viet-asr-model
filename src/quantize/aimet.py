from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

import numpy as np
import onnx

from quantize.types import AimetPackageReport, CalibrationSample

DEFAULT_AIMET_SERVICE_URL = "http://127.0.0.1:18080"
DEFAULT_AIMET_SERVICE_WORKSPACE_ROOT = "/workspace"
DEFAULT_AIMET_HEALTH_TIMEOUT_SECONDS = 30.0
DEFAULT_AIMET_EXPORT_TIMEOUT_SECONDS = 3600.0


def _build_aimet_config_for_op_types(op_types: Sequence[str]) -> dict[str, Any]:
    normalized_op_types = tuple(str(op_type).strip() for op_type in op_types if str(op_type).strip())
    if not normalized_op_types:
        raise ValueError("op_types must not be empty.")
    op_type_config = {
        op_type: {
            "is_input_quantized": "True",
            "is_output_quantized": "True",
            "params": {
                "weight": {
                    "is_quantized": "True",
                }
            },
        }
        for op_type in normalized_op_types
    }
    for op_type in ("Add", "Mul", "Div", "LayerNormalization", "Softmax"):
        if op_type in op_type_config:
            op_type_config[op_type].pop("params", None)
    return {
        "defaults": {
            "ops": {},
            "params": {},
            "strict_symmetric": "False",
            "unsigned_symmetric": "True",
            "per_channel_quantization": "False",
        },
        "params": {
            "bias": {
                "is_quantized": "False",
            }
        },
        "op_type": op_type_config,
        "supergroups": [],
        "model_input": {
            "is_input_quantized": "True",
        },
        "model_output": {
            "is_output_quantized": "True",
        },
    }


def build_matmul_only_aimet_config() -> dict[str, Any]:
    return _build_aimet_config_for_op_types(("MatMul",))


def build_attention_ffn_aimet_config() -> dict[str, Any]:
    return _build_aimet_config_for_op_types(("MatMul", "Add", "Mul", "Div", "LayerNormalization"))


def write_aimet_config(config: Mapping[str, Any], output_path: str | Path) -> Path:
    resolved_output_path = Path(output_path).resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(json.dumps(dict(config), ensure_ascii=False, indent=2), encoding="utf-8")
    return resolved_output_path


def map_local_path_to_service_workspace(
    local_path: str | Path,
    *,
    repo_root: str | Path,
    service_workspace_root: str = DEFAULT_AIMET_SERVICE_WORKSPACE_ROOT,
) -> str:
    resolved_local_path = Path(local_path).resolve()
    resolved_repo_root = Path(repo_root).resolve()
    relative_path = resolved_local_path.relative_to(resolved_repo_root)
    normalized_root = str(service_workspace_root).rstrip("/")
    return f"{normalized_root}/{relative_path.as_posix()}"


def _service_json_request(
    service_url: str,
    *,
    endpoint: str,
    payload: Mapping[str, Any] | None = None,
    timeout_seconds: float,
) -> dict[str, Any]:
    normalized_base_url = str(service_url).rstrip("/")
    url = urllib_parse.urljoin(f"{normalized_base_url}/", endpoint.lstrip("/"))
    data = None
    headers: dict[str, str] = {}
    method = "GET"
    if payload is not None:
        data = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    request = urllib_request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib_request.urlopen(request, timeout=float(timeout_seconds)) as response:
            body = response.read().decode("utf-8")
    except urllib_error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"AIMET service request failed: {exc.code} {exc.reason}: {error_body}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"Could not reach AIMET service at {url}: {exc.reason}") from exc
    if not body:
        return {}
    return json.loads(body)


def healthcheck_aimet_service(
    service_url: str = DEFAULT_AIMET_SERVICE_URL,
    *,
    timeout_seconds: float = DEFAULT_AIMET_HEALTH_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    payload = _service_json_request(
        service_url,
        endpoint="/healthz",
        timeout_seconds=float(timeout_seconds),
    )
    if str(payload.get("status", "")).strip().lower() != "ok":
        raise RuntimeError(f"AIMET service healthcheck returned an unexpected payload: {payload!r}")
    return payload


def request_aimet_service_export(
    *,
    service_url: str = DEFAULT_AIMET_SERVICE_URL,
    export_payload: Mapping[str, Any],
    timeout_seconds: float = DEFAULT_AIMET_EXPORT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    return _service_json_request(
        service_url,
        endpoint="/export",
        payload=export_payload,
        timeout_seconds=float(timeout_seconds),
    )


def build_vpcd_local_quality_policy_manifest(
    *,
    variant_name: str,
    policy_mode: str,
    local_quality_policy: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "variant_name": str(variant_name),
        "policy_mode": str(policy_mode),
        "disable_op_names": list(local_quality_policy.get("excluded_node_names", [])),
        "expected_quantizable_op_types": list(local_quality_policy.get("op_types_to_quantize", [])),
        "local_quality_policy": {
            "preset": local_quality_policy.get("preset"),
            "total_named_nodes": local_quality_policy.get("total_named_nodes"),
            "excluded_node_count": local_quality_policy.get("excluded_node_count"),
            "excluded_decoder_node_count": local_quality_policy.get("excluded_decoder_node_count"),
            "excluded_lm_head_node_count": local_quality_policy.get("excluded_lm_head_node_count"),
            "quantizable_matmul_node_count": local_quality_policy.get("quantizable_matmul_node_count"),
        },
    }


def write_aimet_policy_manifest(policy_manifest: Mapping[str, Any], output_path: str | Path) -> Path:
    resolved_output_path = Path(output_path).resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(json.dumps(dict(policy_manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    return resolved_output_path


def write_calibration_batches(
    calibration_inputs: Sequence[CalibrationSample],
    output_dir: str | Path,
) -> Path:
    normalized_inputs = tuple(calibration_inputs)
    if not normalized_inputs:
        raise ValueError("calibration_inputs must not be empty.")

    resolved_output_dir = Path(output_dir).resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    input_order = tuple(normalized_inputs[0].inputs.keys())
    batch_files: list[str] = []

    for batch_index, sample in enumerate(normalized_inputs):
        current_order = tuple(sample.inputs.keys())
        if current_order != input_order:
            raise ValueError(
                "All calibration inputs must preserve a stable input ordering. "
                f"Expected {input_order}, got {current_order}."
            )
        batch_path = resolved_output_dir / f"batch-{batch_index:05d}.npz"
        np.savez_compressed(batch_path, **{name: np.asarray(sample.inputs[name]) for name in input_order})
        batch_files.append(batch_path.name)

    manifest_path = resolved_output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "input_order": list(input_order),
                "batch_files": batch_files,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


def load_calibration_batches(calibration_dir: str | Path) -> list[dict[str, np.ndarray]]:
    resolved_dir = Path(calibration_dir).resolve()
    manifest_path = resolved_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Could not resolve calibration manifest: {manifest_path}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    input_order = tuple(str(name) for name in payload["input_order"])
    batch_files = [str(name) for name in payload["batch_files"]]
    batches: list[dict[str, np.ndarray]] = []
    for batch_file in batch_files:
        batch_path = resolved_dir / batch_file
        with np.load(batch_path, allow_pickle=False) as arrays:
            batches.append({name: np.asarray(arrays[name]) for name in input_order})
    return batches


def inspect_aimet_package(package_dir: str | Path, *, qdq_reference_model_path: str | Path | None = None) -> dict[str, Any]:
    resolved_package_dir = Path(package_dir).resolve()
    if not resolved_package_dir.exists():
        raise FileNotFoundError(f"Could not resolve AIMET package dir: {resolved_package_dir}")
    if not resolved_package_dir.is_dir():
        raise ValueError(f"AIMET package path must be a directory: {resolved_package_dir}")

    onnx_files = tuple(sorted(path.name for path in resolved_package_dir.glob("*.onnx")))
    encodings_files = tuple(sorted(path.name for path in resolved_package_dir.glob("*.encodings")))
    data_files = tuple(sorted(path.name for path in resolved_package_dir.glob("*.data")))
    notes: list[str] = []
    if len(onnx_files) != 1:
        notes.append("expected_exactly_one_onnx")
    if len(encodings_files) != 1:
        notes.append("expected_exactly_one_encodings")
    package_ready = len(onnx_files) == 1 and len(encodings_files) == 1 and len(data_files) <= 1
    if len(data_files) > 1:
        notes.append("expected_at_most_one_data_file")
    if ".aimet" not in resolved_package_dir.name:
        notes.append("package_dir_name_missing_dot_aimet")

    report = AimetPackageReport(
        package_dir=resolved_package_dir.as_posix(),
        onnx_files=onnx_files,
        encodings_files=encodings_files,
        data_files=data_files,
        package_ready=package_ready,
        package_notes=tuple(notes),
        qdq_reference_model_path=(
            Path(qdq_reference_model_path).resolve().as_posix() if qdq_reference_model_path is not None else None
        ),
    )
    return {
        "package_dir": report.package_dir,
        "onnx_files": list(report.onnx_files),
        "encodings_files": list(report.encodings_files),
        "data_files": list(report.data_files),
        "package_ready": report.package_ready,
        "package_notes": list(report.package_notes),
        "qdq_reference_model_path": report.qdq_reference_model_path,
    }


def _disable_quantizers_for_ops(sim, disabled_op_names: Sequence[str]) -> dict[str, Any]:
    connected_graph = sim.connected_graph
    all_ops = connected_graph.get_all_ops()
    disabled_quantizer_count = 0
    disabled_op_count = 0
    missing_op_names: list[str] = []

    for op_name in disabled_op_names:
        op = all_ops.get(str(op_name))
        if op is None:
            missing_op_names.append(str(op_name))
            continue
        input_quantizers, output_quantizers, param_quantizers = sim.get_op_quantizers(op)
        disabled_op_count += 1
        for quantizer in [*input_quantizers, *output_quantizers, *param_quantizers.values()]:
            if quantizer is None:
                continue
            if bool(getattr(quantizer, "enabled", False)):
                quantizer.enabled = False
                disabled_quantizer_count += 1

    return {
        "disabled_op_count": int(disabled_op_count),
        "disabled_quantizer_count": int(disabled_quantizer_count),
        "missing_op_names": missing_op_names,
    }


def apply_aimet_policy(sim, policy_manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    if not policy_manifest:
        return {
            "policy_mode": "none",
            "variant_name": None,
            "disabled_op_count": 0,
            "disabled_quantizer_count": 0,
            "missing_op_names": [],
        }
    return {
        "policy_mode": str(policy_manifest.get("policy_mode") or "custom"),
        "variant_name": policy_manifest.get("variant_name"),
        **_disable_quantizers_for_ops(sim, tuple(str(name) for name in policy_manifest.get("disable_op_names", []))),
    }


def _resolve_quant_scheme(quant_scheme: str):
    from aimet_common.defs import QuantScheme

    normalized = str(quant_scheme).strip().lower()
    if normalized == "min_max":
        return QuantScheme.min_max
    if normalized == "tf_enhanced":
        return QuantScheme.post_training_tf_enhanced
    raise ValueError(f"Unsupported AIMET quant scheme: {quant_scheme!r}")


def _save_model_with_external_data(model: onnx.ModelProto, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_path = output_path.with_suffix(f"{output_path.suffix}.data")
    if data_path.exists():
        data_path.unlink()
    onnx.save_model(
        model,
        output_path.as_posix(),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_path.name,
        size_threshold=0,
    )
    if data_path.exists() and data_path.stat().st_size == 0:
        data_path.unlink()


def export_aimet_package(
    *,
    fp32_onnx_path: str | Path,
    calibration_dir: str | Path,
    package_dir: str | Path,
    qdq_reference_model_path: str | Path,
    model_prefix: str = "model.option1",
    param_type: str = "int8",
    activation_type: str = "int8",
    quant_scheme: str = "min_max",
    config_file: str = "default",
    policy_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    from aimet_onnx.quantsim import QuantizationSimModel

    resolved_fp32_path = Path(fp32_onnx_path).resolve()
    resolved_package_dir = Path(package_dir).resolve()
    resolved_qdq_reference_path = Path(qdq_reference_model_path).resolve()
    resolved_package_dir.mkdir(parents=True, exist_ok=True)
    calibration_batches = load_calibration_batches(calibration_dir)
    if not calibration_batches:
        raise ValueError("AIMET calibration batches must not be empty.")
    policy_manifest = None
    if policy_manifest_path is not None:
        resolved_policy_manifest_path = Path(policy_manifest_path).resolve()
        policy_manifest = json.loads(resolved_policy_manifest_path.read_text(encoding="utf-8"))

    model = onnx.load(resolved_fp32_path.as_posix())
    normalized_config_file = str(config_file).strip()
    sim_config_file = None if not normalized_config_file or normalized_config_file == "default" else normalized_config_file
    sim = QuantizationSimModel(
        model,
        quant_scheme=_resolve_quant_scheme(quant_scheme),
        default_param_bw=8 if str(param_type).strip().lower() == "int8" else 16,
        default_activation_bw=8 if str(activation_type).strip().lower() == "int8" else 16,
        config_file=sim_config_file,
    )
    policy_report = apply_aimet_policy(sim, policy_manifest)
    sim.compute_encodings(calibration_batches)

    export_root = resolved_package_dir.parent / f"{resolved_package_dir.name}.export"
    export_root.mkdir(parents=True, exist_ok=True)
    sim.export(export_root.as_posix(), model_prefix)

    exported_onnx = export_root / f"{model_prefix}.onnx"
    exported_encodings = export_root / f"{model_prefix}.encodings"
    exported_data = export_root / f"{model_prefix}.onnx.data"
    packaged_onnx = resolved_package_dir / f"{model_prefix}.onnx"
    packaged_encodings = resolved_package_dir / f"{model_prefix}.encodings"
    packaged_data = resolved_package_dir / f"{model_prefix}.onnx.data"

    packaged_onnx.write_bytes(exported_onnx.read_bytes())
    packaged_encodings.write_bytes(exported_encodings.read_bytes())
    if exported_data.exists():
        packaged_data.write_bytes(exported_data.read_bytes())

    qdq_model = sim.to_onnx_qdq(prequantize_constants=False)
    _save_model_with_external_data(qdq_model, resolved_qdq_reference_path)
    report = inspect_aimet_package(resolved_package_dir, qdq_reference_model_path=resolved_qdq_reference_path)
    report["policy_report"] = policy_report
    if policy_manifest is not None:
        report["policy_manifest_path"] = Path(policy_manifest_path).resolve().as_posix()
    return report
