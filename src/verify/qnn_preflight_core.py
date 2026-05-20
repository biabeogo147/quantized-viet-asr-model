from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto

from model_bundle.manifest import ModelBundleManifest


REQUIRED_VPCD_INPUTS = ("input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask")


def _dtype_name(data_type: int) -> str:
    return TensorProto.DataType.Name(data_type)


def _input_shape(value: Any) -> tuple[list[int | str], bool]:
    dims: list[int | str] = []
    symbolic = False
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        elif dim.dim_param:
            dims.append(str(dim.dim_param))
            symbolic = True
        else:
            dims.append("?")
            symbolic = True
    return dims, symbolic


def inspect_onnx_for_qnn_qdq(model_path: str | Path) -> dict[str, Any]:
    model = onnx.load(str(model_path), load_external_data=False)
    op_counts = Counter(node.op_type for node in model.graph.node)
    initializer_dtypes = Counter(_dtype_name(initializer.data_type) for initializer in model.graph.initializer)

    inputs: dict[str, list[int | str]] = {}
    symbolic_inputs: list[str] = []
    for value in model.graph.input:
        shape, symbolic = _input_shape(value)
        inputs[value.name] = shape
        if symbolic:
            symbolic_inputs.append(value.name)

    return {
        "op_counts": dict(op_counts),
        "initializer_dtypes": dict(initializer_dtypes),
        "inputs": inputs,
        "symbolic_inputs": symbolic_inputs,
    }


def _check_manifest_quantization(manifest: ModelBundleManifest) -> dict[str, Any]:
    errors: list[str] = []
    metadata = manifest.metadata
    quantization = metadata.get("quantization")
    if not isinstance(quantization, dict):
        errors.append("metadata.quantization is missing")
        quantization = {}

    expected = {
        "format": "QDQ",
        "activation_type": "quint16",
        "weight_type": "quint8",
    }
    for key, expected_value in expected.items():
        actual = quantization.get(key)
        if actual != expected_value:
            errors.append(f"metadata.quantization.{key} must be {expected_value!r}, got {actual!r}")

    if quantization.get("fixed_shapes") is not True:
        errors.append("metadata.quantization.fixed_shapes must be true")

    return {
        "passed": not errors,
        "errors": errors,
        "quantization": quantization,
    }


def _check_fixed_input_shapes(manifest: ModelBundleManifest, graph_report: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    metadata = manifest.metadata
    qnn_readiness = metadata.get("qnn_readiness")
    if not isinstance(qnn_readiness, dict):
        errors.append("metadata.qnn_readiness is missing")
        qnn_readiness = {}

    expected_readiness = {
        "target_backend": "qnn_htp",
        "model_session_candidate": True,
        "tokenizer_policy": "cpu_only_first_slice",
        "requires_fixed_shapes": True,
        "fixed_shapes_ready": True,
    }
    for key, expected_value in expected_readiness.items():
        actual = qnn_readiness.get(key)
        if actual != expected_value:
            errors.append(f"metadata.qnn_readiness.{key} must be {expected_value!r}, got {actual!r}")

    fixed_input_shapes = metadata.get("fixed_input_shapes")
    model_shapes = fixed_input_shapes.get("model") if isinstance(fixed_input_shapes, dict) else None
    if not isinstance(model_shapes, dict):
        errors.append("metadata.fixed_input_shapes.model is missing")
        model_shapes = {}

    graph_inputs = graph_report.get("inputs", {})
    symbolic_inputs = [name for name in graph_report.get("symbolic_inputs", []) if name in REQUIRED_VPCD_INPUTS]
    if symbolic_inputs:
        errors.append(f"VPCD model inputs must be fixed, symbolic inputs: {symbolic_inputs}")

    for name in REQUIRED_VPCD_INPUTS:
        expected_shape = model_shapes.get(name)
        graph_shape = graph_inputs.get(name)
        if expected_shape is None:
            errors.append(f"metadata.fixed_input_shapes.model.{name} is missing")
            continue
        if graph_shape is None:
            errors.append(f"ONNX graph input {name} is missing")
            continue
        if list(expected_shape) != list(graph_shape):
            errors.append(f"Input shape mismatch for {name}: manifest {expected_shape}, graph {graph_shape}")

    return {
        "passed": not errors,
        "errors": errors,
        "fixed_input_shapes": fixed_input_shapes,
        "graph_inputs": graph_inputs,
        "symbolic_inputs": symbolic_inputs,
        "qnn_readiness": qnn_readiness,
    }


def _check_onnx_qdq_graph(graph_report: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    op_counts = graph_report.get("op_counts", {})
    initializer_dtypes = graph_report.get("initializer_dtypes", {})
    if int(op_counts.get("QuantizeLinear", 0)) <= 0:
        errors.append("ONNX graph has no QuantizeLinear nodes")
    if int(op_counts.get("DequantizeLinear", 0)) <= 0:
        errors.append("ONNX graph has no DequantizeLinear nodes")
    if int(initializer_dtypes.get("UINT16", 0)) <= 0:
        errors.append("ONNX graph has no UINT16 initializers")
    if int(initializer_dtypes.get("UINT8", 0)) <= 0:
        errors.append("ONNX graph has no UINT8 initializers")

    return {
        "passed": not errors,
        "errors": errors,
        "op_counts": op_counts,
        "initializer_dtypes": initializer_dtypes,
    }


def verify_qnn_preflight(*, project: str, bundle_dir: str | Path) -> dict[str, Any]:
    if project != "vpcd":
        raise ValueError(f"Unsupported QNN preflight project: {project}")

    bundle_path = Path(bundle_dir)
    manifest_path = bundle_path / "bundle_manifest.json"
    manifest = ModelBundleManifest.from_path(manifest_path)
    if manifest.project != project:
        raise ValueError(f"Manifest project {manifest.project!r} does not match requested project {project!r}")

    model_artifact = manifest.artifacts.get("model")
    if not model_artifact:
        raise ValueError("VPCD manifest is missing artifacts.model")

    graph_report = inspect_onnx_for_qnn_qdq(bundle_path / model_artifact)
    checks = {
        "manifest_quantization": _check_manifest_quantization(manifest),
        "fixed_input_shapes": _check_fixed_input_shapes(manifest, graph_report),
        "onnx_qdq_graph": _check_onnx_qdq_graph(graph_report),
    }
    return {
        "project": project,
        "bundle_dir": str(bundle_path),
        "model_artifact": model_artifact,
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
    }
