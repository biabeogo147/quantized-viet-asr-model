from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import onnx


@dataclass(frozen=True)
class QdqGraphEvidence:
    """Record observed MatMul and explicit QDQ placement for a benchmark model."""

    total_matmul: int
    selected_matmul: int
    quantized_matmul: int
    quantize_linear: int
    dequantize_linear: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize graph evidence for machine-readable benchmark output.

        Returns:
            JSON-compatible QDQ graph counts.
        """
        return asdict(self)


def inspect_benchmark_qdq(
    model_path: str | Path,
    *,
    selected_matmul_names: Sequence[str],
    expected_total_matmul: int,
) -> QdqGraphEvidence:
    """Require explicit QDQ only on policy-selected MatMul operations.

    Args:
        model_path: Benchmark-only explicit-QDQ ONNX model.
        selected_matmul_names: Exact MatMul node names selected by AIMET policy.
        expected_total_matmul: Canonical total MatMul inventory for the model.

    Returns:
        Observed MatMul and QDQ operator counts.

    Raises:
        ValueError: If graph inventory or quantized MatMul scope differs from policy.
    """
    model = onnx.load(Path(model_path).as_posix(), load_external_data=False)
    nodes = tuple(model.graph.node)
    matmuls = {node.name: node for node in nodes if node.op_type == "MatMul"}
    if len(matmuls) != expected_total_matmul:
        raise ValueError(
            f"QDQ graph has {len(matmuls)} MatMul operations; expected {expected_total_matmul}"
        )
    selected = {str(name) for name in selected_matmul_names}
    missing = selected - set(matmuls)
    if missing:
        raise ValueError(f"QDQ graph is missing policy MatMul operations: {sorted(missing)!r}")
    producer_by_output = {
        output: node
        for node in nodes
        for output in node.output
    }
    consumers_by_input: dict[str, list[object]] = {}
    for node in nodes:
        for input_name in node.input:
            consumers_by_input.setdefault(input_name, []).append(node)
    quantized = {
        name
        for name, node in matmuls.items()
        if any(
            producer_by_output.get(input_name) is not None
            and producer_by_output[input_name].op_type == "DequantizeLinear"
            for input_name in node.input
        )
        or any(
            consumer.op_type == "QuantizeLinear"
            for output_name in node.output
            for consumer in consumers_by_input.get(output_name, ())
        )
    }
    outside = quantized - selected
    not_quantized = selected - quantized
    if outside:
        raise ValueError(f"QDQ graph quantizes MatMul outside the policy: {sorted(outside)!r}")
    if not_quantized:
        raise ValueError(f"Policy MatMul operations lack explicit QDQ: {sorted(not_quantized)!r}")
    quantize_linear = sum(node.op_type == "QuantizeLinear" for node in nodes)
    dequantize_linear = sum(node.op_type == "DequantizeLinear" for node in nodes)
    if not quantize_linear and not dequantize_linear:
        raise ValueError("Benchmark model contains no explicit QDQ operators")
    return QdqGraphEvidence(
        total_matmul=len(matmuls),
        selected_matmul=len(selected),
        quantized_matmul=len(quantized),
        quantize_linear=quantize_linear,
        dequantize_linear=dequantize_linear,
    )
