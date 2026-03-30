import os
from typing import Sequence

import onnx

from quantize.types import QuantizationPlan


def load_model_node_names(path: str | os.PathLike[str]) -> list[str]:
    model = onnx.load(os.fspath(path), load_external_data=False)
    return [node.name for node in model.graph.node if node.name]


def summarize_quantization_plan(plan: QuantizationPlan, node_names: Sequence[str]) -> str:
    preview = ", ".join(plan.nodes_to_exclude[:5]) if plan.nodes_to_exclude else "(none)"
    quantizable_ops = ", ".join(plan.op_types_to_quantize) if plan.op_types_to_quantize else "(n/a)"
    lines = [
        f"Preset: {plan.preset}",
        f"Runner: {plan.runner_kind}",
        f"Total named nodes: {len(node_names)}",
        f"Quantizable ops: {quantizable_ops}",
        f"Quant types: activation={plan.activation_type}, weight={plan.weight_type}",
        f"Excluded nodes: {len(plan.nodes_to_exclude)}",
        f"Excluded preview: {preview}",
    ]
    if plan.preset == "sd8g2_quality":
        lines.append("Quality guard: QNN-targeted PTQ + QDQ voi QUInt16 activations, QUInt8 weights, va decoder + lm_head duoc giu FP32.")
    if plan.preset == "sd8g2_balanced":
        lines.append("Balanced mode: QNN-targeted PTQ + QDQ voi QUInt16 activations va QUInt8 weights.")
    return "\n".join(lines)
