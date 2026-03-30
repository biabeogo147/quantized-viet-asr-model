import os
from typing import Sequence

import onnx

from quantize.types import QuantizationPlan


def load_model_node_names(path: str | os.PathLike[str]) -> list[str]:
    model = onnx.load(os.fspath(path), load_external_data=False)
    return [node.name for node in model.graph.node if node.name]


def summarize_quantization_plan(plan: QuantizationPlan, node_names: Sequence[str]) -> str:
    preview = ", ".join(plan.nodes_to_exclude[:5]) if plan.nodes_to_exclude else "(none)"
    lines = [
        f"Preset: {plan.preset}",
        f"Runner: {plan.runner_kind}",
        f"Total named nodes: {len(node_names)}",
        f"Quantizable ops: {', '.join(plan.op_types_to_quantize)}",
        f"Excluded nodes: {len(plan.nodes_to_exclude)}",
        f"Excluded preview: {preview}",
    ]
    if plan.preset == "sd8g2_quality":
        lines.append("Quality guard: decoder va lm_head duoc giu FP32.")
    return "\n".join(lines)
