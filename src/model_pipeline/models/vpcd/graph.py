from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VpcdMatmulInventory:
    encoder: tuple[str, ...]
    decoder: tuple[str, ...]
    lm_head: tuple[str, ...]
    other: tuple[str, ...]

    @property
    def total(self) -> int:
        """Return the total number of classified MatMul nodes.

        Returns:
            Sum of encoder, decoder, language-head, and other nodes.
        """
        return len(self.encoder) + len(self.decoder) + len(self.lm_head) + len(self.other)

    @property
    def counts(self) -> dict[str, int]:
        """Summarize the MatMul inventory by canonical graph scope.

        Returns:
            Per-scope counts plus the total count.
        """
        return {
            "encoder": len(self.encoder),
            "decoder": len(self.decoder),
            "lm_head": len(self.lm_head),
            "other": len(self.other),
            "total": self.total,
        }

    @property
    def quantized_names(self) -> tuple[str, ...]:
        """Return the only MatMul nodes allowed by the production policy.

        Returns:
            Encoder MatMul node names in graph order.
        """
        return self.encoder


def classify_vpcd_matmul_name(node_name: str) -> str:
    """Classify a VPCD MatMul node name into its model scope.

    Args:
        node_name: ONNX graph node name.

    Returns:
        One of `encoder`, `decoder`, `lm_head`, or `other`.
    """
    if node_name == "/lm_head/MatMul":
        return "lm_head"
    if "/encoder/" in node_name:
        return "encoder"
    if "/decoder/" in node_name:
        return "decoder"
    return "other"


def inspect_vpcd_matmuls(model_path: str | Path) -> VpcdMatmulInventory:
    """Inventory VPCD MatMul nodes without loading external tensor data.

    Args:
        model_path: ONNX model to inspect.

    Returns:
        Immutable node-name groups for all canonical scopes.
    """
    import onnx

    model = onnx.load(Path(model_path).as_posix(), load_external_data=False)
    groups: dict[str, list[str]] = {"encoder": [], "decoder": [], "lm_head": [], "other": []}
    for node in model.graph.node:
        if node.op_type == "MatMul":
            groups[classify_vpcd_matmul_name(str(node.name))].append(str(node.name))
    return VpcdMatmulInventory(**{name: tuple(values) for name, values in groups.items()})
