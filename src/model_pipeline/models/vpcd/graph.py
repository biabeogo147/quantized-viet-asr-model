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
        """Return the encoder MatMul nodes allowed by the quantization policy.

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


def rewrite_encoder_attention_mask_boolean_casts_for_qnn(
    source_path: str | Path,
    destination_path: str | Path,
) -> int:
    """Replace VPCD's floating boolean-mask cast with an integer comparison.

    The exported graph computes ``1.0 - attention_mask`` and then applies two
    consecutive boolean casts. Qualcomm AI Runtime rejects the resulting direct
    floating-point-to-boolean conversion. Attention-mask values are binary by
    contract, so ``bool(1.0 - mask)`` is exactly equivalent to
    ``Equal(Cast(mask, INT32), 0)``. Building the boolean condition from the integer
    mask prevents the compiler from folding consecutive casts back into the
    unsupported conversion.

    Args:
        source_path: Fixed-shape VPCD ONNX model containing the redundant casts.
        destination_path: Output path for the Qualcomm-compatible ONNX model.

    Returns:
        Number of rewritten cast sequences, which is exactly one for the canonical graph.

    Raises:
        ValueError: If the canonical encoder cast sequence is absent or ambiguous.
    """
    import onnx
    from onnx import TensorProto, helper

    source = Path(source_path)
    destination = Path(destination_path)
    model = onnx.load(source.as_posix(), load_external_data=False)
    producers = {output: node for node in model.graph.node for output in node.output}
    consumers: dict[str, list[object]] = {}
    for graph_node in model.graph.node:
        for input_name in graph_node.input:
            consumers.setdefault(input_name, []).append(graph_node)

    matching_casts = []
    for graph_node in model.graph.node:
        if graph_node.op_type != "Cast" or "/encoder/" not in graph_node.name:
            continue
        cast_target = next(
            (
                helper.get_attribute_value(attribute)
                for attribute in graph_node.attribute
                if attribute.name == "to"
            ),
            None,
        )
        producer = producers.get(graph_node.input[0])
        downstream = consumers.get(graph_node.output[0], [])
        if (
            cast_target == TensorProto.BOOL
            and producer is not None
            and producer.op_type == "Sub"
            and len(downstream) == 1
            and downstream[0].op_type == "Cast"
            and next(
                (
                    helper.get_attribute_value(attribute)
                    for attribute in downstream[0].attribute
                    if attribute.name == "to"
                ),
                None,
            )
            == TensorProto.BOOL
        ):
            matching_casts.append(graph_node)

    if len(matching_casts) != 1:
        raise ValueError(
            "Expected exactly one encoder Sub -> Cast(BOOL) -> Cast(BOOL) sequence; "
            f"found {len(matching_casts)}"
        )
    first_boolean_cast = matching_casts[0]
    subtract_node = producers[first_boolean_cast.input[0]]
    mask_float_casts = []
    for input_name in subtract_node.input:
        input_producer = producers.get(input_name)
        if input_producer is None or input_producer.op_type != "Cast":
            continue
        input_target = next(
            (
                helper.get_attribute_value(attribute)
                for attribute in input_producer.attribute
                if attribute.name == "to"
            ),
            None,
        )
        if input_target == TensorProto.FLOAT:
            mask_float_casts.append(input_producer)
    if len(mask_float_casts) != 1:
        raise ValueError(
            "Expected the encoder Sub input to contain one attention-mask Cast(FLOAT); "
            f"found {len(mask_float_casts)}"
        )
    second_boolean_cast = consumers[first_boolean_cast.output[0]][0]
    mask_source = mask_float_casts[0].input[0]
    integer_mask_output = "/model/encoder/AttentionMaskInt32_output_0"
    zero_name = "/model/encoder/AttentionMaskZeroInt32"
    existing_tensor_names = {
        name
        for graph_node in model.graph.node
        for name in (*graph_node.input, *graph_node.output)
    } | {initializer.name for initializer in model.graph.initializer}
    if integer_mask_output in existing_tensor_names or zero_name in existing_tensor_names:
        raise ValueError("QNN attention-mask rewrite tensor names already exist")
    integer_cast = helper.make_node(
        "Cast",
        [mask_source],
        [integer_mask_output],
        name="/model/encoder/AttentionMaskToInt32",
        to=TensorProto.INT32,
    )
    zero_comparison = helper.make_node(
        "Equal",
        [integer_mask_output, zero_name],
        list(second_boolean_cast.output),
        name="/model/encoder/AttentionMaskEqualsZero",
    )
    rewritten_nodes = []
    for graph_node in model.graph.node:
        if graph_node is first_boolean_cast:
            rewritten_nodes.append(integer_cast)
        elif graph_node is second_boolean_cast:
            rewritten_nodes.append(zero_comparison)
        else:
            rewritten_nodes.append(graph_node)
    model.graph.ClearField("node")
    model.graph.node.extend(rewritten_nodes)
    model.graph.initializer.append(
        helper.make_tensor(zero_name, TensorProto.INT32, [], [0])
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, destination.as_posix())
    return 1
