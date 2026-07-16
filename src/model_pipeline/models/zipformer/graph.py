from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BOOLEAN_MASK_SLICE_NODES = (
    "/encoder/Slice_1",
    "/encoder/Slice_3",
    "/encoder/Slice_5",
)
BOOLEAN_MASK_UNSQUEEZE_NODES = (
    "/encoder/1/encoder/0/self_attn_weights/Unsqueeze_15",
    "/encoder/2/encoder/0/self_attn_weights/Unsqueeze_15",
    "/encoder/3/encoder/0/self_attn_weights/Unsqueeze_15",
)
ORT_DISABLED_OPTIMIZERS = ("MatMulAddFusion",)


@dataclass(frozen=True)
class ZipformerGraphContract:
    matmul_by_component: dict[str, int]
    boolean_slice_nodes: tuple[str, ...]
    boolean_unsqueeze_nodes: tuple[str, ...]


ZIPFORMER_GRAPH_CONTRACT = ZipformerGraphContract(
    matmul_by_component={"encoder": 278, "decoder": 0, "joiner": 0},
    boolean_slice_nodes=BOOLEAN_MASK_SLICE_NODES,
    boolean_unsqueeze_nodes=BOOLEAN_MASK_UNSQUEEZE_NODES,
)


def graph_matmul_count(model_path: str | Path) -> int:
    """Count MatMul nodes in an ONNX graph without loading external tensors.

    Args:
        model_path: ONNX component to inspect.

    Returns:
        The number of graph nodes whose operator type is `MatMul`.
    """
    import onnx

    model = onnx.load(Path(model_path).as_posix(), load_external_data=False)
    return sum(node.op_type == "MatMul" for node in model.graph.node)


def rewrite_boolean_mask_for_htp(model):
    """Rewrite only the six verified bool-mask boundary nodes for QNN HTP.

    Args:
        model: Loaded Zipformer encoder model to mutate in memory.

    Returns:
        The same model with UINT8 boundary casts around verified mask nodes.

    Raises:
        ValueError: If any expected Slice or Unsqueeze node is absent.
    """
    from onnx import TensorProto, helper

    slice_names = set(BOOLEAN_MASK_SLICE_NODES)
    unsqueeze_names = set(BOOLEAN_MASK_UNSQUEEZE_NODES)
    present_slices = {node.name for node in model.graph.node if node.name in slice_names}
    present_unsqueezes = {node.name for node in model.graph.node if node.name in unsqueeze_names}
    if present_slices != slice_names or present_unsqueezes != unsqueeze_names:
        raise ValueError(
            "Zipformer bool-mask graph contract mismatch: "
            f"slices={len(present_slices)}/3, unsqueezes={len(present_unsqueezes)}/3"
        )

    shared_output = "/GreaterOrEqual_output_0_u8"
    new_nodes = []
    inserted = False
    stale_value_info_names: set[str] = set()
    for node in model.graph.node:
        if node.name in slice_names and not inserted:
            new_nodes.append(
                helper.make_node(
                    "Cast",
                    ["/GreaterOrEqual_output_0"],
                    [shared_output],
                    name="/GreaterOrEqual_output_0_u8_cast",
                    to=TensorProto.UINT8,
                )
            )
            inserted = True
        if node.name in slice_names:
            node.input[0] = shared_output
            stale_value_info_names.add(node.output[0])
        if node.name in unsqueeze_names:
            original_output = node.output[0]
            temporary_output = f"{original_output}_u8"
            node.output[0] = temporary_output
            stale_value_info_names.add(temporary_output)
            new_nodes.append(node)
            new_nodes.append(
                helper.make_node(
                    "Cast",
                    [temporary_output],
                    [original_output],
                    name=f"{node.name}_cast_bool",
                    to=TensorProto.BOOL,
                )
            )
        else:
            new_nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    _strip_value_info_conflicts(model, stale_value_info_names)
    return model


def prepare_encoder_for_aihub(source_path: str | Path, output_path: str | Path) -> Path:
    """Optimize, infer shapes, and apply the verified HTP bool-mask rewrite.

    Args:
        source_path: Fixed-shape FP32 encoder source.
        output_path: Destination for the AI Hub compile input.

    Returns:
        The validated prepared encoder path.
    """
    import onnx
    import onnxruntime as ort
    from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

    source = Path(source_path).resolve()
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    optimized = destination.with_suffix(".optimized.tmp.onnx")
    try:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        options.optimized_model_filepath = optimized.as_posix()
        ort.InferenceSession(
            source.as_posix(),
            sess_options=options,
            providers=["CPUExecutionProvider"],
            disabled_optimizers=list(ORT_DISABLED_OPTIMIZERS),
        )
        inferred = SymbolicShapeInference.infer_shapes(
            onnx.load(optimized.as_posix()), auto_merge=True, guess_output_rank=True, verbose=0
        )
        _strip_value_info_conflicts(inferred)
        rewrite_boolean_mask_for_htp(inferred)
        onnx.checker.check_model(inferred, full_check=True)
        onnx.save(inferred, destination.as_posix())
    finally:
        optimized.unlink(missing_ok=True)
    return destination


def _strip_value_info_conflicts(model, extra_names: set[str] | None = None) -> None:
    """Remove inferred value-info entries that conflict with graph boundaries.

    Args:
        model: Loaded ONNX model to mutate in memory.
        extra_names: Additional intermediate names whose stale metadata should be removed.

    Returns:
        None.
    """
    names = {value.name for value in model.graph.input} | {value.name for value in model.graph.output}
    names.update(extra_names or ())
    kept = [value for value in model.graph.value_info if value.name not in names]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept)
