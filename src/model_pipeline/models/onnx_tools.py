from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence


def freeze_input_shapes(
    model_path: str | Path,
    output_path: str | Path,
    input_shapes: Mapping[str, Sequence[int]],
) -> Path:
    """Replace selected ONNX input dimensions with fixed integer shapes.

    Args:
        model_path: Source ONNX model.
        output_path: Destination for the fixed-shape model.
        input_shapes: Named input shapes that must exist in the graph.

    Returns:
        The destination model path.

    Raises:
        ValueError: If a named input is missing or has a different rank.
    """
    import onnx

    source = Path(model_path)
    model = onnx.load(source.as_posix())
    available = {value.name for value in model.graph.input}
    missing = set(input_shapes) - available
    if missing:
        raise ValueError(f"Inputs not found in ONNX graph: {sorted(missing)!r}")
    for value in model.graph.input:
        if value.name not in input_shapes:
            continue
        requested = tuple(int(dimension) for dimension in input_shapes[value.name])
        dimensions = value.type.tensor_type.shape.dim
        if len(dimensions) != len(requested):
            raise ValueError(f"Input {value.name!r} has rank {len(dimensions)}, not {len(requested)}")
        for dimension, size in zip(dimensions, requested):
            dimension.ClearField("dim_param")
            dimension.dim_value = size
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, destination.as_posix())
    return destination


def input_shapes(model_path: str | Path) -> dict[str, list[int | str]]:
    """Read static and symbolic input shapes without loading external tensors.

    Args:
        model_path: ONNX model whose input contract should be inspected.

    Returns:
        Input names mapped to integer or symbolic dimensions.
    """
    import onnx

    model = onnx.load(Path(model_path).as_posix(), load_external_data=False)
    result: dict[str, list[int | str]] = {}
    for value in model.graph.input:
        result[value.name] = [
            int(dimension.dim_value)
            if dimension.HasField("dim_value")
            else str(dimension.dim_param)
            for dimension in value.type.tensor_type.shape.dim
        ]
    return result
