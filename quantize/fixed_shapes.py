from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import onnx


def freeze_model_inputs(model_path: str | Path, output_path: str | Path, input_shapes: Mapping[str, Sequence[int]]) -> Path:
    model = onnx.load(Path(model_path).as_posix())
    for value in model.graph.input:
        if value.name not in input_shapes:
            continue
        dims = list(input_shapes[value.name])
        shape = value.type.tensor_type.shape.dim
        if len(shape) != len(dims):
            raise ValueError(f'Input {value.name} expects {len(shape)} dims but got {len(dims)}')
        for dim_proto, dim_value in zip(shape, dims):
            dim_proto.ClearField('dim_param')
            dim_proto.dim_value = int(dim_value)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output.as_posix())
    return output
