import os

import onnx


def load_model_node_names(path: str | os.PathLike[str]) -> list[str]:
    model = onnx.load(os.fspath(path), load_external_data=False)
    return [node.name for node in model.graph.node if node.name]
