from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import onnx

from model_pipeline.core import ArtifactSpec, sha256_file


@dataclass(frozen=True)
class CompiledModelContract:
    """Declare the downloaded ONNX graph contract expected from AI Hub."""

    artifact: ArtifactSpec
    input_dtypes: Mapping[str, str]
    output_dtypes: Mapping[str, str]
    requires_int64_to_int32: bool = False


@dataclass(frozen=True)
class CompiledModelEvidence:
    """Record graph and runtime facts observed in a downloaded ONNX model."""

    checksum: str
    has_ep_context: bool
    input_dtypes: Mapping[str, str]
    output_dtypes: Mapping[str, str]
    execution_target: str
    quantization_scope: str


def validate_compiled_model(
    model_path: str | Path,
    contract: CompiledModelContract,
) -> CompiledModelEvidence:
    """Validate EPContext, I/O dtypes, and artifact target metadata.

    Args:
        model_path: Downloaded primary ONNX model file.
        contract: Expected artifact identity and I/O dtype contract.

    Returns:
        Checksum and observed graph/runtime evidence.

    Raises:
        ValueError: If EPContext, dtype transformation, or target metadata differs.
    """
    path = Path(model_path)
    model = onnx.load(path, load_external_data=False)
    has_ep_context = any(node.op_type == "EPContext" for node in model.graph.node)
    if not has_ep_context:
        raise ValueError("Downloaded model does not contain an EPContext node")
    if contract.artifact.compilation.compiler != "aihub" or contract.artifact.compilation.target != "qnn-htp":
        raise ValueError("Downloaded model artifact does not declare the qnn-htp execution target")

    input_dtypes = _value_info_dtypes(model.graph.input)
    output_dtypes = _value_info_dtypes(model.graph.output)
    if contract.requires_int64_to_int32 and any(dtype == "int64" for dtype in input_dtypes.values()):
        raise ValueError("Downloaded model failed the required int64-to-int32 I/O transform")
    _require_dtypes("input", input_dtypes, contract.input_dtypes)
    _require_dtypes("output", output_dtypes, contract.output_dtypes)
    return CompiledModelEvidence(
        checksum=sha256_file(path),
        has_ep_context=has_ep_context,
        input_dtypes=input_dtypes,
        output_dtypes=output_dtypes,
        execution_target=contract.artifact.compilation.target,
        quantization_scope=contract.artifact.quantization.scope,
    )


def _value_info_dtypes(values: object) -> dict[str, str]:
    """Extract normalized NumPy dtype names from ONNX value information.

    Args:
        values: Iterable of ONNX graph input or output value information.

    Returns:
        Tensor names mapped to normalized dtype names.
    """
    result: dict[str, str] = {}
    for value in values:
        element_type = value.type.tensor_type.elem_type
        result[value.name] = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(element_type)).name
    return result


def _require_dtypes(label: str, actual: Mapping[str, str], expected: Mapping[str, str]) -> None:
    """Require selected graph values to match expected normalized dtypes.

    Args:
        label: Human-readable graph section used in validation errors.
        actual: Observed tensor names and dtype names.
        expected: Required tensor names and dtype names.

    Returns:
        None.

    Raises:
        ValueError: If a required tensor is missing or has a different dtype.
    """
    mismatches = {
        name: {"expected": dtype, "actual": actual.get(name)}
        for name, dtype in expected.items()
        if actual.get(name) != dtype
    }
    if mismatches:
        raise ValueError(f"Compiled model {label} dtype mismatch: {mismatches!r}")
