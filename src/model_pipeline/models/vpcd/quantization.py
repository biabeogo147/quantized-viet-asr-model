from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from model_pipeline.models.vpcd.graph import inspect_vpcd_matmuls


VPCD_FIXED_INPUT_SHAPES: dict[str, tuple[int, int]] = {
    "input_ids": (1, 384),
    "attention_mask": (1, 384),
    "decoder_input_ids": (1, 64),
    "decoder_attention_mask": (1, 64),
}


@dataclass(frozen=True)
class CalibrationBatch:
    inputs: Mapping[str, np.ndarray]


def build_encoder_matmul_policy(
    model_path: str | Path,
    *,
    require_canonical_counts: bool = True,
) -> dict[str, Any]:
    """Build the allow/disable policy for encoder-only MatMul quantization.

    Args:
        model_path: Fixed-shape VPCD ONNX model.
        require_canonical_counts: Whether to require the observed 96/168/1 inventory.

    Returns:
        Named encoder allowlist, non-encoder disable list, and coverage evidence.

    Raises:
        ValueError: If canonical coverage is required but the graph inventory differs.
    """
    inventory = inspect_vpcd_matmuls(model_path)
    counts = inventory.counts
    if require_canonical_counts and (
        counts["encoder"], counts["decoder"], counts["lm_head"], counts["other"]
    ) != (96, 168, 1, 0):
        raise ValueError(
            "VPCD graph does not match canonical MatMul coverage 96/168/1; "
            f"observed {counts['encoder']}/{counts['decoder']}/{counts['lm_head']}"
            f" with {counts['other']} unclassified"
        )
    disabled = (*inventory.decoder, *inventory.lm_head, *inventory.other)
    return {
        "schema_version": 1,
        "quantization_scope": "encoder-matmul",
        "quantize_op_types": ["MatMul"],
        "quantize_op_names": list(inventory.encoder),
        "disable_op_names": list(disabled),
        "quantizer_selection": "operator-name-allowlist",
        "symmetric_activation_encodings": True,
        "coverage": {"quantized": len(inventory.encoder), "total_matmul": inventory.total},
    }


def inspect_encoder_matmul_aimet_encodings(
    encodings_path: str | Path,
) -> dict[str, Any]:
    """Inspect VPCD AIMET encoding precision, symmetry, and graph scope.

    Args:
        encodings_path: AIMET JSON encoding file exported with the VPCD model.

    Returns:
        Encoding counts plus activation, parameter, and encoder-only scope checks.
    """
    payload = json.loads(Path(encodings_path).read_text(encoding="utf-8"))
    activations = list(payload.get("activation_encodings") or ())
    parameters = list(payload.get("param_encodings") or ())
    non_encoder_names = sorted(
        str(encoding.get("name", ""))
        for encoding in (*activations, *parameters)
        if "/decoder/" in str(encoding.get("name", ""))
        or ".decoder." in str(encoding.get("name", ""))
        or "lm_head" in str(encoding.get("name", ""))
    )
    activation_contract = all(
        int(encoding.get("bw", -1)) == 16
        and str(encoding.get("dtype", "")).upper() == "INT"
        and bool(encoding.get("is_sym"))
        and list(encoding.get("offset") or ()) == [-32768.0]
        for encoding in activations
    )
    parameter_contract = all(
        int(encoding.get("bw", -1)) == 8
        and str(encoding.get("dtype", "")).upper() == "INT"
        for encoding in parameters
    )
    return {
        "activation_count": len(activations),
        "parameter_count": len(parameters),
        "activation_contract": bool(activations) and activation_contract,
        "parameter_contract": bool(parameters) and parameter_contract,
        "non_encoder_names": non_encoder_names,
    }


def pad_calibration_batch(batch: CalibrationBatch, *, pad_token_id: int) -> CalibrationBatch:
    """Pad one calibration prefix to the fixed source and decoder shapes.

    Args:
        batch: Ordered source and decoder prefix arrays.
        pad_token_id: Model token ID used for sequence padding.

    Returns:
        A batch whose four inputs match source length 384 and decoder length 64.

    Raises:
        ValueError: If input ordering, rank, batch size, or sequence length is invalid.
    """
    expected_order = tuple(VPCD_FIXED_INPUT_SHAPES)
    if tuple(batch.inputs) != expected_order:
        raise ValueError(f"VPCD calibration input order must be {expected_order!r}")
    pad_values = {
        "input_ids": int(pad_token_id),
        "attention_mask": 0,
        "decoder_input_ids": int(pad_token_id),
        "decoder_attention_mask": 0,
    }
    return CalibrationBatch(
        inputs={
            name: _pad_array(np.asarray(batch.inputs[name]), shape, pad_values[name])
            for name, shape in VPCD_FIXED_INPUT_SHAPES.items()
        }
    )


def _pad_array(values: np.ndarray, target_shape: tuple[int, int], pad_value: int) -> np.ndarray:
    """Right-pad a batch-one sequence array to an exact target shape.

    Args:
        values: Rank-two batch-one source array.
        target_shape: Required batch and sequence dimensions.
        pad_value: Scalar value used for unfilled positions.

    Returns:
        A new array preserving input dtype and prefix values.

    Raises:
        ValueError: If rank, batch size, or sequence length violates the target.
    """
    if values.ndim != 2 or values.shape[0] != 1:
        raise ValueError(f"Expected rank-2 batch-one input, got {values.shape}")
    if any(actual > target for actual, target in zip(values.shape, target_shape)):
        raise ValueError(f"Input {values.shape} exceeds target {target_shape}")
    result = np.full(target_shape, pad_value, dtype=values.dtype)
    result[:, : values.shape[1]] = values
    return result
