from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from model_pipeline.models.vpcd.graph import inspect_vpcd_matmuls


A4_INPUT_SHAPES: dict[str, tuple[int, int]] = {
    "input_ids": (1, 384),
    "attention_mask": (1, 384),
    "decoder_input_ids": (1, 64),
    "decoder_attention_mask": (1, 64),
}


@dataclass(frozen=True)
class CalibrationBatch:
    inputs: Mapping[str, np.ndarray]


def build_matmul_only_aimet_config() -> dict[str, Any]:
    """Build the canonical AIMET W8A16 MatMul-only configuration.

    Returns:
        AIMET configuration fields with per-channel and bias quantization disabled.
    """
    return {
        "defaults": {
            "ops": {},
            "params": {},
            "strict_symmetric": "False",
            "unsigned_symmetric": "True",
            "per_channel_quantization": "False",
        },
        "params": {"bias": {"is_quantized": "False"}},
        "op_type": {
            "MatMul": {
                "is_input_quantized": "True",
                "is_output_quantized": "True",
                "params": {"weight": {"is_quantized": "True"}},
            }
        },
        "supergroups": [],
        "model_input": {"is_input_quantized": "True"},
        "model_output": {"is_output_quantized": "True"},
    }


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
        "coverage": {"quantized": len(inventory.encoder), "total_matmul": inventory.total},
    }


def pad_calibration_batch(batch: CalibrationBatch, *, pad_token_id: int) -> CalibrationBatch:
    """Pad one calibration prefix to the canonical A4 input shapes.

    Args:
        batch: Ordered source and decoder prefix arrays.
        pad_token_id: Model token ID used for sequence padding.

    Returns:
        A batch whose four inputs exactly match the A4 shapes.

    Raises:
        ValueError: If input ordering, rank, batch size, or sequence length is invalid.
    """
    expected_order = tuple(A4_INPUT_SHAPES)
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
            for name, shape in A4_INPUT_SHAPES.items()
        }
    )


def write_calibration_batches(
    batches: Sequence[CalibrationBatch], output_dir: str | Path
) -> Path:
    """Persist ordered calibration batches and their deterministic manifest.

    Args:
        batches: Non-empty sequence of identically ordered calibration batches.
        output_dir: Directory receiving compressed arrays and the manifest.

    Returns:
        Path to the calibration manifest.

    Raises:
        ValueError: If batches are empty or input ordering changes.
    """
    normalized = tuple(batches)
    if not normalized:
        raise ValueError("Calibration batches must not be empty")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    input_order = tuple(normalized[0].inputs)
    batch_files: list[str] = []
    for index, batch in enumerate(normalized):
        if tuple(batch.inputs) != input_order:
            raise ValueError("Calibration input ordering changed between batches")
        batch_path = output / f"batch-{index:05d}.npz"
        np.savez_compressed(batch_path, **{name: np.asarray(batch.inputs[name]) for name in input_order})
        batch_files.append(batch_path.name)
    manifest = output / "manifest.json"
    manifest.write_text(
        json.dumps({"input_order": list(input_order), "batch_files": batch_files}, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_calibration_batches(calibration_dir: str | Path) -> list[dict[str, np.ndarray]]:
    """Restore calibration arrays in manifest-defined input order.

    Args:
        calibration_dir: Directory containing `manifest.json` and batch archives.

    Returns:
        Ordered input mappings suitable for AIMET encoding computation.
    """
    root = Path(calibration_dir).resolve()
    payload = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    input_order = tuple(payload["input_order"])
    batches: list[dict[str, np.ndarray]] = []
    for file_name in payload["batch_files"]:
        with np.load(root / file_name, allow_pickle=False) as arrays:
            batches.append({name: np.asarray(arrays[name]) for name in input_order})
    return batches


def quantize_with_aimet(
    *,
    fp32_model_path: str | Path,
    calibration_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    policy: Mapping[str, Any],
) -> dict[str, Path]:
    """Export the single supported AIMET W8A16 encoder-MatMul package.

    Args:
        fp32_model_path: Fixed-shape FP32 VPCD model.
        calibration_dir: Directory containing serialized calibration batches.
        output_dir: Directory receiving the AIMET model package.
        config_path: Canonical MatMul-only AIMET configuration.
        policy: Encoder allowlist and non-encoder disable policy.

    Returns:
        Exported model, encodings, and optional external-data paths.

    Raises:
        ValueError: If calibration is empty or policy nodes are missing.
    """
    import onnx
    from aimet_common.defs import QuantScheme
    from aimet_onnx.quantsim import QuantizationSimModel

    model_path = Path(fp32_model_path).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    batches = load_calibration_batches(calibration_dir)
    if not batches:
        raise ValueError("AIMET calibration batches must not be empty")
    sim = QuantizationSimModel(
        onnx.load(model_path.as_posix()),
        quant_scheme=QuantScheme.min_max,
        default_param_bw=8,
        default_activation_bw=16,
        config_file=Path(config_path).resolve().as_posix(),
    )
    disabled = _disable_ops(sim, policy.get("disable_op_names", ()))
    if disabled["missing_op_names"]:
        raise ValueError(f"AIMET policy nodes were not found: {disabled['missing_op_names']!r}")
    sim.compute_encodings(batches)
    sim.export(destination.as_posix(), "model")
    outputs = {
        "model": destination / "model.onnx",
        "encodings": destination / "model.encodings",
    }
    external_data = destination / "model.onnx.data"
    if external_data.is_file():
        outputs["external_data"] = external_data
    return outputs


def _disable_ops(sim, op_names: Sequence[str]) -> dict[str, Any]:
    """Disable every enabled quantizer associated with selected graph operations.

    Args:
        sim: AIMET quantization simulation model.
        op_names: Connected-graph operation names that must remain unquantized.

    Returns:
        Disabled quantizer count and any operation names not found.
    """
    all_ops = sim.connected_graph.get_all_ops()
    missing: list[str] = []
    disabled = 0
    for name in op_names:
        op = all_ops.get(str(name))
        if op is None:
            missing.append(str(name))
            continue
        inputs, outputs, parameters = sim.get_op_quantizers(op)
        for quantizer in (*inputs, *outputs, *parameters.values()):
            if quantizer is not None and bool(getattr(quantizer, "enabled", False)):
                quantizer.enabled = False
                disabled += 1
    return {"disabled_quantizer_count": disabled, "missing_op_names": missing}


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
