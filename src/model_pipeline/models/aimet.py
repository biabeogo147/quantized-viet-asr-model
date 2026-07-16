from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def build_matmul_only_aimet_config(
    *,
    select_operators_from_policy: bool = False,
    symmetric_activations: bool = False,
) -> dict[str, Any]:
    """Build signed AIMET MatMul-only quantization configuration.

    Args:
        select_operators_from_policy: Whether the service enables quantizers
            from explicit operator names instead of an AIMET op-type rule.
        symmetric_activations: Whether enabled activation quantizers use a
            signed symmetric range required by the target QNN MatMul contract.

    Returns:
        AIMET configuration with per-channel and bias quantization disabled.
    """
    config = {
        "defaults": {
            "ops": {
                **({"is_symmetric": "True"} if symmetric_activations else {}),
            },
            "params": {},
            "strict_symmetric": "False",
            "unsigned_symmetric": "False",
            "per_channel_quantization": "False",
        },
        "params": {"bias": {"is_quantized": "False"}},
        "op_type": {},
        "supergroups": [],
    }
    if not select_operators_from_policy:
        config["op_type"] = {
            "MatMul": {
                "is_input_quantized": "True",
                "is_output_quantized": "True",
                "params": {"weight": {"is_quantized": "True"}},
            }
        }
    config["model_input"] = {"is_input_quantized": "True"}
    config["model_output"] = {"is_output_quantized": "True"}
    return config


def write_aimet_calibration_inputs(
    batches: Sequence[Mapping[str, np.ndarray]],
    output_dir: str | Path,
) -> Path:
    """Persist ordered calibration inputs for the generic AIMET service.

    Args:
        batches: Non-empty sequence of consistently ordered input mappings.
        output_dir: Directory receiving compressed arrays and manifest.

    Returns:
        Path to the deterministic calibration manifest.

    Raises:
        ValueError: If inputs are empty or input ordering changes.
    """
    normalized = tuple(batches)
    if not normalized:
        raise ValueError("AIMET calibration inputs must not be empty")
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    input_order = tuple(normalized[0])
    batch_files: list[str] = []
    for index, batch in enumerate(normalized):
        if tuple(batch) != input_order:
            raise ValueError("AIMET calibration input ordering changed between batches")
        batch_path = destination / f"batch-{index:05d}.npz"
        np.savez_compressed(
            batch_path,
            **{name: np.asarray(batch[name]) for name in input_order},
        )
        batch_files.append(batch_path.name)
    manifest = destination / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"input_order": list(input_order), "batch_files": batch_files},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def load_aimet_calibration_inputs(
    calibration_dir: str | Path,
) -> list[dict[str, np.ndarray]]:
    """Load AIMET calibration inputs in manifest-defined order.

    Args:
        calibration_dir: Directory containing manifest and compressed arrays.

    Returns:
        Ordered calibration input mappings.
    """
    root = Path(calibration_dir).resolve()
    payload = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    input_order = tuple(payload["input_order"])
    batches: list[dict[str, np.ndarray]] = []
    for file_name in payload["batch_files"]:
        with np.load(root / file_name, allow_pickle=False) as arrays:
            batches.append({name: np.asarray(arrays[name]) for name in input_order})
    return batches
