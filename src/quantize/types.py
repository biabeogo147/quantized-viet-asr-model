from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CalibrationSample:
    inputs: dict[str, np.ndarray]


@dataclass(frozen=True)
class PresetSpec:
    name: str
    runner_kind: str
    op_types_to_quantize: tuple[str, ...]
    exclusion_patterns: tuple[str, ...]
    calibration_method: str
    percentile: float
    per_channel: bool
    activation_type: str
    weight_type: str


@dataclass(frozen=True)
class QuantizationPlan:
    preset: str
    runner_kind: str
    op_types_to_quantize: tuple[str, ...]
    exclusion_patterns: tuple[str, ...]
    nodes_to_exclude: tuple[str, ...]
    calibration_method: str
    percentile: float
    per_channel: bool
    activation_type: str
    weight_type: str


@dataclass(frozen=True)
class AiHubQuantizeRecipe:
    preset: str
    activation_type: str
    weight_type: str
    activations_dtype_name: str
    weights_dtype_name: str
    calibration_dataset: dict[str, list[np.ndarray]]
    calibration_stats: dict[str, Any]
