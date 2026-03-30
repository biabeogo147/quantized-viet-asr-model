from dataclasses import dataclass

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

    @property
    def quantized_node_count(self) -> int:
        return 0
