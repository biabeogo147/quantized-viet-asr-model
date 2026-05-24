from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CalibrationSample:
    inputs: dict[str, np.ndarray]


@dataclass(frozen=True)
class QuantizationPlan:
    preset: str
    op_types_to_quantize: tuple[str, ...]
    nodes_to_exclude: tuple[str, ...]
    per_channel: bool
    activation_type: str
    weight_type: str


@dataclass(frozen=True)
class VpcdLocalQualityPolicySummary:
    preset: str
    total_named_nodes: int
    excluded_node_count: int
    excluded_decoder_node_count: int
    excluded_lm_head_node_count: int
    quantizable_matmul_node_count: int
    quantizable_node_count: int
    op_types_to_quantize: tuple[str, ...]
    excluded_node_names: tuple[str, ...]
    quantizable_matmul_node_names: tuple[str, ...]
    quantizable_node_names: tuple[str, ...]
    quantizable_node_count_by_op_type: dict[str, int]


@dataclass(frozen=True)
class AimetQuantizeRecipe:
    param_type: str
    activation_type: str
    quant_scheme: str
    config_file: str
    calibration_inputs: tuple[CalibrationSample, ...]
    calibration_stats: dict[str, Any]
    variant_name: str
    policy_mode: str
    local_quality_policy: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AimetPackageReport:
    package_dir: str
    onnx_files: tuple[str, ...]
    encodings_files: tuple[str, ...]
    data_files: tuple[str, ...]
    package_ready: bool
    package_notes: tuple[str, ...]
    qdq_reference_model_path: str | None
