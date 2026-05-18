from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class VpcdLocalQualityPolicySummary:
    preset: str
    total_named_nodes: int
    excluded_node_count: int
    excluded_decoder_node_count: int
    excluded_lm_head_node_count: int
    quantizable_matmul_node_count: int
    op_types_to_quantize: tuple[str, ...]
    excluded_node_names: tuple[str, ...]
    quantizable_matmul_node_names: tuple[str, ...]


@dataclass(frozen=True)
class AimetQuantizeRecipe:
    param_type: str
    activation_type: str
    quant_scheme: str
    config_file: str
    calibration_inputs: tuple[CalibrationSample, ...]
    calibration_stats: dict[str, Any]
    variant_name: str = "w8a8_min_max_default"
    policy_mode: str = "broad_default"
    local_quality_policy: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LocalQdqCompatibilityReport:
    model_path: str
    opsets: dict[str, int]
    qdq_domains: dict[str, int]
    ms_qdq_node_count: int
    main_qdq_node_count: int
    uses_uint16_qdq: bool
    uses_int16_qdq: bool
    uses_quantized_weight_initializers: bool
    quantized_weight_initializer_count: int
    initializer_dtypes: dict[str, int]
    packaging_kind: str
    packaging_ready: bool
    packaging_notes: tuple[str, ...]
    aihub_compile_readiness: str
    readiness_flags: tuple[str, ...]


@dataclass(frozen=True)
class AimetPackageReport:
    package_dir: str
    onnx_files: tuple[str, ...]
    encodings_files: tuple[str, ...]
    data_files: tuple[str, ...]
    package_ready: bool
    package_notes: tuple[str, ...]
    qdq_reference_model_path: str | None
