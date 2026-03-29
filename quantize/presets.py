import fnmatch
from collections import OrderedDict
from typing import Sequence

from quantize.types import PresetSpec, QuantizationPlan


_PRESET_SPECS: "OrderedDict[str, PresetSpec]" = OrderedDict(
    [
        (
            "sd8g2_quality",
            PresetSpec(
                name="sd8g2_quality",
                runner_kind="static",
                op_types_to_quantize=("MatMul",),
                exclusion_patterns=(
                    "*/self_attn/MatMul",
                    "*/self_attn/MatMul_1",
                    "*/encoder_attn/MatMul",
                    "*/encoder_attn/MatMul_1",
                    "/lm_head/MatMul",
                ),
                calibration_method="percentile",
                percentile=99.99,
                per_channel=True,
            ),
        ),
        (
            "sd8g2_balanced",
            PresetSpec(
                name="sd8g2_balanced",
                runner_kind="static",
                op_types_to_quantize=("MatMul",),
                exclusion_patterns=(
                    "*/self_attn/MatMul_1",
                    "*/encoder_attn/MatMul_1",
                    "/lm_head/MatMul",
                ),
                calibration_method="percentile",
                percentile=99.95,
                per_channel=True,
            ),
        ),
        (
            "sd8g2_aggressive",
            PresetSpec(
                name="sd8g2_aggressive",
                runner_kind="static",
                op_types_to_quantize=("MatMul",),
                exclusion_patterns=("/lm_head/MatMul",),
                calibration_method="percentile",
                percentile=99.9,
                per_channel=True,
            ),
        ),
        (
            "baseline_dynamic_int8",
            PresetSpec(
                name="baseline_dynamic_int8",
                runner_kind="dynamic",
                op_types_to_quantize=("MatMul",),
                exclusion_patterns=(),
                calibration_method="minmax",
                percentile=99.99,
                per_channel=False,
            ),
        ),
    ]
)


def list_supported_presets() -> tuple[str, ...]:
    return tuple(_PRESET_SPECS.keys())


def get_preset_spec(preset: str) -> PresetSpec:
    try:
        return _PRESET_SPECS[preset]
    except KeyError as exc:
        raise ValueError(f"Unsupported preset: {preset}") from exc


def build_exclusion_patterns(preset: str) -> tuple[str, ...]:
    return get_preset_spec(preset).exclusion_patterns


def _matches_any_pattern(node_name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(node_name, pattern) for pattern in patterns)


def build_quantization_plan(
    node_names: Sequence[str],
    preset: str,
    extra_exclude_patterns: Sequence[str] | None = None,
) -> QuantizationPlan:
    spec = get_preset_spec(preset)
    patterns = list(spec.exclusion_patterns)
    if extra_exclude_patterns:
        patterns.extend(extra_exclude_patterns)

    nodes_to_exclude = tuple(
        node_name for node_name in node_names if _matches_any_pattern(node_name, patterns)
    )
    return QuantizationPlan(
        preset=spec.name,
        runner_kind=spec.runner_kind,
        op_types_to_quantize=spec.op_types_to_quantize,
        exclusion_patterns=tuple(patterns),
        nodes_to_exclude=nodes_to_exclude,
        calibration_method=spec.calibration_method,
        percentile=spec.percentile,
        per_channel=spec.per_channel,
    )
