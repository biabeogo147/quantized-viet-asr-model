import fnmatch
from collections import OrderedDict
from typing import Iterable, Sequence

from quantize.types import PresetSpec, QuantizationPlan


def _attention_matmul_patterns(scope: str) -> tuple[str, ...]:
    return (
        f"*/{scope}/q_proj/MatMul",
        f"*/{scope}/k_proj/MatMul",
        f"*/{scope}/v_proj/MatMul",
        f"*/{scope}/MatMul",
        f"*/{scope}/MatMul_1",
        f"*/{scope}/out_proj/MatMul",
    )


def _decoder_attention_patterns() -> tuple[str, ...]:
    return _attention_matmul_patterns("decoder/*/self_attn") + _attention_matmul_patterns("decoder/*/encoder_attn")


def _attention_score_patterns(scope: str) -> tuple[str, ...]:
    return (
        f"*/{scope}/MatMul",
        f"*/{scope}/MatMul_1",
    )


def _decoder_attention_score_patterns() -> tuple[str, ...]:
    return _attention_score_patterns("decoder/*/self_attn") + _attention_score_patterns("decoder/*/encoder_attn")


def _decoder_ffn_patterns(layer_indices: Iterable[int]) -> tuple[str, ...]:
    patterns: list[str] = []
    for layer_index in layer_indices:
        patterns.append(f"/model/decoder/layers.{layer_index}/fc1/MatMul")
        patterns.append(f"/model/decoder/layers.{layer_index}/fc2/MatMul")
    return tuple(patterns)


QUALITY_REQUIRED_MATCHERS = (
    ("/decoder/", "decoder"),
    ("/lm_head/MatMul", "lm_head"),
)

BALANCED_REQUIRED_MATCHERS = (
    ("/decoder/", "decoder"),
    ("/self_attn/", "decoder self attention"),
    ("/encoder_attn/", "decoder cross attention"),
    ("/lm_head/MatMul", "lm_head"),
)

BALANCED_DEFAULT_EXCLUSION_PATTERNS = _decoder_attention_score_patterns() + _decoder_ffn_patterns(range(8, 12)) + (
    "/lm_head/MatMul",
)

_PRESET_SPECS: "OrderedDict[str, PresetSpec]" = OrderedDict(
    [
        (
            "sd8g2_quality",
            PresetSpec(
                name="sd8g2_quality",
                runner_kind="qnn_static",
                op_types_to_quantize=("MatMul",),
                exclusion_patterns=("*/decoder/*", "/lm_head/MatMul"),
                calibration_method="minmax",
                percentile=99.99,
                per_channel=False,
                activation_type="quint16",
                weight_type="quint8",
            ),
        ),
        (
            "sd8g2_balanced",
            PresetSpec(
                name="sd8g2_balanced",
                runner_kind="qnn_static",
                op_types_to_quantize=("MatMul",),
                exclusion_patterns=BALANCED_DEFAULT_EXCLUSION_PATTERNS,
                calibration_method="minmax",
                percentile=99.95,
                per_channel=False,
                activation_type="quint16",
                weight_type="quint8",
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
                activation_type="qint8",
                weight_type="qint8",
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
                activation_type="quint8",
                weight_type="qint8",
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


def _matches_any_pattern(node_name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(node_name, pattern) for pattern in patterns)


def _validate_required_matches(nodes_to_exclude: Sequence[str], required_matchers: Sequence[tuple[str, str]], preset: str) -> None:
    missing_requirements = [
        label for substring, label in required_matchers if not any(substring in node_name for node_name in nodes_to_exclude)
    ]
    if missing_requirements:
        missing_text = ", ".join(missing_requirements)
        raise ValueError(f"{preset} must keep these regions in FP32: {missing_text}")


def build_quantization_plan(
    node_names: Sequence[str],
    preset: str,
    extra_exclude_patterns: Sequence[str] | None = None,
) -> QuantizationPlan:
    spec = get_preset_spec(preset)
    patterns = list(spec.exclusion_patterns)
    if extra_exclude_patterns:
        patterns.extend(extra_exclude_patterns)

    nodes_to_exclude = tuple(node_name for node_name in node_names if _matches_any_pattern(node_name, patterns))
    if spec.name == "sd8g2_quality":
        _validate_required_matches(nodes_to_exclude, QUALITY_REQUIRED_MATCHERS, spec.name)
    if spec.name == "sd8g2_balanced":
        _validate_required_matches(nodes_to_exclude, BALANCED_REQUIRED_MATCHERS, spec.name)

    return QuantizationPlan(
        preset=spec.name,
        runner_kind=spec.runner_kind,
        op_types_to_quantize=spec.op_types_to_quantize,
        exclusion_patterns=tuple(patterns),
        nodes_to_exclude=nodes_to_exclude,
        calibration_method=spec.calibration_method,
        percentile=spec.percentile,
        per_channel=spec.per_channel,
        activation_type=spec.activation_type,
        weight_type=spec.weight_type,
    )
