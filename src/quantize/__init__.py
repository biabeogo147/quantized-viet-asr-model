from quantize.types import CalibrationSample, PresetSpec, QuantizationPlan


def build_quantization_plan(*args, **kwargs):
    from quantize.presets import build_quantization_plan as _build_quantization_plan

    return _build_quantization_plan(*args, **kwargs)


def get_preset_spec(*args, **kwargs):
    from quantize.presets import get_preset_spec as _get_preset_spec

    return _get_preset_spec(*args, **kwargs)


def list_supported_presets(*args, **kwargs):
    from quantize.presets import list_supported_presets as _list_supported_presets

    return _list_supported_presets(*args, **kwargs)


def list_quantize_projects(*args, **kwargs):
    from quantize.projects import list_quantize_projects as _list_quantize_projects

    return _list_quantize_projects(*args, **kwargs)


def resolve_quantize_project(*args, **kwargs):
    from quantize.projects import resolve_quantize_project as _resolve_quantize_project

    return _resolve_quantize_project(*args, **kwargs)


__all__ = [
    "CalibrationSample",
    "PresetSpec",
    "QuantizationPlan",
    "build_quantization_plan",
    "get_preset_spec",
    "list_supported_presets",
    "list_quantize_projects",
    "resolve_quantize_project",
]
