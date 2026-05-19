from quantize.types import CalibrationSample, QuantizationPlan


def list_quantize_projects(*args, **kwargs):
    from quantize.projects import list_quantize_projects as _list_quantize_projects

    return _list_quantize_projects(*args, **kwargs)


def resolve_quantize_project(*args, **kwargs):
    from quantize.projects import resolve_quantize_project as _resolve_quantize_project

    return _resolve_quantize_project(*args, **kwargs)


__all__ = [
    "CalibrationSample",
    "QuantizationPlan",
    "list_quantize_projects",
    "resolve_quantize_project",
]
