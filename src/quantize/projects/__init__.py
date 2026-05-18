from __future__ import annotations

from importlib import import_module


_PROJECT_MODULES = {
    "vpcd": "quantize.projects.vpcd",
    "zipformer": "quantize.projects.zipformer",
}


def resolve_quantize_project(name: str):
    try:
        module_name = _PROJECT_MODULES[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported quantize project: {name}") from exc
    return import_module(module_name)


def list_quantize_projects() -> tuple[str, ...]:
    return tuple(_PROJECT_MODULES.keys())
