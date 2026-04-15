from __future__ import annotations

from . import vpcd, zipformer

_PROJECTS = {
    'vpcd': vpcd,
    'zipformer': zipformer,
}


def resolve_quantize_project(name: str):
    try:
        return _PROJECTS[name]
    except KeyError as exc:
        raise ValueError(f'Unsupported quantize project: {name}') from exc


def list_quantize_projects() -> tuple[str, ...]:
    return tuple(_PROJECTS.keys())
