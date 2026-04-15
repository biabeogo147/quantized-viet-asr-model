from __future__ import annotations

from . import vpcd, zipformer
from ..contracts import BundleProjectAdapter

_PROJECTS: dict[str, BundleProjectAdapter] = {
    vpcd.ADAPTER.name: vpcd.ADAPTER,
    zipformer.ADAPTER.name: zipformer.ADAPTER,
}


def resolve_bundle_project(name: str) -> BundleProjectAdapter:
    try:
        return _PROJECTS[name]
    except KeyError as exc:
        raise ValueError(f'Unsupported model bundle project: {name}') from exc


def list_bundle_projects() -> tuple[str, ...]:
    return tuple(_PROJECTS.keys())
