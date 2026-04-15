from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from .projects import resolve_bundle_project
from .manifest import ModelBundleManifest


def _filter_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    parameters = inspect.signature(callable_obj).parameters
    return {key: value for key, value in kwargs.items() if key in parameters}


def export_model_bundle(*, project: str, model_dir: str | Path, output_dir: str | Path, **kwargs: Any) -> ModelBundleManifest:
    adapter = resolve_bundle_project(project)
    filtered_kwargs = _filter_kwargs(adapter.export_bundle, kwargs)
    return adapter.export_bundle(model_dir=Path(model_dir), output_dir=Path(output_dir), **filtered_kwargs)
