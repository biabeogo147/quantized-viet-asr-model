from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from .projects import resolve_bundle_project


def _filter_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    parameters = inspect.signature(callable_obj).parameters
    return {key: value for key, value in kwargs.items() if key in parameters}


def verify_model_bundle(*, project: str, **kwargs: Any) -> Any:
    adapter = resolve_bundle_project(project)
    normalized = {}
    for key, value in kwargs.items():
        normalized[key] = Path(value) if isinstance(value, (str, Path)) and key.endswith(('dir', 'bundle', 'manifest', 'path')) else value
    filtered_kwargs = _filter_kwargs(adapter.verify_bundle, normalized)
    return adapter.verify_bundle(**filtered_kwargs)
