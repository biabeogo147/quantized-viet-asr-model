from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol


class BundleRuntimeProtocol(Protocol):
    pass


@dataclass(frozen=True)
class BundleVerificationReport:
    project: str
    passed: bool
    checked_samples: int
    mismatches: list[dict[str, Any]]
    details: dict[str, Any]


@dataclass(frozen=True)
class BundleProjectAdapter:
    name: str
    default_model_dir: str
    default_output_dir: str
    default_asset_namespace: str
    default_variant: str
    export_bundle: Callable[..., Any]
    verify_bundle: Callable[..., Any]
    bundle_runtime_from_manifest: Callable[..., Any]


def normalize_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)
