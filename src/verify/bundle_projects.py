from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from quantize.vpcd_bundle import DEFAULT_MODEL_DIR as VPCD_DEFAULT_MODEL_DIR
from quantize.vpcd_bundle import DEFAULT_OUTPUT_DIR as VPCD_DEFAULT_OUTPUT_DIR
from quantize.vpcd_bundle import verify_bundle as verify_vpcd_bundle
from quantize.zipformer_bundle import DEFAULT_MODEL_DIR as ZIPFORMER_DEFAULT_MODEL_DIR
from quantize.zipformer_bundle import DEFAULT_OUTPUT_DIR as ZIPFORMER_DEFAULT_OUTPUT_DIR
from quantize.zipformer_bundle import verify_bundle as verify_zipformer_bundle


@dataclass(frozen=True)
class BundleVerificationProject:
    name: str
    default_model_dir: str
    default_output_dir: str
    verify_bundle: Callable[..., Any]


_PROJECTS: dict[str, BundleVerificationProject] = {
    "vpcd": BundleVerificationProject(
        name="vpcd",
        default_model_dir=str(VPCD_DEFAULT_MODEL_DIR),
        default_output_dir=str(VPCD_DEFAULT_OUTPUT_DIR),
        verify_bundle=verify_vpcd_bundle,
    ),
    "zipformer": BundleVerificationProject(
        name="zipformer",
        default_model_dir=str(ZIPFORMER_DEFAULT_MODEL_DIR),
        default_output_dir=str(ZIPFORMER_DEFAULT_OUTPUT_DIR),
        verify_bundle=verify_zipformer_bundle,
    ),
}


def resolve_bundle_project(name: str) -> BundleVerificationProject:
    try:
        return _PROJECTS[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported bundle verification project: {name}") from exc


def list_bundle_projects() -> tuple[str, ...]:
    return tuple(_PROJECTS.keys())
