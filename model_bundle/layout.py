from __future__ import annotations

from pathlib import Path


def resolve_bundle_dir(project: str, variant: str) -> Path:
    return Path('build') / 'model_bundle' / project / variant
