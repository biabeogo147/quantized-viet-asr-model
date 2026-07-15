from __future__ import annotations

import json
import shutil
from pathlib import Path


def sync_bundle(bundle_dir: str | Path, destination: str | Path) -> Path:
    """Synchronize exactly the manifest-declared bundle files to Android assets.

    Args:
        bundle_dir: Source directory containing manifest v2 and component files.
        destination: Android asset directory to reconcile deterministically.

    Returns:
        The synchronized destination directory.
    """
    source = Path(bundle_dir)
    manifest = json.loads((source / "artifact-manifest.json").read_text(encoding="utf-8"))
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    expected = {"artifact-manifest.json", *(row["file"] for row in manifest["components"])}
    for name in sorted(expected):
        shutil.copyfile(source / name, target / name)
    for existing in target.iterdir():
        if existing.is_file() and existing.name not in expected:
            existing.unlink()
    return target
