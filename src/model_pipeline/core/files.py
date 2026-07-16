from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Compute the SHA-256 digest of a file without loading it fully into memory.

    Args:
        path: File whose bytes should be hashed.
        chunk_size: Maximum number of bytes read per iteration.

    Returns:
        The lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: str | Path) -> str:
    """Hash a file or directory package without including machine-local paths.

    Args:
        path: File or package directory to hash deterministically.

    Returns:
        The lowercase digest of the file bytes or normalized package inventory.

    Raises:
        FileNotFoundError: If the supplied path is neither a file nor a directory.
    """
    resolved = Path(path)
    if resolved.is_file():
        return sha256_file(resolved)
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    entries = [
        {"path": file.relative_to(resolved).as_posix(), "sha256": sha256_file(file)}
        for file in sorted(path for path in resolved.rglob("*") if path.is_file())
    ]
    return stable_digest(entries)


def stable_digest(value: Any) -> str:
    """Hash a JSON-compatible value using canonical serialization.

    Args:
        value: JSON-compatible value whose ordering should not affect the digest.

    Returns:
        The lowercase hexadecimal SHA-256 digest.
    """
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
