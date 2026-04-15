from __future__ import annotations

from pathlib import Path


def _looks_like_repo_root(candidate: Path) -> bool:
    return (
        (candidate / 'pyproject.toml').is_file()
        and (candidate / 'src').is_dir()
        and (candidate / 'assets').is_dir()
        and (candidate / 'test').is_dir()
    )


def find_repo_root(anchor: str | Path) -> Path:
    path = Path(anchor).resolve()
    current = path if path.is_dir() else path.parent
    for candidate in (current, *current.parents):
        if _looks_like_repo_root(candidate):
            return candidate
    raise RuntimeError(f'Could not resolve python-model-test repo root from: {path}')


def resolve_repo_path(path_like: str | Path, *, anchor: str | Path) -> Path:
    return find_repo_root(anchor) / Path(path_like)

