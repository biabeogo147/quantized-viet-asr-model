from pathlib import Path


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)
