import contextlib
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Iterator


class ManualTemporaryDirectory:
    def __init__(
        self,
        suffix: str | None = None,
        prefix: str | None = None,
        dir: str | os.PathLike[str] | None = None,
    ):
        base_dir = Path(dir) if dir is not None else Path.cwd()
        name_prefix = prefix or "tmp"
        name_suffix = suffix or ""
        self.path = base_dir / f"{name_prefix}{uuid.uuid4().hex}{name_suffix}"

    def __enter__(self) -> str:
        self.path.mkdir(parents=True, exist_ok=False)
        return os.fspath(self.path)

    def __exit__(self, exc_type, exc, tb) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


@contextlib.contextmanager
def temporary_workspace_tempdir(path: Path) -> Iterator[Path]:
    path.mkdir(parents=True, exist_ok=True)
    original_temp = os.environ.get("TEMP")
    original_tmp = os.environ.get("TMP")
    original_tempdir = tempfile.tempdir
    try:
        os.environ["TEMP"] = os.fspath(path)
        os.environ["TMP"] = os.fspath(path)
        tempfile.tempdir = os.fspath(path)
        yield path
    finally:
        if original_temp is None:
            os.environ.pop("TEMP", None)
        else:
            os.environ["TEMP"] = original_temp
        if original_tmp is None:
            os.environ.pop("TMP", None)
        else:
            os.environ["TMP"] = original_tmp
        tempfile.tempdir = original_tempdir
