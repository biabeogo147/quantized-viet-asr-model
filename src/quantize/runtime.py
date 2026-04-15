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
def isolated_model_input(fp32_onnx_path: Path, temp_root: Path) -> Iterator[Path]:
    model_input_root = temp_root / "model_inputs"
    model_input_root.mkdir(parents=True, exist_ok=True)
    staged_input = model_input_root / f"{fp32_onnx_path.stem}.{uuid.uuid4().hex}{fp32_onnx_path.suffix}"
    try:
        os.link(fp32_onnx_path, staged_input)
    except OSError:
        shutil.copy2(fp32_onnx_path, staged_input)

    try:
        yield staged_input
    finally:
        inferred_path = staged_input.with_name(f"{staged_input.stem}-inferred{staged_input.suffix}")
        for cleanup_path in (inferred_path, staged_input):
            with contextlib.suppress(FileNotFoundError, PermissionError):
                cleanup_path.unlink()


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
