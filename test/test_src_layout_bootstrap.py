from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"


@contextmanager
def src_only_import_path():
    original_path = list(sys.path)
    filtered = []
    repo_root_resolved = REPO_ROOT.resolve()
    for entry in original_path:
        try:
            if Path(entry).resolve() == repo_root_resolved:
                continue
        except OSError:
            pass
        filtered.append(entry)
    sys.path = [str(SRC_ROOT)] + filtered
    importlib.invalidate_caches()
    try:
        yield
    finally:
        sys.path = original_path
        importlib.invalidate_caches()


@contextmanager
def fresh_package_import(package_name: str):
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == package_name or name.startswith(f"{package_name}.")
    }
    for name in list(saved_modules):
        sys.modules.pop(name, None)
    importlib.invalidate_caches()
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name == package_name or name.startswith(f"{package_name}."):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
        importlib.invalidate_caches()


def test_src_packages_import_from_src_root():
    with src_only_import_path():
        with fresh_package_import("model_bundle"):
            model_bundle = importlib.import_module("model_bundle")
        with fresh_package_import("quantize"):
            quantize = importlib.import_module("quantize")

    assert str(Path(model_bundle.__file__).resolve()).startswith(str(SRC_ROOT.resolve()))
    assert str(Path(quantize.__file__).resolve()).startswith(str(SRC_ROOT.resolve()))


def test_model_bundle_submodules_import_from_src_root():
    with src_only_import_path():
        with fresh_package_import("model_bundle"):
            manifest = importlib.import_module("model_bundle.manifest")
            projects = importlib.import_module("model_bundle.projects")

    assert str(Path(manifest.__file__).resolve()).startswith(str(SRC_ROOT.resolve()))
    assert str(Path(projects.__file__).resolve()).startswith(str(SRC_ROOT.resolve()))


def test_quantize_submodules_import_from_src_root():
    with src_only_import_path():
        with fresh_package_import("quantize"):
            cli = importlib.import_module("quantize.cli")
            projects = importlib.import_module("quantize.projects")

    assert str(Path(cli.__file__).resolve()).startswith(str(SRC_ROOT.resolve()))
    assert str(Path(projects.__file__).resolve()).startswith(str(SRC_ROOT.resolve()))


def test_export_and_verify_submodules_import_from_src_root():
    with src_only_import_path():
        with fresh_package_import("export"):
            export_model_bundle = importlib.import_module("export.model_bundle")
        with fresh_package_import("verify"):
            verify_model_bundle = importlib.import_module("verify.model_bundle")

    assert str(Path(export_model_bundle.__file__).resolve()).startswith(str(SRC_ROOT.resolve()))
    assert str(Path(verify_model_bundle.__file__).resolve()).startswith(str(SRC_ROOT.resolve()))


def test_tools_submodules_import_from_src_root():
    with src_only_import_path():
        with fresh_package_import("tools"):
            convert_bpe2token = importlib.import_module("tools.convert_bpe2token")

    assert str(Path(convert_bpe2token.__file__).resolve()).startswith(str(SRC_ROOT.resolve()))


def test_legacy_repo_root_package_files_are_removed():
    legacy_paths = [
        REPO_ROOT / "export" / "__init__.py",
        REPO_ROOT / "model_bundle" / "__init__.py",
        REPO_ROOT / "quantize" / "__init__.py",
        REPO_ROOT / "verify" / "__init__.py",
    ]

    for legacy_path in legacy_paths:
        assert not legacy_path.exists()
