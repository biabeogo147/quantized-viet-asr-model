from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
FORBIDDEN_MODEL_BUNDLE_IMPORTS = {
    "model_bundle.contracts",
    "model_bundle.exporter",
    "model_bundle.layout",
    "model_bundle.qnn_preflight",
    "model_bundle.verifier",
    "model_bundle.projects",
    "model_bundle.projects.vpcd",
    "model_bundle.projects.zipformer",
    "model_bundle.projects._vpcd_support",
    "model_bundle.projects.vpcd_shapes",
}
ALLOWED_AIHUB_IMPORTS = {
    "model_bundle.fixtures",
    "model_bundle.manifest",
    "model_bundle.vpcd_runtime",
    "model_bundle.vpcd_shapes",
    "model_bundle.zipformer_runtime",
}


def _module_name(path: Path) -> str:
    return ".".join(path.relative_to(SRC_ROOT).with_suffix("").parts)


def _imports_for(path: Path) -> set[str]:
    imports: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
    return imports


def test_export_package_is_not_importable():
    for name in list(sys.modules):
        if name == "export" or name.startswith("export."):
            sys.modules.pop(name, None)

    try:
        importlib.import_module("export")
    except ModuleNotFoundError:
        return
    raise AssertionError("export package should be removed")


def test_verify_quantize_and_tools_do_not_import_retired_model_bundle_modules():
    checked_roots = ("verify", "quantize", "tools")
    for path in SRC_ROOT.rglob("*.py"):
        module_name = _module_name(path)
        if not module_name.startswith(checked_roots):
            continue
        forbidden = _imports_for(path) & FORBIDDEN_MODEL_BUNDLE_IMPORTS
        assert not forbidden, f"{module_name} still imports retired model_bundle modules: {sorted(forbidden)}"


def test_aihub_uses_only_allowed_model_bundle_surface():
    for path in (SRC_ROOT / "aihub").rglob("*.py"):
        module_name = _module_name(path)
        model_bundle_imports = {item for item in _imports_for(path) if item.startswith("model_bundle")}
        disallowed = model_bundle_imports - ALLOWED_AIHUB_IMPORTS
        assert not disallowed, f"{module_name} imports disallowed model_bundle modules: {sorted(disallowed)}"
