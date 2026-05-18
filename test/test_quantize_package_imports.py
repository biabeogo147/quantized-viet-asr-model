import importlib
import sys


def _clear_quantize_modules() -> None:
    for name in list(sys.modules):
        if name == "quantize" or name.startswith("quantize."):
            sys.modules.pop(name, None)


def test_import_quantize_is_lazy_for_project_modules() -> None:
    _clear_quantize_modules()

    quantize = importlib.import_module("quantize")

    assert "quantize.projects.vpcd" not in sys.modules
    assert "quantize.projects.zipformer" not in sys.modules
    assert callable(quantize.list_quantize_projects)
    assert callable(quantize.resolve_quantize_project)


def test_resolve_quantize_project_imports_only_requested_project() -> None:
    _clear_quantize_modules()

    projects = importlib.import_module("quantize.projects")
    resolved = projects.resolve_quantize_project("vpcd")

    assert resolved.NAME == "vpcd"
    assert "quantize.projects.vpcd" in sys.modules
