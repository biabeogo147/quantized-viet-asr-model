from __future__ import annotations

import ast
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1] / "src" / "model_pipeline"


def test_dependency_boundaries() -> None:
    """Verify core, model, and integration imports respect architecture boundaries.

    Returns:
        None.
    """
    violations: list[str] = []
    for path in PACKAGE.rglob("*.py"):
        relative = path.relative_to(PACKAGE).as_posix()
        imports = _model_pipeline_imports(path)
        if relative.startswith("core/"):
            forbidden = [name for name in imports if not name.startswith("model_pipeline.core")]
        elif relative.startswith("models/"):
            forbidden = [
                name
                for name in imports
                if name.startswith("model_pipeline.integrations") or name == "model_pipeline.pipeline"
            ]
        elif relative.startswith("integrations/"):
            forbidden = [name for name in imports if name.startswith("model_pipeline.models")]
        else:
            forbidden = []
        violations.extend(f"{relative}: {name}" for name in forbidden)

    assert violations == []


def _model_pipeline_imports(path: Path) -> list[str]:
    """Collect internal model-pipeline imports from one Python module.

    Args:
        path: Python source file to parse.

    Returns:
        Imported module names within the public package namespace.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.extend(alias.name for alias in node.names if alias.name.startswith("model_pipeline"))
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("model_pipeline"):
            result.append(str(node.module))
    return result
