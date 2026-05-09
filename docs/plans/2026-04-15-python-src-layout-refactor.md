# Python `src/` Layout Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `python-model-test` from a repo-root import layout into a maintainable `src/` layout while keeping export, verify, quantize, and smoke-test workflows working end to end.

**Architecture:** Move the maintained Python modules under one `src/` root, but keep the existing package names: `export`, `model_bundle`, `quantize`, `verify`, and `tools`. Use a staged migration: first add packaging and `src/`, then move internal modules, then switch tests and docs to the new canonical layout, and only remove repo-root compatibility wrappers after the new flow is fully verified.

**Tech Stack:** Python 3, PEP 621 `pyproject.toml`, editable installs (`pip install -e .`), pytest, ONNX Runtime, ONNX Runtime quantization APIs, NumPy, Torchaudio

---

## Scope To Lock

- This plan is for `python-model-test` only.
- `assets/`, `build/`, `docs/`, and `test/` remain top-level directories for now.
- Only maintained Python modules are moved under `src/`.
- The canonical package paths become:
  - `src/export/...`
  - `src/model_bundle/...`
  - `src/quantize/...`
  - `src/verify/...`
  - `src/tools/...`
- Imports remain the existing package names:
  - `from export...`
  - `from model_bundle...`
  - `from quantize...`
  - `from verify...`
- A short-lived compatibility layer is allowed during migration, but the final state must not depend on repo-root packages as the canonical source.
- Smoke runners may stay in `test/`, but they must import runtime code from the `src/` packages.
- Existing bundle and quantization behavior must remain functionally unchanged during this refactor.

## Decision Summary

### Recommended approach: `src/` with the current top-level package names

Use:

```text
python-model-test/
  src/
    export/
    model_bundle/
    quantize/
    verify/
    tools/
```

Why this is the right target for this repo:

- matches the layout you explicitly want
- keeps import churn smaller because package names stay the same
- still gives the repo a real installable package root
- improves IDE indexing, editable install behavior, and pytest consistency
- keeps the refactor focused on structure, not on naming redesign

### Rejected approach 1: add a namespace package like `src/python_model_test/...`

Why it is not the chosen path:

- it forces every import site to change
- it is a bigger migration than this repo needs right now
- it adds naming churn without directly helping the current model workflows

### Rejected approach 2: keep repo-root imports and only add packaging metadata

Why it is weaker:

- it does not really solve the structure problem
- source code still lives at the repo root
- it keeps cwd-dependent behavior and architectural ambiguity

## Current State Summary

Today the repo behaves as if the repository root is the Python import root:

- `from model_bundle...`
- `from quantize...`
- `from export...`
- `from verify...`

There is currently no packaging file in `python-model-test`:

- no `pyproject.toml`
- no `setup.cfg`
- no `setup.py`
- no `pytest.ini`

That means:

- imports work only because commands are run from the repo root
- IDE and test behavior depend on cwd conventions
- there is no explicit installable source root

## Target File Map

### New package root

- Create: `python-model-test/pyproject.toml` - package metadata, editable-install support, and optional pytest settings.

### Export package

- Create: `python-model-test/src/export/__init__.py`
- Create: `python-model-test/src/export/model_bundle.py`
- Create: `python-model-test/src/export/punctuation_onnx.py`

### Bundle package

- Create: `python-model-test/src/model_bundle/__init__.py`
- Create: `python-model-test/src/model_bundle/contracts.py`
- Create: `python-model-test/src/model_bundle/exporter.py`
- Create: `python-model-test/src/model_bundle/fixtures.py`
- Create: `python-model-test/src/model_bundle/layout.py`
- Create: `python-model-test/src/model_bundle/manifest.py`
- Create: `python-model-test/src/model_bundle/verifier.py`
- Create: `python-model-test/src/model_bundle/projects/__init__.py`
- Create: `python-model-test/src/model_bundle/projects/vpcd.py`
- Create: `python-model-test/src/model_bundle/projects/zipformer.py`
- Create: `python-model-test/src/model_bundle/projects/_vpcd_support.py`

### Quantize package

- Create: `python-model-test/src/quantize/__init__.py`
- Create: `python-model-test/src/quantize/__main__.py`
- Create: `python-model-test/src/quantize/cli.py`
- Create: `python-model-test/src/quantize/calibration.py`
- Create: `python-model-test/src/quantize/config.py`
- Create: `python-model-test/src/quantize/evaluate.py`
- Create: `python-model-test/src/quantize/fixed_shapes.py`
- Create: `python-model-test/src/quantize/model_introspection.py`
- Create: `python-model-test/src/quantize/presets.py`
- Create: `python-model-test/src/quantize/qnn.py`
- Create: `python-model-test/src/quantize/reports.py`
- Create: `python-model-test/src/quantize/runner.py`
- Create: `python-model-test/src/quantize/runtime.py`
- Create: `python-model-test/src/quantize/types.py`
- Create: `python-model-test/src/quantize/projects/__init__.py`
- Create: `python-model-test/src/quantize/projects/vpcd.py`
- Create: `python-model-test/src/quantize/projects/zipformer.py`

### Verify package

- Create: `python-model-test/src/verify/__init__.py`
- Create: `python-model-test/src/verify/model_bundle.py`

### Shared tools and wrappers

- Create: `python-model-test/src/tools/__init__.py`
- Create: `python-model-test/src/tools/convert_bpe2token.py`
- Modify: `python-model-test/convert_bpe2token.py` - temporary wrapper that forwards to `tools.convert_bpe2token`.

### Tests and docs

- Modify: `python-model-test/test/test_model_bundle_core.py`
- Modify: `python-model-test/test/test_vpcd_bundle.py`
- Modify: `python-model-test/test/test_zipformer_bundle.py`
- Modify: `python-model-test/test/test_quantize_projects.py`
- Modify: `python-model-test/test/test_zipformer_quantize.py`
- Modify: `python-model-test/test/test_export_verify_modules.py`
- Modify: `python-model-test/test/test_punctuation_model_onnx.py`
- Modify: `python-model-test/test/test_acoustic_model_onnx.py`
- Modify: `python-model-test/README.md`
- Modify: `python-model-test/export/README.md`
- Modify: `python-model-test/model_bundle/README.md`
- Modify: `python-model-test/model_bundle/projects/README.md`
- Modify: `python-model-test/quantize/README.md`
- Modify: `python-model-test/quantize/projects/README.md`
- Modify: `python-model-test/verify/README.md`
- Modify: `python-model-test/test/README.md`

## Final Target Layout

```text
python-model-test/
  assets/
  build/
  docs/
  src/
    export/
    model_bundle/
    quantize/
    verify/
    tools/
  test/
  pyproject.toml
  README.md
```

## Migration Rules

- Do not move `test/` into `src/` in this phase.
- Do not rewrite runtime logic and packaging logic in the same step unless a failing test demands it.
- Keep imports absolute inside the packages:
  - use `from model_bundle...`
  - use `from quantize...`
  - do not add a new namespace package in this phase
- During the transition, do not keep duplicate long-lived source trees active at the same time longer than necessary.
- The final documentation must clearly distinguish:
  - canonical source paths under `src/...`
  - old repo-root source paths that are being retired

## Verification Gates

Do not mark this refactor complete until all of the following are true:

- `pip install -e .` works from `python-model-test/`
- canonical imports work from an installed environment:
  - `export...`
  - `model_bundle...`
  - `quantize...`
  - `verify...`
- all pytest suites still pass
- `python -m quantize --project vpcd --dry-run` works after editable install
- `python -m quantize --project zipformer --dry-run` works after editable install
- `python -m export.model_bundle --project vpcd ...` works after editable install
- `python -m verify.model_bundle --project vpcd ...` works after editable install
- smoke runners in `test/` still work
- repo-root package files are removed by the end of the migration

## Task 1: Add Packaging And `src/` Scaffolding

**Files:**
- Create: `python-model-test/pyproject.toml`
- Create: `python-model-test/src/export/__init__.py`
- Create: `python-model-test/src/model_bundle/__init__.py`
- Create: `python-model-test/src/quantize/__init__.py`
- Create: `python-model-test/src/verify/__init__.py`
- Create: `python-model-test/src/tools/__init__.py`
- Create: `python-model-test/test/test_src_layout_bootstrap.py`

- [ ] **Step 1: Write the failing packaging bootstrap test**

```python
import importlib


def test_src_packages_import_after_editable_install():
    assert importlib.import_module("model_bundle").__name__ == "model_bundle"
    assert importlib.import_module("quantize").__name__ == "quantize"
```

- [ ] **Step 2: Run the test to confirm it fails**

Run:

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m pytest python-model-test/test/test_src_layout_bootstrap.py -q
```

Expected: FAIL because `pyproject.toml` and `src/` packages do not exist yet.

- [ ] **Step 3: Add minimal packaging**

Add a `pyproject.toml` with:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "python-model-test"
version = "0.1.0"
description = "Model export, bundle, verify, and quantize tooling"
requires-python = ">=3.10"

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 4: Add empty package markers under `src/`**

```python
__all__ = []
```

- [ ] **Step 5: Install editable package**

Run:

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m pip install -e D:\DS-AI\BKMeeting-Research\python-model-test
```

Expected: editable install succeeds.

- [ ] **Step 6: Re-run the bootstrap test**

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add python-model-test/pyproject.toml python-model-test/src/export python-model-test/src/model_bundle python-model-test/src/quantize python-model-test/src/verify python-model-test/src/tools python-model-test/test/test_src_layout_bootstrap.py
git commit -m "build: add src package scaffold for python-model-test"
```

## Task 2: Move `model_bundle` Into `src/model_bundle`

**Files:**
- Create: `python-model-test/src/model_bundle/*`
- Modify: `python-model-test/test/test_model_bundle_core.py`
- Modify: `python-model-test/test/test_vpcd_bundle.py`
- Modify: `python-model-test/test/test_zipformer_bundle.py`

- [ ] **Step 1: Write failing focused tests for installed `model_bundle` imports**

```python
from model_bundle.manifest import ModelBundleManifest
from model_bundle.projects import resolve_bundle_project


def test_installed_model_bundle_imports():
    assert ModelBundleManifest.__name__ == "ModelBundleManifest"
    assert resolve_bundle_project("vpcd").name == "vpcd"
```

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run:

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m pytest `
  python-model-test/test/test_model_bundle_core.py `
  python-model-test/test/test_vpcd_bundle.py `
  python-model-test/test/test_zipformer_bundle.py -q
```

Expected: FAIL until the installed `src/model_bundle` package exists and is used consistently.

- [ ] **Step 3: Copy implementation into `src/model_bundle/`**

Required rule:
- preserve runtime behavior
- keep import names unchanged

- [ ] **Step 4: Re-run the focused tests**

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add python-model-test/src/model_bundle python-model-test/test/test_model_bundle_core.py python-model-test/test/test_vpcd_bundle.py python-model-test/test/test_zipformer_bundle.py
git commit -m "refactor: move model_bundle into src layout"
```

## Task 3: Move `quantize` Into `src/quantize`

**Files:**
- Create: `python-model-test/src/quantize/*`
- Modify: `python-model-test/test/test_quantize_projects.py`
- Modify: `python-model-test/test/test_zipformer_quantize.py`

- [ ] **Step 1: Write failing focused tests for installed `quantize` imports**

```python
from quantize.cli import main
from quantize.projects import resolve_quantize_project


def test_installed_quantize_imports():
    assert callable(main)
    assert resolve_quantize_project("zipformer").name == "zipformer"
```

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run:

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m pytest `
  python-model-test/test/test_quantize_projects.py `
  python-model-test/test/test_zipformer_quantize.py -q
```

Expected: FAIL until `src/quantize` is the real imported package.

- [ ] **Step 3: Copy `quantize/` into `src/quantize/`**

Required rule:
- update imports only where needed for installed `src/` resolution
- keep package names unchanged

- [ ] **Step 4: Re-run the focused tests**

Expected: PASS

- [ ] **Step 5: Smoke-check the CLI**

Run:

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m quantize --project vpcd --dry-run
```

Expected: CLI starts and prints a dry-run summary.

- [ ] **Step 6: Commit**

```bash
git add python-model-test/src/quantize python-model-test/test/test_quantize_projects.py python-model-test/test/test_zipformer_quantize.py
git commit -m "refactor: move quantize into src layout"
```

## Task 4: Move `export`, `verify`, And `tools` Into `src/`

**Files:**
- Create: `python-model-test/src/export/*`
- Create: `python-model-test/src/verify/*`
- Create: `python-model-test/src/tools/*`
- Modify: `python-model-test/test/test_export_verify_modules.py`
- Modify: `python-model-test/convert_bpe2token.py`

- [ ] **Step 1: Write a failing focused test for installed `export` and `verify` imports**

```python
from export.model_bundle import main as export_main
from verify.model_bundle import main as verify_main


def test_installed_export_verify_imports():
    assert callable(export_main)
    assert callable(verify_main)
```

- [ ] **Step 2: Run the focused test and confirm it fails**

Run:

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m pytest python-model-test/test/test_export_verify_modules.py -q
```

Expected: FAIL until the installed `src/export` and `src/verify` packages exist and are used.

- [ ] **Step 3: Move `export` and `verify` under `src/`**

Required rule:
- keep the module names unchanged
- keep CLI behavior unchanged

- [ ] **Step 4: Move `convert_bpe2token.py` logic into `src/tools/convert_bpe2token.py`**

Keep the repo-root file as a temporary wrapper:

```python
from tools.convert_bpe2token import main


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Re-run the focused test**

Expected: PASS

- [ ] **Step 6: Smoke-check the CLIs**

Run:

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m export.model_bundle --help
& D:\Anaconda\envs\speech2text\python.exe -m verify.model_bundle --help
```

Expected: both help commands print successfully.

- [ ] **Step 7: Commit**

```bash
git add python-model-test/src/export python-model-test/src/verify python-model-test/src/tools python-model-test/convert_bpe2token.py python-model-test/test/test_export_verify_modules.py
git commit -m "refactor: move export verify and tools into src layout"
```

## Task 5: Rewire Smoke Runners And End-To-End Commands

**Files:**
- Modify: `python-model-test/test/test_punctuation_model_onnx.py`
- Modify: `python-model-test/test/test_acoustic_model_onnx.py`
- Modify: `python-model-test/README.md`
- Modify: module README files

- [ ] **Step 1: Write failing smoke-runner import tests**

```python
def test_punctuation_smoke_runner_uses_installed_packages():
    import inspect
    import test.test_punctuation_model_onnx as mod
    source = inspect.getsource(mod)
    assert "from model_bundle" in source
```

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run:

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m pytest `
  python-model-test/test/test_export_verify_modules.py `
  python-model-test/test/test_quantize_projects.py -q
```

Expected: FAIL until the runners and docs are updated to the installed `src/` layout.

- [ ] **Step 3: Update smoke runners to rely on the installed packages**

Required rule:
- `--model-dir` and `--bundle-manifest` behavior must stay unchanged

- [ ] **Step 4: Update README commands**

Examples to prefer:

```powershell
python -m export.model_bundle ...
python -m verify.model_bundle ...
python -m quantize --project zipformer ...
```

- [ ] **Step 5: Run smoke-test commands**

Run:

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m export.model_bundle --help
& D:\Anaconda\envs\speech2text\python.exe -m verify.model_bundle --help
& D:\Anaconda\envs\speech2text\python.exe -m quantize --project vpcd --dry-run
```

Expected: all commands succeed.

- [ ] **Step 6: Commit**

```bash
git add python-model-test/test/test_punctuation_model_onnx.py python-model-test/test/test_acoustic_model_onnx.py python-model-test/README.md python-model-test/export/README.md python-model-test/model_bundle/README.md python-model-test/model_bundle/projects/README.md python-model-test/quantize/README.md python-model-test/quantize/projects/README.md python-model-test/verify/README.md python-model-test/test/README.md
git commit -m "docs: switch smoke runners and docs to src layout"
```

## Task 6: Remove Legacy Repo-Root Packages And Finalize The Layout

**Files:**
- Delete: `python-model-test/export/*`
- Delete: `python-model-test/model_bundle/*`
- Delete: `python-model-test/quantize/*`
- Delete: `python-model-test/verify/*`
- Keep: README stubs only if needed, otherwise move docs and delete the directories

- [ ] **Step 1: Write a failing test that disallows legacy repo-root package files**

```python
from pathlib import Path


def test_repo_root_quantize_package_removed():
    assert not Path("python-model-test/quantize/cli.py").exists()
```

- [ ] **Step 2: Run the full test suite and confirm the legacy test fails before cleanup**

Run:

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m pytest python-model-test/test -q
```

Expected: FAIL because repo-root package files still exist.

- [ ] **Step 3: Delete the repo-root package files**

Required rule:
- only delete them after all tests and commands have been switched to the installed `src/` layout

- [ ] **Step 4: Re-run editable install**

Run:

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m pip install -e D:\DS-AI\BKMeeting-Research\python-model-test
```

Expected: editable install still succeeds.

- [ ] **Step 5: Re-run the full test suite**

Run:

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m pytest python-model-test/test -q
```

Expected: PASS

- [ ] **Step 6: Run final end-to-end smoke commands**

Run:

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m export.model_bundle --help
& D:\Anaconda\envs\speech2text\python.exe -m verify.model_bundle --help
& D:\Anaconda\envs\speech2text\python.exe -m quantize --project zipformer --dry-run
```

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: finalize src layout and remove legacy root packages"
```

## Task 7: Write Final Migration Notes

**Files:**
- Modify: `python-model-test/README.md`
- Modify: `python-model-test/docs/superpowers/plans/2026-04-15-python-src-layout-refactor.md`

- [ ] **Step 1: Document the final canonical invocation style**

Required examples:

```powershell
python -m export.model_bundle ...
python -m verify.model_bundle ...
python -m quantize --project ...
```

- [ ] **Step 2: Add a short migration section**

Document:
- old source layout
- new source layout
- whether any temporary wrapper commands remain

- [ ] **Step 3: Re-read the README and plan for consistency**

Expected: no stale command examples remain.

- [ ] **Step 4: Commit**

```bash
git add python-model-test/README.md python-model-test/docs/superpowers/plans/2026-04-15-python-src-layout-refactor.md
git commit -m "docs: record src layout migration guidance"
```

## Exit Criteria

The refactor is complete only when all of these are true:

- the canonical source root is `src/`
- internal imports no longer depend on repo-root package files
- editable install is the normal workflow
- smoke runners still work
- bundle export, verification, and quantization still work
- docs no longer teach repo-root source layout as the primary path
