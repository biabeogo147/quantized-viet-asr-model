# Shared Model Bundle Platform And Quantization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the model-specific Python bundle code with one shared `model_bundle` platform for both `vpcd` and `zipformer`, refactor `quantize` into a multi-project framework for both models, remove `acoustic_bundle`, and deliver the first fixed-shape `PTQ + QDQ` Zipformer candidate bundle.

**Architecture:** Introduce a single `model_bundle` package that owns manifest contracts, layout helpers, export orchestration, verification, fixtures, and project adapters. Move the existing punctuation bundle logic from `android_bundle` and the acoustic bundle logic from `acoustic_bundle` behind project adapters so the top-level CLIs and smoke runners use one shared core. In parallel, refactor `quantize` to resolve a project adapter (`vpcd` or `zipformer`) and share the CLI, planning, reporting, and QNN helpers while letting each project provide its own calibration, exclusions, and quality gates.

**Tech Stack:** Python 3, ONNX Runtime, ONNX Runtime quantization APIs, QNN preprocessing helpers, NumPy, Torchaudio, pytest

---

## Scope To Lock

- This plan is **Python-only**. Do not change `bkmeeting` while executing it.
- The canonical shared bundle module becomes `python-model-test/model_bundle`.
- `python-model-test/acoustic_bundle` must be removed by the end of this plan.
- `python-model-test/android_bundle` must stop being a punctuation-only architectural root; its responsibilities move into `model_bundle`.
- Domain-specific smoke runners may stay for usability:
  - `test/test_punctuation_model_onnx.py`
  - `test/test_acoustic_model_onnx.py`
  But they must call the shared `model_bundle` core under the hood.
- `quantize` must become explicitly multi-project:
  - `vpcd`
  - `zipformer`
- The first quantized target remains Zipformer `QDQ` with `QUInt16` activations + `QUInt8` weights.

## Design Rules

- Core bundle logic lives once in `model_bundle`; project-specific logic lives in small project adapters.
- Core quantization logic lives once in `quantize`; project-specific logic lives in `quantize/projects`.
- Bundle manifests should be generic enough for both text-generation style models and RNNT component groups.
- Verification and export CLIs should be generic, with the project selected by explicit metadata instead of hidden assumptions.
- Existing behavior for punctuation must remain reproducible after migration.
- Existing FP32 Zipformer bundle stays as a **reference bundle**, not a deployment artifact.

## Planned File Map

### Shared Model Bundle Platform

- Create: `python-model-test/model_bundle/__init__.py` - package boundary for the shared bundle framework.
- Create: `python-model-test/model_bundle/contracts.py` - shared dataclasses and protocol interfaces for bundle projects, artifacts, and verification reports.
- Create: `python-model-test/model_bundle/manifest.py` - generic manifest schema and path resolution helpers.
- Create: `python-model-test/model_bundle/layout.py` - canonical directory and file layout helpers.
- Create: `python-model-test/model_bundle/fixtures.py` - shared sample and expected-output fixture serialization.
- Create: `python-model-test/model_bundle/exporter.py` - generic export orchestration that dispatches to a project adapter.
- Create: `python-model-test/model_bundle/verifier.py` - generic verification runner that dispatches to a project adapter.
- Create: `python-model-test/model_bundle/projects/__init__.py` - registry for bundle project adapters.
- Create: `python-model-test/model_bundle/projects/vpcd.py` - punctuation-specific adapter using tokenizer graphs and bridge maps.
- Create: `python-model-test/model_bundle/projects/zipformer.py` - RNNT-specific adapter using encoder/decoder/joiner plus tokens.
- Create: `python-model-test/model_bundle/README.md` - operator docs for the shared bundle system.

### CLI And Smoke Runner Integration

- Create: `python-model-test/export/model_bundle.py` - generic CLI to export a bundle by project.
- Create: `python-model-test/verify/model_bundle.py` - generic CLI to verify a bundle by project.
- Modify: `python-model-test/test/test_punctuation_model_onnx.py` - use the shared `model_bundle` runtime path for bundle-manifest mode.
- Modify: `python-model-test/test/test_acoustic_model_onnx.py` - use the shared `model_bundle` runtime path for bundle-manifest mode.
- Modify: `python-model-test/test/test_vpcd_bundle.py` - migrate tests to the shared model bundle adapter model.
- Modify: `python-model-test/test/test_zipformer_bundle.py` - rename/reframe around shared bundle behavior.
- Modify: `python-model-test/test/README.md` - document the shared bundle commands and domain-specific smoke runners.

### Legacy Module Removal

- Delete: `python-model-test/acoustic_bundle/__init__.py`
- Delete: `python-model-test/acoustic_bundle/manifest.py`
- Delete: `python-model-test/acoustic_bundle/runtime.py`
- Delete: `python-model-test/acoustic_bundle/exporter.py`
- Delete: `python-model-test/acoustic_bundle/verifier.py`
- Delete: `python-model-test/acoustic_bundle/eval_fixtures.py`
- Modify or delete: `python-model-test/android_bundle/*` after migrating code into `model_bundle`.
- Modify: `python-model-test/acoustic_bundle/README.md` - replace with redirect or remove if empty after migration.
- Modify: `python-model-test/android_bundle/README.md` - replace with redirect or remove if empty after migration.

### Multi-Project Quantization Framework

- Create: `python-model-test/quantize/projects/__init__.py` - registry for quantization project adapters.
- Create: `python-model-test/quantize/projects/vpcd.py` - punctuation-specific calibration and exclusion logic.
- Create: `python-model-test/quantize/projects/zipformer.py` - Zipformer-specific component metadata, calibration, fixed-shape policy, and quality gates.
- Create: `python-model-test/quantize/fixed_shapes.py` - reusable fixed-shape helpers for component-based models.
- Create: `python-model-test/quantize/reports.py` - shared report dataclasses for quantization and evaluation.
- Create: `python-model-test/quantize/evaluate.py` - project-dispatched evaluation entry points.
- Modify: `python-model-test/quantize/cli.py` - add `--project` and remove punctuation-only assumptions.
- Modify: `python-model-test/quantize/config.py` - move defaults behind project adapters.
- Modify: `python-model-test/quantize/types.py` - add project and component metadata.
- Modify: `python-model-test/quantize/calibration.py` - split text and audio calibration through project adapters.
- Modify: `python-model-test/quantize/presets.py` - move preset definitions behind project adapters.
- Modify: `python-model-test/quantize/qnn.py` - support both single-model and component-wise QDQ workflows.
- Modify: `python-model-test/quantize/runner.py` - orchestrate project-aware quantization and output staging.
- Modify: `python-model-test/quantize/README.md` - document `vpcd` and `zipformer` flows side by side.

### Plans And Repo Docs

- Modify: `python-model-test/docs/superpowers/plans/README.md` - make this plan the canonical architecture and quantization plan.
- Modify: `python-model-test/docs/superpowers/plans/2026-04-01-zipformer-python-first-bundle-qnn.md` - this file becomes the canonical shared-platform plan.

## Standardized Output Layout

All exported bundles should share one root pattern:

```text
python-model-test/build/model_bundle/<project>/<variant>/
  bundle_manifest.json
  fixtures.jsonl or sample_manifest.jsonl
  expected_outputs.jsonl
  ...project-specific artifacts...
```

Examples:

```text
python-model-test/build/model_bundle/vpcd/fp32/
  bundle_manifest.json
  tokenizer.encode.onnx
  tokenizer.decode.onnx
  tokenizer.to_model_id_map.json
  tokenizer.from_model_id_map.json
  golden_samples.jsonl
  model.mobile.onnx

python-model-test/build/model_bundle/zipformer/fp32/
  bundle_manifest.json
  encoder.onnx
  decoder.onnx
  joiner.onnx
  tokens.txt
  sample_manifest.jsonl
  expected_outputs.jsonl

python-model-test/build/model_bundle/zipformer/qnn_u16u8/
  bundle_manifest.json
  encoder.onnx
  decoder.onnx
  joiner.onnx
  tokens.txt
  sample_manifest.jsonl
  expected_outputs.jsonl
  quantization_report.json
  evaluation_report.json
```

## Success Gates

Do not call this phase complete until all of these are true:

- `model_bundle` is the only architectural bundle root in Python.
- `acoustic_bundle` has been removed.
- `android_bundle` no longer contains logic that exists only for punctuation; any remaining files are thin compatibility wrappers or are removed.
- `python -m export.model_bundle --project vpcd` works.
- `python -m export.model_bundle --project zipformer` works.
- `python -m verify.model_bundle --project vpcd` works.
- `python -m verify.model_bundle --project zipformer` works.
- `python -m quantize --project vpcd` works.
- `python -m quantize --project zipformer` works.
- `test/test_punctuation_model_onnx.py` and `test/test_acoustic_model_onnx.py` still pass in both model-dir and bundle-manifest modes.
- `build/model_bundle/zipformer/qnn_u16u8` can be produced with reports.

## Task 1: Introduce The Shared `model_bundle` Core

**Files:**
- Create: `python-model-test/model_bundle/__init__.py`
- Create: `python-model-test/model_bundle/contracts.py`
- Create: `python-model-test/model_bundle/manifest.py`
- Create: `python-model-test/model_bundle/layout.py`
- Create: `python-model-test/model_bundle/fixtures.py`
- Create: `python-model-test/model_bundle/projects/__init__.py`
- Create: `python-model-test/test/test_model_bundle_core.py`

- [ ] **Step 1: Write a failing test for generic manifest round-trip**

```python
def test_model_bundle_manifest_round_trips_generic_artifacts():
    manifest = ModelBundleManifest(...)
    restored = ModelBundleManifest.from_dict(manifest.to_dict())
    assert restored.project == "vpcd"
```

- [ ] **Step 2: Write a failing test for project registry lookup**

```python
def test_model_bundle_project_registry_resolves_vpcd_and_zipformer():
    assert resolve_bundle_project("vpcd").name == "vpcd"
    assert resolve_bundle_project("zipformer").name == "zipformer"
```

- [ ] **Step 3: Run the tests and confirm they fail**

Run:

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m pytest python-model-test/test/test_model_bundle_core.py -q
```

Expected: FAIL because the shared bundle core does not exist yet.

- [ ] **Step 4: Implement the shared manifest, contracts, fixtures, and project registry**
- [ ] **Step 5: Re-run the tests**

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add python-model-test/model_bundle python-model-test/test/test_model_bundle_core.py
git commit -m "feat: add shared model bundle core"
```

## Task 2: Migrate VPCD From `android_bundle` To `model_bundle`

**Files:**
- Create: `python-model-test/model_bundle/projects/vpcd.py`
- Modify: `python-model-test/test/test_punctuation_model_onnx.py`
- Modify: `python-model-test/test/test_vpcd_bundle.py`
- Modify: `python-model-test/android_bundle/README.md`
- Modify or delete: `python-model-test/android_bundle/*.py`

- [ ] **Step 1: Write a failing test that the shared bundle runtime can export and verify VPCD**
- [ ] **Step 2: Run the test and confirm it fails**
- [ ] **Step 3: Move punctuation-specific export, verify, runtime, and tokenizer-bridge logic into the `vpcd` project adapter**
- [ ] **Step 4: Rewire punctuation smoke tests to use `model_bundle`**
- [ ] **Step 5: Re-run the tests**

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add python-model-test/model_bundle/projects/vpcd.py python-model-test/test/test_punctuation_model_onnx.py python-model-test/test/test_vpcd_bundle.py python-model-test/android_bundle
 git commit -m "refactor: migrate vpcd bundle flow to shared model bundle"
```

## Task 3: Migrate Zipformer And Remove `acoustic_bundle`

**Files:**
- Create: `python-model-test/model_bundle/projects/zipformer.py`
- Modify: `python-model-test/test/test_acoustic_model_onnx.py`
- Modify: `python-model-test/test/test_zipformer_bundle.py`
- Delete: `python-model-test/acoustic_bundle/*`
- Modify: `python-model-test/acoustic_bundle/README.md`

- [ ] **Step 1: Write a failing test that the shared bundle runtime can export and verify Zipformer**
- [ ] **Step 2: Run the test and confirm it fails**
- [ ] **Step 3: Move acoustic bundle manifest, runtime, exporter, verifier, and fixtures into the `zipformer` project adapter**
- [ ] **Step 4: Delete `acoustic_bundle` once all imports are moved**
- [ ] **Step 5: Re-run the tests**

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add python-model-test/model_bundle/projects/zipformer.py python-model-test/test/test_acoustic_model_onnx.py python-model-test/test/test_zipformer_bundle.py python-model-test/acoustic_bundle
git commit -m "refactor: migrate zipformer bundle flow and remove acoustic_bundle"
```

## Task 4: Replace Per-Model Bundle CLIs With Generic CLIs

**Files:**
- Create: `python-model-test/export/model_bundle.py`
- Create: `python-model-test/verify/model_bundle.py`
- Modify: `python-model-test/export_acoustic_bundle.py`
- Modify: `python-model-test/verify_acoustic_bundle.py`
- Modify: `python-model-test/verify_android_punctuation_bundle.py`
- Modify: `python-model-test/test/README.md`

- [ ] **Step 1: Write a failing test for `python -m export.model_bundle --project vpcd`**
- [ ] **Step 2: Write a failing test for `python -m export.model_bundle --project zipformer`**
- [ ] **Step 3: Run the tests and confirm they fail**
- [ ] **Step 4: Implement the generic export and verify CLIs**
- [ ] **Step 5: Turn old per-model scripts into thin wrappers or deprecate them explicitly**
- [ ] **Step 6: Re-run the tests**

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add python-model-test/export python-model-test/verify python-model-test/export_acoustic_bundle.py python-model-test/verify_acoustic_bundle.py python-model-test/verify_android_punctuation_bundle.py python-model-test/test/README.md
git commit -m "refactor: add generic model bundle CLIs"
```

## Task 5: Refactor `quantize` Into A Multi-Project Framework

**Files:**
- Create: `python-model-test/quantize/projects/__init__.py`
- Create: `python-model-test/quantize/projects/vpcd.py`
- Create: `python-model-test/quantize/projects/zipformer.py`
- Modify: `python-model-test/quantize/cli.py`
- Modify: `python-model-test/quantize/config.py`
- Modify: `python-model-test/quantize/types.py`
- Modify: `python-model-test/quantize/calibration.py`
- Modify: `python-model-test/quantize/presets.py`
- Modify: `python-model-test/quantize/qnn.py`
- Modify: `python-model-test/quantize/runner.py`
- Create: `python-model-test/test/test_quantize_projects.py`

- [ ] **Step 1: Write a failing test for `quantize --project vpcd` dry-run**
- [ ] **Step 2: Write a failing test for `quantize --project zipformer` dry-run**
- [ ] **Step 3: Run the tests and confirm they fail**
- [ ] **Step 4: Add project adapters and move project-specific defaults out of shared files**
- [ ] **Step 5: Make the CLI route calibration, presets, and output planning through the selected project**
- [ ] **Step 6: Re-run the tests**

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add python-model-test/quantize python-model-test/test/test_quantize_projects.py
git commit -m "refactor: make quantize multi-project for vpcd and zipformer"
```

## Task 6: Implement Zipformer Fixed-Shape PTQ QDQ On Top Of The New Framework

**Files:**
- Create: `python-model-test/quantize/fixed_shapes.py`
- Create: `python-model-test/quantize/reports.py`
- Create: `python-model-test/quantize/evaluate.py`
- Modify: `python-model-test/quantize/projects/zipformer.py`
- Modify: `python-model-test/model_bundle/projects/zipformer.py`
- Create: `python-model-test/test/test_zipformer_quantize.py`

- [ ] **Step 1: Write a failing test for fixed-shape Zipformer planning**
- [ ] **Step 2: Write a failing test for quantization/evaluation report emission**
- [ ] **Step 3: Run the tests and confirm they fail**
- [ ] **Step 4: Implement fixed-shape prep, component-wise QDQ, and evaluation through the shared framework**
- [ ] **Step 5: Stage the candidate bundle at `build/model_bundle/zipformer/qnn_u16u8` only after evaluation passes**
- [ ] **Step 6: Re-run the tests**

Expected: PASS

- [ ] **Step 7: Verify the real workflow**

Run:

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m quantize --project zipformer --preset zipformer_sd8g2_qnn_u16u8 --model-dir python-model-test/assets/zipformer --output-root python-model-test/build/quantize/zipformer/qnn_u16u8

& D:\Anaconda\envs\speech2text\python.exe -m verify.model_bundle --project zipformer --reference-bundle python-model-test/build/model_bundle/zipformer/fp32 --candidate-bundle python-model-test/build/model_bundle/zipformer/qnn_u16u8
```

Expected: quantized component outputs and both reports are produced, and candidate bundle verification completes.

- [ ] **Step 8: Commit**

```bash
git add python-model-test/quantize python-model-test/model_bundle/projects/zipformer.py python-model-test/test/test_zipformer_quantize.py
git commit -m "feat: add zipformer fixed-shape ptq qdq candidate bundle"
```

## Task 7: Update Docs And Remove Legacy Architectural Drift

**Files:**
- Modify: `python-model-test/model_bundle/README.md`
- Modify: `python-model-test/quantize/README.md`
- Modify: `python-model-test/test/README.md`
- Modify: `python-model-test/docs/superpowers/plans/README.md`
- Modify or remove: `python-model-test/android_bundle/README.md`
- Modify or remove: `python-model-test/acoustic_bundle/README.md`

- [ ] **Step 1: Document the new shared `model_bundle` architecture and project adapters**
- [ ] **Step 2: Document the multi-project `quantize` workflow for `vpcd` and `zipformer`**
- [ ] **Step 3: Document compatibility wrappers or explicit removals for legacy scripts and modules**
- [ ] **Step 4: Re-run the core verification commands**

Run:

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m pytest python-model-test/test/test_model_bundle_core.py -q
& D:\Anaconda\envs\speech2text\python.exe -m pytest python-model-test/test/test_quantize_projects.py -q
& D:\Anaconda\envs\speech2text\python.exe -m pytest python-model-test/test/test_zipformer_quantize.py -q
& D:\Anaconda\envs\speech2text\python.exe -m quantize --project vpcd --dry-run --model-dir python-model-test/assets/vietnamese-punc-cap-denorm-v1
& D:\Anaconda\envs\speech2text\python.exe -m quantize --project zipformer --dry-run --model-dir python-model-test/assets/zipformer
```

Expected: PASS and both dry-runs print project-specific plans without leaking assumptions from the other model.

- [ ] **Step 5: Commit**

```bash
git add python-model-test/model_bundle/README.md python-model-test/quantize/README.md python-model-test/test/README.md python-model-test/docs/superpowers/plans/README.md python-model-test/android_bundle python-model-test/acoustic_bundle
git commit -m "docs: finalize shared bundle and multi-project quantize architecture"
```

## Final Verification Checklist

- [ ] `pytest python-model-test/test/test_model_bundle_core.py -q`
- [ ] `pytest python-model-test/test/test_vpcd_bundle.py -q`
- [ ] `pytest python-model-test/test/test_zipformer_bundle.py -q`
- [ ] `pytest python-model-test/test/test_quantize_projects.py -q`
- [ ] `pytest python-model-test/test/test_zipformer_quantize.py -q`
- [ ] `python -m quantize --project vpcd --dry-run --model-dir python-model-test/assets/vietnamese-punc-cap-denorm-v1`
- [ ] `python -m quantize --project zipformer --dry-run --model-dir python-model-test/assets/zipformer`
- [ ] `python -m verify.model_bundle --project vpcd --bundle-dir python-model-test/build/model_bundle/vpcd/fp32`
- [ ] `python -m verify.model_bundle --project zipformer --reference-bundle python-model-test/build/model_bundle/zipformer/fp32 --candidate-bundle python-model-test/build/model_bundle/zipformer/qnn_u16u8`
- [ ] Confirm `acoustic_bundle` no longer exists as an implementation module

## Notes For `bkmeeting`

- Android should target the shared Python `model_bundle` contract instead of model-specific bundle modules.
- The first Android bundle migration should reuse the punctuation runtime behavior through a shared bundle core, then consume Zipformer through the same contract.
- Android should stage `build/model_bundle/zipformer/qnn_u16u8` only after the Python evaluation gate passes.


