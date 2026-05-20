# Phase 5.5 Export Retirement And Model Bundle Prune Plan

> Status: completed on 2026-05-20. Execution notes live in `docs/plans/archive/2026-05-20-phase-5-5-export-retirement-and-model-bundle-prune-results.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retire `src/export/` completely and reduce `src/model_bundle/` to only the support surface still required by the retained AI Hub lane and the Android handoff surface, while migrating every remaining non-AIHub caller to a more appropriate home under `src/quantize/`, `src/verify/`, or `src/tools/`.

**Architecture:** Make `src/aihub/` the owner of AI Hub compile, evaluation, deployment, and Android-ready bundle materialization. Remove `src/export/` as an obsolete wrapper layer. Re-home non-AIHub bundle export, bundle verification, and QNN preflight logic out of `src/model_bundle/`. Keep `src/model_bundle/` only as a thin internal contract package for the AI Hub path and shared Android handoff pieces. If that keep-set collapses far enough by the end of the refactor, open a follow-up plan to delete `src/model_bundle/` entirely.

**Tech Stack:** Python 3, `src/aihub`, `src/quantize`, `src/verify`, `src/tools`, pytest, AST/import-scan tests, manifest JSON contract, Android handoff docs

---

## Current Reality

This cleanup should start from the real import graph, not from the intended future shape.

### `src/export/` is easy to retire

Current package contents:

- `src/export/model_bundle.py`
- `src/export/punctuation_onnx.py`

Current reality:

- `export.model_bundle` is only a thin wrapper over `model_bundle.exporter` and the project adapter registry.
- `export.punctuation_onnx` is a standalone helper with no live code imports; it is currently referenced only by docs.

### `src/model_bundle/` is not yet AI-Hub-only

Live `src/` callers still importing `model_bundle` today:

- `src/aihub/session.py`
- `src/aihub/evaluation.py`
- `src/verify/model_bundle.py`
- `src/verify/qnn_preflight.py`
- `src/quantize/evaluate.py`
- `src/quantize/projects/zipformer.py`
- `src/quantize/projects/vpcd.py`
- `src/tools/sync_android_bundle.py`
- `src/tools/prepare_vpcd_qnn_candidate.py`

That means the cleanup cannot safely prune `model_bundle` by AI Hub usage alone until the non-AIHub callers have been migrated first.

### Current AI Hub keep-set

The retained AI Hub lane currently imports these `model_bundle` surfaces:

- `model_bundle.fixtures`
  - `AudioSampleFixture`
  - `read_jsonl`
- `model_bundle.manifest`
  - `ModelBundleManifest`
- `model_bundle.projects.vpcd_shapes`
  - `resolve_vpcd_model_input_shapes`
  - `pad_token_row`
  - `attention_mask_for_length`
- `model_bundle.projects.zipformer`
  - `ModelDirAcousticRuntime`
  - `BundleAcousticRuntime`
  - `prepare_encoder_inputs`
  - `resolve_fixed_encoder_frames`
  - `trim_encoder_frames`
  - `decode_encoder_frames_greedy`
- `model_bundle.projects._vpcd_support`
  - `BundleOnnxRuntime`

### Current non-AIHub `model_bundle` surfaces that should not stay there

These pieces are currently in `model_bundle`, but they are not part of the AI Hub keep-set:

- `model_bundle.exporter`
- `model_bundle.verifier`
- `model_bundle.contracts`
- `model_bundle.projects.__init__`
- `model_bundle.projects.vpcd`
- `model_bundle.qnn_preflight`
- bundle-export paths inside `model_bundle.projects.zipformer`
- bundle-verification paths inside `model_bundle.projects.zipformer`
- helper layout routing in `model_bundle.layout`

## Scope Lock

In scope:

- delete `src/export/`
- replace or retire `python -m export.*` command surfaces
- migrate non-AIHub imports off `model_bundle`
- split mixed `model_bundle` files so AI Hub keeps only the helpers it actually needs
- update tests and docs to enforce the new ownership boundaries

Out of scope:

- changing AI Hub compile or deployment semantics
- changing BKMeeting Android runtime code
- deleting archived docs or historical plans
- deleting the remaining `model_bundle` keep-set if it is still materially used after this phase

## Target State

After Phase 5.5, the intended module ownership is:

- `src/aihub/`
  - owns AI Hub session, evaluation, deployment, and Android-ready bundle synthesis
- `src/quantize/`
  - owns local candidate generation and any retained bundle-production helpers needed by quantize flows
- `src/verify/`
  - owns bundle verification CLI and QNN preflight logic
- `src/tools/`
  - owns standalone helper CLIs that are not AI Hub or quantize framework concerns
- `src/model_bundle/`
  - keeps only the minimal manifest, fixtures, shape helpers, and tiny runtime helpers still needed by AI Hub or Android handoff

### Explicit retirement goals

- `src/export/` no longer exists
- `python -m export.model_bundle` no longer exists
- `python -m export.punctuation_onnx` no longer exists
- no non-AIHub `src/` package imports `model_bundle.exporter`, `model_bundle.verifier`, `model_bundle.contracts`, `model_bundle.projects.__init__`, or `model_bundle.projects.vpcd`

### Recommended post-cleanup command surface

- AI Hub Android bundle materialization:
  - `python -m aihub.android_bundle ...`
- retained standalone punctuation ONNX export, if still needed:
  - move to `python -m tools.punctuation_onnx ...`
- bundle verification CLI:
  - keep under `python -m verify.model_bundle ...`, but make it own its logic instead of routing through `model_bundle.verifier`
- QNN preflight CLI:
  - keep under `python -m verify.qnn_preflight ...`, but make it own its core logic instead of routing through `model_bundle.qnn_preflight`

## Planned File Map

### Delete

- Delete: `python-model-test/src/export/__init__.py`
- Delete: `python-model-test/src/export/model_bundle.py`
- Delete: `python-model-test/src/export/punctuation_onnx.py`
- Delete after migration: `python-model-test/src/model_bundle/exporter.py`
- Delete after migration: `python-model-test/src/model_bundle/verifier.py`
- Delete after migration: `python-model-test/src/model_bundle/contracts.py`
- Delete after migration: `python-model-test/src/model_bundle/projects/__init__.py`
- Delete after migration: `python-model-test/src/model_bundle/projects/vpcd.py`
- Delete after migration: `python-model-test/src/model_bundle/qnn_preflight.py`

### Create or move

- Create: `python-model-test/src/aihub/android_bundle.py`
- Create if retained: `python-model-test/src/tools/punctuation_onnx.py`
- Create: `python-model-test/src/verify/bundle_projects.py`
- Create: `python-model-test/src/verify/bundle_runtime.py`
- Create: `python-model-test/src/verify/qnn_preflight_core.py`
- Create: `python-model-test/src/model_bundle/zipformer_runtime.py`
- Create: `python-model-test/src/model_bundle/vpcd_runtime.py`
- Create: `python-model-test/src/tools/bundle_paths.py`

### Modify

- Modify: `python-model-test/src/verify/model_bundle.py`
- Modify: `python-model-test/src/verify/qnn_preflight.py`
- Modify: `python-model-test/src/quantize/evaluate.py`
- Modify: `python-model-test/src/quantize/projects/zipformer.py`
- Modify: `python-model-test/src/quantize/projects/vpcd.py`
- Modify: `python-model-test/src/tools/sync_android_bundle.py`
- Modify: `python-model-test/src/tools/prepare_vpcd_qnn_candidate.py`
- Modify: `python-model-test/src/aihub/session.py`
- Modify: `python-model-test/src/aihub/evaluation.py`
- Modify: `python-model-test/src/model_bundle/__init__.py`
- Modify: `python-model-test/src/model_bundle/README.md`
- Modify: `python-model-test/src/verify/README.md`
- Modify: `python-model-test/docs/architecture/overview.md`
- Modify: `python-model-test/docs/workflows/export-verify-smoke.md`
- Modify: `python-model-test/docs/workflows/android-handoff.md`

### Tests to add or update

- Modify or replace: `python-model-test/test/test_export_verify_modules.py`
- Create: `python-model-test/test/test_phase55_import_boundaries.py`
- Modify: `python-model-test/test/test_src_layout_bootstrap.py`
- Modify: `python-model-test/test/test_sync_android_bundle.py`
- Modify: `python-model-test/test/test_qnn_preflight.py`
- Modify: `python-model-test/test/test_aihub_session.py`
- Modify: `python-model-test/test/test_aihub_evaluation.py`
- Modify: `python-model-test/test/test_zipformer_bundle.py`
- Modify: `python-model-test/test/test_vpcd_bundle.py`
- Modify: `python-model-test/test/test_zipformer_quantize.py`

## Success Gates

Do not call Phase 5.5 complete until all of these are true:

- `src/export/` is fully deleted
- no maintained doc still recommends `python -m export.model_bundle`
- no maintained doc still recommends `python -m export.punctuation_onnx`
- `src/verify/`, `src/quantize/`, and `src/tools/` no longer import `model_bundle` surfaces outside the approved keep-set
- `model_bundle` no longer contains generic exporter, verifier, project-registry, or QNN-preflight logic
- `src/aihub/` still passes its current tests after the prune
- `tools.sync_android_bundle` and the new Android handoff lane still work after the ownership change

## Task 1: Freeze The Keep-Set And Add Boundary Tests

**Files:**

- Create: `python-model-test/test/test_phase55_import_boundaries.py`
- Modify: `python-model-test/test/test_src_layout_bootstrap.py`

- [ ] **Step 1: Write failing import-boundary tests**

Cover at least:

- `export` package must not be importable in the final state
- `src/verify`, `src/quantize`, and `src/tools` must not import forbidden `model_bundle` modules
- only approved `model_bundle` modules remain imported by `src/aihub`

- [ ] **Step 2: Run the boundary tests and confirm failure**

Run:

```bash
pytest python-model-test/test/test_phase55_import_boundaries.py -v
```

Expected: FAIL because the old layout is still live.

- [ ] **Step 3: Lock the approved `model_bundle` keep-set**

Approved keep-set for this phase:

- `model_bundle.manifest`
- `model_bundle.fixtures`
- `model_bundle.projects.vpcd_shapes`
- the minimal zipformer and VPCD runtime helpers that `src/aihub` still imports

Everything else should migrate or be deleted.

## Task 2: Retire `src/export/` Completely

**Files:**

- Delete: `python-model-test/src/export/__init__.py`
- Delete: `python-model-test/src/export/model_bundle.py`
- Delete or move: `python-model-test/src/export/punctuation_onnx.py`
- Modify: `python-model-test/docs/workflows/export-verify-smoke.md`
- Modify: `python-model-test/docs/architecture/overview.md`
- Modify: `python-model-test/test/test_export_verify_modules.py`

- [ ] **Step 1: Decide the fate of `punctuation_onnx`**

Recommended rule:

- if the helper is still useful, move it to `src/tools/punctuation_onnx.py`
- if it is no longer a maintained flow, delete it and remove the docs

- [ ] **Step 2: Remove the export wrapper entrypoints**

Requirements:

- no maintained code path shells through `export.model_bundle`
- docs point to the new owners instead of to `src/export/`

- [ ] **Step 3: Replace the old wrapper tests**

Replace “delegates to shared exporter” tests with:

- package removal tests
- command-surface tests for the replacement modules

- [ ] **Step 4: Run the focused tests**

Run:

```bash
pytest python-model-test/test/test_export_verify_modules.py python-model-test/test/test_phase55_import_boundaries.py -v
```

Expected: PASS.

## Task 3: Re-Home Verification-Owned Logic Out Of `model_bundle`

**Files:**

- Modify: `python-model-test/src/verify/model_bundle.py`
- Modify: `python-model-test/src/verify/qnn_preflight.py`
- Create: `python-model-test/src/verify/bundle_projects.py`
- Create: `python-model-test/src/verify/bundle_runtime.py`
- Create: `python-model-test/src/verify/qnn_preflight_core.py`
- Delete after cutover: `python-model-test/src/model_bundle/verifier.py`
- Delete after cutover: `python-model-test/src/model_bundle/qnn_preflight.py`
- Delete after cutover: `python-model-test/src/model_bundle/contracts.py`
- Delete after cutover: `python-model-test/src/model_bundle/projects/__init__.py`
- Delete after cutover: `python-model-test/src/model_bundle/projects/vpcd.py`

- [ ] **Step 1: Write failing tests for verify-owned imports**

Cover at least:

- `verify.model_bundle` no longer routes through `model_bundle.verifier`
- `verify.qnn_preflight` no longer routes through `model_bundle.qnn_preflight`
- project dispatch for bundle verification lives under `src/verify`

- [ ] **Step 2: Move verification dispatch and preflight logic**

Requirements:

- keep CLI behavior stable where practical
- make `verify` own its registry and verification implementation
- keep QNN preflight under `src/verify` because it is not part of the retained AI Hub keep-set

- [ ] **Step 3: Re-run focused verify tests**

Run:

```bash
pytest python-model-test/test/test_qnn_preflight.py -v
```

Expected: PASS.

## Task 4: Re-Home Quantize-Owned Bundle Logic Out Of `model_bundle`

**Files:**

- Modify: `python-model-test/src/quantize/evaluate.py`
- Modify: `python-model-test/src/quantize/projects/zipformer.py`
- Modify: `python-model-test/src/quantize/projects/vpcd.py`
- Modify: `python-model-test/src/tools/prepare_vpcd_qnn_candidate.py`

- [ ] **Step 1: Split mixed runtime and export code in `model_bundle.projects.zipformer`**

Move out of `model_bundle`:

- `export_bundle`
- verification helpers
- default bundle-production ownership

Keep only if AI Hub still uses them:

- runtime helpers
- encoder input preparation helpers

- [ ] **Step 2: Re-home VPCD bundle-production logic**

Move out of `model_bundle`:

- bundle export path
- bundle verification path

Keep only if AI Hub still uses them:

- runtime helpers needed by retained AI Hub evaluation

- [ ] **Step 3: Stop `quantize` from calling `model_bundle.verifier`**

`quantize.evaluate` should call verify-owned code directly under `src/verify`.

- [ ] **Step 4: Re-run focused quantize tests**

Run:

```bash
pytest python-model-test/test/test_zipformer_quantize.py python-model-test/test/test_vpcd_bundle.py python-model-test/test/test_zipformer_bundle.py -v
```

Expected: PASS.

## Task 5: Prune `model_bundle` Down To The AI Hub Keep-Set

**Files:**

- Create: `python-model-test/src/model_bundle/zipformer_runtime.py`
- Create: `python-model-test/src/model_bundle/vpcd_runtime.py`
- Create: `python-model-test/src/tools/bundle_paths.py`
- Modify: `python-model-test/src/aihub/session.py`
- Modify: `python-model-test/src/aihub/evaluation.py`
- Modify: `python-model-test/src/tools/sync_android_bundle.py`
- Modify: `python-model-test/src/model_bundle/__init__.py`
- Modify: `python-model-test/src/model_bundle/README.md`

- [ ] **Step 1: Split mixed helper files**

Recommended end state:

- `manifest.py` stays
- `fixtures.py` stays
- `projects/vpcd_shapes.py` stays
- `zipformer_runtime.py` keeps only the zipformer helpers still imported by AI Hub
- `vpcd_runtime.py` keeps only the VPCD runtime helpers still imported by AI Hub

- [ ] **Step 2: Move `resolve_bundle_dir` out of `model_bundle.layout`**

Recommended target:

- `src/tools/bundle_paths.py`

Reason:

- `layout.py` is not part of the retained AI Hub keep-set
- `tools.sync_android_bundle` should not keep `model_bundle.layout` alive by accident

- [ ] **Step 3: Update AI Hub imports to the reduced `model_bundle` surface**

Requirements:

- `src/aihub/session.py` and `src/aihub/evaluation.py` must import only the narrowed runtime helpers
- no AI Hub code should depend on deleted adapter or exporter abstractions

- [ ] **Step 4: Re-run focused AI Hub and sync tests**

Run:

```bash
pytest python-model-test/test/test_aihub_session.py python-model-test/test/test_aihub_evaluation.py python-model-test/test/test_sync_android_bundle.py -v
```

Expected: PASS.

## Task 6: Documentation Refresh And Final Verification

**Files:**

- Modify: `python-model-test/docs/architecture/overview.md`
- Modify: `python-model-test/docs/workflows/export-verify-smoke.md`
- Modify: `python-model-test/docs/workflows/android-handoff.md`
- Modify: `python-model-test/src/model_bundle/README.md`
- Modify: `python-model-test/src/verify/README.md`

- [ ] **Step 1: Rewrite the maintained command surface**

Docs must explain:

- `src/export/` is gone
- AI Hub Android bundle export lives under `src/aihub`
- verification lives under `src/verify`
- any retained standalone ONNX helper lives under `src/tools`

- [ ] **Step 2: Run the final Python verification sweep**

Run:

```bash
pytest python-model-test/test/test_phase55_import_boundaries.py -v
pytest python-model-test/test/test_aihub_session.py python-model-test/test/test_aihub_evaluation.py python-model-test/test/test_aihub_deployment.py -v
pytest python-model-test/test/test_sync_android_bundle.py python-model-test/test/test_qnn_preflight.py -v
python -m compileall python-model-test/src
```

Expected: PASS.

## Acceptance Criteria

- `src/export/` is deleted
- `verify`, `quantize`, and `tools` no longer depend on forbidden `model_bundle` modules
- the only `model_bundle` surface left is the minimal keep-set still needed by AI Hub or Android handoff
- docs no longer present `export` as a maintained package
- AI Hub tests and Android handoff tests still pass after the prune

## Recommended Execution Order

1. Add boundary tests and freeze the keep-set.
2. Delete `src/export/` and replace its maintained command surface.
3. Move verification and preflight ownership into `src/verify`.
4. Move quantize-owned bundle helpers out of `model_bundle`.
5. Split and prune `model_bundle` to the AI Hub keep-set.
6. Refresh docs and run the final verification sweep.

## Notes For The Implementer

- The key risk in this cleanup is not file deletion. The key risk is leaving one quiet live import behind and discovering it only after the package has already been deleted. Bias toward import-boundary tests early.
- Do not treat archived plans or notebooks as proof that a module is still maintained. Base this cleanup on live `src/` imports, maintained docs, and active tests.
- If `model_bundle` becomes small enough that it is effectively just `manifest + fixtures + runtime helpers`, stop there for Phase 5.5. Full package deletion can be a later cleanup once the remaining AI Hub keep-set is intentionally re-homed.
