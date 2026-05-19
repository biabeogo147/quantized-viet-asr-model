# 2026-05-19 Option 1 Phase 6 Contract Sync Plan

## Goal

Implement the next handoff phase after `Phase 5` packaging:

- consume the retained Phase 5 contract packages
- sync them into BKMeeting through one repeatable Python command
- preserve `bundle_manifest.json` as the Android entrypoint
- attach enough Option 1 metadata and packaged evidence for future runtime integration work

This plan assumes the next clean rerun will use one shared label:

- Zipformer:
  - `RUN_LABEL = 20260519-option1-final-rerun`
- VPCD:
  - `RUN_LABEL = 20260519-option1-final-rerun`

## Starting Point

What already exists:

- the contract packaging helpers and notebooks are ready
- retained candidate bundles:
  - `build/model_bundle/zipformer/qnn_u16u8/`
  - `build/model_bundle/vpcd/qnn_fixed_1024x128/`
- existing BKMeeting asset namespaces already present:
  - `BKMeeting/modelassets/src/main/assets/models/asr/zipformer/qnn_u16u8/`
  - `BKMeeting/modelassets/src/main/assets/models/punctuation/vpcd/qnn_fixed_1024x128/`

What does not exist yet:

- `build/aihub/` was intentionally reset for the final rerun, so fresh Phase 5 packages must be regenerated first
- `sync_android_bundle.py` only knows how to copy a classic bundle directory
- it does not accept a Phase 5 contract package as an input
- it does not copy packaged Option 1 evidence into BKMeeting
- it does not rewrite `bundle_manifest.json` with Option 1 contract metadata
- tests only cover the legacy bundle-only sync mode

## Phase 6 Scope

### In scope

- extend `sync_android_bundle.py` with a contract-aware sync mode
- define one Android-facing on-disk layout for packaged Option 1 evidence
- write tests for both:
  - `deployment_candidate` Zipformer contract sync
  - `research_only` VPCD contract sync
- dry-run the new sync mode into the sibling `BKMeeting` repo
- update operator docs to use the new contract-aware command

### Out of scope

- BKMeeting runtime code changes
- Android benchmark runs
- revisiting archived VPCD quantize fallback paths

## Phase 6 Payload Decision

The Android-facing payload should stay bundle-first and add a colocated contract subtree.

Recommended target layout inside each synced asset namespace:

- existing bundle payload files remain at the asset root
- add:
  - `option1_contract/contract_manifest.json`
  - `option1_contract/io_contract.json`
  - `option1_contract/contract_summary.md`
  - `option1_contract/evidence/...`

Recommended manifest enrichment:

- keep top-level bundle fields unchanged
- write new `metadata.option1_contract` with:
  - `enabled = true`
  - `run_label`
  - `promotion_status`
  - `phase4_recommendation`
  - `target_runtime = precompiled_qnn_onnx`
  - `target_model_id`
  - `contract_dir = option1_contract`
  - `contract_manifest = option1_contract/contract_manifest.json`
  - `io_contract = option1_contract/io_contract.json`

Why this layout:

- Android still enters through `bundle_manifest.json`
- Option 1 evidence remains colocated and versioned with the synced bundle
- `research_only` VPCD can still be synced for experimentation without pretending it is deployable

## Implementation Tasks

### Task 1: Add contract-aware sync inputs

Files:

- `src/tools/sync_android_bundle.py`
- `test/test_sync_android_bundle.py`

Changes:

- add a new CLI / API input for a Phase 5 contract package path
- keep legacy bundle-only mode working
- require that contract-aware mode still receives a source bundle so the classic payload stays intact

Expected API shape:

- keep:
  - `project`
  - `variant`
  - `source_bundle`
  - `bkmeeting_root`
  - `overwrite`
- add something like:
  - `contract_package`

### Task 2: Copy the contract subtree into BKMeeting

Files:

- `src/tools/sync_android_bundle.py`
- `test/test_sync_android_bundle.py`

Changes:

- copy `contract_manifest.json`
- copy `io_contract.json`
- copy `contract_summary.md`
- copy the entire `evidence/` subtree
- place them under `option1_contract/` in the destination asset namespace

Pass condition:

- one sync command leaves a self-contained asset directory with:
  - bundle payload
  - rewritten bundle manifest
  - Option 1 contract subtree

### Task 3: Enrich the synced manifest

Files:

- `src/tools/sync_android_bundle.py`
- possibly `src/model_bundle/manifest.py` only if helper ergonomics need to improve
- `test/test_sync_android_bundle.py`

Changes:

- preserve existing bundle metadata
- add `metadata.option1_contract`
- populate it from the packaged Phase 5 `contract_manifest.json`

Important rule:

- do not flatten the entire Phase 5 contract into top-level manifest fields
- keep all Option 1-specific additions under one nested metadata object

### Task 4: Support both promotion states cleanly

Files:

- `src/tools/sync_android_bundle.py`
- `test/test_sync_android_bundle.py`

Changes:

- allow sync for:
  - `deployment_candidate`
  - `research_only`
- do not block VPCD just because the current contract is `research_only`
- expose promotion state in the synced manifest metadata
- print a clear summary in CLI output when the synced contract is not deployable

### Task 5: Update docs and operator flow

Files:

- `docs/workflows/android-handoff.md`
- `src/tools/README.md`
- `docs/workflows/aihub-option1-phase5-contract.md`
- `docs/plans/active/2026-05-19-bkmeeting-android-option1-export-plan.md`

Changes:

- document the new contract-aware sync mode
- show one canonical command per model family
- explain where `option1_contract/` lands under BKMeeting assets
- record that current VPCD sync is valid for experimentation, but remains `research_only`

### Task 6: Run a real BKMeeting dry run

Target:

- `D:/DS-AI/BKMeeting-Research/BKMeeting`

Checks:

- Zipformer contract sync lands in:
  - `modelassets/src/main/assets/models/asr/zipformer/qnn_u16u8/`
- VPCD contract sync lands in:
  - `modelassets/src/main/assets/models/punctuation/vpcd/qnn_fixed_1024x128/`
- each target contains:
  - bundle payload
  - `bundle_manifest.json`
  - `option1_contract/`

## Verification Plan

Minimum verification for the implementation pass:

1. `pytest test/test_sync_android_bundle.py -v`
2. `pytest test/test_aihub_option1_phase5_contract.py -v`
3. one BKMeeting dry run for Zipformer
4. one BKMeeting dry run for VPCD
5. inspect the rewritten `bundle_manifest.json` files under BKMeeting

## Expected Risks

- overloading `bundle_manifest.json` with too much contract detail
- accidentally mixing absolute Python-side evidence paths into Android-facing metadata
- letting contract-aware sync break legacy bundle-only sync

Mitigation:

- keep contract payload on disk under `option1_contract/`
- store only relative pointers in Android-facing manifest metadata
- keep legacy tests and add explicit contract-mode tests rather than replacing them

## Success Criteria

Phase 6 is complete when all of these are true:

- `sync_android_bundle.py` can sync a bundle plus a Phase 5 contract package together
- BKMeeting receives the payload without manual copying
- `bundle_manifest.json` remains the entrypoint
- the synced manifest clearly records whether the contract is:
  - `deployment_candidate`
  - `research_only`
- docs explain the exact dry-run command and resulting asset layout
