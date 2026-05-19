# 2026-05-19 BKMeeting Android Option 1 Export Plan

## Goal

Export the currently verified `Option 1` Qualcomm AI Hub artifacts into a BKMeeting-consumable Android bundle flow without manual file shuffling.

The export path must stay aligned with the existing Python-first contract:

- `Phase 2`: compile and cloud inference evidence
- `Phase 3`: hybrid correctness evidence
- `Phase 4`: promotion gate
- `Phase 5`: package contract
- Android handoff: `python -m tools.sync_android_bundle`

## Starting Point

### Zipformer

- Current bundle candidate:
  - `build/model_bundle/zipformer/qnn_u16u8`
- Current AI Hub proof:
  - encoder-first prepared source graph
  - compile-run records under `build/aihub/records/zipformer_encoder_option1/`
- Caveat:
  - the full quantized Zipformer bundle is not yet proven end-to-end on NPU

### VPCD

- Current local bundle candidate:
  - `build/model_bundle/vpcd/qnn_fixed_1024x128`
- Current leading AI Hub-compatible local quantize lane:
  - `local_aimet_compile_candidate`
  - `w8a16 + min_max + local_quality_parity`
- Current final rerun label:
  - `20260519-option1-final-rerun`
- Current state:
  - `build/aihub/` has been intentionally cleaned
  - the next operator rerun must regenerate the fresh VPCD evidence set

## Status Update: 2026-05-19

Phase 4 and Phase 5 are now runnable against the promoted evidence layout.

What changed:

- pilot, Phase 4, and Phase 5 notebooks now all default to the clean rerun label:
  - `20260519-option1-final-rerun`
- VPCD now resolves the active compile lane directly through:
  - `vpcd_option1_local_aimet`
- VPCD Phase 4 still preserves the bounded debug contract:
  - `max_decode_steps = 5`
- all stale `build/aihub` evidence was removed so the next rerun produces one clean version only

## Export Principle

Android should keep using `bundle_manifest.json` as the entrypoint, but the BKMeeting-facing payload must grow from "just candidate bundle files" into "candidate bundle plus Option 1 evidence-backed compiled payload" where appropriate.

That means:

- do not hand-copy AI Hub artifacts into BKMeeting
- do not bypass `Phase 5` packaging
- do not treat raw AI Hub records as the Android artifact contract

Instead:

1. choose the promoted `RUN_LABEL`
2. materialize a `Phase 5` contract package
3. sync the contract-backed payload into BKMeeting asset paths
4. let BKMeeting runtime read the manifest-driven payload

## Decisions Already Locked

- The archived AI Hub quantize `A/B/C/D` investigation is no longer part of the active execution path.
- The retained VPCD direction is the AIMET parity lane only.
- The legacy `prefer_fp32_fixed -> AI Hub quantize` VPCD path has been removed from the active notebook and helper defaults.
- Do not promote the historical local-QDQ probe lane.
- Do not remove the current bundle sync entrypoint:
  - `src/tools/sync_android_bundle.py`

## Current Next Step

The remaining implementation work is now concentrated in `Phase 6`:

- first, run the fresh clean rerun from the current notebook defaults
- extend `sync_android_bundle.py` so it can consume Phase 5 contract packages
- sync those contract-backed payloads into BKMeeting
- keep Android entry through `bundle_manifest.json`

See:

- [2026-05-19-option1-vpcd-default-lane-and-workflow-refresh-plan.md](/D:/DS-AI/BKMeeting-Research/python-model-test/docs/plans/active/2026-05-19-option1-vpcd-default-lane-and-workflow-refresh-plan.md)
- [2026-05-19-option1-phase6-contract-sync-plan.md](/D:/DS-AI/BKMeeting-Research/python-model-test/docs/plans/active/2026-05-19-option1-phase6-contract-sync-plan.md)

## Scope

### In scope

- define the exact Android-facing source artifacts for Zipformer and VPCD
- define the record prerequisites that must exist before packaging
- extend the packaging and sync flow where the current tooling only covers legacy candidate bundles
- preserve `bundle_manifest.json` as the single Android entrypoint

### Out of scope

- Android runtime code changes inside BKMeeting
- Snapdragon device benchmarking inside BKMeeting
- final promotion of Zipformer as an end-to-end NPU-ready ASR stack

## Phase 1: Freeze Promotion Inputs

### Task 1.1

Choose the concrete `RUN_LABEL` per model family.

Expected initial labels:

- Zipformer:
  - `20260519-option1-final-rerun`
- VPCD:
  - `20260519-option1-final-rerun`

### Task 1.2

Verify the required records exist for each selected label:

- `prepared-artifact-<RUN_LABEL>.json`
- `compile-run-<RUN_LABEL>.json`
- `live-run-<RUN_LABEL>.json`
- `hybrid-run-<RUN_LABEL>.json`
- `phase4-gate-<RUN_LABEL>.json`

### Task 1.3

If `phase4-gate-<RUN_LABEL>.json` is missing, run the Phase 4 notebook before packaging.

Files:

- `On_device_Ai_option1_phase4_gate.ipynb`
- `src/tools/aihub_option1_phase4_gate.py`

## Phase 2: Materialize Phase 5 Contract Packages

### Task 2.1

Use the existing package-only flow to materialize contract packages:

- notebook:
  - `On_device_Ai_option1_phase5_contract.ipynb`
- helper:
  - `src/tools/aihub_option1_phase5_contract.py`

### Task 2.2

Confirm each package contains:

- `contract_manifest.json`
- `io_contract.json`
- `contract_summary.md`
- `evidence/`

### Task 2.3

Verify the package summary records the right promotion status:

- `GO` -> `deployment_candidate`
- `WARN` -> `deployment_candidate`
- `NO_GO` -> `research_only`

## Phase 3: Extend Android Handoff To Option 1 Contracts

### Why this phase is needed

`sync_android_bundle.py` currently syncs classic candidate bundles from:

- `build/model_bundle/zipformer/...`
- `build/model_bundle/vpcd/...`

That is enough for legacy candidate-bundle handoff, but not yet enough for full `Option 1` AI Hub contract packaging where Android needs compiled `precompiled_qnn_onnx` payload context as part of the handoff story.

### Task 3.1

Decide whether to:

- extend `sync_android_bundle.py` with an `Option 1 contract` input mode, or
- add a thin new sync entrypoint that consumes Phase 5 packages and writes the Android-facing payload

Default recommendation:

- extend `sync_android_bundle.py`

Reason:

- one canonical sync command is simpler for operators
- it preserves the existing BKMeeting asset namespace map

### Task 3.2

Map Phase 5 package inputs to BKMeeting asset locations:

- Zipformer:
  - `modelassets/src/main/assets/models/asr/zipformer/qnn_u16u8`
- VPCD:
  - `modelassets/src/main/assets/models/punctuation/vpcd/qnn_fixed_1024x128`

### Task 3.3

Define the Android-facing manifest contract additions needed for Option 1:

- selected `target_runtime = precompiled_qnn_onnx`
- source `RUN_LABEL`
- copied compile-run evidence pointer
- enough component metadata for BKMeeting runtime capability detection

## Phase 4: BKMeeting Sync Dry Run

### Task 4.1

Run the Python-side sync into the sibling repo:

- BKMeeting root:
  - `D:/DS-AI/BKMeeting-Research/BKMeeting`

### Task 4.2

Verify after sync:

- target asset namespace exists
- `bundle_manifest.json` is present
- Option 1 contract files are present where expected
- no manual post-copy edits are required

## Phase 5: BKMeeting Runtime Readiness Checklist

This phase is still planning-only, but it must be explicit before claiming export success.

BKMeeting still needs to prove:

- asset packaging picks up the synced files
- ORT/QNN runtime packaging is present
- `precompiled_qnn_onnx` is recognized by runtime capability detection
- session creation succeeds on Snapdragon
- fallback behavior stays correct when NPU is unavailable

## Files Expected To Change In The Future Implementation

- `src/tools/sync_android_bundle.py`
- `test/test_sync_android_bundle.py`
- `src/tools/README.md`
- `docs/workflows/android-handoff.md`
- `docs/workflows/aihub-option1-phase5-contract.md`
- possibly `src/tools/aihub_option1_phase5_contract.py` if the current package layout needs small manifest additions

## Verification Plan

Minimum verification once implementation starts:

1. `pytest test/test_sync_android_bundle.py -v`
2. `pytest test/test_aihub_option1_phase5_contract.py -v`
3. notebook JSON sanity for:
   - `On_device_Ai_option1_phase4_gate.ipynb`
   - `On_device_Ai_option1_phase5_contract.ipynb`
4. one real sync dry-run into `BKMeeting`

## Success Criteria

This plan is complete when all of these are true:

- the selected Zipformer and VPCD `RUN_LABEL`s have Phase 4 and Phase 5 evidence
- a Phase 5 package can be produced without recompiling
- BKMeeting receives the payload through a repeatable sync command
- Android still enters through `bundle_manifest.json`
- the handoff docs explain exactly which model family is:
  - deployment candidate
  - research-only

## Related Docs

- `docs/workflows/model-quantization-status.md`
- `docs/workflows/aihub-option1-npu-pilots.md`
- `docs/workflows/aihub-option1-phase5-contract.md`
- `docs/workflows/android-handoff.md`
