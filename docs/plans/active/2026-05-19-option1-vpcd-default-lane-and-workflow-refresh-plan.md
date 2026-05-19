# Option 1 VPCD Default Lane And Workflow Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote the proven VPCD AIMET parity lane to the default `Option 1` pilot path, remove the retired VPCD AI Hub-quantize lane from active code and `build/aihub`, prepare a clean final rerun setup for the operator, and rewrite workflow docs so the current NPU path is short, consistent, and easy to follow.

**Architecture:** Treat `local_aimet_compile_candidate` as the only active VPCD Phase 2/3 path. Align notebook defaults, helper defaults, pilot names, and record expectations around `vpcd_option1_local_aimet`, then delete the unused VPCD AI Hub-quantize branch and its leftover artifacts. Finish by resetting `build/aihub` to a clean rerun state and pre-filling notebook configs so the operator can execute one final clean run without mixing old versions.

**Tech Stack:** Python, Jupyter notebook JSON, Qualcomm AI Hub record helpers, Markdown docs, pytest

---

## File Structure

### Keep as active runtime/code path

- `On_device_Ai_option1_pilots.ipynb`
  - day-to-day Phase 2 + Phase 3 operator notebook
- `src/tools/aihub_option1_pilots.py`
  - Phase 2 prepare / compile / record helpers
- `src/tools/aihub_option1_hybrid_pipeline.py`
  - Phase 3 hybrid / teacher-forced runtime helpers
- `src/tools/aihub_option1_phase4_gate.py`
  - Phase 4 gate layout and record resolution
- `src/tools/aihub_option1_phase5_contract.py`
  - Phase 5 package materialization and evidence resolution

### Tests that must move with the new default

- `test/test_option1_notebook_layout.py`
- `test/test_aihub_option1_pilots.py`
- `test/test_aihub_option1_hybrid_pipeline.py`
- `test/test_aihub_option1_phase4_gate.py`
- `test/test_aihub_option1_phase5_contract.py`

### Workflow docs to rewrite

- `docs/workflows/aihub-option1-npu-pilots.md`
- `docs/workflows/aihub-option1-hybrid-pipeline.md`
- `docs/workflows/model-quantization-status.md`
- `docs/workflows/aihub-option1-phase4-gate.md`
- `docs/workflows/aihub-option1-phase5-contract.md`
- `docs/workflows/android-handoff.md`

### Build paths to prune

- `build/aihub/records/vpcd_option1/`
- any regenerated `build/aihub/vpcd_option1/`
- any notebook outputs or duplicate VPCD artifacts tied only to the retired `prefer_fp32_fixed -> AI Hub quantize` lane
- any stale `build/aihub/records/vpcd_*` or `build/aihub/notebook_runs/*` versions that are not part of the final rerun keep-set

## Task 1: Promote AIMET parity as the only active VPCD pilot lane

**Files:**
- Modify: `On_device_Ai_option1_pilots.ipynb`
- Modify: `src/tools/aihub_option1_pilots.py`
- Modify: `src/tools/aihub_option1_hybrid_pipeline.py`
- Modify: `src/tools/aihub_option1_phase4_gate.py`
- Modify: `src/tools/aihub_option1_phase5_contract.py`
- Test: `test/test_option1_notebook_layout.py`
- Test: `test/test_aihub_option1_pilots.py`
- Test: `test/test_aihub_option1_hybrid_pipeline.py`
- Test: `test/test_aihub_option1_phase4_gate.py`
- Test: `test/test_aihub_option1_phase5_contract.py`

- [ ] **Step 1: Write or update failing tests for the new VPCD defaults**

Cover:
- notebook default must be `VPCD_SOURCE_STRATEGY = "local_aimet_compile_candidate"`
- Phase 2 compile pilot default for VPCD must resolve to `vpcd_option1_local_aimet`
- Phase 3 VPCD helpers must no longer default to `vpcd_option1`
- Phase 4 / Phase 5 should no longer need a VPCD compile-pilot override for the active path

- [ ] **Step 2: Run the targeted tests to confirm current defaults are still old**

Run:
- `pytest test/test_option1_notebook_layout.py -k "vpcd" -v`
- `pytest test/test_aihub_option1_phase4_gate.py -k "vpcd" -v`
- `pytest test/test_aihub_option1_phase5_contract.py -k "vpcd" -v`

Expected:
- failures or assertions still tied to `prefer_fp32_fixed` / `vpcd_option1`

- [ ] **Step 3: Change the pilot notebook default to AIMET parity**

In `On_device_Ai_option1_pilots.ipynb`:
- set `VPCD_SOURCE_STRATEGY = "local_aimet_compile_candidate"`
- replace the stale `RUN_LABEL = "20260513-1am"` VPCD narrative/output references with the current retained AIMET parity lane
- keep the bounded guardrails:
  - `VPCD_HYBRID_MAX_SAMPLES = 2`
  - `VPCD_HYBRID_MAX_STEPS = 5`
  - `VPCD_TEACHER_FORCED_SAMPLE_INDEX = 0`

- [ ] **Step 4: Promote VPCD helper defaults to `vpcd_option1_local_aimet`**

Change the active defaults so VPCD no longer points at the retired AI Hub-quantize lane:
- `src/tools/aihub_option1_hybrid_pipeline.py`
  - change `VPCD_PHASE2_PILOT`
- `src/tools/aihub_option1_phase4_gate.py`
  - change the VPCD `phase2_compile_pilot_name` in `PILOT_LAYOUTS`
- `src/tools/aihub_option1_phase5_contract.py`
  - ensure default evidence resolution follows the new Phase 2 pilot without requiring override wiring

- [ ] **Step 5: Run the targeted tests again**

Run:
- `pytest test/test_option1_notebook_layout.py -k "vpcd" -v`
- `pytest test/test_aihub_option1_pilots.py -k "vpcd and aimet" -v`
- `pytest test/test_aihub_option1_hybrid_pipeline.py -k "vpcd" -v`
- `pytest test/test_aihub_option1_phase4_gate.py -k "vpcd" -v`
- `pytest test/test_aihub_option1_phase5_contract.py -k "vpcd" -v`

Expected:
- PASS

- [ ] **Step 6: Commit**

```bash
git add On_device_Ai_option1_pilots.ipynb src/tools/aihub_option1_pilots.py src/tools/aihub_option1_hybrid_pipeline.py src/tools/aihub_option1_phase4_gate.py src/tools/aihub_option1_phase5_contract.py test/test_option1_notebook_layout.py test/test_aihub_option1_pilots.py test/test_aihub_option1_hybrid_pipeline.py test/test_aihub_option1_phase4_gate.py test/test_aihub_option1_phase5_contract.py
git commit -m "feat: promote vpcd aimet parity as option1 default lane"
```

## Task 2: Remove the retired VPCD AI Hub-quantize lane from active code

**Files:**
- Modify: `src/tools/aihub_option1_pilots.py`
- Modify: `On_device_Ai_option1_pilots.ipynb`
- Test: `test/test_aihub_option1_pilots.py`
- Test: `test/test_option1_notebook_layout.py`

- [ ] **Step 1: Write failing tests for the removed branch**

Cover:
- `prepare_vpcd_option1_source_model(...)` should no longer accept `prefer_fp32_fixed` as an active VPCD source strategy
- notebook text/config/output should no longer expose the old VPCD lane as the day-to-day path

- [ ] **Step 2: Run those tests to verify the old branch still exists**

Run:
- `pytest test/test_aihub_option1_pilots.py -k "prepare_vpcd_option1_source_model" -v`
- `pytest test/test_option1_notebook_layout.py -k "vpcd" -v`

- [ ] **Step 3: Delete the retired VPCD branch from active code**

In `src/tools/aihub_option1_pilots.py`:
- remove the VPCD `prefer_fp32_fixed` branch from `prepare_vpcd_option1_source_model(...)`
- remove code paths that create:
  - `build/aihub/vpcd_fp32_fixed/`
  - `build/aihub/vpcd_option1/`
  - VPCD `quantize-run` as an active output path
- keep ZIPFORMER behavior untouched

In `On_device_Ai_option1_pilots.ipynb`:
- remove notebook text that presents AI Hub quantize as any supported VPCD compile lane
- remove stale output text that mentions `vpcd_option1`

- [ ] **Step 4: Run the VPCD tests**

Run:
- `pytest test/test_aihub_option1_pilots.py -k "vpcd" -v`
- `pytest test/test_option1_notebook_layout.py -k "vpcd" -v`

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add On_device_Ai_option1_pilots.ipynb src/tools/aihub_option1_pilots.py test/test_aihub_option1_pilots.py test/test_option1_notebook_layout.py
git commit -m "refactor: remove retired vpcd ai hub quantize lane"
```

## Task 3: Reset `build/aihub` for one final clean rerun

**Files:**
- Delete: `build/aihub/records/vpcd_option1/`
- Delete: `build/aihub/vpcd_option1/`
- Delete: any regenerated notebook runs or duplicate artifacts that only exist for the retired lane
- Delete: stale VPCD AIMET records, notebook outputs, and contract packages that would cause version confusion during the final rerun
- Modify: workflow docs only if they still reference deleted paths

- [ ] **Step 1: Enumerate the exact paths to delete and verify they are inside the repo**

Record the final deletion list in the session notes before deleting.

- [ ] **Step 2: Define the final rerun keep-set**

Before deleting, choose the minimal rerun keep-set.

Recommended keep-set:
- Docker support:
  - `docker/aimet-onnx-ubuntu2204/Dockerfile`
- source assets and local bundle sources:
  - `assets/vietnamese-punc-cap-denorm-v1/...`
  - `build/model_bundle/vpcd/qnn_fixed_1024x128/`
  - `build/model_bundle/zipformer/qnn_u16u8/`
- no retained VPCD AI Hub records
- no retained VPCD notebook runs
- no retained VPCD contract packages
- optionally keep the latest Zipformer evidence if Zipformer will not be rerun in this cleanup pass

- [ ] **Step 3: Delete the retired VPCD lane artifacts and stale rerun clutter**

Delete only the retired VPCD AI Hub-quantize artifacts, not:
- repo assets and local bundle sources needed to regenerate the lane

Also delete stale VPCD AIMET outputs that would pollute a fresh rerun:
- `build/aihub/records/vpcd_option1_local_aimet/`
- `build/aihub/records/vpcd_quantized_teacher_forced_option1/`
- `build/aihub/records/vpcd_teacher_forced_option1/`
- `build/aihub/records/vpcd_hybrid_option1/`
- `build/aihub/records/vpcd_phase4_option1/`
- `build/aihub/contracts/option1/vpcd/`
- VPCD-specific notebook runs under `build/aihub/notebook_runs/`
- regenerated `build/aihub/vpcd_option1_local_aimet/` outputs so the final run rebuilds them cleanly

If Zipformer will also be rerun as part of the final pass, apply the same cleanup principle to stale Zipformer `records/`, `contracts/`, and notebook runs.

- [ ] **Step 4: Verify the clean rerun baseline**

Check:
- notebook source files still exist
- Dockerfile still exists
- source assets and local bundle sources still exist
- `build/aihub` no longer contains stale VPCD run versions

- [ ] **Step 5: Commit**

```bash
git add -A build/aihub
git commit -m "chore: reset aihub state for final option1 rerun"
```

## Task 4: Preconfigure the final rerun notebooks for the operator

**Files:**
- Modify: `On_device_Ai_option1_pilots.ipynb`
- Modify: `On_device_Ai_option1_phase4_gate.ipynb`
- Modify: `On_device_Ai_option1_phase5_contract.ipynb`
- Modify: `docs/workflows/aihub-option1-npu-pilots.md`
- Modify: `docs/workflows/aihub-option1-phase4-gate.md`
- Modify: `docs/workflows/aihub-option1-phase5-contract.md`

- [ ] **Step 1: Choose one clean final rerun label**

Pre-fill one fresh label that is not reused by old evidence, for example:
- `FINAL_RUN_LABEL = "20260519-vpcd-aimet-parity-final"`

Use that label consistently across:
- `On_device_Ai_option1_pilots.ipynb`
- `On_device_Ai_option1_phase4_gate.ipynb`
- `On_device_Ai_option1_phase5_contract.ipynb`

- [ ] **Step 2: Pre-fill the pilot notebook with the exact VPCD rerun config**

Set and document clearly:
- `RUN_LABEL = "20260519-vpcd-aimet-parity-final"` for the final VPCD rerun pass
- `VPCD_SOURCE_STRATEGY = "local_aimet_compile_candidate"`
- `VPCD_HYBRID_MAX_SAMPLES = 2`
- `VPCD_HYBRID_MAX_STEPS = 5`
- `VPCD_TEACHER_FORCED_SAMPLE_INDEX = 0`
- keep any explicit VPCD compile pilot mapping aligned with `vpcd_option1_local_aimet`

- [ ] **Step 3: Pre-fill Phase 4 and Phase 5 notebooks for the same rerun**

Set and document clearly:
- `VPCD_RUN_LABEL = "20260519-vpcd-aimet-parity-final"`
- `VPCD_PHASE2_COMPILE_PILOT_NAME = "vpcd_option1_local_aimet"`
- `max_decode_steps = 5` where the gate notebook uses bounded VPCD reruns

- [ ] **Step 4: Add a short operator rerun checklist**

Document a compact rerun order in the workflow docs:
1. run VPCD cells in `On_device_Ai_option1_pilots.ipynb`
2. inspect teacher-forced + bounded hybrid outputs
3. run `On_device_Ai_option1_phase4_gate.ipynb`
4. run `On_device_Ai_option1_phase5_contract.ipynb`

The checklist must say the rerun is expected to regenerate all VPCD `build/aihub` outputs from a clean state.

- [ ] **Step 5: Commit**

```bash
git add On_device_Ai_option1_pilots.ipynb On_device_Ai_option1_phase4_gate.ipynb On_device_Ai_option1_phase5_contract.ipynb docs/workflows/aihub-option1-npu-pilots.md docs/workflows/aihub-option1-phase4-gate.md docs/workflows/aihub-option1-phase5-contract.md
git commit -m "chore: preconfigure final option1 rerun notebooks"
```

## Task 5: Rewrite workflow docs around old lanes, outcomes, and the kept lane

**Files:**
- Modify: `docs/workflows/aihub-option1-npu-pilots.md`
- Modify: `docs/workflows/aihub-option1-hybrid-pipeline.md`
- Modify: `docs/workflows/model-quantization-status.md`
- Modify: `docs/workflows/aihub-option1-phase4-gate.md`
- Modify: `docs/workflows/aihub-option1-phase5-contract.md`
- Modify: `docs/workflows/android-handoff.md`

- [ ] **Step 1: Rewrite the top-level workflow story**

For both Zipformer and VPCD, document in short form:
- old lanes
- result of each old lane
- current kept lane
- what is proven
- what is still not proven

Required tone:
- short
- tight
- operational
- no long historical narrative in workflow docs

- [ ] **Step 2: Make Zipformer status explicit**

Docs must say clearly:
- old / available lane:
  - local full QDQ bundle exists
  - AI Hub proof is still encoder-first
- current kept lane for NPU phase:
  - `zipformer_encoder_option1`
- limitation:
  - full end-to-end NPU parity is not proven yet

- [ ] **Step 3: Make VPCD status explicit**

Docs must say clearly:
- retired lanes:
  - `prefer_fp32_fixed -> AI Hub quantize -> compile`
  - local-QDQ compile probe
  - broad AIMET `w8a8 + min_max`
- result:
  - AI Hub quantize baseline failed at teacher-forced step `2`
  - local-QDQ was compile-incompatible
  - broad AIMET was compile-compatible but semantically wrong
- current kept lane:
  - `local_aimet_compile_candidate`
  - `w8a16 + min_max + local_quality_parity`
  - `vpcd_option1_local_aimet`
- current limitation:
  - bounded proof window is still `max_decode_steps = 5`

- [ ] **Step 4: Align workflow docs with the new record layout**

Remove or rewrite references that imply:
- `vpcd_option1` is still a normal active pilot
- AI Hub quantize is still part of the active VPCD operator workflow in any form

- [ ] **Step 5: Run doc-facing sanity checks**

Run:
- `pytest test/test_option1_notebook_layout.py -v`
- any JSON sanity command already used for:
  - `On_device_Ai_option1_pilots.ipynb`
  - `On_device_Ai_option1_phase4_gate.ipynb`
  - `On_device_Ai_option1_phase5_contract.ipynb`

- [ ] **Step 6: Commit**

```bash
git add docs/workflows/aihub-option1-npu-pilots.md docs/workflows/aihub-option1-hybrid-pipeline.md docs/workflows/model-quantization-status.md docs/workflows/aihub-option1-phase4-gate.md docs/workflows/aihub-option1-phase5-contract.md docs/workflows/android-handoff.md
git commit -m "docs: refresh option1 workflows around kept npu lanes"
```

## Task 6: Final verification and readiness handoff

**Files:**
- Verify only

- [ ] **Step 1: Run the combined targeted verification**

Run:
- `pytest test/test_aihub_option1_pilots.py test/test_aihub_option1_hybrid_pipeline.py test/test_aihub_option1_phase4_gate.py test/test_aihub_option1_phase5_contract.py test/test_option1_notebook_layout.py -k "vpcd or aimet or option1" -v -p no:cacheprovider`

- [ ] **Step 2: Verify the retained build layout**

Confirm the rerun baseline is clean:
- no stale VPCD `build/aihub/records/vpcd_*` directories remain
- no stale VPCD `build/aihub/contracts/option1/vpcd/*` rerun versions remain
- notebooks are pre-filled with the final rerun label and AIMET parity config

Confirm these do not exist:
- `build/aihub/vpcd_option1/`
- `build/aihub/records/vpcd_option1/`

- [ ] **Step 3: Commit any remaining metadata or doc touch-ups**

```bash
git add -A
git commit -m "chore: finalize option1 vpcd lane refresh"
```

## Success Criteria

This plan is complete when all of these are true:

- `On_device_Ai_option1_pilots.ipynb` defaults to the AIMET parity VPCD lane
- active code no longer treats any AI Hub quantize path as the VPCD operator path
- retired VPCD lane artifacts are gone from `build/aihub`
- `build/aihub` is reset so the operator can run one final clean pass without old version clutter
- workflow docs clearly distinguish:
  - old lanes
  - lane outcomes
  - current kept lane
  - remaining proof gaps
- the notebooks are preconfigured so the operator can rerun the retained VPCD lane directly
- the repo is ready to continue the NPU implementation phase from the retained Zipformer and VPCD lanes only
