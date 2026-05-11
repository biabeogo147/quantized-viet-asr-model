# AI Hub Option 1 Roadmap And Phase 2 Hardening Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move BKMeeting's Qualcomm AI Hub `Option 1` work from isolated NPU pilot wins to a reproducible, deployment-ready `precompiled_qnn_onnx` pipeline without leaving the ONNX Runtime + QNN lane.

**Architecture:** Keep the work Python-first until the artifact contract is stable. Treat `src/tools/aihub_option1_pilots.py` plus `On_device_Ai_option1_pilots.ipynb` as the canonical preparation and execution lane for Zipformer encoder and VPCD. Harden the compile path first, then graduate to hybrid CPU/NPU Python pipelines, then artifact packaging, and only after that move to Android integration.

**Tech Stack:** Python 3.10+, NumPy, ONNX, ONNX Runtime, Qualcomm AI Hub (`qai-hub`), Jupyter notebook (`.ipynb`), existing `python-model-test` bundle helpers, SHA256/file metadata recording, JSON run records.

---

## Option 1 Boundaries

- Stay inside the `Option 1` lane:
  - source models come from ONNX,
  - compilation goes through Qualcomm AI Hub,
  - target artifact is `precompiled_qnn_onnx`,
  - downstream runtime stays conceptually aligned with ORT + QNN.
- Do not switch this roadmap to:
  - local QAIRT-native packaging as the primary runtime boundary,
  - `Option 2` custom-QDQ-first as the long-term answer,
  - `Option 3` QAIRT-native app integration.
- The roadmap may use prepared ONNX source artifacts when the raw graph is not directly acceptable to AI Hub/HTP, as long as the final lane remains `ONNX -> AI Hub -> precompiled_qnn_onnx`.

## Current Status Snapshot

### Phase 1 Status: Completed

- `Zipformer encoder` has a verified NPU lane through a prepared ONNX source artifact.
  - compile success job: `jp3qe2vl5`
  - profile success job: `j5mv3y6w5`
  - inference success job: `jgle3r28p`
  - verified output shapes:
    - `output_0`: `(1, 501, 512)`
    - `output_1`: `(1,)`
- `VPCD` has a verified NPU lane through the same Option 1 workflow class.
- The notebook already contains both pilots:
  - `On_device_Ai_option1_pilots.ipynb`
- The current helper module already supports:
  - source resolution,
  - prepared Zipformer upload artifact generation,
  - VPCD source preparation,
  - compile/profile/inference option helpers,
  - `int64 -> int32` coercion for compiled artifacts using `truncate_64bit_io`.

### Known Constraints

- The verified Zipformer lane currently uses a prepared source model and direct compile.
- AI Hub quantize is **not** the current default Zipformer path because the graph still collides with control-flow outputs during QAIRT conversion.
- ASR scope is still `encoder-only on NPU`; decoder/joiner stay out of scope for the current pilot.
- No end-to-end Python pipeline result has been promoted to a formal benchmark record yet.
- No Android integration work is included in this phase plan.

## Roadmap Overview

### Phase 1: Single-Model NPU Proof

**Status:** Completed

**Purpose:**
- prove each selected graph can compile, profile, and infer on a Qualcomm cloud device with NPU requested

**Exit Criteria:**
- Zipformer encoder compile/profile/inference succeeds
- VPCD compile/profile/inference succeeds
- notebook and helper module exist

### Phase 2: Compile Pipeline Hardening

**Status:** Next phase

**Purpose:**
- turn the current working pilot lane into a deterministic, repeatable, reviewable build-and-run flow

**Exit Criteria:**
- one canonical helper path regenerates the prepared upload artifact for each pilot
- notebook writes structured run records for every live AI Hub run
- artifact hashes, input specs, compile options, device name, and QAIRT version are recorded beside each run
- docs explain exactly what files to keep after a successful run
- focused and regression tests cover the new helper behavior without requiring live AI Hub credentials

### Phase 3: Hybrid Python Pipeline Proof

**Status:** Planned

**Purpose:**
- move from isolated graph success to pipeline-level success while staying in Python

**Exit Criteria:**
- ASR Python flow runs `Zipformer encoder on NPU + decoder/joiner on CPU`
- punctuation Python flow runs `tokenizer on CPU + VPCD on NPU + host decode`
- at least a small fixed sample set produces final outputs, not only intermediate tensors

### Phase 4: Quality And Performance Gate

**Status:** Planned

**Purpose:**
- decide whether the Option 1 artifacts are good enough to justify deployment work

**Exit Criteria:**
- latency, warmup, and basic memory observations are recorded
- output sanity checks are compared against CPU baselines
- a go/no-go recommendation exists for:
  - `Zipformer encoder-only on NPU`
  - `VPCD on NPU`

### Phase 5: Deployment Contract Packaging

**Status:** Planned

**Purpose:**
- freeze the artifact contract that downstream integration will consume

**Exit Criteria:**
- compiled artifacts are stored in predictable locations
- each artifact has a manifest containing:
  - source model path and hash,
  - prepared upload model path and hash,
  - input specs,
  - compile flags,
  - device family,
  - QAIRT version,
  - expected tensor outputs
- docs explain how to refresh or validate the package

### Phase 6: Android Integration Under Option 1

**Status:** Planned

**Purpose:**
- integrate only after the Python-side artifact contract is stable

**Exit Criteria:**
- Android uses the AI Hub-generated `precompiled_qnn_onnx` artifacts
- runtime input/output contracts match the Python-side manifests
- the app can load and execute the selected NPU-backed models with the intended provider configuration

## Recommended Strategic Rules

- Do not block Phase 2 on recovering AI Hub quantize for Zipformer.
- Treat `prepared-source direct compile` as the baseline until data shows it is insufficient.
- Keep `Zipformer encoder-first` as the ASR production candidate until a later phase proves that pushing more RNNT graphs to NPU is worth the cost.
- Record every successful run as if it will later be handed to Android integration or another engineer with no prior context.

## Phase 2 Scope

Phase 2 is the next coding phase and is intentionally narrow:

- harden the current successful compile path
- make the outputs reproducible
- make successful runs reviewable and easy to hand off
- avoid expanding model scope

### Phase 2 Non-Goals

- full ASR transcript generation in this phase
- full punctuation decode quality evaluation in this phase
- Android asset packaging
- restoring Zipformer AI Hub quantize as a done criterion

## Phase 2 File Structure

**Files:**

- Modify: `src/tools/aihub_option1_pilots.py`
- Modify: `test/test_aihub_option1_pilots.py`
- Modify: `On_device_Ai_option1_pilots.ipynb`
- Modify: `docs/workflows/aihub-option1-npu-pilots.md`
- Modify: `docs/plans/active/2026-05-11-aihub-option1-npu-pilots.md`

## Phase 2 Detailed Tasks

### Task 1: Freeze The Canonical Pilot Runtime Configuration

**Files:**

- Modify: `src/tools/aihub_option1_pilots.py`
- Test: `test/test_aihub_option1_pilots.py`

- [ ] **Step 1: Write a failing test for runtime configuration normalization**

Test behavior:

- a helper builds one canonical config object for:
  - `device_name`,
  - `qairt_version`,
  - `compute_unit`,
  - artifact root directory,
  - record output directory

- [ ] **Step 2: Run the test to confirm it fails**

Run: `pytest test/test_aihub_option1_pilots.py -k runtime_config -v`

Expected: failure because the canonical config helper does not exist yet.

- [ ] **Step 3: Implement the minimal runtime configuration helper**

Add a small dataclass or equivalent helper in `src/tools/aihub_option1_pilots.py` that:

- centralizes Option 1 run settings,
- avoids notebook-only hidden defaults,
- resolves deterministic output directories under `build/aihub/`.

- [ ] **Step 4: Run the focused test to verify it passes**

Run: `pytest test/test_aihub_option1_pilots.py -k runtime_config -v`

Expected: pass.

### Task 2: Persist Prepared Artifact Records

**Files:**

- Modify: `src/tools/aihub_option1_pilots.py`
- Test: `test/test_aihub_option1_pilots.py`

- [ ] **Step 1: Write a failing test for prepared artifact record generation**

Test behavior:

- given a prepared source model path and input specs,
- the helper writes a JSON record containing:
  - pilot name,
  - source model path,
  - prepared upload model path,
  - file sizes,
  - SHA256 hashes,
  - input specs,
  - compile options

- [ ] **Step 2: Run the test to confirm it fails**

Run: `pytest test/test_aihub_option1_pilots.py -k prepared_record -v`

Expected: failure because the record writer does not exist yet.

- [ ] **Step 3: Implement the prepared artifact record helper**

Add helper logic that writes deterministic JSON beside the prepared artifact, for example under:

- `build/aihub/zipformer_encoder_option1/records/`
- `build/aihub/vpcd_option1/records/`

- [ ] **Step 4: Run the focused test to verify it passes**

Run: `pytest test/test_aihub_option1_pilots.py -k prepared_record -v`

Expected: pass.

### Task 3: Persist Live AI Hub Run Records

**Files:**

- Modify: `src/tools/aihub_option1_pilots.py`
- Test: `test/test_aihub_option1_pilots.py`

- [ ] **Step 1: Write a failing test for run record summaries**

Test behavior:

- given fake compile/profile/inference job metadata and output tensors,
- the helper writes a JSON run record containing:
  - job URLs or job ids,
  - device name,
  - QAIRT version,
  - compile options,
  - output tensor names,
  - output tensor shapes,
  - timestamp or run label

- [ ] **Step 2: Run the test to confirm it fails**

Run: `pytest test/test_aihub_option1_pilots.py -k run_record -v`

Expected: failure because the run record helper does not exist yet.

- [ ] **Step 3: Implement the run record writer and tensor summary helper**

Keep the implementation independent from notebook cell state as much as possible:

- one helper summarizes tensor names, shapes, and dtypes
- one helper writes JSON run records to disk

- [ ] **Step 4: Run the focused test to verify it passes**

Run: `pytest test/test_aihub_option1_pilots.py -k run_record -v`

Expected: pass.

### Task 4: Update The Notebook To Use The Hardened Helpers

**Files:**

- Modify: `On_device_Ai_option1_pilots.ipynb`
- Modify if needed: `src/tools/aihub_option1_pilots.py`

- [ ] **Step 1: Add a canonical runtime configuration cell**

Include:

- `DEVICE_NAME`
- optional `QAIRT_VERSION`
- canonical `Option 1` runtime config object

- [ ] **Step 2: Update the Zipformer cells to emit records**

After:

- preparing the upload artifact,
- compiling,
- profiling,
- inferring

write the prepared artifact record and live run record to disk.

- [ ] **Step 3: Update the VPCD cells to emit records**

Mirror the same pattern for VPCD so both pilots produce the same class of outputs.

- [ ] **Step 4: Add a final notebook summary cell**

Print:

- record locations,
- artifact locations,
- compile/profile/inference URLs,
- reminder that Phase 2 exits at reproducibility, not at Android integration.

### Task 5: Update The Workflow Doc For Handoff Quality

**Files:**

- Modify: `docs/workflows/aihub-option1-npu-pilots.md`

- [ ] **Step 1: Document the canonical run outputs**

Document the exact folders a successful run should leave behind:

- prepared upload model
- prepared artifact JSON record
- live run JSON record
- downloaded profile/output artifacts when available

- [ ] **Step 2: Document failure handling**

Explain how to classify failures:

- source-model preparation failure,
- AI Hub compile failure,
- AI Hub profile failure,
- AI Hub inference failure

- [ ] **Step 3: Document the minimum evidence required to call a pilot hardened**

List:

- record files exist,
- artifact hashes exist,
- job URLs are stored,
- output tensor summaries are stored.

### Task 6: Run Full Phase 2 Verification

**Files:**

- Test: `test/test_aihub_option1_pilots.py`
- Test: existing regression slice
- Test: notebook JSON sanity

- [ ] **Step 1: Run the focused helper tests**

Run: `pytest test/test_aihub_option1_pilots.py -v`

Expected: pass.

- [ ] **Step 2: Run the broader regression slice**

Run: `pytest test/test_zipformer_quantize.py test/test_vpcd_bundle.py test/test_vpcd_qnn_candidate.py -v`

Expected: pass.

- [ ] **Step 3: Run Python compile verification**

Run: `python -m compileall src`

Expected: pass.

- [ ] **Step 4: Verify the notebook is still valid JSON**

Run: `python - <<'PY'\nimport json\nfrom pathlib import Path\njson.loads(Path('On_device_Ai_option1_pilots.ipynb').read_text(encoding='utf-8'))\nprint('ok')\nPY`

Expected: prints `ok`.

## Phase 2 Acceptance Criteria

- The notebook no longer relies on hidden, ad-hoc per-cell settings for critical runtime choices.
- Each pilot can regenerate its prepared upload artifact in a deterministic output directory.
- Each pilot writes a prepared artifact record with hash and compile metadata.
- Each live AI Hub run writes a run record with job links and tensor summaries.
- The workflow doc tells another engineer exactly what to keep and inspect after a successful run.
- Focused and regression tests pass locally.

## Planned Phase 3 Preview

Phase 3 should begin only after Phase 2 acceptance criteria are met.

The next implementation targets will be:

- `Zipformer hybrid pipeline`
  - encoder on NPU
  - decoder/joiner on CPU
  - final transcript sanity checks on a tiny sample set
- `VPCD hybrid pipeline`
  - tokenizer encode on CPU
  - model on NPU
  - host-side decode to final punctuated text
- `shared evaluation harness`
  - fixed sample list
  - recorded final outputs
  - latency summary at pipeline level

## Decision Gates After Phase 2

1. Keep the current Zipformer prepared-source direct compile lane as the official baseline, unless a later experiment proves it blocks deployment.
2. Treat recovery of AI Hub quantize for Zipformer as an optimization branch, not as a blocker.
3. Do not start Android integration until the prepared artifact and run record contract is stable.

## Recommended Execution Order

1. Implement `Task 1` to freeze runtime configuration.
2. Implement `Task 2` to persist prepared artifact records.
3. Implement `Task 3` to persist live run records.
4. Implement `Task 4` to wire the notebook to the hardened helpers.
5. Implement `Task 5` to lock documentation to the new output contract.
6. Run `Task 6` verification before calling Phase 2 complete.
