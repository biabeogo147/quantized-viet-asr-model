# AI Hub Option 1 Roadmap, Phase 2 Hardening, And Phase 3 Hybrid Pipeline Plan

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

### Phase 2 Status: Completed

- the pilot notebook now supports:
  - `prepare`,
  - `compile only`,
  - `resolve existing compiled target`,
  - `run + compare`
- structured JSON records now exist for:
  - prepared artifacts,
  - compile-only runs,
  - live runs
- compile reuse is now a first-class flow through:
  - `RUN_LABEL`,
  - compile-run records,
  - optional explicit target model ids
- local verification already exists for the hardened helper path:
  - focused helper tests,
  - regression slice,
  - notebook JSON sanity,
  - `compileall`

### Phase 3 Status: Implemented And Executed In The Shared Notebook

- the shared hybrid runtime now exists in:
  - `src/tools/aihub_option1_hybrid_pipeline.py`
- the existing notebook now hosts both:
  - Phase 2 tensor diagnostics,
  - Phase 3 hybrid e2e sections,
  - final compare gates against `expected_outputs.jsonl` and `golden_samples.jsonl`
- local verification now covers:
  - compiled target resolution,
  - compiled inference normalization,
  - Zipformer hybrid transcript flow with injected cloud inference,
  - VPCD hybrid restore flow with injected cloud inference,
  - hybrid run record writing
- a dedicated workflow document now exists in:
  - `docs/workflows/aihub-option1-hybrid-pipeline.md`
- the latest shared-notebook execution now provides live evidence for both pilots:
  - `Zipformer`
    - final compare is now using the correct `expected_outputs.jsonl` fixture audio
    - exact-match status is currently `0 / 2`
    - current mismatches look small and localized, for example:
      - `NĂM` vs `LĂM`
      - `CLIENT` vs `CLIAN`
  - `VPCD`
    - final compare is currently `0 / 2`
    - current outputs collapse to placeholder-like text with `generated_ids = [0, 1, 2]`
    - tensor diagnostic already shows large drift versus CPU baseline, so this lane must be treated as a likely `NO_GO` candidate until proven otherwise

### Phase 4 Status: Implemented In Code And Notebook, Pending Live Gate Refresh

- the dedicated gate module now exists in:
  - `src/tools/aihub_option1_phase4_gate.py`
- focused tests now cover:
  - benchmark sweep summaries
  - correctness severity classification
  - footprint summaries
  - per-pilot recommendations
  - deterministic Phase 4 gate record writing
- the shared notebook now includes:
  - `Phase 4 Config`
  - `Zipformer Phase 4 Benchmark And Gate`
  - `VPCD Phase 4 Benchmark And Gate`
  - `Phase 4 Recommendation Summary`
- the Phase 4 workflow doc now exists in:
  - `docs/workflows/aihub-option1-phase4-gate.md`

### Phase 5 Status: Implemented In Code And Notebook, Pending Live Packaging Refresh

- the dedicated contract packager now exists in:
  - `src/tools/aihub_option1_phase5_contract.py`
- focused tests now cover:
  - manifest generation
  - evidence resolution
  - package materialization
  - promotion-status mapping
  - normalized I/O contract export
- the shared notebook now includes:
  - `Phase 5 Config`
  - `Package Zipformer Phase 5 Contract`
  - `Package VPCD Phase 5 Contract`
  - `Phase 5 Packaging Summary`
- workflow docs now exist in:
  - `docs/workflows/aihub-option1-phase5-contract.md`

### Known Constraints

- The verified Zipformer lane currently uses a prepared source model and direct compile.
- AI Hub quantize is **not** the current default Zipformer path because the graph still collides with control-flow outputs during QAIRT conversion.
- ASR scope is still `encoder-only on NPU`; decoder/joiner stay out of scope for the current pilot.
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

**Status:** Completed

**Purpose:**
- turn the current working pilot lane into a deterministic, repeatable, reviewable build-and-run flow

**Exit Criteria:**
- one canonical helper path regenerates the prepared upload artifact for each pilot
- notebook writes structured run records for every live AI Hub run
- artifact hashes, input specs, compile options, device name, and QAIRT version are recorded beside each run
- docs explain exactly what files to keep after a successful run
- focused and regression tests cover the new helper behavior without requiring live AI Hub credentials

### Phase 3: Hybrid Python Pipeline Proof

**Status:** Implemented And Executed, Pending Phase 4 Judgment

**Purpose:**
- move from isolated graph success to pipeline-level success while staying in Python

**Exit Criteria:**
- ASR Python flow runs `Zipformer encoder on NPU + decoder/joiner on CPU`
- punctuation Python flow runs `tokenizer on CPU + VPCD on NPU + host decode`
- at least a small fixed sample set produces final outputs, not only intermediate tensors

### Phase 4: Quality And Performance Gate

**Status:** Implemented In Code And Notebook, Pending Live Gate Refresh

**Purpose:**
- decide whether the Option 1 artifacts are good enough to justify deployment work

**Exit Criteria:**
- latency, warmup, and basic memory observations are recorded
- output sanity checks are compared against CPU baselines
- a formal gate record and go/no-go recommendation exist for:
  - `Zipformer encoder-only on NPU`
  - `VPCD on NPU`

### Phase 5: Deployment Contract Packaging

**Status:** Implemented In Code And Notebook, Pending Live Packaging Refresh

**Purpose:**
- freeze the artifact contract that downstream integration and research handoff will consume, even when a pilot is not promotable

**Exit Criteria:**
- per-pilot contract packages are stored in predictable locations
- each package has a manifest containing:
  - source model path and hash,
  - prepared upload model path and hash,
  - input specs,
  - compile flags,
  - device family,
  - QAIRT version,
  - Phase 4 recommendation,
  - promotion status,
  - evidence record paths
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

## Phase 3 Scope

Phase 3 is now implemented in code, notebook structure, and local tests.

The remaining work for this phase is operational:

- run the notebook against real compiled cloud targets,
- review the generated hybrid run records,
- decide whether any e2e mismatches or latency problems need follow-up before moving to Phase 4.

The purpose of Phase 3 is to prove the real hybrid Python pipelines, not only isolated graph inference:

- `Zipformer`
  - features on host
  - encoder on Qualcomm cloud NPU through the compiled target model from Phase 2
  - decoder and joiner on CPU
  - final transcript output recorded first
  - compare against bundle expectations only after the full hybrid e2e path is working
- `VPCD`
  - tokenizer encode on CPU
  - punctuation model on Qualcomm cloud NPU through the compiled target model from Phase 2
  - tokenizer decode on CPU
  - final punctuated text recorded first
  - compare against golden samples only after the full hybrid e2e path is working

### Phase 3 Non-Goals

- moving Zipformer decoder or joiner onto NPU
- replacing the current prepared-source direct compile baseline for Zipformer
- Android asset packaging or app integration
- large-scale benchmark automation
- model retuning or quantization experiments outside the proven Option 1 lane

## Phase 3 File Structure

**Files:**

- Create: `src/tools/aihub_option1_hybrid_pipeline.py`
- Create: `test/test_aihub_option1_hybrid_pipeline.py`
- Create: `docs/workflows/aihub-option1-hybrid-pipeline.md`
- Modify: `src/model_bundle/projects/zipformer.py`
- Modify: `src/model_bundle/projects/_vpcd_support.py`
- Modify if needed: `src/tools/aihub_option1_pilots.py`
- Modify: `On_device_Ai_option1_pilots.ipynb`
- Modify: `docs/plans/active/2026-05-11-aihub-option1-npu-pilots.md`

### File Responsibilities

- `src/tools/aihub_option1_hybrid_pipeline.py`
  - Option 1-specific hybrid runtime orchestration
  - resolve compiled target models from Phase 2 records or explicit ids
  - submit AI Hub inference for compiled targets
  - bridge cloud outputs into CPU-side decode loops
  - write hybrid run records with final outputs and timing summaries
- `src/model_bundle/projects/zipformer.py`
  - expose a reusable helper for CPU-side RNNT decode from encoder frames so the hybrid runtime does not duplicate the greedy loop
- `src/model_bundle/projects/_vpcd_support.py`
  - expose a reusable helper for host-side punctuation restore where the model-step runner can be swapped from local ORT to AI Hub cloud inference
- `On_device_Ai_option1_pilots.ipynb`
  - remains the single human-facing notebook
  - keeps the existing Phase 2 compile/reuse flow
  - gains additional Phase 3 e2e sections without forcing a second notebook
- `docs/workflows/aihub-option1-hybrid-pipeline.md`
  - canonical operator instructions for running Phase 3 and preserving evidence

## Phase 3 Detailed Tasks

### Notebook Refactor Plan

Phase 3 must execute inside the existing notebook:

- `On_device_Ai_option1_pilots.ipynb`

Do not create a second notebook for the first implementation of Phase 3.

The notebook should be refactored in-place so that:

- the current Phase 2 compile / reuse flow still works
- tensor-level inspection remains available as an optional diagnostic path
- full hybrid e2e execution is added after the compiled target reuse flow
- final gold-sample comparison appears only after e2e outputs exist

### Notebook Refactor Principles

- keep setup, authentication, imports, and Phase 2 compile reuse at the top
- keep `RUN_LABEL` and explicit target model ids as the single control plane for compile reuse
- rename any section whose current title implies final correctness if it only checks intermediate tensors
- make the e2e path visually separate from the tensor-debug path
- do not compare against final expected outputs until the notebook already has final pipeline outputs in memory

### Target Cell Layout

The target notebook should read in this order:

1. title and scope
2. environment notes
3. dependency install
4. AI Hub authentication bootstrap
5. imports
6. runtime config
7. notebook usage guide
8. `Zipformer` Phase 2 prepare cell
9. `Zipformer` compile-only section
10. `Zipformer` resolve existing compiled target section
11. `Zipformer` compiled-graph inference section
12. `Zipformer` tensor inspection section
13. `Zipformer` hybrid e2e run section
14. `Zipformer` final transcript-vs-expected comparison section
15. `VPCD` Phase 2 prepare cell
16. `VPCD` compile-only section
17. `VPCD` resolve existing compiled target section
18. `VPCD` compiled-graph inference section
19. `VPCD` tensor inspection section
20. `VPCD` hybrid e2e run section
21. `VPCD` final text-vs-golden comparison section
22. final summary and evidence paths

### Required Cell-Level Refactor Changes

For `Zipformer`:

- keep the current prepare / compile / resolve cells
- keep the existing cloud encoder tensor inspection, but relabel it as an intermediate diagnostic
- add a new e2e cell that:
  - runs encoder inference on the compiled target
  - sends encoder frames into CPU decoder and joiner
  - produces final transcript rows
- add a new comparison cell after the e2e cell that:
  - loads `expected_outputs.jsonl`
  - compares transcript text only after the e2e outputs are available
  - prints mismatch summaries

For `VPCD`:

- keep the current prepare / compile / resolve cells
- keep the existing logits inspection, but relabel it as an intermediate diagnostic
- add a new e2e cell that:
  - keeps CPU tokenizer encode / decode
  - runs only the model step on the compiled target
  - produces final punctuated text rows
- add a new comparison cell after the e2e cell that:
  - loads `golden_samples.jsonl`
  - compares final punctuated text only after the e2e outputs are available
  - prints mismatch summaries

For the final summary:

- print Phase 2 compile-run record paths used by the notebook
- print Phase 3 hybrid run record paths
- print how many samples matched vs mismatched for:
  - Zipformer final transcript comparison
  - VPCD final text comparison
- keep the reminder that this is still Python-only and not Android integration

### Task 1: Extract CPU-Side Decode Seams From The Existing Runtimes

**Files:**

- Modify: `src/model_bundle/projects/zipformer.py`
- Modify: `src/model_bundle/projects/_vpcd_support.py`
- Test: `test/test_zipformer_bundle.py`
- Test: `test/test_vpcd_bundle.py`

- [ ] **Step 1: Write a failing Zipformer test for decoder/joiner-only greedy decode**

Test behavior:

- given trimmed encoder frames plus fake decoder and joiner sessions,
- a helper reproduces the same token sequence and transcript as the current `transcribe(...)` path.

- [ ] **Step 2: Run the focused Zipformer test to confirm it fails**

Run: `pytest test/test_zipformer_bundle.py -k greedy_decode -v`

Expected: failure because no standalone CPU decode helper exists yet.

- [ ] **Step 3: Implement the minimal Zipformer decode helper**

Add a helper in `src/model_bundle/projects/zipformer.py` that:

- accepts encoder frames, decoder session, joiner session, token table, `blank_id`, and `context_size`
- returns final text and token count without owning feature extraction or encoder execution
- becomes the single implementation used by both:
  - the existing local runtime
  - the new Phase 3 hybrid runtime

- [ ] **Step 4: Write a failing VPCD test for a pluggable model-step runner**

Test behavior:

- given CPU tokenizer sessions plus a fake model-step callable that returns logits,
- a helper reproduces the same final punctuated text as the current `restore(...)` loop.

- [ ] **Step 5: Run the focused VPCD test to confirm it fails**

Run: `pytest test/test_vpcd_bundle.py -k pluggable_restore -v`

Expected: failure because the current restore path is still hardwired to `model_session.run(...)`.

- [ ] **Step 6: Implement the minimal VPCD restore seam**

Refactor `src/model_bundle/projects/_vpcd_support.py` so that:

- CPU tokenizer encode and decode stay exactly where they are
- the model step can be delegated through a callable or helper instead of being permanently tied to a local ORT session
- the local bundle runtime still uses the same helper so Phase 3 does not fork behavior

- [ ] **Step 7: Run the focused runtime tests to verify both helpers pass**

Run: `pytest test/test_zipformer_bundle.py -k greedy_decode -v`

Expected: pass.

Run: `pytest test/test_vpcd_bundle.py -k pluggable_restore -v`

Expected: pass.

### Task 2: Create The Shared Option 1 Hybrid Runtime Module

**Files:**

- Create: `src/tools/aihub_option1_hybrid_pipeline.py`
- Test: `test/test_aihub_option1_hybrid_pipeline.py`
- Modify if needed: `src/tools/aihub_option1_pilots.py`

- [ ] **Step 1: Write a failing test for compiled target resolution and cloud inference normalization**

Test behavior:

- the hybrid module can:
  - resolve a target model id from:
    - explicit override, or
    - Phase 2 `compile-run-<run_label>.json`
  - submit or simulate compiled-model inference
  - normalize returned output tensors into a deterministic mapping

- [ ] **Step 2: Run the focused hybrid-module test to confirm it fails**

Run: `pytest test/test_aihub_option1_hybrid_pipeline.py -k resolve_target_or_inference_adapter -v`

Expected: failure because the hybrid runtime module does not exist yet.

- [ ] **Step 3: Implement the minimal shared hybrid helpers**

In `src/tools/aihub_option1_hybrid_pipeline.py`, add helpers for:

- loading Phase 2 compile records
- resolving compiled target models from:
  - `RUN_LABEL`, or
  - explicit target model ids
- coercing inference inputs for compiled models where `truncate_64bit_io` applies
- normalizing AI Hub output tensors for downstream CPU decode

- [ ] **Step 4: Add hybrid run record writing**

Record output should include:

- pilot name
- run label
- device name
- source compile record path or explicit model id
- profile and inference job metadata
- final output text
- final output comparison summary
- latency summary

- [ ] **Step 5: Run the focused hybrid-module tests**

Run: `pytest test/test_aihub_option1_hybrid_pipeline.py -k resolve_target_or_inference_adapter -v`

Expected: pass.

### Task 3: Implement The Zipformer Hybrid Python Pipeline

**Files:**

- Modify: `src/tools/aihub_option1_hybrid_pipeline.py`
- Test: `test/test_aihub_option1_hybrid_pipeline.py`

- [ ] **Step 1: Write a failing test for `Zipformer encoder on NPU + decoder/joiner on CPU`**

Test behavior:

- given:
  - a fake compiled encoder target output,
  - fake decoder and joiner sessions,
  - a sample manifest row,
- the hybrid helper returns:
  - a final transcript,
  - token count,
  - encoder vs host decode timing fields.

- [ ] **Step 2: Run the focused Zipformer hybrid test to confirm it fails**

Run: `pytest test/test_aihub_option1_hybrid_pipeline.py -k zipformer_hybrid -v`

Expected: failure because the hybrid Zipformer runner does not exist yet.

- [ ] **Step 3: Implement the minimal Zipformer hybrid runner**

Implementation should:

- reuse the Phase 2 prepared compile record / target model resolution
- build encoder inputs from the bundle fixture audio path
- submit encoder inference on the compiled target model
- trim encoder frames from `output_0` and `output_1`
- feed those frames into the extracted CPU decoder/joiner helper
- return a structured result with:
  - `sample_id`
  - `audio_path`
  - `text`
  - `encoder_latency`
  - `decode_latency`

- [ ] **Step 4: Add a tiny fixed sample-set runner**

Default behavior for the first implementation:

- read Zipformer `sample_manifest.jsonl`
- evaluate the first `2` samples by default
- allow `max_samples` override for later experiments

- [ ] **Step 5: Run the focused Zipformer hybrid tests**

Run: `pytest test/test_aihub_option1_hybrid_pipeline.py -k zipformer_hybrid -v`

Expected: pass.

### Task 4: Implement The VPCD Hybrid Python Pipeline

**Files:**

- Modify: `src/tools/aihub_option1_hybrid_pipeline.py`
- Test: `test/test_aihub_option1_hybrid_pipeline.py`

- [ ] **Step 1: Write a failing test for `tokenizer on CPU + VPCD model on NPU + host decode`**

Test behavior:

- given:
  - CPU tokenizer sessions,
  - fake compiled-model logits from the cloud step,
  - one or more golden samples,
- the hybrid helper returns:
  - final punctuated text,
  - step count or decode length,
  - timing summary.

- [ ] **Step 2: Run the focused VPCD hybrid test to confirm it fails**

Run: `pytest test/test_aihub_option1_hybrid_pipeline.py -k vpcd_hybrid -v`

Expected: failure because the hybrid VPCD runner does not exist yet.

- [ ] **Step 3: Implement the minimal VPCD hybrid runner**

Implementation should:

- reuse the Phase 2 prepared compile record / target model resolution
- keep CPU tokenizer encode and decode from the bundle runtime
- use the cloud inference helper only for the `model.mobile.onnx` step
- iterate decode steps exactly as the local runtime does:
  - build `decoder_input_ids`
  - build `decoder_attention_mask`
  - read logits at the active decoder position
  - append next token id
  - stop at `eos_token_id` or `max_decode_length`

- [ ] **Step 4: Add a tiny fixed sample-set runner**

Default behavior for the first implementation:

- read `golden_samples.jsonl`
- evaluate the first `4` samples by default, or all rows if fewer exist
- allow `max_samples` override for later experiments

- [ ] **Step 5: Run the focused VPCD hybrid tests**

Run: `pytest test/test_aihub_option1_hybrid_pipeline.py -k vpcd_hybrid -v`

Expected: pass.

### Task 5: Add The Shared Evaluation Harness And Hybrid Evidence Contract

**Files:**

- Modify: `src/tools/aihub_option1_hybrid_pipeline.py`
- Test: `test/test_aihub_option1_hybrid_pipeline.py`
- Create: `docs/workflows/aihub-option1-hybrid-pipeline.md`

- [ ] **Step 1: Write a failing test for hybrid evaluation summaries**

Test behavior:

- given multiple sample results,
- the harness writes a deterministic record containing:
  - target model id
  - compile record path
  - sample-level outputs
  - sample-level expected outputs
  - mismatch summaries
  - aggregate counts
  - simple latency summary

- [ ] **Step 2: Run the focused evaluation-record test to confirm it fails**

Run: `pytest test/test_aihub_option1_hybrid_pipeline.py -k hybrid_record -v`

Expected: failure because no hybrid evidence contract exists yet.

- [ ] **Step 3: Implement the hybrid evidence contract**

Write records under deterministic locations such as:

- `build/aihub/records/zipformer_hybrid_option1/hybrid-run-<run_label>.json`
- `build/aihub/records/vpcd_hybrid_option1/hybrid-run-<run_label>.json`

Each record should preserve:

- resolved target model id
- originating compile-run record path when used
- sample count
- sample-level final output text
- expected text
- match / mismatch classification
- latency fields

- [ ] **Step 4: Add final-output comparison only after e2e pipeline outputs exist**

Comparison rules:

- `Zipformer`
  - compare final transcript rows against `expected_outputs.jsonl`
  - do not compare expected transcripts at the encoder-tensor stage
- `VPCD`
  - compare final punctuated text rows against `golden_samples.jsonl`
  - do not compare gold text at the logits-only stage

- [ ] **Step 5: Document the record contract in the new workflow doc**

Explain:

- how to reuse Phase 2 compile records
- how to override with explicit target model ids
- what files must be preserved after a successful Phase 3 run

- [ ] **Step 6: Run the focused hybrid record tests**

Run: `pytest test/test_aihub_option1_hybrid_pipeline.py -k hybrid_record -v`

Expected: pass.

### Task 6: Add The Phase 3 Execution Notebook

**Files:**

- Modify: `On_device_Ai_option1_pilots.ipynb`
- Modify: `docs/workflows/aihub-option1-hybrid-pipeline.md`

- [ ] **Step 1: Refactor the existing notebook in-place instead of creating a second notebook**

The notebook must continue to support:

- Phase 2 compile-only flow
- Phase 2 compiled-target reuse flow
- new Phase 3 hybrid e2e flow

- [ ] **Step 2: Keep and relabel the Phase 2 tensor-diagnostic cells**

Requirements:

- current tensor inspection cells remain available
- labels should make clear these are intermediate diagnostics, not final correctness gates
- final expected-output comparison must not happen inside these cells

- [ ] **Step 3: Add the Zipformer hybrid e2e and final comparison sections**

The notebook should:

- reuse the compiled target from Phase 2
- run the tiny sample set through:
  - encoder on cloud NPU
  - decoder and joiner on CPU
- print final transcripts first
- compare against `expected_outputs.jsonl` only in the next section
- print the hybrid run record path

- [ ] **Step 4: Add the VPCD hybrid e2e and final comparison sections**

The notebook should:

- reuse the compiled target from Phase 2
- run the tiny sample set through:
  - tokenizer encode on CPU
  - model on cloud NPU
  - tokenizer decode on CPU
- print final punctuated outputs first
- compare against `golden_samples.jsonl` only in the next section
- print the hybrid run record path

- [ ] **Step 5: Keep a clear notebook usage guide**

Document these flows:

- compile already exists and should be reused through `RUN_LABEL`
- explicit target model id override
- final gold comparison runs only after the e2e section has completed

- [ ] **Step 6: Add a final notebook summary cell**

Print:

- input compile-run record paths
- hybrid run record paths
- mismatch counts
- reminder that Phase 3 is still Python-only and not Android integration

### Task 7: Run Full Phase 3 Verification

**Files:**

- Test: `test/test_aihub_option1_hybrid_pipeline.py`
- Test: existing regression slices
- Test: notebook JSON sanity

- [ ] **Step 1: Run the focused Phase 3 tests**

Run: `pytest test/test_aihub_option1_hybrid_pipeline.py -v`

Expected: pass.

- [ ] **Step 2: Run the runtime regression slices**

Run: `pytest test/test_zipformer_bundle.py test/test_vpcd_bundle.py test/test_aihub_option1_pilots.py -v`

Expected: pass.

- [ ] **Step 3: Run the broader regression slice**

Run: `pytest test/test_zipformer_quantize.py test/test_vpcd_qnn_candidate.py -v`

Expected: pass.

- [ ] **Step 4: Run Python compile verification**

Run: `python -m compileall src`

Expected: pass.

- [ ] **Step 5: Verify the Phase 3 notebook is valid JSON**

Run: `python - <<'PY'\nimport json\nfrom pathlib import Path\njson.loads(Path('On_device_Ai_option1_pilots.ipynb').read_text(encoding='utf-8'))\nprint('ok')\nPY`

Expected: prints `ok`.

## Phase 3 Acceptance Criteria

- a dedicated hybrid runtime module exists and reuses Phase 2 compile records instead of recompiling
- Zipformer can produce final transcript outputs through:
  - NPU encoder inference on AI Hub
  - CPU decoder and joiner on the host
- VPCD can produce final punctuated outputs through:
  - CPU tokenizer encode
  - NPU model inference on AI Hub
  - CPU tokenizer decode
- both pipelines run on deterministic tiny sample sets without requiring a fresh compile
- final expected/gold comparison happens only after the hybrid e2e outputs exist
- both pipelines write hybrid run records with:
  - target model id
  - sample-level final outputs
  - expected outputs
  - mismatch summaries
  - timing summaries
- `On_device_Ai_option1_pilots.ipynb` remains the single execution notebook and can run the hybrid flows by:
  - reusing `RUN_LABEL`, or
  - accepting explicit target model ids
- focused and regression tests pass locally

## Decision Gates After Phase 3

1. Keep the current Zipformer prepared-source direct compile lane as the official baseline unless the hybrid pipeline exposes a concrete blocker.
2. Treat any attempt to move additional Zipformer graphs onto NPU as a separate optimization branch, not part of the first deployment candidate.
3. Enter Phase 4 only after the hybrid run records show final outputs on deterministic sample sets for both Zipformer and VPCD.
4. Do not start Android integration until the hybrid Python pipeline outputs and evidence contract are stable.

## Recommended Execution Order For Phase 3

1. Implement `Task 1` to extract CPU-side decode seams from the existing runtimes.
2. Implement `Task 2` to create the shared hybrid runtime module and compiled-model adapter.
3. Implement `Task 3` to land the Zipformer hybrid pipeline.
4. Implement `Task 4` to land the VPCD hybrid pipeline.
5. Implement `Task 5` to lock the hybrid evidence contract and workflow doc.
6. Implement `Task 6` to refactor `On_device_Ai_option1_pilots.ipynb` into the shared Phase 2 + Phase 3 notebook.
7. Run `Task 7` verification before calling Phase 3 complete.

---

## Phase 4 Scope

Phase 4 is now implemented in the shared Python lane.

The purpose of Phase 4 is to turn the raw Phase 3 notebook outputs into a formal quality and performance gate for `Option 1`.

This phase does **not** try to improve model quality yet.
It answers these questions in a deterministic, record-backed way:

- Is the current `Zipformer encoder on NPU + decoder/joiner on CPU` lane accurate enough to advance?
- Is the current `VPCD model-step on NPU + tokenizer on CPU` lane accurate enough to advance?
- How expensive are these lanes in practice for:
  - warmup,
  - steady-state latency,
  - basic memory or footprint observations?
- Which pilot gets:
  - `GO`,
  - `WARN`,
  - `NO_GO`
  before Android work begins, while still allowing Phase 5 packaging for all pilots?

Phase 4 should be designed around the current evidence snapshot:

- `Zipformer`
  - final compare now works against the correct gold fixture
  - current mismatches look small and localized
  - this pilot needs a severity classifier and a promotion rule, not just exact-match counting
- `VPCD`
  - current hybrid output is catastrophically wrong on the tested samples
  - this pilot needs an explicit negative gate path instead of being averaged into a neutral benchmark report

### Phase 4 Non-Goals

- changing model architecture
- re-quantizing models as part of the gate itself
- moving more Zipformer graphs onto NPU
- Android integration
- hiding failures behind averaged metrics
- inventing ad-hoc judgment outside a deterministic record contract

## Phase 4 File Structure

**Files:**

- Create: `src/tools/aihub_option1_phase4_gate.py`
- Test: `test/test_aihub_option1_phase4_gate.py`
- Modify: `src/tools/aihub_option1_hybrid_pipeline.py`
- Modify: `On_device_Ai_option1_pilots.ipynb`
- Create: `docs/workflows/aihub-option1-phase4-gate.md`
- Modify: `docs/workflows/aihub-option1-hybrid-pipeline.md`
- Modify: `docs/plans/active/2026-05-11-aihub-option1-npu-pilots.md`

### Notebook Refactor Plan

Phase 4 must stay inside:

- `On_device_Ai_option1_pilots.ipynb`

The notebook should remain the single operator entrypoint.
Do not create a second benchmark notebook for the first implementation.

Target cell layout after the current Phase 3 sections:

1. `## Phase 4 Config`
   - benchmark iteration counts
   - per-pilot sample limits
   - recommendation thresholds
   - optional skip flags for rerunning expensive sections
2. `### Zipformer Phase 4 Benchmark Sweep`
   - rerun the hybrid pipeline for a small fixed iteration count without recompiling
   - collect warmup and steady-state timings
3. `### Zipformer Phase 4 Gate Summary`
   - classify per-sample mismatch severity
   - print `GO / WARN / NO_GO` recommendation with reasons
4. `### VPCD Phase 4 Benchmark Sweep`
   - rerun the hybrid pipeline for a small fixed iteration count without recompiling
   - collect timings and collapse/failure signals
5. `### VPCD Phase 4 Gate Summary`
   - classify per-sample failure severity
   - print `GO / WARN / NO_GO` recommendation with reasons
6. `## Phase 4 Recommendation Summary`
   - print both pilot recommendations together
   - print all generated Phase 4 record paths

Operator rule:

- compile sections must stay skippable
- Phase 4 must reuse:
  - `RUN_LABEL`
  - existing compile records
  - existing compiled target ids
- Phase 4 may rerun inference and hybrid flows, but must never force a fresh compile

## Phase 4 Detailed Tasks

### Task 1: Lock The Phase 4 Gate Vocabulary And Record Contract

**Files:**

- Create: `src/tools/aihub_option1_phase4_gate.py`
- Test: `test/test_aihub_option1_phase4_gate.py`

- [ ] **Step 1: Write a failing test for per-pilot gate summaries**

Test behavior:

- given sample-level Phase 3 style outputs,
- a helper produces a deterministic Phase 4 summary containing:
  - `sample_count`
  - `comparable_samples`
  - `matched_samples`
  - `mismatched_samples`
  - `severity_counts`
  - `recommendation`
  - `recommendation_reasons`

- [ ] **Step 2: Run the focused summary test to confirm it fails**

Run: `pytest test/test_aihub_option1_phase4_gate.py -k gate_summary -v`

Expected: failure because the Phase 4 module does not exist yet.

- [ ] **Step 3: Implement the minimal Phase 4 gate vocabulary**

In `src/tools/aihub_option1_phase4_gate.py`, add:

- one canonical recommendation vocabulary:
  - `GO`
  - `WARN`
  - `NO_GO`
- one canonical severity vocabulary:
  - `exact_match`
  - `minor_text_drift`
  - `major_text_drift`
  - `catastrophic_decode_failure`
  - `comparison_unavailable`
- one deterministic summary builder for sample rows

- [ ] **Step 4: Add the Phase 4 record writer**

Write records under deterministic locations such as:

- `build/aihub/records/zipformer_phase4_option1/phase4-gate-<RUN_LABEL>.json`
- `build/aihub/records/vpcd_phase4_option1/phase4-gate-<RUN_LABEL>.json`

Each record should preserve:

- pilot name
- run label
- target model id
- compile record path
- live run record path when available
- hybrid run record path
- correctness summary
- latency summary
- memory or footprint summary
- final recommendation and reasons

- [ ] **Step 5: Run the focused Phase 4 summary tests**

Run: `pytest test/test_aihub_option1_phase4_gate.py -k gate_summary -v`

Expected: pass.

### Task 2: Add Repeated Benchmark Sweep Helpers Without Recompiling

**Files:**

- Modify: `src/tools/aihub_option1_phase4_gate.py`
- Modify if needed: `src/tools/aihub_option1_hybrid_pipeline.py`
- Test: `test/test_aihub_option1_phase4_gate.py`

- [ ] **Step 1: Write a failing test for repeated hybrid benchmark sweeps**

Test behavior:

- given a fake hybrid runner that returns deterministic per-iteration timings,
- a helper records:
  - per-iteration total time
  - per-iteration cloud inference time
  - per-iteration host decode time
  - first-iteration warmup time
  - steady-state mean, min, and max

- [ ] **Step 2: Run the focused benchmark test to confirm it fails**

Run: `pytest test/test_aihub_option1_phase4_gate.py -k benchmark_sweep -v`

Expected: failure because no repeated benchmark helper exists yet.

- [ ] **Step 3: Implement the benchmark sweep helpers**

Add helpers that:

- rerun the existing hybrid evaluation without recompiling
- accept:
  - `iterations`
  - `max_samples`
  - `RUN_LABEL`
  - explicit target model id override if present
- produce:
  - one per-iteration result row
  - warmup timing from the first iteration
  - steady-state timing summary from the remaining iterations

Default first-pass benchmark sizes:

- `Zipformer`
  - `iterations = 3`
  - `max_samples = 2`
- `VPCD`
  - `iterations = 2`
  - `max_samples = 2`

- [ ] **Step 4: Run the focused benchmark tests**

Run: `pytest test/test_aihub_option1_phase4_gate.py -k benchmark_sweep -v`

Expected: pass.

### Task 3: Add Correctness Severity Classification

**Files:**

- Modify: `src/tools/aihub_option1_phase4_gate.py`
- Test: `test/test_aihub_option1_phase4_gate.py`

- [ ] **Step 1: Write a failing test for Zipformer drift classification**

Test behavior:

- exact text match classifies as `exact_match`
- tiny localized transcript drift, such as one word or one token substitution, classifies as `minor_text_drift`
- broader transcript divergence classifies as `major_text_drift`

- [ ] **Step 2: Write a failing test for VPCD catastrophic failure classification**

Test behavior:

- placeholder-like output or extremely short generated ids such as `[0, 1, 2]` against a much longer expected sentence classifies as `catastrophic_decode_failure`

- [ ] **Step 3: Run the focused classification tests to confirm they fail**

Run: `pytest test/test_aihub_option1_phase4_gate.py -k classify -v`

Expected: failure because the classifier logic does not exist yet.

- [ ] **Step 4: Implement the minimal severity classifiers**

Implementation rules for the first version:

- use exact string equality for `exact_match`
- use a deterministic normalized string-distance heuristic for:
  - `minor_text_drift`
  - `major_text_drift`
- use explicit failure heuristics for:
  - empty outputs
  - placeholder-like outputs
  - extremely short generated ids versus long expected text
  as `catastrophic_decode_failure`

Important:

- the classifier must support negative outcomes cleanly
- Phase 4 must not assume every pilot is salvageable

- [ ] **Step 5: Run the focused classification tests**

Run: `pytest test/test_aihub_option1_phase4_gate.py -k classify -v`

Expected: pass.

### Task 4: Add Basic Memory And Footprint Observations

**Files:**

- Modify: `src/tools/aihub_option1_phase4_gate.py`
- Test: `test/test_aihub_option1_phase4_gate.py`

- [ ] **Step 1: Write a failing test for footprint summaries**

Test behavior:

- given prepared artifact, live-run record, and hybrid-run record metadata,
- a helper records:
  - prepared model size
  - output tensor footprint
  - generated token footprint where relevant
  - optional host RSS delta when available

- [ ] **Step 2: Run the focused footprint test to confirm it fails**

Run: `pytest test/test_aihub_option1_phase4_gate.py -k footprint -v`

Expected: failure because no footprint helper exists yet.

- [ ] **Step 3: Implement the minimal footprint observation helpers**

Use the following strategy:

- always record artifact and tensor size observations
- attempt host process RSS observation only when the environment supports it
- if host RSS is unavailable, write a structured reason instead of failing the gate

This is important because Qualcomm AI Hub does not expose true device memory directly through the current notebook lane.

- [ ] **Step 4: Run the focused footprint tests**

Run: `pytest test/test_aihub_option1_phase4_gate.py -k footprint -v`

Expected: pass.

### Task 5: Build Per-Pilot Recommendations And Overall Gate Decisions

**Files:**

- Modify: `src/tools/aihub_option1_phase4_gate.py`
- Test: `test/test_aihub_option1_phase4_gate.py`

- [ ] **Step 1: Write a failing test for per-pilot recommendations**

Test behavior:

- `Zipformer`
  - exact match plus acceptable timings can become `GO`
  - minor drift with otherwise healthy behavior becomes `WARN`
  - broad drift or catastrophic behavior becomes `NO_GO`
- `VPCD`
  - catastrophic output collapse becomes `NO_GO`
  - exact match plus acceptable timings can become `GO`

- [ ] **Step 2: Run the focused recommendation test to confirm it fails**

Run: `pytest test/test_aihub_option1_phase4_gate.py -k recommendation -v`

Expected: failure because recommendation logic does not exist yet.

- [ ] **Step 3: Implement the recommendation rules**

The first version should:

- accept threshold values from config rather than hardcoding them into notebook cells
- produce:
  - one recommendation per pilot
  - one overall Phase 4 recommendation summary
- preserve the exact reasons that drove:
  - `GO`
  - `WARN`
  - `NO_GO`

- [ ] **Step 4: Run the focused recommendation tests**

Run: `pytest test/test_aihub_option1_phase4_gate.py -k recommendation -v`

Expected: pass.

### Task 6: Refactor The Shared Notebook For Phase 4 Execution

**Files:**

- Modify: `On_device_Ai_option1_pilots.ipynb`
- Modify: `src/tools/aihub_option1_phase4_gate.py`
- Test: notebook JSON sanity

- [ ] **Step 1: Add the Phase 4 config cell**

The config cell should expose:

- `PHASE4_ZIPFORMER_ITERATIONS`
- `PHASE4_VPCD_ITERATIONS`
- `PHASE4_ZIPFORMER_MAX_SAMPLES`
- `PHASE4_VPCD_MAX_SAMPLES`
- threshold settings for:
  - minor drift
  - catastrophic collapse
  - recommendation cutoffs

- [ ] **Step 2: Add the Zipformer Phase 4 benchmark section**

This section should:

- reuse the existing compiled target
- rerun hybrid evaluation for the configured iteration count
- write one benchmark or gate record
- print:
  - warmup time
  - steady-state summary
  - per-sample severity labels
  - recommendation

- [ ] **Step 3: Add the VPCD Phase 4 benchmark section**

This section should:

- reuse the existing compiled target
- rerun hybrid evaluation for the configured iteration count
- write one benchmark or gate record
- print:
  - warmup time
  - steady-state summary
  - per-sample severity labels
  - recommendation

- [ ] **Step 4: Add the final Phase 4 recommendation summary section**

This section should print:

- `Zipformer` recommendation and reasons
- `VPCD` recommendation and reasons
- all generated Phase 4 record paths
- a reminder that Phase 5 still packages every pilot, but only `GO` or justified `WARN` candidates may later advance to Android-facing work

- [ ] **Step 5: Verify the notebook still supports compile skipping**

Expected behavior:

- if a compile record already exists for `RUN_LABEL`, the notebook still skips compile
- Phase 4 can run without forcing any fresh compile job

### Task 7: Add The Phase 4 Workflow Doc And Run Full Verification

**Files:**

- Create: `docs/workflows/aihub-option1-phase4-gate.md`
- Modify: `docs/workflows/aihub-option1-hybrid-pipeline.md`
- Test: `test/test_aihub_option1_phase4_gate.py`
- Test: existing regression slices
- Test: notebook JSON sanity

- [ ] **Step 1: Document the Phase 4 operator workflow**

The new workflow doc should explain:

- what Phase 4 consumes from Phase 2 and Phase 3
- how to rerun benchmark sweeps without recompiling
- what each recommendation level means
- what artifacts must be preserved for handoff

- [ ] **Step 2: Run the focused Phase 4 tests**

Run: `pytest test/test_aihub_option1_phase4_gate.py -v`

Expected: pass.

- [ ] **Step 3: Run the regression slices that protect the existing helpers**

Run: `pytest test/test_aihub_option1_hybrid_pipeline.py test/test_aihub_option1_pilots.py test/test_zipformer_bundle.py test/test_vpcd_bundle.py -v`

Expected: pass.

- [ ] **Step 4: Run Python compile verification**

Run: `python -m compileall src`

Expected: pass.

- [ ] **Step 5: Verify the shared notebook is valid JSON**

Run: `python - <<'PY'\nimport json\nfrom pathlib import Path\njson.loads(Path('On_device_Ai_option1_pilots.ipynb').read_text(encoding='utf-8'))\nprint('ok')\nPY`

Expected: prints `ok`.

## Phase 4 Acceptance Criteria

- a dedicated Phase 4 module exists and can consume Phase 2 and Phase 3 records without forcing recompilation
- benchmark sweeps can rerun both pilots on the shared notebook with compile skipping intact
- each pilot receives:
  - per-sample severity labels
  - latency summary
  - basic memory or footprint summary
  - one final recommendation
- recommendation levels are deterministic and record-backed
- the notebook remains the single operator entrypoint for:
  - Phase 2
  - Phase 3
  - Phase 4
- Phase 4 records are written under deterministic paths
- focused and regression tests pass locally

## Decision Gates After Phase 4

1. Always enter Phase 5 after Phase 4 so the contract package exists for every pilot, including `NO_GO` lanes.
2. Treat any `catastrophic_decode_failure` result as an immediate `NO_GO` for that pilot until the lane is changed or repaired.
3. Treat the current VPCD lane as expected to fail the gate unless benchmark reruns and correctness summaries prove otherwise.
4. Treat Zipformer minor transcript drift as a product decision, not an automatic promotion. The Phase 4 record must make that tradeoff visible.
5. Use Phase 4 verdicts to label Phase 5 packages as either:
   - `deployment_candidate`
   - `research_only`
6. Do not start Android integration for any pilot whose Phase 4 recommendation is `NO_GO`.

## Recommended Execution Order For Phase 4

1. Implement `Task 1` to lock the gate vocabulary and record contract.
2. Implement `Task 2` to add repeated benchmark sweep helpers.
3. Implement `Task 3` to classify correctness severity.
4. Implement `Task 4` to add memory and footprint observations.
5. Implement `Task 5` to produce deterministic recommendations.
6. Implement `Task 6` to refactor `On_device_Ai_option1_pilots.ipynb` with Phase 4 sections.
7. Run `Task 7` verification before calling Phase 4 complete.

---

## Phase 5 Scope

Phase 5 is now implemented as the packaging phase for `Option 1`.

The purpose of Phase 5 is to turn the working research lane and its evidence into a deterministic contract package that later consumers can trust.

This phase now proceeds regardless of the Phase 4 verdict.
That means:

- `GO` pilots are packaged as likely deployment candidates
- `WARN` pilots are packaged with explicit risk notes
- `NO_GO` pilots are still packaged, but clearly marked as `research_only`

Phase 5 is not a loophole around Phase 4.
It does not override the gate result.
It preserves the gate result inside the package so later Android work, reviews, or experiments do not lose context.

The package should answer, without opening the notebook:

- what artifact was compiled
- what runtime contract it expects
- what evidence exists
- what the latest quality gate decided
- whether this pilot is:
  - `deployment_candidate`
  - `research_only`

### Phase 5 Non-Goals

- changing model quality or performance
- recompiling models
- rerunning large cloud benchmarks unless a required evidence file is missing
- Android integration
- hiding `NO_GO` results behind packaging success

## Phase 5 File Structure

**Files:**

- Create: `src/tools/aihub_option1_phase5_contract.py`
- Test: `test/test_aihub_option1_phase5_contract.py`
- Modify: `src/tools/aihub_option1_phase4_gate.py`
- Modify if needed: `src/tools/aihub_option1_hybrid_pipeline.py`
- Modify: `On_device_Ai_option1_pilots.ipynb`
- Create: `docs/workflows/aihub-option1-phase5-contract.md`
- Modify: `docs/workflows/aihub-option1-phase4-gate.md`
- Modify: `docs/plans/active/2026-05-11-aihub-option1-npu-pilots.md`

### Notebook Refactor Plan

Phase 5 must stay inside:

- `On_device_Ai_option1_pilots.ipynb`

The notebook should remain the single operator entrypoint.

Target new sections after the Phase 4 gate sections:

1. `## Phase 5 Config`
   - package root
   - package label override
   - optional per-pilot include flags
2. `### Package Zipformer Phase 5 Contract`
   - gather Phase 2, Phase 3, and Phase 4 evidence
   - materialize one per-pilot contract package
3. `### Package VPCD Phase 5 Contract`
   - gather the same evidence for VPCD
   - materialize one per-pilot contract package
4. `## Phase 5 Packaging Summary`
   - print package paths
   - print promotion status
   - print missing optional artifacts, if any

Operator rule:

- Phase 5 may package a `NO_GO` pilot
- Phase 5 must never silently present a `NO_GO` package as deployable
- the notebook must keep compile skipping intact

## Phase 5 Detailed Tasks

### Task 1: Lock The Contract Package Layout And Manifest Schema

**Files:**

- Create: `src/tools/aihub_option1_phase5_contract.py`
- Test: `test/test_aihub_option1_phase5_contract.py`

- [ ] **Step 1: Write a failing test for contract manifest generation**

Test behavior:

- given one pilot and a set of Phase 2 to Phase 4 records,
- a helper produces one deterministic manifest containing:
  - pilot name
  - run label
  - target model id
  - promotion status
  - Phase 4 recommendation
  - source artifact metadata
  - evidence record paths
  - input and output contract summary

- [ ] **Step 2: Run the focused manifest test to confirm it fails**

Run: `pytest test/test_aihub_option1_phase5_contract.py -k manifest -v`

Expected: failure because the Phase 5 module does not exist yet.

- [ ] **Step 3: Implement the minimal manifest schema**

In `src/tools/aihub_option1_phase5_contract.py`, add:

- one canonical manifest builder
- one canonical promotion status vocabulary:
  - `deployment_candidate`
  - `research_only`
- deterministic package layout helpers

- [ ] **Step 4: Run the focused manifest tests**

Run: `pytest test/test_aihub_option1_phase5_contract.py -k manifest -v`

Expected: pass.

### Task 2: Resolve And Validate Upstream Evidence Inputs

**Files:**

- Modify: `src/tools/aihub_option1_phase5_contract.py`
- Test: `test/test_aihub_option1_phase5_contract.py`

- [ ] **Step 1: Write a failing test for evidence resolution**

Test behavior:

- a helper can resolve all required upstream inputs from deterministic locations:
  - prepared artifact record
  - compile-run record
  - live-run record
  - hybrid-run record
  - Phase 4 gate record
- if an optional input is missing, the package still builds with a recorded warning
- if a required input is missing, the packager fails clearly

- [ ] **Step 2: Run the focused evidence-resolution test to confirm it fails**

Run: `pytest test/test_aihub_option1_phase5_contract.py -k resolve_inputs -v`

Expected: failure because no evidence resolver exists yet.

- [ ] **Step 3: Implement the evidence resolvers**

Implementation rules:

- Phase 5 should prefer deterministic record paths from:
  - `RUN_LABEL`
  - pilot name
- Phase 5 should preserve missing-optional notes instead of crashing
- required inputs should fail loudly with actionable messages

- [ ] **Step 4: Run the focused evidence-resolution tests**

Run: `pytest test/test_aihub_option1_phase5_contract.py -k resolve_inputs -v`

Expected: pass.

### Task 3: Build The Per-Pilot Contract Package Materializer

**Files:**

- Modify: `src/tools/aihub_option1_phase5_contract.py`
- Test: `test/test_aihub_option1_phase5_contract.py`

- [ ] **Step 1: Write a failing test for package materialization**

Test behavior:

- a packager creates one deterministic directory such as:
  - `build/aihub/contracts/option1/zipformer/<RUN_LABEL>/`
  - `build/aihub/contracts/option1/vpcd/<RUN_LABEL>/`
- and writes:
  - `contract_manifest.json`
  - `contract_summary.md`
  - one normalized `io_contract.json`
  - copied or linked evidence records under an `evidence/` folder

- [ ] **Step 2: Run the focused materialization test to confirm it fails**

Run: `pytest test/test_aihub_option1_phase5_contract.py -k materialize_package -v`

Expected: failure because the package writer does not exist yet.

- [ ] **Step 3: Implement the minimal package writer**

Package contents for the first implementation:

- local contract metadata files are always materialized
- upstream JSON evidence records are copied into the package
- large source artifacts may be referenced in the manifest instead of duplicated if copying them would be wasteful
- every copied or referenced file must be described by:
  - path
  - size
  - hash when available

- [ ] **Step 4: Run the focused materialization tests**

Run: `pytest test/test_aihub_option1_phase5_contract.py -k materialize_package -v`

Expected: pass.

### Task 4: Add Promotion Status Rules That Respect Phase 4

**Files:**

- Modify: `src/tools/aihub_option1_phase5_contract.py`
- Test: `test/test_aihub_option1_phase5_contract.py`

- [ ] **Step 1: Write a failing test for promotion status mapping**

Test behavior:

- `GO` maps to `deployment_candidate`
- `WARN` may still map to `deployment_candidate`, but must preserve risk notes
- `NO_GO` maps to `research_only`

- [ ] **Step 2: Run the focused promotion-status test to confirm it fails**

Run: `pytest test/test_aihub_option1_phase5_contract.py -k promotion_status -v`

Expected: failure because the mapping rules do not exist yet.

- [ ] **Step 3: Implement the promotion status mapper**

The first version should:

- consume the Phase 4 recommendation
- preserve the recommendation reasons in the package manifest
- never allow `NO_GO` to appear as deployable

- [ ] **Step 4: Run the focused promotion-status tests**

Run: `pytest test/test_aihub_option1_phase5_contract.py -k promotion_status -v`

Expected: pass.

### Task 5: Export Normalized I/O Contracts And Deployment Notes

**Files:**

- Modify: `src/tools/aihub_option1_phase5_contract.py`
- Test: `test/test_aihub_option1_phase5_contract.py`

- [ ] **Step 1: Write a failing test for normalized I/O contract export**

Test behavior:

- the package contains one normalized `io_contract.json` per pilot with:
  - input tensor names
  - input shapes
  - input dtypes
  - output tensor names
  - output shapes
  - dtype expectations
  - special handling notes such as `truncate_64bit_io`

- [ ] **Step 2: Run the focused I/O contract test to confirm it fails**

Run: `pytest test/test_aihub_option1_phase5_contract.py -k io_contract -v`

Expected: failure because the export helper does not exist yet.

- [ ] **Step 3: Implement the normalized I/O contract export**

Implementation rules:

- prefer data already recorded in:
  - prepared artifact records
  - live-run records
  - bundle manifests
- add deployment notes for:
  - Zipformer `encoder-only on NPU`
  - VPCD `model-step on NPU`
  - any dtype coercion required by the compiled target path

- [ ] **Step 4: Run the focused I/O contract tests**

Run: `pytest test/test_aihub_option1_phase5_contract.py -k io_contract -v`

Expected: pass.

### Task 6: Refactor The Shared Notebook For Phase 5 Packaging

**Files:**

- Modify: `On_device_Ai_option1_pilots.ipynb`
- Modify: `src/tools/aihub_option1_phase5_contract.py`
- Test: notebook JSON sanity

- [ ] **Step 1: Add the Phase 5 config cell**

The config cell should expose:

- `PHASE5_PACKAGE_LABEL`
- `PHASE5_OUTPUT_ROOT`
- `PHASE5_INCLUDE_ZIPFORMER`
- `PHASE5_INCLUDE_VPCD`

- [ ] **Step 2: Add the Zipformer package section**

This section should:

- consume the latest records for the chosen `RUN_LABEL`
- build one Zipformer contract package
- print:
  - package path
  - promotion status
  - recommendation snapshot
  - warnings, if any

- [ ] **Step 3: Add the VPCD package section**

This section should:

- consume the latest records for the chosen `RUN_LABEL`
- build one VPCD contract package
- print:
  - package path
  - promotion status
  - recommendation snapshot
  - warnings, if any

- [ ] **Step 4: Add the final Phase 5 package summary section**

This section should print:

- package path per pilot
- promotion status per pilot
- which pilots are:
  - `deployment_candidate`
  - `research_only`
- a reminder that package creation does not override the Phase 4 verdict

- [ ] **Step 5: Verify the notebook still supports compile skipping and package-only reruns**

Expected behavior:

- the operator can rerun Phase 5 packaging without rerunning compile
- the operator can rerun Phase 5 packaging without rerunning Phase 4 if the gate records already exist

### Task 7: Add The Phase 5 Workflow Doc And Run Full Verification

**Files:**

- Create: `docs/workflows/aihub-option1-phase5-contract.md`
- Modify: `docs/workflows/aihub-option1-phase4-gate.md`
- Test: `test/test_aihub_option1_phase5_contract.py`
- Test: existing regression slices
- Test: notebook JSON sanity

- [ ] **Step 1: Document the Phase 5 operator workflow**

The new workflow doc should explain:

- what Phase 5 consumes from Phases 2, 3, and 4
- how package layout is organized
- what `deployment_candidate` and `research_only` mean
- what downstream consumers should read first in the package

- [ ] **Step 2: Run the focused Phase 5 tests**

Run: `pytest test/test_aihub_option1_phase5_contract.py -v`

Expected: pass.

- [ ] **Step 3: Run the regression slices that protect upstream helpers**

Run: `pytest test/test_aihub_option1_phase4_gate.py test/test_aihub_option1_hybrid_pipeline.py test/test_aihub_option1_pilots.py -v`

Expected: pass.

- [ ] **Step 4: Run Python compile verification**

Run: `python -m compileall src`

Expected: pass.

- [ ] **Step 5: Verify the shared notebook is valid JSON**

Run: `python - <<'PY'\nimport json\nfrom pathlib import Path\njson.loads(Path('On_device_Ai_option1_pilots.ipynb').read_text(encoding='utf-8'))\nprint('ok')\nPY`

Expected: prints `ok`.

## Phase 5 Acceptance Criteria

- a dedicated Phase 5 module exists and can package both pilots without forcing recompilation
- every package includes:
  - manifest
  - normalized I/O contract
  - evidence copies or references
  - promotion status
  - Phase 4 recommendation snapshot
- `GO`, `WARN`, and `NO_GO` pilots can all be packaged
- `NO_GO` pilots are always marked `research_only`
- the notebook remains the single operator entrypoint for:
  - Phase 2
  - Phase 3
  - Phase 4
  - Phase 5
- focused and regression tests pass locally

## Decision Gates After Phase 5

1. Start Android integration only from packages marked `deployment_candidate`.
2. Preserve `research_only` packages for debugging, comparison, and future rework, but do not treat them as app-ready.
3. Do not let package existence be mistaken for production readiness.
4. If multiple packages exist for the same pilot, later phases must select one explicitly by:
   - `RUN_LABEL`
   - package label
   - recommendation snapshot

## Recommended Execution Order For Phase 5

1. Implement `Task 1` to lock the package manifest schema.
2. Implement `Task 2` to resolve and validate upstream evidence.
3. Implement `Task 3` to materialize deterministic contract packages.
4. Implement `Task 4` to map Phase 4 verdicts into promotion status.
5. Implement `Task 5` to export normalized I/O contracts and deployment notes.
6. Implement `Task 6` to refactor `On_device_Ai_option1_pilots.ipynb` with Phase 5 sections.
7. Run `Task 7` verification before calling Phase 5 complete.

## Phase 6 Scope

Phase 6 is the first Android-app phase in the Option 1 roadmap.

The purpose is narrow and explicit:

- take the Phase 5 contract package as the only promoted Python-side input
- sync that package into `BKMeeting/modelassets`
- teach the existing ORT + QNN Android runtime to load the `precompiled_qnn_onnx` artifact through the current manifest-driven asset flow
- validate the selected pilot on device without changing the runtime boundary away from ONNX Runtime

Phase 6 should start with `Zipformer` first because its current evidence is stronger.

`VPCD` can still be wired in the same phase, but if its Phase 4 verdict remains weak it must stay experimental:

- present in assets for reproducibility
- available only behind explicit selection
- not promoted as the default punctuation path

### Phase 6 Non-Goals

- moving the runtime boundary to QAIRT-native integration
- changing the current CPU/NPU split:
  - `Zipformer`: encoder only on NPU
  - `VPCD`: model session only on NPU, tokenizer stays CPU
- redesigning the provider stack from scratch if the current ORT + QNN layer can already load the compiled artifacts
- using notebook records directly inside the Android app
- treating `research_only` packages as production-ready

## Phase 6 File Structure

**Python-side sync and contract handoff**

- Modify: `src/tools/sync_android_bundle.py`
- Test: `test/test_sync_android_bundle.py`

**Android asset catalogs and capability detection**

- Modify: `../BKMeeting/app/src/main/java/com/navis/bkacs/asr/catalog/AsrModelCatalog.java`
- Modify: `../BKMeeting/app/src/main/java/com/navis/bkacs/postprocess/catalog/PunctuationModelCatalog.java`
- Modify: `../BKMeeting/app/src/main/java/com/navis/bkacs/runtime/ModelBundleQnnCapability.java`
- Modify: `../BKMeeting/app/src/main/java/com/navis/bkacs/runtime/AsrProviderAssignment.java`
- Modify: `../BKMeeting/app/src/main/java/com/navis/bkacs/runtime/PunctuationProviderAssignment.java`

**Android session loading**

- Modify: `../BKMeeting/app/src/main/java/com/navis/bkacs/asr/model/OnnxSessionManager.java`
- Modify: `../BKMeeting/app/src/main/java/com/navis/bkacs/postprocess/model/PunctuationOnnxSessionManager.java`
- Modify only if needed: `../BKMeeting/app/src/main/java/com/navis/bkacs/runtime/OrtSessionOptionsFactory.java`
- Modify only if needed: `../BKMeeting/app/src/main/java/com/navis/bkacs/runtime/QnnProviderOptions.java`

**Android tests**

- Modify: `../BKMeeting/app/src/test/java/com/navis/bkacs/runtime/AsrProviderAssignmentTest.java`
- Modify: `../BKMeeting/app/src/test/java/com/navis/bkacs/runtime/PunctuationProviderAssignmentTest.java`
- Modify: `../BKMeeting/app/src/test/java/com/navis/bkacs/modelbundle/ModelAssetsContractTest.java`
- Modify if needed: `../BKMeeting/app/src/test/java/com/navis/bkacs/asr/model/OnnxSessionManagerTest.java`
- Modify if needed: `../BKMeeting/app/src/test/java/com/navis/bkacs/postprocess/model/PunctuationOnnxSessionManagerTest.java`
- Modify if needed: `../BKMeeting/app/src/androidTest/java/com/navis/bkacs/asr/bundle/ZipformerBundleParityTest.java`
- Modify if needed: `../BKMeeting/app/src/androidTest/java/com/navis/bkacs/asr/runtime/QnnHtpStrictDeviceTest.java`
- Modify if needed: `../BKMeeting/app/src/androidTest/java/com/navis/bkacs/postprocess/runtime/VpcdQnnHtpStrictDeviceTest.java`

## Phase 6 Asset Contract Shape

Phase 6 should not ask Android to interpret the full Phase 5 research package directly.

Instead, the sync tool should materialize an Android-ready payload under `BKMeeting/modelassets`, keeping the existing `bundle_manifest.json` entrypoint and adding only the extra files Android truly needs.

Minimum Android payload per pilot:

- `bundle_manifest.json`
- compiled `precompiled_qnn_onnx` model file
- `io_contract.json`
- any runtime-required fixtures already expected by the current bundle flow

The synthesized Android-facing `bundle_manifest.json` should carry enough metadata for runtime capability detection:

- `asset_namespace`
- `model_name`
- `model_variant`
- `metadata.quantization`
- `metadata.qnn_readiness`
- `metadata.option1`
  - `target_runtime = precompiled_qnn_onnx`
  - `run_label`
  - `promotion_status`
  - `device_name`
  - `qairt_version`
  - `io_contract_file = io_contract.json`

`io_contract.json` remains the precise place for:

- input tensor names
- input dtypes
- input shapes
- output tensor names
- output dtypes
- output shapes
- special handling such as `truncate_64bit_io`

## Phase 6 Detailed Tasks

### Task 1: Extend The Android Sync Tool For Phase 5 Contract Packages

**Files:**

- Modify: `src/tools/sync_android_bundle.py`
- Test: `test/test_sync_android_bundle.py`

- [ ] **Step 1: Write a failing test for syncing a Phase 5 Option 1 contract package**

Test behavior:

- given a Phase 5 contract package directory for `zipformer` or `vpcd`
- the sync tool copies the compiled model into the right Android asset namespace
- the sync tool writes an Android-facing `bundle_manifest.json`
- the sync tool copies `io_contract.json`

- [ ] **Step 2: Run the test to confirm it fails**

Run: `pytest test/test_sync_android_bundle.py -k option1 -v`

Expected: failure because the sync tool only understands bundle variants today.

- [ ] **Step 3: Implement contract-package-aware sync**

The simplest acceptable implementation is:

- extend `sync_android_bundle.py` with an Option 1 contract input mode
- keep namespace mapping deterministic
- refuse to sync packages without the minimum Phase 5 evidence files

- [ ] **Step 4: Run the focused sync test to verify it passes**

Run: `pytest test/test_sync_android_bundle.py -k option1 -v`

Expected: pass.

### Task 2: Define The Android-Facing Option 1 Manifest Contract

**Files:**

- Modify: `src/tools/sync_android_bundle.py`
- Test: `test/test_sync_android_bundle.py`
- Modify: `../BKMeeting/app/src/test/java/com/navis/bkacs/modelbundle/ModelAssetsContractTest.java`

- [ ] **Step 1: Write a failing test for the synthesized manifest metadata**

Test behavior:

- the synced manifest contains:
  - `metadata.option1.target_runtime`
  - `metadata.option1.run_label`
  - `metadata.option1.promotion_status`
  - `metadata.option1.io_contract_file`

- [ ] **Step 2: Run the tests to confirm they fail**

Run:

- `pytest test/test_sync_android_bundle.py -k option1_manifest -v`
- `.\gradlew.bat :app:testDebugUnitTest --tests "com.navis.bkacs.modelbundle.ModelAssetsContractTest" --no-daemon`

Expected: failures because the new manifest contract does not exist yet.

- [ ] **Step 3: Implement the Android-facing Option 1 metadata block**

Keep this rule:

- Android still enters through `bundle_manifest.json`
- `io_contract.json` is referenced from the manifest, not guessed by filename conventions alone

- [ ] **Step 4: Re-run the focused tests**

Expected: pass.

### Task 3: Register Option 1 Packages In The Android Catalogs

**Files:**

- Modify: `../BKMeeting/app/src/main/java/com/navis/bkacs/asr/catalog/AsrModelCatalog.java`
- Modify: `../BKMeeting/app/src/main/java/com/navis/bkacs/postprocess/catalog/PunctuationModelCatalog.java`
- Modify: corresponding catalog tests

- [ ] **Step 1: Write failing catalog tests for new Option 1 entries**

Test behavior:

- `Zipformer` gets an explicit catalog entry for the synced Phase 6 Option 1 asset namespace
- `VPCD` gets an explicit catalog entry only when the package is intentionally exposed

- [ ] **Step 2: Run the focused tests to confirm they fail**

Run:

- `.\gradlew.bat :app:testDebugUnitTest --tests "com.navis.bkacs.asr.catalog.AsrModelCatalogTest" --tests "com.navis.bkacs.postprocess.catalog.PunctuationModelCatalogTest" --no-daemon`

Expected: failure because catalog entries do not exist yet.

- [ ] **Step 3: Implement catalog entries and promotion rules**

Recommended rule:

- `deployment_candidate` packages may appear in the normal selectable catalog
- `research_only` packages must either stay hidden or be clearly marked experimental

- [ ] **Step 4: Re-run the focused catalog tests**

Expected: pass.

### Task 4: Teach Runtime Capability Detection To Recognize Option 1 Precompiled Bundles

**Files:**

- Modify: `../BKMeeting/app/src/main/java/com/navis/bkacs/runtime/ModelBundleQnnCapability.java`
- Modify: `../BKMeeting/app/src/main/java/com/navis/bkacs/runtime/AsrProviderAssignment.java`
- Modify: `../BKMeeting/app/src/main/java/com/navis/bkacs/runtime/PunctuationProviderAssignment.java`
- Test: corresponding runtime unit tests

- [ ] **Step 1: Write failing runtime tests for precompiled bundle recognition**

Test behavior:

- the runtime treats `metadata.option1.target_runtime = precompiled_qnn_onnx` as QNN-capable
- this recognition must not depend on the old `QDQ + fixed_shapes` check alone

- [ ] **Step 2: Run the focused runtime tests to confirm they fail**

Run:

- `.\gradlew.bat :app:testDebugUnitTest --tests "com.navis.bkacs.runtime.AsrProviderAssignmentTest" --tests "com.navis.bkacs.runtime.PunctuationProviderAssignmentTest" --no-daemon`

Expected: failure because Option 1 precompiled bundles are not recognized yet.

- [ ] **Step 3: Implement precompiled capability detection**

Keep the logic narrow:

- existing QDQ candidate detection remains untouched for the current local-QNN lane
- add a second path for `precompiled_qnn_onnx`
- do not let malformed manifests silently fall back to “QNN-capable”

- [ ] **Step 4: Re-run the focused runtime tests**

Expected: pass.

### Task 5: Load Precompiled Option 1 Artifacts Through The Existing Session Managers

**Files:**

- Modify: `../BKMeeting/app/src/main/java/com/navis/bkacs/asr/model/OnnxSessionManager.java`
- Modify: `../BKMeeting/app/src/main/java/com/navis/bkacs/postprocess/model/PunctuationOnnxSessionManager.java`
- Modify only if needed: `../BKMeeting/app/src/main/java/com/navis/bkacs/runtime/OrtSessionOptionsFactory.java`
- Modify only if needed: `../BKMeeting/app/src/main/java/com/navis/bkacs/runtime/QnnProviderOptions.java`

- [ ] **Step 1: Write failing unit tests for precompiled artifact loading**

Test behavior:

- the session managers can resolve the synced Option 1 model asset
- the runtime keeps using the existing ORT + QNN provider stack
- integer inputs follow `io_contract.json`, including `truncate_64bit_io` when required

- [ ] **Step 2: Run the focused session-manager tests to confirm they fail**

Run:

- `.\gradlew.bat :app:testDebugUnitTest --tests "com.navis.bkacs.asr.model.OnnxSessionManagerTest" --tests "com.navis.bkacs.postprocess.model.PunctuationOnnxSessionManagerTest" --no-daemon`

Expected: failure because the session managers do not know the Phase 6 contract yet.

- [ ] **Step 3: Implement precompiled loading with minimal provider changes**

Preferred rule:

- reuse the existing provider-option plumbing first
- only touch `OrtSessionOptionsFactory` or `QnnProviderOptions` if the current path cannot create valid sessions for the compiled artifact

- [ ] **Step 4: Re-run the focused unit tests**

Expected: pass.

### Task 6: Validate The Selected Option 1 Asset On Real Android Hardware

**Files:**

- Modify if needed: `../BKMeeting/app/src/androidTest/java/com/navis/bkacs/asr/bundle/ZipformerBundleParityTest.java`
- Modify if needed: `../BKMeeting/app/src/androidTest/java/com/navis/bkacs/asr/runtime/QnnHtpStrictDeviceTest.java`
- Modify if needed: `../BKMeeting/app/src/androidTest/java/com/navis/bkacs/postprocess/runtime/VpcdQnnHtpStrictDeviceTest.java`

- [ ] **Step 1: Add or update instrumentation coverage for the synced Option 1 asset**

Test behavior:

- the strict device test loads the new asset namespace
- session creation succeeds on the real Snapdragon target
- fallback behavior remains explicit and observable

- [ ] **Step 2: Run Android instrumentation verification**

Run:

- `.\gradlew.bat :app:connectedDebugAndroidTest --no-daemon`

Expected:

- `Zipformer` strict load succeeds for the promoted Option 1 package
- `VPCD` succeeds only if its promoted package is genuinely app-ready; otherwise the failure is recorded without redefining the contract

### Task 7: Refresh The Playbook And Close The Android Handoff Loop

**Files:**

- Modify: `../BKMeeting/docs/qnn/playbook.md`
- Modify: this plan file

- [ ] **Step 1: Document the exact Phase 6 operator flow**

Document:

- which Phase 5 package to sync
- which catalog entry to select
- which unit tests to run
- which device test to run

- [ ] **Step 2: Record the minimum evidence for “Android-integrated Option 1”**

At minimum:

- synced asset namespace path
- manifest metadata snapshot
- unit-test pass
- device strict-test result

- [ ] **Step 3: Re-read the playbook and ensure it matches the code path**

Expected: no contradictions between Python-side packaging and Android-side loading.

## Phase 6 Acceptance Criteria

- a Phase 5 contract package can be synced into `BKMeeting/modelassets` without manual file shuffling
- Android catalogs can point to the synced Option 1 asset namespace intentionally
- runtime capability detection recognizes `precompiled_qnn_onnx` bundles without breaking the current QDQ lane
- the existing ORT + QNN session managers can attempt to load the synced precompiled artifact
- at least the promoted `Zipformer` Option 1 package can be exercised through Android validation
- `VPCD` remains explicitly labeled experimental if its gate/package evidence is still weak

## Decision Gates After Phase 6

1. Promote only the Android-validated `deployment_candidate` package into default app flows.
2. Keep `research_only` or failing packages available only for explicit debug selection.
3. Do not let Android asset sync become the only evidence source; keep Phase 4 and Phase 5 records linked.
4. If the current provider stack cannot load the compiled artifact cleanly, treat that as a Phase 6 blocker instead of silently switching Option 1 to another runtime lane.

## Recommended Execution Order For Phase 6

1. Implement `Task 1` first so the Android asset payload shape is deterministic.
2. Implement `Task 2` immediately after, because the manifest contract drives every downstream runtime decision.
3. Implement `Task 3` to expose the synced asset in catalogs only after the asset payload is stable.
4. Implement `Task 4` before touching session creation so provider routing is explicit.
5. Implement `Task 5` with the current ORT + QNN stack kept as intact as possible.
6. Run `Task 6` on real hardware only after the unit-test gates are clean.
7. Finish with `Task 7` so the Android handoff path is reproducible for the next engineer.
