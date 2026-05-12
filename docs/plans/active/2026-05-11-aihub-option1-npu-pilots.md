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

**Status:** Next phase

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

## Phase 3 Scope

Phase 3 is now the next coding phase.

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
