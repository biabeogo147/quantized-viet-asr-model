# VPCD Quantize Vs Compile Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Isolate whether the current VPCD punctuation-collapse failure originates in the AI Hub quantize stage or only after AI Hub compile, while proving that the calibration dataset used on AI Hub matches the local autoregressive calibration recipe.

**Architecture:** Reuse the existing fixed-shape FP32 source lane, but split the current opaque cloud path into two observable checkpoints. First, build one canonical autoregressive calibration dataset locally, fingerprint it, and upload exactly that dataset to AI Hub quantize. Second, download the resulting quantized ONNX model and run the same teacher-forced diagnostic locally with ONNX Runtime before testing the compiled cloud target. This creates a three-way comparison: FP32 local, quantized local, compiled cloud.

**Tech Stack:** Python 3.10+, Jupyter notebook (`.ipynb`), NumPy, ONNX, ONNX Runtime, Qualcomm AI Hub (`qai-hub`), JSON run records, pytest.

---

## Scope And Boundaries

- This plan is only for the VPCD Option 1 lane.
- Keep the source lane unchanged:
  - `FP32 fixed-shape prepare -> AI Hub quantize -> AI Hub compile`
- Do not reopen local QDQ-source selection work.
- Do not broaden this plan into generic quantization tuning.
- The main deliverable is observability and root-cause attribution:
  - `quantize` fault,
  - `compile / QNN execution` fault,
  - or neither.
- The notebook remains the operator-facing entrypoint:
  - `On_device_Ai_option1_pilots.ipynb`

## Root-Cause Question To Answer

After this plan is implemented, the team must be able to answer the following with evidence:

1. Does the downloaded AI Hub quantized ONNX already diverge from FP32 at teacher-forced step `2`?
2. If not, does divergence appear only after AI Hub compile and cloud inference?
3. Was the AI Hub quantize job calibrated with the same locally built autoregressive dataset that the notebook intended to use?

## Success Criteria

- The VPCD compile flow writes a quantize-run record that includes:
  - quantize job metadata,
  - quantized target model metadata,
  - downloaded quantized artifact metadata,
  - calibration stats,
  - a stable calibration dataset fingerprint.
- The notebook can resolve the downloaded quantized ONNX for a given `RUN_LABEL` and run local teacher-forced diagnostics against it.
- The local quantized teacher-forced record uses the same evidence style as the existing cloud teacher-forced record:
  - expected token,
  - FP32 argmax,
  - quantized argmax,
  - top-k summaries,
  - first divergent step.
- The calibration dataset uploaded to AI Hub is built once locally and fingerprinted before upload.
- The final decision rule becomes concrete:
  - quantized local diverges => `quantize` is the root-cause stage
  - quantized local matches but compiled cloud diverges => `compile / QNN execution` is the root-cause stage
- If `quantize` is confirmed as the failing stage, the plan immediately provides one bounded follow-up matrix with calibration held constant and only quantize options changed:
  - `A`: `w8a16 + auto + calibration giữ nguyên`
  - `B`: `w8a16 + min_max + calibration giữ nguyên`
  - `C`: `w8a8 + auto + calibration giữ nguyên`
  - `D`: `w8a8 + min_max + calibration giữ nguyên`

## Official Assumptions To Preserve

- AI Hub quantize produces a quantized ONNX model in QDQ-like ONNX format and exposes it as the quantize job target model.
- AI Hub supports downloading the quantized target model from the quantize job.
- The current notebook must continue using the same local autoregressive calibration builder as the source of truth for AI Hub calibration data.

## File Structure

**Files:**

- Modify: `src/quantize/projects/vpcd.py`
- Modify: `src/quantize/types.py`
- Modify: `src/tools/aihub_option1_pilots.py`
- Modify: `src/tools/aihub_option1_hybrid_pipeline.py`
- Modify: `On_device_Ai_option1_pilots.ipynb`
- Modify: `docs/workflows/aihub-option1-npu-pilots.md`
- Modify: `docs/workflows/aihub-option1-hybrid-pipeline.md`
- Modify: `docs/plans/active/2026-05-13-vpcd-option1-debug-results.md`
- Test: `test/test_vpcd_quantize_aihub.py`
- Test: `test/test_aihub_option1_pilots.py`
- Test: `test/test_aihub_option1_hybrid_pipeline.py`
- Test: `test/test_option1_notebook_layout.py`

### File Responsibilities

- `src/quantize/projects/vpcd.py`
  - remains the source of truth for building the AI Hub calibration dataset
  - must surface a stable fingerprint and enough metadata to prove local/AI Hub parity
- `src/quantize/types.py`
  - carries the expanded quantize recipe metadata if needed
- `src/tools/aihub_option1_pilots.py`
  - owns quantize submission, quantized artifact download, quantize-run record writing, and record resolution helpers
- `src/tools/aihub_option1_hybrid_pipeline.py`
  - owns teacher-forced diagnostics
  - must add a local quantized-teacher-forced comparator without duplicating the existing cloud comparator logic unnecessarily
- `On_device_Ai_option1_pilots.ipynb`
  - must run the new diagnostic order:
    - build/upload quantize dataset
    - download quantized ONNX
    - local quantized teacher-forced
    - compiled cloud teacher-forced
    - bounded free-run hybrid
- docs
  - must explain the new decision tree and record locations
- tests
  - must lock calibration parity evidence, quantized-artifact download behavior, notebook ordering, and attribution logic

## Proposed Record Additions

- New record kind under `vpcd_option1`:
  - `quantize-run-<RUN_LABEL>.json`
- New teacher-forced pilot record:
  - `build/aihub/records/vpcd_quantized_teacher_forced_option1/hybrid-run-<RUN_LABEL>.json`
- New quantized artifact download path convention:
  - `build/aihub/vpcd_option1/model.quantized.<RUN_LABEL>.onnx`

## Quantize Fallback Matrix

Only use this matrix after the attribution step proves that the downloaded quantized ONNX already diverges from the FP32 reference.

Rules for this matrix:

- keep the source model unchanged:
  - fixed-shape FP32 prepared artifact
- keep the calibration dataset unchanged:
  - same local autoregressive builder
  - same text source
  - same sample cap
  - same generation-length cap
  - same dataset fingerprint
- change only these two axes:
  - activation dtype
  - AI Hub quantize range scheme
- test each variant in the smallest useful loop:
  - quantize
  - download quantized ONNX
  - run local quantized teacher-forced on sample `0`
  - stop immediately if step `2` still diverges

Variant table:

- `A`
  - weights: `INT8`
  - activations: `INT16`
  - quantize options: `auto`
  - calibration: unchanged
- `B`
  - weights: `INT8`
  - activations: `INT16`
  - quantize options: `--range_scheme min_max`
  - calibration: unchanged
- `C`
  - weights: `INT8`
  - activations: `INT8`
  - quantize options: `auto`
  - calibration: unchanged
- `D`
  - weights: `INT8`
  - activations: `INT8`
  - quantize options: `--range_scheme min_max`
  - calibration: unchanged

Expected interpretation:

- if `A` fails but `B` passes:
  - the main suspect is the AI Hub range-selection policy, not the calibration dataset itself
- if `A/B` fail but `C` or `D` passes:
  - the main suspect is the `INT16` activation lane
- if all four variants fail at the same early step:
  - the problem is likely deeper than a simple dtype or range-scheme choice
  - move next to source-graph / quantize compatibility investigation instead of spending more time on compile

## Detailed Tasks

### Task 1: Fingerprint The Local Calibration Dataset Before Upload

**Files:**

- Modify: `src/quantize/projects/vpcd.py`
- Modify: `src/quantize/types.py`
- Test: `test/test_vpcd_quantize_aihub.py`

- [ ] **Step 1: Write the failing test for calibration fingerprinting**

Test behavior:

- `build_vpcd_aihub_quantize_recipe(...)` must return calibration stats with a stable fingerprint and structural summary.
- The fingerprint must be derived from the ordered arrays actually uploaded to AI Hub.

```python
recipe = build_vpcd_aihub_quantize_recipe(...)

assert recipe.calibration_stats["records"] == 3
assert recipe.calibration_stats["dataset_fingerprint"]
assert recipe.calibration_stats["input_order"] == [
    "input_ids",
    "attention_mask",
    "decoder_input_ids",
    "decoder_attention_mask",
]
```

- [ ] **Step 2: Run the focused quantize recipe test to confirm it fails**

Run: `pytest test/test_vpcd_quantize_aihub.py -k "fingerprint or autoregressive_records" -v`

Expected: failure because the recipe does not yet expose a stable dataset fingerprint.

- [ ] **Step 3: Implement the minimal calibration-signature helper**

Implementation rules:

- build the AI Hub dataset exactly once
- compute one stable hash over:
  - ordered input names,
  - per-input sample count,
  - sample shapes,
  - sample dtypes,
  - raw array bytes in order
- store the result in `recipe.calibration_stats`
- keep the existing calibration builder as the single source of truth

- [ ] **Step 4: Re-run the focused quantize recipe tests**

Run: `pytest test/test_vpcd_quantize_aihub.py -k "fingerprint or autoregressive_records" -v`

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/quantize/projects/vpcd.py src/quantize/types.py test/test_vpcd_quantize_aihub.py
git commit -m "feat: fingerprint vpcd aihub calibration datasets"
```

### Task 2: Persist And Download The AI Hub Quantized ONNX Artifact

**Files:**

- Modify: `src/tools/aihub_option1_pilots.py`
- Test: `test/test_aihub_option1_pilots.py`

- [ ] **Step 1: Write the failing tests for quantize-run persistence**

Test behavior:

- the VPCD lane writes a `quantize-run` record
- the record includes:
  - quantize job metadata,
  - target model metadata,
  - downloaded quantized file metadata,
  - calibration stats including the fingerprint
- the helper can resolve the downloaded quantized ONNX path later by `RUN_LABEL`

```python
record_path = write_quantize_run_record(...)
payload = json.loads(record_path.read_text(encoding="utf-8"))

assert payload["record_kind"] == "quantize_run"
assert payload["quantized_model"]["path"].endswith(".onnx")
assert payload["calibration"]["dataset_fingerprint"]
```

- [ ] **Step 2: Run the focused pilot tests to confirm they fail**

Run: `pytest test/test_aihub_option1_pilots.py -k "quantize_run or downloaded_quantized" -v`

Expected: failure because no quantize-run record or artifact resolver exists yet.

- [ ] **Step 3: Implement quantize-run record writing and artifact download**

Implementation rules:

- after `submit_quantize_job(...)` completes, download the quantized target model to:
  - `build/aihub/vpcd_option1/model.quantized.<RUN_LABEL>.onnx`
- prefer the quantize job API to download the produced target model directly
- persist:
  - `weights_dtype_name`
  - `activations_dtype_name`
  - `quantize_options`
  - calibration stats and fingerprint
  - downloaded file metadata
- add a resolver helper for the downloaded quantized ONNX path

- [ ] **Step 4: Re-run the focused pilot tests**

Run: `pytest test/test_aihub_option1_pilots.py -k "quantize_run or downloaded_quantized" -v`

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/tools/aihub_option1_pilots.py test/test_aihub_option1_pilots.py
git commit -m "feat: persist and download vpcd quantized artifacts"
```

### Task 3: Add Local Teacher-Forced Diagnostics For The Downloaded Quantized ONNX

**Files:**

- Modify: `src/tools/aihub_option1_hybrid_pipeline.py`
- Modify: `src/tools/aihub_option1_pilots.py`
- Test: `test/test_aihub_option1_hybrid_pipeline.py`

- [ ] **Step 1: Write the failing test for local quantized teacher-forced diagnostics**

Test behavior:

- a new helper runs the existing teacher-forced pattern against a local quantized ONNX session
- the record shows the first divergent step against the FP32 reference

```python
report = run_vpcd_quantized_teacher_forced_diagnostics(
    runtime_config=runtime_config,
    run_label="phase3",
    sample_index=0,
    max_decode_steps=5,
    cpu_model_step_runner=fake_fp32_runner,
    quantized_model_step_runner=fake_quantized_runner,
)

assert report["steps"][1]["quantized_argmax_token_id"] == 4
assert report["steps"][1]["matches_fp32_argmax"] is False
```

- [ ] **Step 2: Run the focused hybrid-pipeline tests to confirm they fail**

Run: `pytest test/test_aihub_option1_hybrid_pipeline.py -k "quantized_teacher_forced" -v`

Expected: failure because the helper does not exist yet.

- [ ] **Step 3: Implement the local quantized diagnostic helper**

Implementation rules:

- reuse as much of the existing teacher-forced path as possible
- the only changing model lane should be:
  - FP32 local reference
  - quantized local comparison session
- the helper must resolve the downloaded quantized ONNX by `RUN_LABEL` when no explicit path override is provided
- write a dedicated record under:
  - `vpcd_quantized_teacher_forced_option1`

- [ ] **Step 4: Re-run the focused hybrid-pipeline tests**

Run: `pytest test/test_aihub_option1_hybrid_pipeline.py -k "quantized_teacher_forced" -v`

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/tools/aihub_option1_hybrid_pipeline.py src/tools/aihub_option1_pilots.py test/test_aihub_option1_hybrid_pipeline.py
git commit -m "feat: add local quantized vpcd teacher-forced diagnostics"
```

### Task 4: Reorder The Notebook Around Quantize-Then-Local-Then-Cloud Diagnosis

**Files:**

- Modify: `On_device_Ai_option1_pilots.ipynb`
- Test: `test/test_option1_notebook_layout.py`

- [ ] **Step 1: Write the failing notebook-layout test**

Test behavior:

- the VPCD compile cell prints the calibration fingerprint and quantized artifact path
- the notebook contains a new `VPCD Quantized Local Teacher-Forced Diagnostics` section
- that section appears before the existing cloud teacher-forced section

```python
markdown_text = "\n".join(_cell_texts(notebook, cell_type="markdown"))
code_text = "\n".join(_cell_texts(notebook, cell_type="code"))

assert "### VPCD Quantized Local Teacher-Forced Diagnostics" in markdown_text
assert "vpcd quantized model path:" in code_text
assert markdown_text.index("### VPCD Quantized Local Teacher-Forced Diagnostics") < markdown_text.index("### VPCD Teacher-Forced Diagnostics")
```

- [ ] **Step 2: Run the focused notebook-layout test to confirm it fails**

Run: `pytest test/test_option1_notebook_layout.py -k "quantized_local_teacher_forced" -v`

Expected: failure because the notebook does not yet expose the quantized-local checkpoint.

- [ ] **Step 3: Update the notebook flow**

Notebook changes:

- in `VPCD Compile Only`:
  - build one local autoregressive calibration dataset
  - print calibration stats and dataset fingerprint
  - submit quantize job
  - download quantized ONNX
  - write the quantize-run record
- add a new VPCD quantized-local diagnostic cell
- keep the existing cloud teacher-forced cell after the local quantized cell
- keep bounded free-run hybrid last

- [ ] **Step 4: Re-run the focused notebook-layout test**

Run: `pytest test/test_option1_notebook_layout.py -k "quantized_local_teacher_forced" -v`

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add On_device_Ai_option1_pilots.ipynb test/test_option1_notebook_layout.py
git commit -m "feat: add quantized-local vpcd notebook diagnostics"
```

### Task 5: Update Workflow Docs And Root-Cause Reporting

**Files:**

- Modify: `docs/workflows/aihub-option1-npu-pilots.md`
- Modify: `docs/workflows/aihub-option1-hybrid-pipeline.md`
- Modify: `docs/plans/active/2026-05-13-vpcd-option1-debug-results.md`

- [ ] **Step 1: Update the workflow docs**

Doc requirements:

- explain the new three-way diagnostic order:
  - FP32 local
  - quantized local
  - compiled cloud
- document the new record paths
- document the attribution rule:
  - quantized local fails => quantize issue
  - quantized local passes and compiled cloud fails => compile / QNN issue

- [ ] **Step 2: Add an execution-results template to the debug-results note**

Required result sections:

- calibration fingerprint used
- quantize job id and downloaded model path
- local quantized teacher-forced outcome
- compiled cloud teacher-forced outcome
- final attribution and next action

- [ ] **Step 3: Commit**

```bash
git add docs/workflows/aihub-option1-npu-pilots.md docs/workflows/aihub-option1-hybrid-pipeline.md docs/plans/active/2026-05-13-vpcd-option1-debug-results.md
git commit -m "docs: add vpcd quantize vs compile triage workflow"
```

### Task 6: Verify End-To-End And Write The Attribution Result

**Files:**

- Verify: `test/test_vpcd_quantize_aihub.py`
- Verify: `test/test_aihub_option1_pilots.py`
- Verify: `test/test_aihub_option1_hybrid_pipeline.py`
- Verify: `test/test_option1_notebook_layout.py`
- Modify: `docs/plans/active/2026-05-13-vpcd-option1-debug-results.md`

- [ ] **Step 1: Run the focused local test slices**

Run:

- `pytest test/test_vpcd_quantize_aihub.py -v`
- `pytest test/test_aihub_option1_pilots.py -k "vpcd" -v`
- `pytest test/test_aihub_option1_hybrid_pipeline.py -k "vpcd" -v`
- `pytest test/test_option1_notebook_layout.py -k "vpcd" -v`

Expected: pass.

- [ ] **Step 2: Run the VPCD notebook cells only**

Run the notebook path needed for:

- AI Hub auth/setup
- VPCD prepare
- VPCD compile-only with quantize artifact download
- VPCD quantized-local teacher-forced diagnostics
- VPCD compiled cloud teacher-forced diagnostics
- VPCD bounded hybrid
- summary

Expected: notebook completes without requiring manual reruns.

- [ ] **Step 3: Attribute the failure source**

Decision rule:

- if local quantized teacher-forced diverges at step `2`, write `quantize` as the failing stage
- if local quantized teacher-forced matches FP32 but compiled cloud diverges, write `compile / QNN execution` as the failing stage
- if both match, continue investigating only free-run behavior

- [ ] **Step 4: If quantize is the failing stage, execute the bounded quantize matrix**

Run the variants in this exact order:

- `A`: `w8a16 + auto + calibration giữ nguyên`
- `B`: `w8a16 + min_max + calibration giữ nguyên`
- `C`: `w8a8 + auto + calibration giữ nguyên`
- `D`: `w8a8 + min_max + calibration giữ nguyên`

Execution rules:

- reuse the same calibration fingerprint for all four variants
- use a new `RUN_LABEL` per variant
- run only local quantized teacher-forced first
- compile is allowed only for the first passing quantized-local variant, if one exists
- if none passes, stop and document that the quantize stage remains unresolved across the bounded matrix

- [ ] **Step 5: Update the debug-results note with real evidence**

Required evidence:

- run label
- calibration fingerprint
- downloaded quantized model metadata
- first divergent step for local quantized
- first divergent step for compiled cloud
- final attribution
- recommended next fix
- matrix outcomes for variants `A/B/C/D` if the quantize branch was exercised

- [ ] **Step 6: Commit**

```bash
git add docs/plans/active/2026-05-13-vpcd-option1-debug-results.md
git commit -m "test: verify vpcd quantize vs compile attribution"
```

## Expected Outcome

At the end of this plan, the team should no longer be guessing whether AI Hub quantize or AI Hub compile caused the punctuation-collapse failure. The notebook and records should provide one reproducible calibration fingerprint, one downloadable quantized ONNX checkpoint, and one explicit attribution path from FP32 local to quantized local to compiled cloud.
