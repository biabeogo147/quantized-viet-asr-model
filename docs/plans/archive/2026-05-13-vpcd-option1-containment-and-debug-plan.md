# VPCD Option 1 Containment And Debug Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the VPCD Option 1 lane from a multi-hour, low-signal hybrid rerun into a bounded debug loop that finishes quickly enough to isolate the root cause of runaway punctuation decoding.

**Architecture:** Split the problem into two independent concerns: cost containment and root-cause diagnosis. First, cap the VPCD hybrid decode loop to a small, explicit step budget so a rerun cannot fan out into dozens of cloud jobs per sample. Second, preserve enough per-step evidence to decide whether the repeated punctuation output comes from the compiled AI Hub target or from autoregressive free-run drift after several steps. This plan intentionally narrows the compile path to a single lane: `FP32 fixed-shape -> AI Hub quantize -> AI Hub compile -> AI Hub inference`.

**Tech Stack:** Python 3.10+, Jupyter notebook (`.ipynb`), NumPy, ONNX, ONNX Runtime, Qualcomm AI Hub (`qai-hub`), existing `python-model-test` helper modules, JSON run records, pytest.

---

## Scope And Boundaries

- This plan is only for the VPCD Option 1 hybrid lane.
- Do not expand scope to:
  - Zipformer transcript quality work,
  - Android integration,
  - Phase 4 gate tuning,
  - Phase 5 contract packaging,
  - broad quantization experiments unrelated to the current runaway decode failure.
- Keep the notebook as the human-facing control surface:
  - `On_device_Ai_option1_pilots.ipynb`
- Keep helper logic in Python modules under `src/tools/`.
- Prefer the smallest useful code changes that reduce runtime and increase observability.
- Treat the local QDQ candidate as out of scope for this debug slice.
- Do not add any new source-selection branching for VPCD in this plan.

## Current Evidence Snapshot

- `build/aihub/records/vpcd_hybrid_option1/hybrid-run-20260513-1am.json` shows:
  - `sample_count = 2`
  - `decode_steps = 48` and `55`
  - repeated `generated_ids` dominated by token ids `4` and `382`
  - output text collapsed into repeated punctuation such as `",,,,,,...,..."`
  - average cloud inference time over three hours per sample
- `build/model_bundle/vpcd/qnn_fixed_1024x128/bundle_manifest.json` shows:
  - `max_decode_length = 128`
  - fixed decoder shape `1 x 128`
- `build/aihub/records/vpcd_option1/prepared-artifact-20260513-1am.json` shows the notebook prepared a fixed-shape FP32 upload artifact near `1.75 GB`, which strongly suggests the current compile lane is paying for:
  - large upload volume,
  - AI Hub quantize work,
  - AI Hub compile work
- The current investigation goal is not to compare source lanes.
- To avoid ambiguity, this plan must use only:
  - a fixed-shape FP32 prepared source model,
  - autoregressive calibration built locally,
  - AI Hub quantize,
  - AI Hub compile,
  - AI Hub inference.

## Implementation Status Snapshot

- `Task 1` code path is implemented:
  - `run_vpcd_hybrid_evaluation(...)` accepts `max_decode_steps`
  - the helper clamps and records `decode_step_limit`
- `Task 2` notebook wiring is implemented:
  - `On_device_Ai_option1_pilots.ipynb` exposes `VPCD_HYBRID_MAX_STEPS = 5`
  - the VPCD hybrid cell passes `max_decode_steps=VPCD_HYBRID_MAX_STEPS`
- Focused verification completed:
  - `test_vpcd_hybrid_runner_restores_expected_text`
  - `test_vpcd_hybrid_runner_passes_decode_step_limit_to_bundle_runtime`
  - `test_pilot_notebook_limits_vpcd_hybrid_decode_steps`
  - notebook JSON sanity
- Remaining work starts at:
  - `Task 3` FP32-only source-lane enforcement
  - `Task 4` teacher-forced diagnosis

## Success Criteria

- A VPCD hybrid rerun with the shared notebook can be bounded to at most `5` decode steps per sample.
- A two-sample debug rerun submits no more than `10` compiled cloud inference jobs.
- The hybrid run record preserves enough information to answer:
  - did the compiled target diverge immediately,
  - or only after several decode steps,
  - and which token choices caused the divergence.
- The compile path is documented and tested as a single explicit lane:
  - `FP32 fixed-shape -> AI Hub quantize -> AI Hub compile`
- The notebook exposes a teacher-forced diagnostic path that can show whether divergence starts in the first few decode steps before the operator pays for a long free-run hybrid decode.
- Focused local tests and notebook layout checks pass.

## File Structure

**Files:**

- Modify: `src/tools/aihub_option1_hybrid_pipeline.py`
- Modify: `src/tools/aihub_option1_pilots.py`
- Modify: `On_device_Ai_option1_pilots.ipynb`
- Modify: `docs/workflows/aihub-option1-hybrid-pipeline.md`
- Modify: `docs/workflows/aihub-option1-npu-pilots.md`
- Modify: `test/test_aihub_option1_hybrid_pipeline.py`
- Modify: `test/test_aihub_option1_pilots.py`
- Modify: `test/test_option1_notebook_layout.py`

### File Responsibilities

- `src/tools/aihub_option1_hybrid_pipeline.py`
  - owns the VPCD hybrid runner
  - must accept an explicit decode-step cap
  - must surface bounded step counts and step-level metadata into the hybrid run record
- `src/tools/aihub_option1_pilots.py`
  - owns VPCD source-model preparation policy
  - must force the VPCD debug lane onto a fixed-shape FP32 prepared source model
  - must not silently switch to a local QDQ source for this plan
  - must surface the quantize recipe and calibration stats clearly enough for debug records
- `On_device_Ai_option1_pilots.ipynb`
  - remains the operator entrypoint
  - must expose `VPCD_HYBRID_MAX_STEPS = 5`
  - must pass the cap into the VPCD hybrid helper
  - must expose a teacher-forced debug section before free-run hybrid
- workflow docs
  - explain that this debug plan uses one source lane only: `FP32 -> AI Hub quantize -> AI Hub compile`
  - explain when to run teacher-forced diagnosis versus free-run hybrid
  - explain the expected debug evidence after a bounded rerun
- tests
  - lock the new notebook knobs, helper behavior, and FP32-only source-path rules

## Detailed Tasks

### Task 1: Add A Hard Decode-Step Guardrail To The Hybrid VPCD Lane

**Files:**

- Modify: `src/tools/aihub_option1_hybrid_pipeline.py`
- Test: `test/test_aihub_option1_hybrid_pipeline.py`

- [x] **Step 1: Write the failing test for the decode-step cap**

Test behavior:

- `run_vpcd_hybrid_evaluation(...)` accepts `max_decode_steps`
- the helper clamps the runtime step budget to the smaller of:
  - requested `max_decode_steps`
  - bundle `decoder_sequence`
- the bounded value is passed into `BundleOnnxRuntime.restore_with_model_step(...)`

```python
report = run_vpcd_hybrid_evaluation(
    runtime_config=runtime_config,
    run_label="phase3",
    max_samples=1,
    max_decode_steps=5,
    bundle_runtime=FakeRuntime(),
)

assert seen["max_length"] == 5
assert report["results"][0]["decode_steps"] == 5
assert report["results"][0]["decode_step_limit"] == 5
```

- [x] **Step 2: Run the focused test to confirm it fails**

Run: `pytest test/test_aihub_option1_hybrid_pipeline.py -k vpcd_hybrid -v`

Expected: `TypeError` or assertion failure because `max_decode_steps` is not yet part of the helper contract.

- [x] **Step 3: Implement the minimal runner change**

Implementation rules:

- add `max_decode_steps: int | None = None` to `run_vpcd_hybrid_evaluation(...)`
- compute one normalized `decode_step_limit`
- pass that value to `restore_with_model_step(..., max_length=decode_step_limit)`
- persist `decode_step_limit` into the hybrid run result so the record explains why a run stopped

- [x] **Step 4: Re-run the focused test to verify it passes**

Run: `pytest test/test_aihub_option1_hybrid_pipeline.py -k vpcd_hybrid -v`

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/tools/aihub_option1_hybrid_pipeline.py test/test_aihub_option1_hybrid_pipeline.py
git commit -m "feat: cap vpcd hybrid decode steps"
```

### Task 2: Expose The Guardrail In The Shared Notebook

**Files:**

- Modify: `On_device_Ai_option1_pilots.ipynb`
- Test: `test/test_option1_notebook_layout.py`

- [x] **Step 1: Write the failing notebook-layout test**

Test behavior:

- the notebook declares `VPCD_HYBRID_MAX_STEPS = 5`
- the VPCD hybrid cell passes `max_decode_steps=VPCD_HYBRID_MAX_STEPS`

```python
code_text = "\n".join(_cell_texts(notebook, cell_type="code"))
assert "VPCD_HYBRID_MAX_STEPS = 5" in code_text
assert "max_decode_steps=VPCD_HYBRID_MAX_STEPS" in code_text
```

- [x] **Step 2: Run the focused notebook-layout test to confirm it fails**

Run: `pytest test/test_option1_notebook_layout.py -k limits_vpcd_hybrid_decode_steps -v`

Expected: failure because the notebook currently only caps `VPCD_HYBRID_MAX_SAMPLES`.

- [x] **Step 3: Update the notebook config and hybrid cell**

Notebook changes:

- add `VPCD_HYBRID_MAX_STEPS = 5` near the existing VPCD sample/calibration knobs
- print the configured step cap in the summary/config cell
- pass the cap into the VPCD hybrid helper call

- [x] **Step 4: Re-run the focused notebook-layout test**

Run: `pytest test/test_option1_notebook_layout.py -k limits_vpcd_hybrid_decode_steps -v`

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add On_device_Ai_option1_pilots.ipynb test/test_option1_notebook_layout.py
git commit -m "feat: add notebook vpcd hybrid step cap"
```

### Task 3: Force The VPCD Debug Lane To Use FP32 Prepare -> AI Hub Quantize

**Files:**

- Modify: `src/tools/aihub_option1_pilots.py`
- Test: `test/test_aihub_option1_pilots.py`

- [ ] **Step 1: Write the failing test for explicit FP32-only source selection**

Test behavior:

- the default preparation path for this plan always resolves to the fixed-shape FP32 source when that source exists
- the helper must report `is_quantized_source is False`
- the local QDQ bundle must not become the selected upload artifact for this plan

```python
prepared_model_path, is_quantized_source = prepare_vpcd_option1_source_model(
    source,
    output_path=repo_root / "build" / "aihub" / "vpcd_option1" / "model.option1.onnx",
)

assert is_quantized_source is False
assert prepared_model_path.name == "model.option1.onnx"
```

- [ ] **Step 2: Run the focused VPCD source-prep tests to confirm the new case fails**

Run: `pytest test/test_aihub_option1_pilots.py -k "prepare_vpcd_option1_source_model" -v`

Expected: failure in the new FP32-only case if the helper still allows a quantized-source default or keeps misleading strategy semantics.

- [ ] **Step 3: Implement the narrow source-selection change**

Implementation rules:

- default strategy:
  - `prefer_fp32_fixed` for this plan
- do not expose local-QDQ-first behavior in the shared notebook path
- keep any direct-QDQ experimentation out of this plan and out of the default notebook lane
- surface enough metadata or documentation so the operator can tell that the upload artifact is:
  - fixed-shape FP32 prepared locally,
  - then AI Hub-quantized later

- [ ] **Step 4: Re-run the focused source-prep tests**

Run: `pytest test/test_aihub_option1_pilots.py -k "prepare_vpcd_option1_source_model" -v`

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/tools/aihub_option1_pilots.py test/test_aihub_option1_pilots.py
git commit -m "feat: force fp32 vpcd source for debug compiles"
```

### Task 4: Add Teacher-Forced Step Diagnostics Before Free-Run Hybrid

**Files:**

- Modify: `src/tools/aihub_option1_hybrid_pipeline.py`
- Modify if needed: `src/tools/aihub_option1_pilots.py`
- Modify: `On_device_Ai_option1_pilots.ipynb`
- Test: `test/test_aihub_option1_hybrid_pipeline.py`

- [ ] **Step 1: Write the failing test for teacher-forced bounded diagnostic output**

Test behavior:

- a teacher-forced helper accepts:
  - one sample,
  - a small step cap,
  - the compiled target
- each step uses the gold decoder prefix instead of the previously generated token stream
- each step stores enough metadata to compare CPU-vs-cloud next-token behavior
- at minimum store:
  - `step_index`
  - `decoder_prefix_ids`
  - `cpu_top_tokens`
  - `cloud_top_tokens`
  - `cpu_argmax_token_id`
  - `cloud_argmax_token_id`
  - `job_id`

```python
result = report["steps"][0]
assert result["step_index"] == 1
assert result["cpu_argmax_token_id"] is not None
assert result["cloud_argmax_token_id"] is not None
assert "cloud_top_tokens" in result
```

- [ ] **Step 2: Run the focused test to confirm it fails or is incomplete**

Run: `pytest test/test_aihub_option1_hybrid_pipeline.py -k vpcd_teacher_forced -v`

Expected: failure or missing-field assertion because the teacher-forced diagnostic path does not exist yet.

- [ ] **Step 3: Implement the minimal diagnostics**

Recommended implementation:

- keep the record schema flat and reviewable
- avoid dumping full logits tensors into JSON
- store only:
  - `active_index`
  - `top_tokens`
  - `token_id`
  - `score`
- add a notebook section that runs teacher-forced debug before the free-run hybrid cell
- keep free-run hybrid available, but not as the first diagnostic step

- [ ] **Step 4: Re-run the focused test**

Run: `pytest test/test_aihub_option1_hybrid_pipeline.py -k vpcd_teacher_forced -v`

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/tools/aihub_option1_hybrid_pipeline.py On_device_Ai_option1_pilots.ipynb test/test_aihub_option1_hybrid_pipeline.py
git commit -m "feat: add teacher forced vpcd diagnostics"
```

### Task 5: Document The FP32-Only Debug Lane And The Diagnosis Workflow

**Files:**

- Modify: `docs/workflows/aihub-option1-npu-pilots.md`
- Modify: `docs/workflows/aihub-option1-hybrid-pipeline.md`

- [ ] **Step 1: Document the single compile lane in the NPU pilot workflow**

Add a short operator-facing section that fixes the lane definition to:

- `debug lane`
  - freeze or prepare fixed-shape FP32 locally
  - build autoregressive calibration data locally
  - submit AI Hub quantize
  - submit AI Hub compile
  - submit AI Hub inference
- Add an explicit note that local QDQ experimentation is intentionally excluded from this plan to keep root-cause attribution clean.

- [ ] **Step 2: Document the bounded hybrid rerun in the hybrid workflow**

Document the recommended knobs for the failure at hand:

- `VPCD_HYBRID_MAX_SAMPLES = 2` or smaller
- `VPCD_HYBRID_MAX_STEPS = 5`
- run teacher-forced debug before free-run hybrid
- preserve the resulting `hybrid-run-<RUN_LABEL>.json`

- [ ] **Step 3: Document the expected decision tree after the rerun**

Spell out:

- if bounded output already loops on punctuation:
  - inspect teacher-forced step divergence first
  - if divergence starts immediately, suspect `AI Hub quantize -> compile`
- if bounded output is initially reasonable and degrades only with longer runs:
  - investigate stopping/EOS behavior next

- [ ] **Step 4: Commit**

```bash
git add docs/workflows/aihub-option1-npu-pilots.md docs/workflows/aihub-option1-hybrid-pipeline.md
git commit -m "docs: document vpcd containment and debug lanes"
```

### Task 6: Run The Focused Verification Slice

**Files:**

- Test: `test/test_aihub_option1_pilots.py`
- Test: `test/test_aihub_option1_hybrid_pipeline.py`
- Test: `test/test_option1_notebook_layout.py`

- [ ] **Step 1: Run the targeted verification suite**

Run:

```bash
pytest test/test_aihub_option1_pilots.py test/test_aihub_option1_hybrid_pipeline.py test/test_option1_notebook_layout.py -v
```

Expected:

- all VPCD source-prep tests pass
- all VPCD hybrid helper tests pass
- notebook-layout checks pass

- [ ] **Step 2: Run notebook JSON sanity**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
json.loads(Path("On_device_Ai_option1_pilots.ipynb").read_text(encoding="utf-8"))
print("ok")
PY
```

Expected: prints `ok`.

- [ ] **Step 3: Re-read the hybrid and compile workflow docs**

Expected:

- the FP32-only debug lane is clearly documented
- teacher-forced debug is documented before free-run hybrid
- notebook knob names match the docs exactly

- [ ] **Step 4: Commit**

```bash
git add test/test_aihub_option1_pilots.py test/test_aihub_option1_hybrid_pipeline.py test/test_option1_notebook_layout.py On_device_Ai_option1_pilots.ipynb docs/workflows/aihub-option1-*.md
git commit -m "test: verify vpcd containment debug slice"
```

## Acceptance Checklist

- [x] The notebook exposes `VPCD_HYBRID_MAX_STEPS = 5`.
- [x] The VPCD hybrid helper accepts and records `max_decode_steps`.
- [ ] A bounded two-sample rerun can no longer spawn dozens of cloud jobs.
- [ ] The default debug compile path for this plan always uses the FP32 prepared source model and AI Hub quantize.
- [ ] The notebook exposes and documents a teacher-forced debug path before free-run hybrid.
- [ ] Workflow docs explain the FP32-only debug lane and the teacher-forced decision flow.
- [ ] Focused tests and notebook JSON validation pass.

## Recommended Execution Order

1. Implement `Task 1` first so the runaway rerun cost is capped in code.
2. Implement `Task 2` immediately after so the notebook actually uses the cap.
3. Implement `Task 3` before any further live compile so the source lane is unambiguous.
4. Implement `Task 4` before the next real bounded rerun so the resulting record is diagnostic, not just shorter.
5. Finish with `Task 5` and `Task 6` so the workflow and tests remain aligned.

## Notes For The Engineer Executing This Plan

- The primary goal is not to make VPCD correct in one pass.
- The first win is to reduce one VPCD debug cycle from hours to minutes.
- Resist the temptation to widen scope into general VPCD quality fixes before the bounded rerun proves where the divergence starts.
- If the first bounded rerun still emits repeated punctuation in the first `3-5` steps, stop extending decode length and treat the compile/source lane as the next root-cause target.
- Be precise in language:
  - this plan is intentionally **not** comparing source lanes
  - this plan assumes the only supported debug lane is `FP32 -> AI Hub quantize -> AI Hub compile`
  - if teacher-forced divergence appears in the first few steps, treat the quantize/compile lane as the primary suspect before investigating longer free-run behavior
