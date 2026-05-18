# VPCD Local QDQ To AI Hub Compile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current `FP32 -> AI Hub quantize -> AI Hub compile` VPCD lane with a `local QDQ -> AI Hub compile` lane, but only after proving the local quantized artifact is in a format AI Hub can compile safely and that it improves correctness over the AI Hub quantize path.

**Architecture:** Keep the fixed-shape VPCD artifact flow, but move quantization ownership local. First, add a strict format-audit stage for the local QDQ model so the repo stops assuming that “QDQ” automatically means “AI Hub compile-ready.” Second, add a compile-probe lane that uploads the local QDQ artifact directly to AI Hub, records the result, and runs the same teacher-forced diagnostics already used for attribution. Only after the local-QDQ compile lane passes those gates should the notebook disable AI Hub quantize by default.

**Tech Stack:** Python 3.11, Jupyter notebook, ONNX, ONNX Runtime QNN quantization tooling, Qualcomm AI Hub Workbench, JSON run records, pytest.

---

## Why This Direction Is Reasonable

The current evidence says baseline `A` already fails in the AI Hub quantize stage, so spending more time on compile jobs built from that quantized output is low leverage.

Official documentation also supports trying a locally quantized source, but with an important constraint:

- Qualcomm AI Hub compile supports ONNX source models and says that if the ONNX source is quantized, its quantization parameters will be respected.
- Qualcomm AI Hub quantization docs also say that the only ONNX quantization format they officially support as compile input is the fake-quant QDQ format produced by AI Hub quantize.
- Qualcomm AI Hub compile docs explicitly support AIMET-quantized model packages as another official path.

That means the local-QDQ direction is valid to investigate, but the current local VPCD QDQ artifact cannot be treated as a drop-in replacement without a compatibility audit first.

## Official Findings That Must Drive The Plan

### Qualcomm AI Hub

- Quantize job output:
  - Qualcomm AI Hub says `submit_quantize_job()` takes unquantized ONNX and produces a quantized ONNX model.
  - It describes that output as ONNX fake-quant format where weights remain floating point and quantization bottlenecks are represented with `QuantizeLinear` / `DequantizeLinear`.
- Compile input expectations:
  - Qualcomm AI Hub compile docs say quantized ONNX models may be compiled and their quantization parameters will be respected.
  - The same quantization page says the AI Hub fake-quant ONNX is the only ONNX quantization format they officially support as input to compile jobs.
- Alternative officially supported quantized source:
  - Qualcomm AI Hub compile docs explicitly support `.aimet` packages containing ONNX plus `.encodings`.
- ONNX packaging rule:
  - If an ONNX model uses external weights, AI Hub expects a directory ending in `.onnx` containing exactly one `.onnx` file and exactly one `.data` file.

### ONNX Runtime / ONNX

- ONNX Runtime QNN EP supports QDQ models and mixed precision boundaries.
- ONNX Runtime quantization source says `UseQDQContribOps=True` forces `com.microsoft` Q/DQ operators and may be needed for features not standardized in ONNX, including 16-bit quantization types.
- ONNX operator docs show `uint16` / `int16` support in main-domain `QuantizeLinear` is standardized in ONNX opset `21+`.

## Current Repo Assessment

### What The Local VPCD Quantizer Is Good At

- `src/quantize/qnn.py`
  - uses ONNX Runtime QNN preprocessing plus `get_qnn_qdq_config(...)`
  - this is aligned with generating ORT/QNN-friendly QDQ artifacts locally
- `src/quantize/projects/vpcd.py`
  - already owns VPCD preset logic, calibration generation, and fixed-shape bundling
- the bundle manifest already treats local VPCD as a QDQ model candidate:
  - [bundle_manifest.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/model_bundle/vpcd/qnn_fixed_1024x128/bundle_manifest.json)

### Why The Current Local QDQ Artifact Is Not Yet AI Hub-Ready By Default

Observed from the current bundled model:

- artifact:
  - [model.mobile.onnx](/D:/DS-AI/BKMeeting-Research/python-model-test/build/model_bundle/vpcd/qnn_fixed_1024x128/model.mobile.onnx)
- opsets:
  - `main=17`
  - `com.microsoft=1`
- QDQ shape:
  - `842` Q/DQ nodes in `com.microsoft`
  - `329` `UINT16` initializers
  - `368` `UINT8` initializers
- weight representation:
  - many quantized weights appear as quantized initializers feeding `DequantizeLinear`
  - this is not the same representation AI Hub documents for its own fake-quant ONNX output

### Current Gap In Our Helper Layer

- `prepare_vpcd_option1_source_model(..., strategy="direct_qdq_sanitized")`
  - currently only strips `com.microsoft` from Q/DQ nodes
  - it does **not**:
    - upgrade main-domain opset to `21+`
    - convert weight representation to AI Hub fake-quant style
    - validate whether the resulting graph is still semantically valid for `uint16`
    - produce an explicit compatibility report

Conclusion:

- the current local QDQ module is suitable for local ORT/QNN usage
- it is **not yet** sufficient as an AI Hub compile input compatibility layer

## Scope And Boundaries

- This plan only covers the VPCD Option 1 lane.
- Do not modify Zipformer.
- Do not assume we can keep the exact current local QDQ graph unchanged.
- Do not assume we must force local QDQ into AI Hub if the official compatibility signals remain weak.
- The plan ends with one of two supported outcomes:
  - local QDQ compile lane becomes the default and AI Hub quantize is disabled
  - local QDQ compile lane is rejected by evidence, and the repo keeps AI Hub quantize or pivots to AIMET packaging instead

## File Structure

**Files:**

- Modify: `src/quantize/qnn.py`
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

- `src/quantize/qnn.py`
  - local ORT/QNN quantization implementation
  - must surface enough metadata about the produced graph for AI Hub compatibility decisions
- `src/quantize/projects/vpcd.py`
  - VPCD-specific quantization presets and output artifact contract
  - must define which local QDQ artifact is the compile candidate
- `src/quantize/types.py`
  - expanded report/recipe types for compatibility decisions
- `src/tools/aihub_option1_pilots.py`
  - compile packaging, source-strategy selection, record writing, and AI Hub upload behavior
- `src/tools/aihub_option1_hybrid_pipeline.py`
  - diagnostics over the compiled result
- notebook
  - operator-facing lane selection and execution order
- docs
  - explain the new local-QDQ-first decision tree and its risks

## Decision Gates

The implementation must stop at the first failing gate and record why.

### Gate 1: Local QDQ Compatibility Audit

The compile candidate must produce a report that clearly answers:

- does it still contain `com.microsoft` Q/DQ?
- does it rely on main-domain `uint16` / `int16` QDQ while staying on opset `<21`?
- are weights stored as quantized initializers plus `DQ`, or as fake-quant float weights plus `Q -> DQ`?
- does it satisfy AI Hub ONNX packaging rules?

If the answer still indicates “ORT/QNN-specific only,” do **not** silently switch the notebook default.

### Gate 2: AI Hub Compile Acceptance

The local QDQ artifact must compile on AI Hub to the target runtime we already use for VPCD:

- precompiled QNN ONNX

If compile fails, record the failure and stop the default-switch work.

### Gate 3: Teacher-Forced Correctness

The compiled local-QDQ lane must be tested with the same teacher-forced diagnostic we already trust.

Success rule:

- local-QDQ compiled lane must no longer diverge at teacher-forced step `2`

If it still diverges at step `2`, the local-QDQ switch is not justified.

## Detailed Tasks

### Task 1: Add A Strict Local-QDQ Compatibility Report

**Files:**

- Modify: `src/quantize/qnn.py`
- Modify: `src/quantize/projects/vpcd.py`
- Modify: `src/quantize/types.py`
- Test: `test/test_vpcd_quantize_aihub.py`

- [ ] **Step 1: Write the failing tests for local-QDQ compatibility reporting**

Test behavior:

- local VPCD quantization must expose a report that includes:
  - opset imports
  - Q/DQ domains
  - whether `uint16` / `int16` QDQ is present
  - whether weight initializers are already quantized
  - a compatibility verdict that is conservative

```python
report = inspect_vpcd_qdq_compile_candidate(model_path)

assert report["opsets"]["main"] == 17
assert report["ms_qdq_node_count"] > 0
assert report["uses_uint16_qdq"] is True
assert report["uses_quantized_weight_initializers"] is True
assert report["aihub_compile_readiness"] in {"unsafe", "experimental", "ready"}
```

- [ ] **Step 2: Run the focused test to confirm it fails**

Run: `pytest test/test_vpcd_quantize_aihub.py -k "compile_candidate or compatibility_report" -v`

Expected: failure because no strict report exists yet.

- [ ] **Step 3: Implement the minimal report helper**

Implementation rules:

- do not mutate the graph in this task
- report only what the current graph actually is
- make the readiness verdict fail-closed:
  - `unsafe`
  - `experimental`
  - `ready`

- [ ] **Step 4: Re-run the focused test**

Run: `pytest test/test_vpcd_quantize_aihub.py -k "compile_candidate or compatibility_report" -v`

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/quantize/qnn.py src/quantize/projects/vpcd.py src/quantize/types.py test/test_vpcd_quantize_aihub.py
git commit -m "feat: add local qdq compatibility report for vpcd"
```

### Task 2: Replace Blind QDQ Sanitization With An Explicit Compile Candidate Strategy

**Files:**

- Modify: `src/tools/aihub_option1_pilots.py`
- Test: `test/test_aihub_option1_pilots.py`

- [ ] **Step 1: Write the failing tests for source-strategy selection**

Test behavior:

- the helper must stop treating `direct_qdq_sanitized` as automatically safe
- the helper must package a local-QDQ compile candidate explicitly
- the helper must preserve enough metadata to know whether the upload is:
  - `as_is`
  - `domain_rewritten`
  - `standardized_qdq`
  - `aimet_fallback`

```python
prepared = prepare_vpcd_option1_source_model(source, strategy="local_qdq_compile_candidate")

assert prepared.report["aihub_compile_readiness"] in {"experimental", "ready"}
assert prepared.source_kind == "local_qdq"
assert prepared.packaging_kind in {"onnx_file", "onnx_dir"}
```

- [ ] **Step 2: Run the focused pilot tests to confirm they fail**

Run: `pytest test/test_aihub_option1_pilots.py -k "local_qdq_source_strategy or qdq_packaging" -v`

Expected: failure because the strategy does not exist yet.

- [ ] **Step 3: Implement the new strategy contract**

Implementation rules:

- add a new explicit strategy name:
  - `local_qdq_compile_candidate`
- deprecate blind reliance on `direct_qdq_sanitized`
- package the model according to AI Hub ONNX upload rules
- preserve a machine-readable report describing what transformations, if any, were applied
- do not yet disable the FP32 lane

- [ ] **Step 4: Re-run the focused pilot tests**

Run: `pytest test/test_aihub_option1_pilots.py -k "local_qdq_source_strategy or qdq_packaging" -v`

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/tools/aihub_option1_pilots.py test/test_aihub_option1_pilots.py
git commit -m "feat: add explicit local qdq compile candidate strategy"
```

### Task 3: Add Records For Local-QDQ Prepare And Compile-Probe

**Files:**

- Modify: `src/tools/aihub_option1_pilots.py`
- Test: `test/test_aihub_option1_pilots.py`

- [ ] **Step 1: Write the failing tests for local-QDQ records**

Test behavior:

- preparing a local QDQ compile candidate must write a record with:
  - source strategy
  - compatibility report
  - packaging path
- compile-only must write a record showing that AI Hub quantize was skipped

```python
payload = json.loads(record_path.read_text(encoding="utf-8"))

assert payload["record_kind"] == "prepared_artifact"
assert payload["source_strategy"] == "local_qdq_compile_candidate"
assert payload["compatibility"]["aihub_compile_readiness"]
assert payload["quantize_stage"] == "disabled"
```

- [ ] **Step 2: Run the focused test to confirm it fails**

Run: `pytest test/test_aihub_option1_pilots.py -k "local_qdq_record or quantize_disabled" -v`

Expected: failure because the records do not carry this information yet.

- [ ] **Step 3: Implement record extensions**

Implementation rules:

- do not overload the old quantize-run record for this path
- clearly encode that the local-QDQ lane skipped AI Hub quantize
- keep record names stable by `RUN_LABEL`

- [ ] **Step 4: Re-run the focused test**

Run: `pytest test/test_aihub_option1_pilots.py -k "local_qdq_record or quantize_disabled" -v`

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/tools/aihub_option1_pilots.py test/test_aihub_option1_pilots.py
git commit -m "feat: record local qdq prepare and compile probe metadata"
```

### Task 4: Add Notebook Support For Local-QDQ Compile And Disable AI Hub Quantize Behind A Strategy Flag

**Files:**

- Modify: `On_device_Ai_option1_pilots.ipynb`
- Test: `test/test_option1_notebook_layout.py`

- [ ] **Step 1: Write the failing notebook-layout tests**

Test behavior:

- notebook config must expose a VPCD source strategy flag
- compile-only must skip `submit_quantize_job()` when local-QDQ strategy is selected
- notebook text must explain that AI Hub quantize is disabled for this lane

```python
assert "VPCD_SOURCE_STRATEGY" in code_text
assert "local_qdq_compile_candidate" in code_text
assert "Skipping AI Hub quantize for local QDQ lane" in code_text
```

- [ ] **Step 2: Run the focused notebook test to confirm it fails**

Run: `pytest test/test_option1_notebook_layout.py -k "local_qdq" -v`

Expected: failure because the notebook still assumes AI Hub quantize.

- [ ] **Step 3: Update the notebook**

Notebook changes:

- add:
  - `VPCD_SOURCE_STRATEGY`
- support at least:
  - `prefer_fp32_fixed`
  - `local_qdq_compile_candidate`
- when local-QDQ is selected:
  - prepare local QDQ artifact
  - skip `submit_quantize_job()`
  - submit compile directly on the packaged local QDQ model
  - keep teacher-forced and hybrid cells unchanged downstream

- [ ] **Step 4: Re-run the focused notebook test**

Run: `pytest test/test_option1_notebook_layout.py -k "local_qdq" -v`

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add On_device_Ai_option1_pilots.ipynb test/test_option1_notebook_layout.py
git commit -m "feat: add notebook support for local qdq compile lane"
```

### Task 5: Reuse Teacher-Forced Diagnostics To Gate The New Compile Lane

**Files:**

- Modify: `src/tools/aihub_option1_hybrid_pipeline.py`
- Test: `test/test_aihub_option1_hybrid_pipeline.py`

- [ ] **Step 1: Write the failing test for local-QDQ compiled attribution**

Test behavior:

- the same teacher-forced diagnostic flow must work when the compiled model came from the local-QDQ lane
- attribution output must make clear that AI Hub quantize was skipped

```python
report = run_vpcd_teacher_forced_diagnostics(..., compile_pilot_name="vpcd_option1_local_qdq")

assert report["target_reference"].target_model_id
assert report["results"][0]["reference_stats"]["source_strategy"] == "local_qdq_compile_candidate"
```

- [ ] **Step 2: Run the focused hybrid test to confirm it fails**

Run: `pytest test/test_aihub_option1_hybrid_pipeline.py -k "local_qdq_compiled" -v`

Expected: failure because the metadata path is incomplete.

- [ ] **Step 3: Implement the minimal metadata plumbing**

Implementation rules:

- do not fork a second diagnostic framework
- reuse existing teacher-forced and hybrid record writers
- add only the source-strategy metadata needed to explain the lane

- [ ] **Step 4: Re-run the focused hybrid test**

Run: `pytest test/test_aihub_option1_hybrid_pipeline.py -k "local_qdq_compiled" -v`

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/tools/aihub_option1_hybrid_pipeline.py test/test_aihub_option1_hybrid_pipeline.py
git commit -m "feat: gate local qdq compile lane with teacher forced diagnostics"
```

### Task 6: Execute The Compile Probe And Decide Whether To Switch Defaults

**Files:**

- Verify: `test/test_vpcd_quantize_aihub.py`
- Verify: `test/test_aihub_option1_pilots.py`
- Verify: `test/test_aihub_option1_hybrid_pipeline.py`
- Verify: `test/test_option1_notebook_layout.py`
- Modify: `docs/workflows/aihub-option1-npu-pilots.md`
- Modify: `docs/workflows/aihub-option1-hybrid-pipeline.md`
- Modify: `docs/plans/active/2026-05-13-vpcd-option1-debug-results.md`

- [ ] **Step 1: Run the focused local tests**

Run:

- `pytest test/test_vpcd_quantize_aihub.py -v`
- `pytest test/test_aihub_option1_pilots.py -k "vpcd" -v`
- `pytest test/test_aihub_option1_hybrid_pipeline.py -k "vpcd" -v`
- `pytest test/test_option1_notebook_layout.py -k "vpcd" -v`

Expected: pass.

- [ ] **Step 2: Run only the VPCD notebook cells with `VPCD_SOURCE_STRATEGY=local_qdq_compile_candidate`**

Required notebook path:

- auth/setup
- VPCD prepare
- VPCD compile-only without AI Hub quantize
- resolve compiled target
- teacher-forced diagnostics
- bounded hybrid
- summary

Expected: notebook completes without touching `submit_quantize_job()`.

- [ ] **Step 3: Apply the switch decision**

Decision rule:

- if compile fails:
  - keep AI Hub quantize as the default
  - record the local-QDQ lane as unsupported or inconclusive
- if compile succeeds but teacher-forced still diverges at step `2`:
  - keep AI Hub quantize as the default
  - record that local-QDQ compile did not fix correctness
- if compile succeeds and teacher-forced step `2` is fixed:
  - switch notebook default to local-QDQ lane
  - disable AI Hub quantize for VPCD by default

- [ ] **Step 4: Update docs with the real result**

Required doc updates:

- local-QDQ compatibility findings
- whether the current local artifact was accepted by AI Hub compile
- whether `direct_qdq_sanitized` was retained, replaced, or removed
- whether AI Hub quantize is now disabled by default
- if local QDQ was rejected, record the official fallback:
  - keep AI Hub quantize
  - or prepare an AIMET export lane

- [ ] **Step 5: Commit**

```bash
git add docs/workflows/aihub-option1-npu-pilots.md docs/workflows/aihub-option1-hybrid-pipeline.md docs/plans/active/2026-05-13-vpcd-option1-debug-results.md
git commit -m "docs: record local qdq ai hub compile decision for vpcd"
```

## Expected Outcome

At the end of this plan, we should stop guessing about whether “local QDQ” is a viable replacement for AI Hub quantize in VPCD. We will either have:

- a working default lane:
  - `local QDQ -> AI Hub compile -> teacher-forced pass`

or a documented rejection with clear reasons:

- current local QDQ remains ORT/QNN-specific and should not replace AI Hub quantize yet
- the next official-compatible route should then be AIMET packaging rather than more blind QDQ rewriting

## Official Sources To Keep Open While Executing

- [Qualcomm AI Hub Quantization docs](https://workbench.aihub.qualcomm.com/docs/hub/quantize_examples.html)
- [Qualcomm AI Hub Compile docs](https://workbench.aihub.qualcomm.com/docs/hub/compile_examples.html)
- [Qualcomm AI Hub `submit_compile_job()` API](https://workbench.aihub.qualcomm.com/docs/hub/generated/qai_hub.submit_compile_job.html)
- [Qualcomm AI Hub `submit_quantize_job()` API](https://workbench.aihub.qualcomm.com/docs/hub/generated/qai_hub.submit_quantize_job.html)
- [Qualcomm AI Hub FAQ](https://workbench.aihub.qualcomm.com/docs/hub/faq.html)
- [ONNX Runtime quantization docs](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [ONNX Runtime QNN Execution Provider docs](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)
- [ONNX Runtime quantize source](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/quantize.py)
- [ONNX QuantizeLinear operator docs](https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html)
