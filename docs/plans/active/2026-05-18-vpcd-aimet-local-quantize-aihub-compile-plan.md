# VPCD AIMET Local Quantize To AI Hub Compile Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new official local quantization lane for VPCD based on `AIMET -> .aimet package -> AI Hub compile`, and use that lane as the primary candidate for replacing the current `FP32 -> AI Hub quantize -> AI Hub compile` flow.

**Primary decision:** The new local quantization lane should be **AIMET-based**, not a continuation of the current ORT/QNN-specific local QDQ lane.

**Important scoping rule:** This phase is about **adding and validating** the new AIMET lane. Do **not** remove the older quantize lanes in this phase. Old lanes may become deprecated after the AIMET lane passes, but actual code cleanup must happen in a later phase called something like `remove redundant code`.

**Tech Stack:** Python 3.11, Jupyter notebook, ONNX, AIMET ONNX QuantizationSimModel, Qualcomm AI Hub Workbench, JSON run records, pytest.

**Status on 2026-05-18:** Implemented through `Task 6`.

- Docker-backed AIMET export is working and reusable
- `.aimet` package compile on AI Hub is proven
- current switch decision:
  - keep existing lanes as default
  - do not switch to AIMET yet
- reason:
  - the default official variant `w8a8 + min_max` still fails local teacher-forced step `2`
  - compiled cloud reproduces that same divergence
- detailed evidence lives in:
  - [2026-05-13-vpcd-option1-debug-results.md](/D:/DS-AI/BKMeeting-Research/python-model-test/docs/plans/active/2026-05-13-vpcd-option1-debug-results.md)

**Status on 2026-05-19:** Keep this plan active for a new parity track.

- working hypothesis:
  - the current AIMET lane is quantizing too much of the decoder stack compared with the proven local VPCD/QNN policy
- local policy facts gathered from the current `sd8g2_quality` preset:
  - total graph nodes: `2946`
  - excluded nodes: `1765`
  - excluded decoder nodes: `1764`
  - excluded `lm_head` nodes: `1`
  - remaining quantized `MatMul` nodes: about `96`
- implication:
  - the next AIMET attempt must aim for policy parity with the local quality lane, not just dtype parity
- new switch decision:
  - keep the new AIMET infrastructure
  - do not promote AIMET to default until a policy-constrained variant passes bounded local and compiled-cloud teacher-forced checks

**Updated status on 2026-05-19 after the parity rerun:**

- `Task 7` is implemented
- `Task 8` is implemented
- `Task 10` is implemented
- `Task 9` is still deferred because the parity variant passed the bounded gates without requiring extra AIMET sensitivity tooling
- current best official variant:
  - `w8a16 + min_max + local_quality_parity`
  - custom config `vpcd_matmul_only`
- bounded evidence status:
  - local quantized teacher-forced `5/5` match
  - compiled-cloud teacher-forced `5/5` match
  - bounded hybrid prefix correct
- remaining switch blocker:
  - validate a longer free-run window beyond the current `max_decode_steps = 5`

---

## Execution Contract For The Next Implementation Pass

The next implementation pass under this plan must be fully operator-free once started.

- implement the planned AIMET parity work end-to-end without requiring manual notebook reruns
- run only the VPCD-relevant notebook cells after implementation, not the Zipformer cells
- if a foreground notebook execution times out, switch to a more reliable execution method:
  - selective cell runner
  - background process with progress logging
  - save-after-each-cell flow
- after the notebook run finishes:
  - inspect the outputs
  - analyze any failure
  - apply fixes if needed
  - rerun the affected VPCD cells until the lane is stable enough to summarize
- update docs with:
  - the root cause observed for the current VPCD output
  - every AIMET variant attempted
  - every Docker command or image requirement used
  - the result of each approach
  - the recommended next step if the parity attempt still fails
- keep the `B/C/D` AI Hub quantize fallback matrix documented and runnable throughout

## Why This Is The Right Direction

The current evidence gives us a very specific constraint:

- the current AI Hub quantize baseline fails correctness at teacher-forced step `2`
- the current local ORT/QNN QDQ artifact is semantically promising in bounded local teacher-forced checks
- but AI Hub compile rejects that artifact because it still contains `com.microsoft:DequantizeLinear`

So the next local-quantize attempt should not be “more graph rewriting on the ORT/QNN artifact.” It should follow an **officially supported format**.

## Official Findings That Drive This Plan

### Qualcomm AI Hub

- Qualcomm AI Hub compile officially supports:
  - ONNX
  - AIMET quantized models
- Qualcomm AI Hub quantization docs say the quantized ONNX produced by AI Hub is in ONNX fake-quant format and that this is the only ONNX quantization format officially supported as compile input.
- Qualcomm AI Hub compile docs explicitly document the `.aimet` package structure:
  - one `.onnx`
  - one `.encodings`
  - one `.data` if and only if the ONNX uses external weights

### AIMET ONNX

- AIMET ONNX `QuantizationSimModel` supports:
  - `param_type=int8`
  - `activation_type=int8` or `int16`
  - `quant_scheme=min_max` or `tf_enhanced`
- AIMET `compute_encodings(...)` supports calibration from an iterable of input dictionaries or a forward-pass callback.
- AIMET `export(...)` can emit the ONNX model and encodings needed for packaging.

### ONNX / Current Local QDQ Constraint

- Main-domain `QuantizeLinear/DequantizeLinear` support for `int16/uint16` is standardized only in newer ONNX opsets.
- Our current local QDQ artifact is still:
  - `opset 17`
  - `com.microsoft` Q/DQ
  - quantized initializer + `DQ` heavy
- That is the wrong place to keep investing if the target is official AI Hub compatibility.

## Scope And Boundaries

- This plan only covers the VPCD Option 1 lane.
- Do not modify Zipformer.
- Do not remove the existing AI Hub quantize lane in this phase.
- Do not remove the existing ORT/QNN local-QDQ probe lane in this phase.
- Keep the `B/C/D` AI Hub quantize fallback matrix alive.
- Keep all old lanes runnable until the AIMET lane passes the required gates.
- Deleting or collapsing redundant code belongs to a later dedicated cleanup phase.

## Architecture Decision

### New Lane To Add

The new candidate lane is:

- fixed-shape FP32 prepare locally
- local AIMET quantization using the same VPCD autoregressive calibration data
- export `.aimet` package locally
- compile the `.aimet` package on AI Hub
- run the existing VPCD teacher-forced and bounded hybrid diagnostics

### Default Variant To Start With

Start with the simplest official-compatible variant:

- `w8a8`
- `quant_scheme=min_max`

Reason:

- this minimizes format risk
- it aligns with the AI Hub quantization examples
- it gives us a lower-entropy first compile target than starting with `int16` activations

### Follow-up Variant

Only after the first variant compiles and runs should we test:

- `w8a16`

Reason:

- it may recover quality
- but it adds format complexity that is not worth paying before compile acceptance is proven

### Updated Follow-up Direction After The First AIMET Probe

The next AIMET pass should no longer be a broad default-quantize attempt.

It should target a **policy-parity variant** that is intentionally shaped to resemble the current local VPCD/QNN quality preset.

That follow-up variant should start with:

- `w8a16`
- `quant_scheme=min_max`
- a custom AIMET config file
- a quantization policy that behaves like:
  - quantize `MatMul`-heavy encoder regions first
  - keep the decoder stack conservative
  - keep `lm_head` conservative

Reason:

- the proven local quality preset is not simply "8-bit weights plus 16-bit activations"
- it also keeps almost the entire decoder and `lm_head` out of the quantized set
- the first AIMET probe did not respect that policy shape, so it is not yet a fair parity comparison

## New Phase Structure

### Phase 1: Add AIMET Lane And Prove It

This plan covers only this phase.

### Phase 1B: Align AIMET With The Local Quality Policy

This follow-up phase is now in scope for the next implementation pass.

Goal:

- preserve the official AIMET -> `.aimet` -> AI Hub compile route
- change the AIMET quantization policy so it resembles the local `sd8g2_quality` lane closely enough to be a meaningful comparison

Primary idea:

- use official AIMET configuration mechanisms to avoid quantizing sensitive regions that the local QNN lane already keeps in FP32

### Phase 2: Switch Defaults If Proven

Only if the AIMET lane compiles and passes the bounded correctness gates.

### Phase 3: Remove Redundant Code

Out of scope for this plan.

That future phase may:

- remove dead helper branches
- archive deprecated notebook knobs
- simplify source-strategy handling
- remove duplicated record and packaging helpers

But only after the AIMET lane is accepted as stable.

### What Counts As Redundant In The Future Cleanup Phase

The cleanup phase should target concrete duplication or now-obsolete branches, not broad refactors.

Candidate redundant areas already visible today:

- `src/tools/aihub_option1_pilots.py`
  - `prepare_vpcd_option1_source_model(...)` currently mixes three responsibilities in one function:
    - `prefer_fp32_fixed`
    - `direct_qdq_sanitized`
    - `local_qdq_compile_candidate`
  - after AIMET is proven, the old `direct_qdq_sanitized` branch is a strong candidate for removal
  - the `local_qdq_compile_candidate` branch may be demoted to a reference-only helper or moved behind an explicitly legacy path
- `src/tools/aihub_option1_pilots.py`
  - `_build_local_qdq_compile_candidate_report(...)` is highly specific to the rejected local-QDQ lane
  - if AIMET becomes the preferred local lane, this helper may become redundant except as historical/debug support
- `src/quantize/qnn.py`
  - the current local ORT/QNN quantize path may remain useful for device-local experiments, but it is redundant as an AI Hub compile candidate if AIMET takes over that role
  - cleanup must be careful not to delete reusable local-QNN functionality that still serves non-AI-Hub workflows
- `src/quantize/projects/vpcd.py`
  - VPCD quantize config currently serves two competing ideas:
    - AI Hub quantize recipes
    - local ORT/QNN quantize recipes
  - once AIMET is stable, strategy-selection glue that only exists to bridge the rejected local-QDQ compile experiment may be removable
- `On_device_Ai_option1_pilots.ipynb`
  - current VPCD source-strategy handling contains local-QDQ-specific naming and branches:
    - `vpcd_option1_local_qdq`
    - `model.option1.qdq.onnx`
    - explicit `local_qdq_compile_candidate` string checks
  - after AIMET is accepted, these should be replaced by a cleaner strategy table or a smaller set of supported source strategies
- `src/tools/aihub_option1_hybrid_pipeline.py`
  - metadata plumbing for `source_strategy` and `quantize_stage` should stay
  - but lane-specific conditionals added only for the rejected local-QDQ compile probe may become removable if they no longer serve an active diagnostic workflow
- record/docs surface area
  - if AIMET becomes primary, some local-QDQ-specific record shapes and explanatory notebook text may become redundant
  - however, retain enough historical evidence to preserve reproducibility of the rejected experiment

### What Must Not Be Deleted Even In Cleanup

The future cleanup phase should still preserve:

- the baseline `prefer_fp32_fixed -> AI Hub quantize` path until AIMET has multiple successful reruns
- the `B/C/D` fallback matrix plan and enough hooks to execute it
- the shared teacher-forced and bounded hybrid diagnostics
- generic record-writing utilities that are lane-agnostic

### Cleanup Entry Criteria

Do not start the cleanup phase until all of these are true:

- AIMET compile succeeds at least once on AI Hub
- compiled-cloud teacher-forced no longer diverges at step `2`
- bounded hybrid no longer collapses into punctuation for the bounded test window
- notebook docs and records clearly indicate AIMET is the preferred local quantize lane
- no active investigation still depends on `direct_qdq_sanitized` as a primary path

## File Structure

**Files:**

- Add: `src/quantize/aimet.py`
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

- `src/quantize/aimet.py`
  - local AIMET quantization and export helpers
  - `.aimet` packaging contract
- `src/quantize/projects/vpcd.py`
  - VPCD-specific AIMET config, calibration bridge, and variant defaults
- `src/quantize/types.py`
  - AIMET recipe and report types
- `src/tools/aihub_option1_pilots.py`
  - source strategy selection, `.aimet` prepare, packaging, records, AI Hub compile integration
- `src/tools/aihub_option1_hybrid_pipeline.py`
  - diagnostics metadata for the new lane
- notebook
  - operator-facing strategy selection and execution flow
- docs
  - decision tree, risks, and real run outcomes

## Decision Gates

The implementation must stop at the first failing gate and record why.

### Gate 1: Local AIMET Export Contract

The exported artifact must:

- produce one `.onnx`
- produce one `.encodings`
- produce one `.data` only when external weights are used
- sit inside a directory with `.aimet` in the name

If not, do not submit compile.

### Gate 2: Local Bounded Teacher-Forced Sanity

Before AI Hub compile, the local AIMET-exported artifact must be checked with the existing local teacher-forced diagnostic.

Success rule:

- no earlier divergence than the current FP32 reference in the bounded `5`-step window

Minimum target:

- step `2` must not collapse to punctuation like the AI Hub quantize baseline did

### Gate 3: AI Hub Compile Acceptance

The `.aimet` package must compile on AI Hub for the runtime we already use:

- `precompiled_qnn_onnx`

If compile fails, record the failure and keep the old lanes as default.

### Gate 4: Compiled Teacher-Forced Correctness

The compiled AIMET lane must be checked with the existing compiled-cloud teacher-forced diagnostic.

Success rule:

- step `2` must no longer diverge

### Gate 5: Bounded Hybrid Correctness

Only if teacher-forced passes:

- run bounded hybrid with:
  - `VPCD_HYBRID_MAX_SAMPLES = 1 or 2`
  - `VPCD_HYBRID_MAX_STEPS = 5`

Success rule:

- no punctuation-collapse pattern like `[0, 4, 4, 4, 4]`

### Gate 6: Policy-Parity Evidence

Before claiming the AIMET parity attempt is representative of the local quality lane, record all of these:

- how many nodes the local quality preset excludes
- which major regions those exclusions cover
- what the AIMET config keeps conservative
- whether the tested AIMET variant is:
  - broad default quantization
  - MatMul-focused
  - decoder-conservative

If this evidence is missing, do not conclude that "AIMET cannot match local quality."

## Detailed Tasks

## Follow-up Tasks After The First AIMET Probe

These tasks extend the already-implemented AIMET lane. They are the next tasks to execute before any cleanup phase.

### Task 7: Capture Local-Quality Quantization Intent Explicitly

**Files:**

- Modify: `src/quantize/projects/vpcd.py`
- Modify: `src/quantize/types.py`
- Test: `test/test_vpcd_quantize_aihub.py`

- [x] **Step 1: Add tests that describe the local quality policy**

Test behavior:

- the VPCD helper exposes a summary of the current local `sd8g2_quality` quantization plan
- the summary reports:
  - excluded node count
  - excluded decoder coverage
  - excluded `lm_head` coverage
  - quantized `MatMul` count

- [x] **Step 2: Implement a reusable local-policy summary helper**

Implementation rules:

- do not duplicate the preset logic in a second place
- derive the summary from the existing preset and actual model node names
- make the result serializable into records and docs

- [x] **Step 3: Re-run the focused tests**

Run: `pytest test/test_vpcd_quantize_aihub.py -k "quality or aimet" -v`

- [x] **Step 4: Commit**

```bash
git add src/quantize/projects/vpcd.py src/quantize/types.py test/test_vpcd_quantize_aihub.py
git commit -m "feat: record vpcd local quality quantization intent"
```

### Task 8: Add A Policy-Constrained AIMET Variant

**Files:**

- Modify: `src/quantize/aimet.py`
- Modify: `src/quantize/projects/vpcd.py`
- Modify: `src/tools/aihub_option1_pilots.py`
- Test: `test/test_vpcd_quantize_aihub.py`
- Test: `test/test_aihub_option1_pilots.py`

- [x] **Step 1: Write failing tests for the AIMET parity config**

Test behavior:

- the AIMET recipe can request:
  - `w8a16`
  - `min_max`
  - a custom config file path
- the helper writes or resolves a custom AIMET config that is intended to be decoder-conservative
- records clearly state this is a policy-parity variant, not the old broad default variant

- [x] **Step 2: Implement the custom AIMET config flow**

Implementation rules:

- stay on official AIMET mechanisms:
  - config file
  - supported quantsim options
- do not rely on ad-hoc graph rewriting as the main strategy
- keep the original broad `w8a8 + min_max` path available for comparison

- [x] **Step 3: Re-run the focused tests**

Run: `pytest test/test_vpcd_quantize_aihub.py test/test_aihub_option1_pilots.py -k "aimet" -v`

- [x] **Step 4: Commit**

```bash
git add src/quantize/aimet.py src/quantize/projects/vpcd.py src/tools/aihub_option1_pilots.py test/test_vpcd_quantize_aihub.py test/test_aihub_option1_pilots.py
git commit -m "feat: add policy-constrained aimet variant for vpcd"
```

### Task 9: Add AIMET Sensitivity Analysis Before Cloud Compile

**Files:**

- Modify: `src/quantize/aimet.py`
- Modify: `src/tools/aihub_option1_pilots.py`
- Modify: `docs/workflows/aihub-option1-hybrid-pipeline.md`
- Test: `test/test_aihub_option1_pilots.py`

- [ ] **Step 1: Add a bounded local analysis step**

Goal:

- if the parity variant still fails local teacher-forced step `2`, produce layer-sensitivity evidence before spending more cloud compile cycles

- [ ] **Step 2: Implement an AIMET analysis helper**

Implementation rules:

- prefer official AIMET analysis tooling such as `QuantAnalyzer`
- scope the first pass to the VPCD regions most likely to be sensitive:
  - decoder
  - decoder attention
  - `lm_head`
- write the results into local records so docs can compare variants

- [ ] **Step 3: Re-run the focused tests**

Run: `pytest test/test_aihub_option1_pilots.py -k "aimet" -v`

- [ ] **Step 4: Commit**

```bash
git add src/quantize/aimet.py src/tools/aihub_option1_pilots.py docs/workflows/aihub-option1-hybrid-pipeline.md test/test_aihub_option1_pilots.py
git commit -m "feat: add aimet sensitivity analysis for vpcd"
```

### Task 10: Execute The Parity Variant End-To-End

**Files:**

- Modify: `On_device_Ai_option1_pilots.ipynb`
- Modify: `docs/plans/active/2026-05-13-vpcd-option1-debug-results.md`
- Modify: `docs/workflows/aihub-option1-npu-pilots.md`
- Modify: `docs/workflows/aihub-option1-hybrid-pipeline.md`
- Test: `test/test_option1_notebook_layout.py`

- [x] **Step 1: Run the VPCD-only notebook path**

Execution rules:

- run only the VPCD cells needed for:
  - AIMET prepare
  - compile or compile reuse
  - local quantized teacher-forced
  - compiled-cloud teacher-forced when local gate passes
  - bounded hybrid when teacher-forced gates pass
- save the executed notebook and log under `build/aihub/notebook_runs/`

- [x] **Step 2: Analyze the output and fix issues if needed**

Decision rules:

- if local teacher-forced still fails at step `2`, stop cloud expansion and document the failure signature
- if local passes but compiled cloud fails, attribute the issue to compile/runtime and document that
- if both pass, run bounded hybrid and summarize the result

- [x] **Step 3: Update docs with full evidence**

Docs must include:

- the exact AIMET parity variant used
- whether the custom config improved step `2`
- all generated records and notebook logs
- the root cause of the observed output
- the recommended next step

- [x] **Step 4: Re-run notebook layout tests**

Run: `pytest test/test_option1_notebook_layout.py -k "aimet or vpcd" -v`

- [x] **Step 5: Commit**

```bash
git add On_device_Ai_option1_pilots.ipynb docs/plans/active/2026-05-13-vpcd-option1-debug-results.md docs/workflows/aihub-option1-npu-pilots.md docs/workflows/aihub-option1-hybrid-pipeline.md test/test_option1_notebook_layout.py
git commit -m "docs: record aimet parity results for vpcd"
```

### Task 1: Add AIMET Quantization Types And Export Helpers

**Files:**

- Add: `src/quantize/aimet.py`
- Modify: `src/quantize/types.py`
- Test: `test/test_vpcd_quantize_aihub.py`

- [ ] **Step 1: Write failing tests for AIMET export contract**

Test behavior:

- helper can build an AIMET export plan for VPCD
- helper writes a package directory with:
  - one `.onnx`
  - one `.encodings`
  - optional `.data`
- helper returns structured metadata

- [ ] **Step 2: Run the focused test to confirm it fails**

Run: `pytest test/test_vpcd_quantize_aihub.py -k "aimet" -v`

- [ ] **Step 3: Implement the minimal AIMET helper layer**

Implementation rules:

- create a new AIMET-specific module instead of overloading `qnn.py`
- start with `w8a8 + min_max`
- keep the API recipe-driven so `w8a16` can be added later

- [ ] **Step 4: Re-run the focused test**

Run: `pytest test/test_vpcd_quantize_aihub.py -k "aimet" -v`

- [ ] **Step 5: Commit**

```bash
git add src/quantize/aimet.py src/quantize/types.py test/test_vpcd_quantize_aihub.py
git commit -m "feat: add aimet export helpers for vpcd"
```

### Task 2: Reuse The Existing VPCD Calibration Data For AIMET

**Files:**

- Modify: `src/quantize/projects/vpcd.py`
- Test: `test/test_vpcd_quantize_aihub.py`

- [ ] **Step 1: Write failing tests for calibration parity**

Test behavior:

- the AIMET lane must reuse the same autoregressive calibration entries and fingerprint already used for AI Hub quantize
- calibration order and fixed-shape padding must stay identical

- [ ] **Step 2: Run the focused test to confirm it fails**

Run: `pytest test/test_vpcd_quantize_aihub.py -k "aimet and calibration" -v`

- [ ] **Step 3: Implement the calibration bridge**

Implementation rules:

- do not invent a second calibration dataset
- adapt the existing calibration entries into the iterable format AIMET expects
- preserve the calibration fingerprint in records

- [ ] **Step 4: Re-run the focused test**

Run: `pytest test/test_vpcd_quantize_aihub.py -k "aimet and calibration" -v`

- [ ] **Step 5: Commit**

```bash
git add src/quantize/projects/vpcd.py test/test_vpcd_quantize_aihub.py
git commit -m "feat: reuse vpcd calibration data for aimet export"
```

### Task 3: Add A New AIMET Source Strategy To The AI Hub Pilot Helpers

**Files:**

- Modify: `src/tools/aihub_option1_pilots.py`
- Test: `test/test_aihub_option1_pilots.py`

- [ ] **Step 1: Write failing tests for the new source strategy**

Test behavior:

- add a new strategy:
  - `local_aimet_compile_candidate`
- prepare helper must return:
  - source kind
  - packaging kind
  - packaging path
  - AIMET metadata
- compile lane must skip `submit_quantize_job()`

- [ ] **Step 2: Run the focused pilot tests to confirm they fail**

Run: `pytest test/test_aihub_option1_pilots.py -k "aimet_source_strategy or aimet_packaging" -v`

- [ ] **Step 3: Implement the source-strategy contract**

Implementation rules:

- do not remove `prefer_fp32_fixed`
- do not remove `local_qdq_compile_candidate`
- clearly mark AIMET as the new official local-quantize candidate
- package according to `.aimet` directory rules from Qualcomm docs

- [ ] **Step 4: Re-run the focused pilot tests**

Run: `pytest test/test_aihub_option1_pilots.py -k "aimet_source_strategy or aimet_packaging" -v`

- [ ] **Step 5: Commit**

```bash
git add src/tools/aihub_option1_pilots.py test/test_aihub_option1_pilots.py
git commit -m "feat: add aimet source strategy for vpcd pilots"
```

### Task 4: Add AIMET Lane Support To The Notebook

**Files:**

- Modify: `On_device_Ai_option1_pilots.ipynb`
- Test: `test/test_option1_notebook_layout.py`

- [ ] **Step 1: Write failing notebook-layout tests**

Test behavior:

- notebook config exposes:
  - `VPCD_SOURCE_STRATEGY = "local_aimet_compile_candidate"`
- compile-only cell skips AI Hub quantize for the AIMET lane
- notebook text explains that AIMET lane is the preferred official local-quantize experiment

- [ ] **Step 2: Run the focused notebook test to confirm it fails**

Run: `pytest test/test_option1_notebook_layout.py -k "aimet" -v`

- [ ] **Step 3: Update the notebook**

Notebook changes:

- add support for:
  - `prefer_fp32_fixed`
  - `local_qdq_compile_candidate`
  - `local_aimet_compile_candidate`
- when AIMET is selected:
  - prepare `.aimet`
  - skip `submit_quantize_job()`
  - compile the `.aimet` package directly
- keep the existing diagnostics order

- [ ] **Step 4: Re-run the focused notebook test**

Run: `pytest test/test_option1_notebook_layout.py -k "aimet" -v`

- [ ] **Step 5: Commit**

```bash
git add On_device_Ai_option1_pilots.ipynb test/test_option1_notebook_layout.py
git commit -m "feat: add aimet lane support to vpcd notebook"
```

### Task 5: Extend Diagnostics Metadata For AIMET Runs

**Files:**

- Modify: `src/tools/aihub_option1_hybrid_pipeline.py`
- Test: `test/test_aihub_option1_hybrid_pipeline.py`

- [ ] **Step 1: Write failing tests for AIMET metadata**

Test behavior:

- teacher-forced reports must show:
  - `source_strategy = local_aimet_compile_candidate`
  - `quantize_stage = disabled`
  - AIMET package metadata when present

- [ ] **Step 2: Run the focused hybrid test to confirm it fails**

Run: `pytest test/test_aihub_option1_hybrid_pipeline.py -k "aimet" -v`

- [ ] **Step 3: Implement the minimal metadata plumbing**

Implementation rules:

- reuse the same diagnostic framework
- do not fork the teacher-forced record format
- add only the source-strategy and package context needed for attribution

- [ ] **Step 4: Re-run the focused hybrid test**

Run: `pytest test/test_aihub_option1_hybrid_pipeline.py -k "aimet" -v`

- [ ] **Step 5: Commit**

```bash
git add src/tools/aihub_option1_hybrid_pipeline.py test/test_aihub_option1_hybrid_pipeline.py
git commit -m "feat: add aimet metadata to vpcd diagnostics"
```

### Task 6: Execute The AIMET Probe End To End

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

- [ ] **Step 2: Run only the VPCD notebook cells with `VPCD_SOURCE_STRATEGY=local_aimet_compile_candidate`**

Required path:

- setup
- VPCD prepare
- VPCD compile-only without AI Hub quantize
- local quantized teacher-forced
- compiled-cloud teacher-forced
- bounded hybrid
- summary

- [ ] **Step 3: Apply the switch decision**

Decision rule:

- if AIMET compile fails:
  - keep existing lanes as default
  - document AIMET failure clearly
- if AIMET compiles but teacher-forced still diverges at step `2`:
  - keep existing lanes as default
  - document that AIMET packaging solved compatibility but not correctness
- if AIMET compiles and fixes step `2`:
  - promote AIMET lane to preferred local quantize lane
  - do not remove old lanes yet

- [ ] **Step 4: Update docs with the real result**

Required doc updates:

- exact AIMET package structure used
- calibration fingerprint used
- local bounded teacher-forced result
- AI Hub compile result
- compiled-cloud teacher-forced result
- whether the AIMET lane is now the preferred local quantize candidate
- explicit note that redundant-code removal is deferred to a later cleanup phase

- [ ] **Step 5: Commit**

```bash
git add docs/workflows/aihub-option1-npu-pilots.md docs/workflows/aihub-option1-hybrid-pipeline.md docs/plans/active/2026-05-13-vpcd-option1-debug-results.md
git commit -m "docs: record aimet local quantize decision for vpcd"
```

## Expected Outcome

At the end of this plan, we should have one of these outcomes:

- a working official local-quantize lane:
  - `fixed-shape FP32 -> local AIMET -> .aimet -> AI Hub compile -> teacher-forced pass`

or:

- a documented AIMET failure with strong evidence
- the repo still keeps:
  - the current AI Hub quantize lane
  - the `B/C/D` fallback matrix
  - the local-QDQ probe lane for reference only

## Explicit Non-Goal For This Plan

Do **not** remove the older quantize lanes in this phase.

Even if the AIMET lane succeeds, removal work should happen later in a separate plan, after we are confident we no longer need:

- `prefer_fp32_fixed` as the current baseline
- `local_qdq_compile_candidate` as a reference probe
- old helper branches that still provide useful regression comparison points

If AIMET succeeds, create a **new cleanup plan** rather than quietly deleting old paths while implementing this one.

## Official Sources To Keep Open While Executing

- [Qualcomm AI Hub Quantization docs](https://workbench.aihub.qualcomm.com/docs/hub/quantize_examples.html)
- [Qualcomm AI Hub Compile docs](https://workbench.aihub.qualcomm.com/docs/hub/compile_examples.html)
- [Qualcomm AI Hub `submit_compile_job()` API](https://workbench.aihub.qualcomm.com/docs/hub/generated/qai_hub.submit_compile_job.html)
- [AIMET ONNX `QuantizationSimModel`](https://quic.github.io/aimet-pages/releases/latest/apiref/onnx/quantsim.html)
- [ONNX `QuantizeLinear` operator docs](https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html)
