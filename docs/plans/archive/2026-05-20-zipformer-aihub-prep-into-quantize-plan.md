# Zipformer AI Hub Prep Into Quantize Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the retained Zipformer AI Hub compile-prep work out of `On_device_Ai_option1_pilots.ipynb` and into `python -m quantize --project zipformer`, so the notebook only reads one prebuilt AI Hub-ready encoder from `build/quantize` and submits compile/run jobs.

**Architecture:** Keep `build/quantize/zipformer/qnn_u16u8/` as the canonical producer root. Extend the Zipformer quantize pipeline so it emits one retained AI Hub-ready encoder artifact and records its path in the quantize report. Remove the Zipformer prepare helper from `src/tools/aihub_option1_pilots.py` and let the notebook point directly at `build/quantize`, matching the retained VPCD producer/consumer split as closely as possible.

**Tech Stack:** Python, ONNX Runtime, ONNX, existing Zipformer quantize CLI, existing Option 1 AI Hub helper code, notebook JSON, `pytest`

**Execution note (`2026-05-20`):** implemented. `python -m quantize --project zipformer ... --provider CUDAExecutionProvider` now emits the retained AI Hub-ready encoder under `build/quantize/.../aihub_compile/`, notebook Zipformer cells run post-quantize-only, and the `20260520-zipformer-post-quantize` hybrid output matched the prior retained Zipformer output exactly.

---

## Recommended Approach

Use `python -m quantize --project zipformer` as the single producer for:

- fixed-shape encoder / decoder / joiner
- QDQ quantized bundle
- retained AI Hub-ready encoder for `zipformer_encoder_option1`

Write the AI Hub-ready encoder under:

- `build/quantize/zipformer/qnn_u16u8/aihub_compile/encoder.aihub.option1.onnx`

Record it in `quantization_report.json` and keep the file itself under `build/quantize` as the notebook input.

Why this approach:

- keeps `build/quantize` as the canonical producer root for both retained lanes
- removes producer-side model mutation from the notebook
- avoids storing Zipformer producer artifacts under `build/aihub`
- matches the retained VPCD pattern closely enough without forcing Zipformer into VPCD-specific packaging

## Do Not Do

- do not keep generating `encoder.fixed.optimized.onnx` and `encoder.fixed.optimized.symshape.onnx` inside `build/aihub/zipformer_encoder_option1/`
- do not move the AI Hub-ready encoder into `build/model_bundle/zipformer/qnn_u16u8/`
- do not let the notebook rebuild the AI Hub-ready encoder on the fly

## Target Output After This Plan

Producer output:

- `build/quantize/zipformer/qnn_u16u8/fixed_shapes/encoder.fixed.onnx`
- `build/quantize/zipformer/qnn_u16u8/quantized/encoder.onnx`
- `build/quantize/zipformer/qnn_u16u8/aihub_compile/encoder.aihub.option1.onnx`
- `build/quantize/zipformer/qnn_u16u8/quantization_report.json`

Notebook behavior:

- read the prepared encoder path from `build/quantize/...`
- submit compile/run
- write only AI Hub evidence under `build/aihub/records/...`

## Task 1: Add Retained AI Hub-Prep Step To Zipformer Quantize Producer

**Files:**

- Modify: `src/quantize/projects/zipformer.py`
- Modify: `test/test_zipformer_quantize.py`

- [x] Add a retained producer step that starts from the canonical fixed-shape encoder:
  - optimize graph
  - run symbolic shape inference
  - rewrite bool-mask slices for HTP
  - save one final retained artifact:
    - `aihub_compile/encoder.aihub.option1.onnx`

- [x] Keep the transformation code in the producer path, not in the notebook helper path.

- [x] Extend the Zipformer quantize report with stable fields:
  - `aihub_prepared_encoder_path`
  - `aihub_prepare_applied = true`
  - `aihub_prepare_steps = ["ort_optimize", "symbolic_shape_inference", "zipformer_bool_mask_rewrite"]`

- [x] Add tests that prove:
  - the AI Hub-ready encoder is emitted under `build/quantize/.../aihub_compile/`
  - the quantize report points to that file
  - the retained prepare steps are reflected in report metadata

## Task 2: Remove Zipformer Prepare Helper And Point Notebook Directly At `build/quantize`

**Files:**

- Modify: `src/tools/aihub_option1_pilots.py`
- Modify: `test/test_aihub_option1_pilots.py`

- [x] Delete `prepare_zipformer_encoder_option1_source_model(...)`.

- [x] Delete Zipformer-specific AI Hub prep ownership from `src/tools/aihub_option1_pilots.py`.

- [x] Keep only the minimum shared utilities needed by notebook compile/run record writing.

- [x] Add tests that prove:
  - stale `build/aihub/zipformer_encoder_option1/*.onnx` is not required
  - the notebook path now depends on `build/quantize/zipformer/qnn_u16u8/aihub_compile/encoder.aihub.option1.onnx`
  - missing producer artifacts fail with a clear `python -m quantize --project zipformer ...` command hint

## Task 3: Make The Notebook Zipformer Path Post-Quantize Only

**Files:**

- Modify: `On_device_Ai_option1_pilots.ipynb`
- Modify: `test/test_option1_notebook_layout.py`

- [x] Rewrite the Zipformer `Model-Session-First` markdown and code expectations so the notebook clearly starts after local quantize producer output exists.

- [x] Remove wording that implies the notebook is still doing:
  - graph optimization
  - symbolic shape inference
  - Zipformer-specific graph rewrite

- [x] Replace notebook usage of the deleted helper with a direct path under:
  - `build/quantize/zipformer/qnn_u16u8/aihub_compile/encoder.aihub.option1.onnx`

- [x] Prefer one explicit notebook variable over extra abstraction, for example:
  - `zipformer_aihub_prepared_encoder_path`

- [x] Keep the retained compile pilot name:
  - `zipformer_encoder_option1`

- [x] Update notebook layout tests so they lock the new post-quantize-only Zipformer behavior.

## Task 4: Refresh Retained Workflow Docs

**Files:**

- Modify: `docs/workflows/option1-overview.md`
- Modify: `docs/workflows/option1-rerun.md`
- Modify: `docs/workflows/option1-retained-lanes.md`
- Modify: `src/quantize/README.md`

- [x] Update docs to say:
  - Zipformer quantize producer now emits the retained AI Hub-ready encoder
  - the notebook reads that file directly from `build/quantize`
  - `build/aihub` is evidence-only for Zipformer, just like the retained VPCD lane

- [x] Keep the wording short and operational.

## Task 5: Prune Stale Zipformer AI Hub Producer Layout

**Files:**

- Modify: `src/tools/aihub_option1_pilots.py`
- Modify: implementation helpers as needed
- Verify: `build/aihub/zipformer_encoder_option1/`

- [x] Stop treating `build/aihub/zipformer_encoder_option1/` as a canonical producer directory.

- [x] After the new producer path is verified, remove stale assumptions in code/tests/docs that require intermediate files:
  - `encoder.fixed.optimized.onnx`
  - `encoder.fixed.optimized.symshape.onnx`
  - `encoder.aihub.option1.onnx` under `build/aihub/zipformer_encoder_option1/`

- [x] Remove Zipformer-only AI Hub prep code from `src/tools/aihub_option1_pilots.py` once the same logic lives in `src/quantize/projects/zipformer.py`.

- [ ] The expected dead code list in `src/tools/aihub_option1_pilots.py` is:
  - `ZIPFORMER_BOOL_SLICE_NODE_NAMES`
  - `ZIPFORMER_BOOL_UNSQUEEZE_NODE_NAMES`
  - `rewrite_zipformer_bool_mask_slices_for_htp(...)`
  - `prepare_zipformer_encoder_option1_source_model(...)`
  - `_optimize_onnx_model_for_aihub(...)`
  - `_run_symbolic_shape_inference(...)`

- [ ] Keep `build/aihub/records/zipformer_encoder_option1/` as the retained evidence root.

## Verification

- [x] Run:
  - `python -m quantize --project zipformer ...`
- [x] Verify:
  - `build/quantize/zipformer/qnn_u16u8/aihub_compile/encoder.aihub.option1.onnx` exists
  - `quantization_report.json` points to it
- [x] Run:
  - `pytest test/test_zipformer_quantize.py test/test_aihub_option1_pilots.py test/test_option1_notebook_layout.py -k "zipformer or option1" -v`
- [x] Run the Zipformer cells in:
  - `On_device_Ai_option1_pilots.ipynb`
- [x] Verify the notebook writes:
  - `build/aihub/records/zipformer_encoder_option1/prepared-artifact-<RUN_LABEL>.json`
  - `build/aihub/records/zipformer_encoder_option1/compile-run-<RUN_LABEL>.json`
  - `build/aihub/records/zipformer_encoder_option1/live-run-<RUN_LABEL>.json`
  - `build/aihub/records/zipformer_hybrid_option1/hybrid-run-<RUN_LABEL>.json`

## Done Criteria

- [x] `python -m quantize --project zipformer` emits one retained AI Hub-ready encoder under `build/quantize`
- [x] `On_device_Ai_option1_pilots.ipynb` no longer rebuilds or resolves the Zipformer AI Hub-ready encoder through a helper
- [x] `On_device_Ai_option1_pilots.ipynb` reads the retained encoder directly from `build/quantize`
- [x] Zipformer notebook path is post-quantize only, like retained VPCD
- [x] Zipformer compile/run evidence still lands under `build/aihub/records/...`
- [x] stale Zipformer producer assumptions under `build/aihub/zipformer_encoder_option1/` are gone
