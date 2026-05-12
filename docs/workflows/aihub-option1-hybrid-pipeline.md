# AI Hub Option 1 Hybrid Pipeline Workflow

This document describes the Phase 3 Python-only workflow for `Option 1`.

Scope:

- keep Qualcomm AI Hub compiled targets from Phase 2
- reuse `On_device_Ai_option1_pilots.ipynb`
- run hybrid e2e pipelines without recompiling
- compare final outputs only after the full pipeline finishes
- hand off stable evidence to Phase 4 and Phase 5 without creating a second notebook

## What Phase 3 Proves

Phase 2 proved that isolated graphs can compile and run on cloud NPU targets.

Phase 3 proves the real hybrid pipelines:

- `Zipformer`: host feature extraction -> compiled encoder on cloud NPU -> decoder and joiner on host CPU -> final transcript
- `VPCD`: tokenizer encode on host CPU -> compiled model step on cloud NPU -> host decode loop -> final punctuated text

## Notebook Contract

Phase 3 runs inside:

- [On_device_Ai_option1_pilots.ipynb](/D:/DS-AI/BKMeeting-Research/python-model-test/On_device_Ai_option1_pilots.ipynb)

The notebook now has three layers per pilot:

1. `Prepare / Compile / Resolve / Run` for Phase 2 reproducibility
2. `Output Inspection (Intermediate Diagnostic Only)` for tensor sanity checks
3. `Hybrid E2E Run` and `Final Compare` for Phase 3 correctness

The same notebook now continues into:

4. `Phase 4` benchmark and gate sections
5. `Phase 5` packaging sections

## Common Reuse Pattern

If compile already succeeded before, do not rerun compile.

Keep the same:

- `RUN_LABEL`
- `ZIPFORMER_TARGET_MODEL_ID = None`
- `VPCD_TARGET_MODEL_ID = None`

Then the notebook will resolve the target model id from:

- `build/aihub/records/zipformer_encoder_option1/compile-run-<RUN_LABEL>.json`
- `build/aihub/records/vpcd_option1/compile-run-<RUN_LABEL>.json`

If you want to force a specific compiled target manually, paste the id into:

- `ZIPFORMER_TARGET_MODEL_ID`
- `VPCD_TARGET_MODEL_ID`

## Zipformer Phase 3 Flow

Sections to run in order:

1. `Prepare`
2. `Resolve Existing Compiled Target`
3. `Run And Compare Against The Compiled Target`
4. `Zipformer Output Inspection (Intermediate Diagnostic Only)`
5. `Zipformer Hybrid E2E Run`
6. `Zipformer Final Compare Against Expected Outputs`

Notes:

- the tensor inspection section is not the final pass/fail gate
- the final pass/fail decision comes from transcript rows compared against `expected_outputs.jsonl`
- the hybrid run writes:
  - `build/aihub/records/zipformer_hybrid_option1/hybrid-run-<RUN_LABEL>.json`

## VPCD Phase 3 Flow

Sections to run in order:

1. `Prepare`
2. `Resolve Existing Compiled Target`
3. `Run And Compare Against The Compiled Target`
4. `VPCD Output Inspection (Intermediate Diagnostic Only)`
5. `VPCD Hybrid E2E Run`
6. `VPCD Final Compare Against Gold Samples`

Notes:

- the logits inspection section is not the final pass/fail gate
- the final pass/fail decision comes from punctuated outputs compared against `golden_samples.jsonl`
- the hybrid run writes:
  - `build/aihub/records/vpcd_hybrid_option1/hybrid-run-<RUN_LABEL>.json`

## Evidence Contract

Each hybrid run record preserves:

- target model id
- compile pilot name
- compile record path when compile-record lookup was used
- sample-level final outputs
- expected text per sample
- match or mismatch classification
- latency summaries

Current sample-level fields:

- `Zipformer`
  - `sample_id`
  - `audio_path`
  - `text`
  - `expected_text`
  - `matches_expected`
  - `num_tokens`
  - `cloud_inference_seconds`
  - `decode_seconds`

- `VPCD`
  - `sample_index`
  - `raw_text`
  - `text`
  - `expected_text`
  - `matches_expected`
  - `decode_steps`
  - `cloud_inference_seconds`
  - `decode_seconds`

## Current Limits

- still Python-only, not Android integration
- still reuses Phase 2 compiled targets
- `Zipformer` is still `encoder on NPU`, not full RNNT on NPU
- `VPCD` still keeps tokenizer encode/decode on CPU
- final promotion for deployment now depends on Phase 4 and is preserved by Phase 5 packages
