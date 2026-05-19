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
3. `Quantized Local Teacher-Forced Diagnostics` for bounded FP32-vs-quantized next-token comparison on VPCD
4. `Teacher-Forced Diagnostics` for bounded FP32-vs-compiled-cloud comparison on VPCD
5. `Hybrid E2E Run` and `Final Compare` for Phase 3 correctness

For VPCD, a fourth bounded probe path now also exists:

- `local AIMET .aimet compile candidate -> local teacher-forced diagnostics -> compiled-cloud teacher-forced -> bounded hybrid`

That probe is used to answer whether the current local VPCD quantized artifact is semantically healthier than the AI Hub-quantized artifact and whether the official local quantize lane should replace the older AI Hub quantize baseline.

The same notebook now continues into:

6. `Phase 4` benchmark and gate sections
7. `Phase 5` packaging sections

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
5. `VPCD Quantized Local Teacher-Forced Diagnostics`
6. `VPCD Teacher-Forced Diagnostics`
7. `VPCD Hybrid E2E Run`
8. `VPCD Final Compare Against Gold Samples`

Notes:

- the logits inspection section is not the final pass/fail gate
- the quantized-local teacher-forced section is the first bounded diagnostic step and should run before the compiled-cloud teacher-forced section
- the compiled-cloud teacher-forced section should run before the free-run hybrid loop
- when `VPCD_SOURCE_STRATEGY = "local_aimet_compile_candidate"`, the notebook uses the exported local QDQ diagnostic model for the quantized-local teacher-forced section
- in the local-AIMET reuse case, the notebook now preserves:
  - the local AIMET compile pilot name in teacher-forced and hybrid records
  - the local QDQ diagnostic model path when compile records are reused
- the final pass/fail decision comes from punctuated outputs compared against `golden_samples.jsonl`
- recommended knobs for the current runaway decode failure are:
  - `VPCD_HYBRID_MAX_SAMPLES = 2`
  - `VPCD_HYBRID_MAX_STEPS = 5`
  - `VPCD_TEACHER_FORCED_SAMPLE_INDEX = 0`
- quantize runs now also preserve:
  - `build/aihub/records/vpcd_option1/quantize-run-<RUN_LABEL>.json`
  - `build/aihub/vpcd_option1/model.quantized.<RUN_LABEL>.onnx`
- the hybrid run writes:
  - `build/aihub/records/vpcd_hybrid_option1/hybrid-run-<RUN_LABEL>.json`
- the quantized-local teacher-forced run writes:
  - `build/aihub/records/vpcd_quantized_teacher_forced_option1/hybrid-run-<RUN_LABEL>.json`
- the teacher-forced run writes:
  - `build/aihub/records/vpcd_teacher_forced_option1/hybrid-run-<RUN_LABEL>.json`
- a recent local-AIMET probe completed with:
  - [prepared-artifact-20260518-aimet-local-w8a8-minmax.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/records/vpcd_option1_local_aimet/prepared-artifact-20260518-aimet-local-w8a8-minmax.json)
  - [compile-run-20260518-aimet-local-w8a8-minmax.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/records/vpcd_option1_local_aimet/compile-run-20260518-aimet-local-w8a8-minmax.json)
  - [hybrid-run-20260518-aimet-local-w8a8-minmax.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/records/vpcd_quantized_teacher_forced_option1/hybrid-run-20260518-aimet-local-w8a8-minmax.json)
  - [hybrid-run-20260518-aimet-local-w8a8-minmax.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/records/vpcd_teacher_forced_option1/hybrid-run-20260518-aimet-local-w8a8-minmax.json)
  - [hybrid-run-20260518-aimet-local-w8a8-minmax.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/records/vpcd_hybrid_option1/hybrid-run-20260518-aimet-local-w8a8-minmax.json)

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
  - `matches_expected_prefix`
  - `comparison_note`
  - `decode_steps`
  - `decode_step_limit_reached`
  - `truncated_by_decode_step_limit`
  - `ended_with_eos`
  - `cloud_inference_seconds`
  - `decode_seconds`

- `VPCD teacher-forced`
  - `sample_index`
  - `raw_text`
  - `decode_step_limit`
  - `gold_decoder_ids`
  - `reference_stats`
  - `steps[*].decoder_prefix_ids`
  - `steps[*].expected_next_token_id`
  - `steps[*].cpu_top_tokens`
  - `steps[*].cloud_top_tokens`
  - `steps[*].cpu_argmax_token_id`
  - `steps[*].cloud_argmax_token_id`
  - `steps[*].matches_cpu_argmax`
  - `steps[*].job_id`

- `VPCD quantized-local teacher-forced`
  - `sample_index`
  - `raw_text`
  - `decode_step_limit`
  - `reference_stats.quantized_model_path`
  - `steps[*].decoder_prefix_ids`
  - `steps[*].expected_next_token_id`
  - `steps[*].cpu_top_tokens`
  - `steps[*].quantized_top_tokens`
  - `steps[*].cpu_argmax_token_id`
  - `steps[*].quantized_argmax_token_id`
  - `steps[*].matches_fp32_argmax`

## VPCD Decision Tree

Use the current VPCD notebook flow to separate two questions:

1. Does the downloaded AI Hub quantized ONNX already diverge from FP32 on teacher-forced prefixes?
2. If we bypass AI Hub quantize and use the local AIMET parity artifact, does the local quantized artifact still align with FP32 on teacher-forced prefixes?
3. If a quantized artifact looks healthy locally, does divergence appear only after AI Hub compile and cloud inference?
4. Or does the model only drift later during free-run autoregressive decoding?

Interpret the evidence this way:

- if quantized-local divergence appears in the first few steps:
  - treat `AI Hub quantize` as the primary suspect
  - keep the calibration fingerprint fixed while trying bounded quantize variants
- if quantized-local steps stay aligned but compiled-cloud teacher-forced diverges:
  - treat `AI Hub compile / QNN execution` as the primary suspect
  - do not spend more time on quantize variants yet
- if teacher-forced steps look reasonable but free-run hybrid later collapses into punctuation loops:
  - treat stopping behavior or autoregressive drift as the next suspect
- if both teacher-forced and free-run stay aligned for the bounded `5`-step window:
  - only then consider increasing `VPCD_HYBRID_MAX_STEPS`

Current known result:

- AI Hub quantize baseline `A` diverges at teacher-forced step `2`
- the broad official AIMET variant `w8a8 + min_max` compiles on AI Hub
- that broad AIMET variant still diverges at local teacher-forced step `2`
- the newer policy-constrained AIMET parity variant:
  - `w8a16 + min_max + local_quality_parity`
  - compiles on AI Hub
  - matches FP32 locally for teacher-forced steps `1..5`
  - matches FP32 on compiled cloud for teacher-forced steps `1..5`
  - produces the correct bounded hybrid prefix instead of punctuation collapse
- when that bounded hybrid run stops at the debug guardrail before EOS, the record now treats the full-text comparison as unavailable rather than as a real mismatch
- the notebook final compare cell now mirrors that rule:
  - `matches_expected is False` => real mismatch
  - `matches_expected is None` with a truncation note => bounded comparison unavailable

## Current Limits

- still Python-only, not Android integration
- still reuses Phase 2 compiled targets
- `Zipformer` is still `encoder on NPU`, not full RNNT on NPU
- `VPCD` still keeps tokenizer encode/decode on CPU
- final promotion for deployment now depends on Phase 4 and is preserved by Phase 5 packages
