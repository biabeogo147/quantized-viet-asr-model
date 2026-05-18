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

- `local QDQ compile candidate -> local teacher-forced diagnostics`

A fifth bounded probe path now also exists:

- `local AIMET .aimet compile candidate -> local teacher-forced diagnostics -> compiled-cloud teacher-forced -> bounded hybrid`

That probe is used only to answer whether the current local VPCD quantized artifact is semantically healthier than the AI Hub-quantized artifact and whether AI Hub compile will even accept it.

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
- when `VPCD_SOURCE_STRATEGY = "local_qdq_compile_candidate"`, the notebook still runs the local quantized teacher-forced section even if AI Hub compile fails
- when `VPCD_SOURCE_STRATEGY = "local_aimet_compile_candidate"`, the notebook uses the exported local QDQ diagnostic model for the quantized-local teacher-forced section
- in that local-QDQ failure case, the notebook intentionally skips:
  - compiled target resolution
  - live run
  - compiled-cloud teacher-forced
  - hybrid free-run
  - final compare
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
- a recent local-QDQ probe completed with:
  - [prepared-artifact-20260518-local-qdq-probe.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/records/vpcd_option1_local_qdq/prepared-artifact-20260518-local-qdq-probe.json)
  - [compile-run-20260518-local-qdq-probe.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/records/vpcd_option1_local_qdq/compile-run-20260518-local-qdq-probe.json)
  - [hybrid-run-20260518-local-qdq-probe.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/records/vpcd_quantized_teacher_forced_option1/hybrid-run-20260518-local-qdq-probe.json)
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
  - `decode_steps`
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
2. If we bypass AI Hub quantize and use the local-QDQ compile candidate, does the local quantized artifact still align with FP32 on teacher-forced prefixes?
3. If a quantized artifact looks healthy locally, does divergence appear only after AI Hub compile and cloud inference?
4. Or does the model only drift later during free-run autoregressive decoding?

Interpret the evidence this way:

- if quantized-local divergence appears in the first few steps:
  - treat `AI Hub quantize` as the primary suspect
  - keep the calibration fingerprint fixed while trying bounded quantize variants
- if the local-QDQ artifact stays aligned locally but AI Hub compile rejects it before runtime:
  - treat `artifact compatibility with AI Hub compile` as the primary blocker
  - do not switch notebook defaults away from the FP32 + AI Hub quantize lane yet
  - keep the `B/C/D` AI Hub quantize fallback matrix active
- if quantized-local steps stay aligned but compiled-cloud teacher-forced diverges:
  - treat `AI Hub compile / QNN execution` as the primary suspect
  - do not spend more time on quantize variants yet
- if teacher-forced steps look reasonable but free-run hybrid later collapses into punctuation loops:
  - treat stopping behavior or autoregressive drift as the next suspect
- if both teacher-forced and free-run stay aligned for the bounded `5`-step window:
  - only then consider increasing `VPCD_HYBRID_MAX_STEPS`

Current known result:

- AI Hub quantize baseline `A` diverges at teacher-forced step `2`
- the current local QDQ artifact matches FP32 locally for the bounded `5`-step teacher-forced probe
- AI Hub compile rejects that same local QDQ artifact because `com.microsoft:DequantizeLinear` is unsupported in the input model
- the current official AIMET variant `w8a8 + min_max` compiles on AI Hub
- that same AIMET variant already diverges at local teacher-forced step `2`
- compiled cloud reproduces the same AIMET divergence and bounded hybrid exits early with empty text

## Current Limits

- still Python-only, not Android integration
- still reuses Phase 2 compiled targets
- `Zipformer` is still `encoder on NPU`, not full RNNT on NPU
- `VPCD` still keeps tokenizer encode/decode on CPU
- final promotion for deployment now depends on Phase 4 and is preserved by Phase 5 packages
