# VPCD Option 1 Debug Results

Date: `2026-05-18`

This note captures the real outcome after implementing the VPCD quantize-vs-compile isolation work, rerunning the VPCD notebook path, and comparing FP32 local, quantized local, and compiled cloud behavior.

## Executive Conclusion

The VPCD punctuation-collapse bug is now attributed to the `quantize` stage, not only to `compile` or to long free-run decoding.

Evidence:

- the downloaded AI Hub quantized ONNX already diverges from the FP32 local reference at teacher-forced step `2`
- the compiled cloud target diverges at the same step and with the same top-token pattern
- bounded hybrid free-run then collapses to `",,,,"` because that earlier next-token error keeps feeding punctuation tokens back into the decoder

Short version:

- `max_step = 5` fixed the runaway runtime cost
- it did not fix correctness
- correctness is already broken in the quantized ONNX before cloud compile/runtime enters the picture

Update after the local-QDQ compile probe on `2026-05-18`:

- the current local ORT/QNN-flavored QDQ artifact does not reproduce the old step-`2` divergence when it is run locally with teacher-forced prefixes
- however, AI Hub compile rejects that same artifact before runtime because the graph still contains `com.microsoft:DequantizeLinear`
- this means the local-QDQ lane is currently semantically healthier than the AI Hub-quantized baseline for the bounded `5`-step check, but it is not yet a valid AI Hub compile input

## Notebook Status

The VPCD notebook path completed successfully and wrote outputs back into:

- [On_device_Ai_option1_pilots.ipynb](/D:/DS-AI/BKMeeting-Research/python-model-test/On_device_Ai_option1_pilots.ipynb)

Cells executed for VPCD:

- auth and setup
- imports and config
- VPCD prepare
- VPCD compile-only
- resolve compiled target
- VPCD live run
- VPCD quantized-local teacher-forced diagnostics
- VPCD compiled-cloud teacher-forced diagnostics
- VPCD bounded hybrid
- final compare
- summary

Important implementation note:

- a fresh explicit quantize submission `jp0ekj665` stayed in `QUANTIZING_MODEL` long enough to block the first notebook attempt
- to avoid manual reruns, the notebook was unblocked with the historical successful quantized artifact from `jp8wwq1op`
- later, `jp0ekj665` also completed successfully and was checked separately as the current `A` baseline
- a later local-QDQ probe run was executed by selecting only the VPCD cells needed for:
  - setup
  - VPCD prepare
  - VPCD compile-only
  - target resolution
  - local quantized teacher-forced diagnostics
  - compiled-cloud teacher-forced diagnostics
  - bounded hybrid
  - summary
- executed notebook copy:
  - [On_device_Ai_option1_pilots.local_qdq_probe.executed.ipynb](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/notebook_runs/On_device_Ai_option1_pilots.local_qdq_probe.executed.ipynb)
- probe log:
  - [local_qdq_probe.log](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/notebook_runs/local_qdq_probe.log)

## Primary Evidence

### Calibration Fingerprint

- fingerprint: `873ef0635a0bb4b29d6f37080517908a275f7961196676e9a1a7ab59d77c0510`
- text samples: `24`
- autoregressive records: `715`
- fixed shapes:
  - encoder `1 x 1024`
  - decoder `1 x 128`

### Quantized Artifact Evidence

Historical successful AI Hub quantize job used to unblock the notebook:

- quantize job: `jp8wwq1op`
- target model: `mnolrjpkn`
- quantized record:
  - [quantize-run-20260513-1am.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/records/vpcd_option1/quantize-run-20260513-1am.json)
- quantized-local teacher-forced record:
  - [hybrid-run-20260513-1am.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/records/vpcd_quantized_teacher_forced_option1/hybrid-run-20260513-1am.json)

Current baseline `A` rerun using the same current calibration recipe:

- quantize job: `jp0ekj665`
- target model: `mq8r4jw3n`
- quantize record:
  - [quantize-run-20260513-1am-currentA.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/records/vpcd_option1/quantize-run-20260513-1am-currentA.json)
- quantized-local teacher-forced record:
  - [hybrid-run-20260513-1am-currentA.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/records/vpcd_quantized_teacher_forced_option1/hybrid-run-20260513-1am-currentA.json)

Compiled cloud record for the failing production lane:

- compile record:
  - [compile-run-20260513-1am.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/records/vpcd_option1/compile-run-20260513-1am.json)
- compiled-cloud teacher-forced record:
  - [hybrid-run-20260513-1am.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/records/vpcd_teacher_forced_option1/hybrid-run-20260513-1am.json)
- bounded hybrid record:
  - [hybrid-run-20260513-1am.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/records/vpcd_hybrid_option1/hybrid-run-20260513-1am.json)

## Attribution Result

### FP32 Local vs Quantized Local

Local quantized teacher-forced result:

- first divergent step: `2`
- expected next token at step `2`: `2232`
- FP32 local argmax at step `2`: `2232`
- quantized local argmax at step `2`: `4`

Top quantized tokens at first divergence:

- token `4` score about `28.09`
- token `2` score about `26.72`
- token `382` score about `25.50`

This result reproduced in both:

- historical quantized artifact `jp8wwq1op`
- current baseline `A` artifact `jp0ekj665`

Conclusion:

- the quantized ONNX is already wrong before compile/cloud inference

### FP32 Local vs Compiled Cloud

Compiled cloud teacher-forced result:

- first divergent step: `2`
- expected next token at step `2`: `2232`
- FP32 local argmax at step `2`: `2232`
- compiled cloud argmax at step `2`: `4`

Top cloud tokens at first divergence:

- token `4` score about `28.43`
- token `2` score about `27.16`
- token `382` score about `25.99`

Conclusion:

- compiled cloud preserves the same failure signature already present in the quantized ONNX
- compile/cloud may still contribute noise, but they are no longer the first failing stage

### Bounded Hybrid

Bounded hybrid output after `max_decode_steps = 5`:

- sample `0`: `",,,,"`
- sample `1`: `",,,,"`
- generated ids for both samples:
  - `[0, 4, 4, 4, 4]`

Conclusion:

- the punctuation loop is a downstream symptom of the earlier quantized next-token collapse

## Approaches Tried

### Approach 1: Containment First

What changed:

- kept `VPCD_HYBRID_MAX_SAMPLES = 2`
- enforced `VPCD_HYBRID_MAX_STEPS = 5`

Result:

- notebook reruns no longer spend many hours inside runaway free-run decode
- cloud hybrid now stops after `5` decode steps
- correctness remained broken

### Approach 2: Teacher-Forced Cloud Before Free-Run

What changed:

- added a compiled-cloud teacher-forced checkpoint

Result:

- divergence appears at step `2`, not after a long autoregressive chain
- this ruled out “late drift only” as the main explanation

### Approach 3: Download And Run The Quantized ONNX Locally

What changed:

- downloaded the quantized ONNX produced by the AI Hub quantize job
- ran the same teacher-forced diagnostic locally with ONNX Runtime CPU

Result:

- the downloaded quantized ONNX diverged at the same step `2`
- the first wrong argmax was the same token `4`
- the same punctuation-like top-token pattern reappeared locally

Conclusion:

- this is the decisive attribution step
- the root-cause stage is `quantize`

### Approach 4: Re-submit Baseline `A` With Current Calibration

What changed:

- re-submitted baseline `A`
  - `w8a16 + auto + calibration giữ nguyên`

Observed behavior:

- the new quantize job `jp0ekj665` initially stayed in `QUANTIZING_MODEL` long enough to block notebook execution
- after completion, its downloaded artifact still diverged at teacher-forced step `2`

Conclusion:

- the historical artifact was not a one-off anomaly
- the current calibration recipe still reproduces the quantize-stage failure on baseline `A`

### Approach 5: Local QDQ Compile Probe

What changed:

- prepared the bundled local VPCD QDQ artifact as an explicit `local_qdq_compile_candidate`
- added a strict compatibility report before upload
- skipped AI Hub quantize and submitted the packaged local QDQ graph directly to AI Hub compile
- ran local quantized teacher-forced diagnostics against the same prepared artifact

Compatibility report from the prepared local-QDQ probe artifact:

- prepared artifact record:
  - [prepared-artifact-20260518-local-qdq-probe.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/records/vpcd_option1_local_qdq/prepared-artifact-20260518-local-qdq-probe.json)
- compile record:
  - [compile-run-20260518-local-qdq-probe.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/records/vpcd_option1_local_qdq/compile-run-20260518-local-qdq-probe.json)
- local quantized teacher-forced record:
  - [hybrid-run-20260518-local-qdq-probe.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/records/vpcd_quantized_teacher_forced_option1/hybrid-run-20260518-local-qdq-probe.json)

Observed graph facts:

- `main` opset: `17`
- `com.microsoft` opset: `1`
- `com.microsoft` Q/DQ nodes: `842`
- `uses_uint16_qdq`: `true`
- `uses_quantized_weight_initializers`: `true`
- conservative graph readiness: `unsafe`
- candidate readiness: `experimental`

First failure encountered during implementation:

- blindly rewriting `com.microsoft` Q/DQ domains on this graph produced an invalid ONNX graph for `uint16` QDQ under `main` opset `17`
- the fix was to preserve the local QDQ graph `as_is` for the compile probe whenever `uint16` or `int16` QDQ is present under opset `<21`
- this fixed the local ONNX-checker failure, but it did not make the graph AI Hub-compatible yet

Compile probe result:

- compile job: `jp3qz24m5`
- AI Hub failure:
  - `Layer 'DequantizeLinear' with domain 'com.microsoft' in input model is not supported by Qualcomm AI Hub Workbench.`

Local teacher-forced result on the same local-QDQ artifact:

- steps `1` through `5` matched the FP32 argmax
- step `2` matched token `2232`
- no punctuation collapse appeared in the bounded local test window

Conclusion:

- the current local VPCD QDQ artifact is useful as evidence and as a semantically healthier local reference
- it is not yet a valid replacement for AI Hub quantize because AI Hub compile rejects the artifact format before runtime

## Matrix Status

The bounded follow-up matrix is only partially exercised.

Completed:

- `A`: `w8a16 + auto + calibration giữ nguyên`
  - historical quantize job `jp8wwq1op`: fails at local quantized step `2`
  - current quantize job `jp0ekj665`: fails at local quantized step `2`

Not yet executed:

- `B`: `w8a16 + min_max + calibration giữ nguyên`
- `C`: `w8a8 + auto + calibration giữ nguyên`
- `D`: `w8a8 + min_max + calibration giữ nguyên`

Reason not yet executed:

- the main implementation and attribution goal is complete
- each extra variant is another full AI Hub quantize cycle
- once quantize was proven to be the first failing stage, the highest-value next work became bounded matrix exploration rather than more compile reruns on the known-bad baseline

## What Is Fixed Now

- the notebook VPCD path is implemented and runnable end-to-end
- the notebook completed without requiring user reruns
- VPCD free-run cost is bounded
- quantize/local/cloud attribution is no longer guesswork
- the failure signature is now reproducible and documented
- the notebook now completes cleanly even when the local-QDQ compile probe fails to produce a target model
- the repo now records an explicit compatibility report for local-QDQ compile candidates

## What Is Not Fixed Yet

- VPCD punctuation correctness on AI Hub is still failing
- baseline `A` remains bad even with the current calibration fingerprint
- no passing quantized variant has been found yet
- the current local QDQ artifact still cannot be compiled by AI Hub in its present ORT/QNN-specific format

## Recommended Next Steps

Keep two next-step tracks alive:

### Track 1: AI Hub Quantize Fallback Matrix

Run the remaining bounded matrix in this order and stop early on the first passing local quantized result:

1. `B`: `w8a16 + --range_scheme min_max`
2. `C`: `w8a8 + auto`
3. `D`: `w8a8 + --range_scheme min_max`

Execution rules:

- keep the same calibration fingerprint `873ef0635a0bb4b29d6f37080517908a275f7961196676e9a1a7ab59d77c0510`
- use a new `RUN_LABEL` per variant
- run only local quantized teacher-forced first
- compile cloud only for the first variant whose local quantized step `2` stops diverging

Interpretation rules:

- if `B` passes and `A` fails:
  - range selection is the leading suspect
- if `C` or `D` passes while `A/B` fail:
  - the `INT16` activation lane is the leading suspect
- if all four fail at the same early step:
  - stop spending time on compile
  - move next to source-graph / AI Hub quantize compatibility investigation

### Track 2: Official-Compatible Local Quantization Path

If the team continues pursuing local quantization as the default source lane, the next work should target an artifact format that is closer to what AI Hub documents as officially supported:

1. standard main-domain QDQ with `main` opset `21+`, without `com.microsoft` Q/DQ
2. or an official `.aimet` package lane containing ONNX plus encodings

Do not keep spending time on blind domain rewriting of the current `opset 17 + com.microsoft + uint16` graph. The compile probe already showed that AI Hub rejects it as input.

## Practical Decision Rule

- quantized local diverges:
  - blame `quantize`
- quantized local matches FP32 but compiled cloud diverges:
  - blame `compile / QNN execution`
- both match and only free-run fails:
  - investigate stopping and autoregressive drift

For VPCD baseline `A`, the verdict is now:

- final attribution: `quantize`

For the current local-QDQ compile probe, the verdict is now:

- local semantics in the bounded teacher-forced check: `promising`
- AI Hub compile compatibility: `rejected`
