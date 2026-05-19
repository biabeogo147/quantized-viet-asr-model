# VPCD Option 1 Debug Results

Date: `2026-05-19`

This note captures the real outcome after implementing the VPCD quantize-vs-compile isolation work, the local-QDQ compile probe, the official AIMET probe, and the new AIMET parity rerun that keeps the quantization policy close to the proven local `sd8g2_quality` lane.

## Executive Conclusion

There are now two distinct conclusions, and both matter:

- the original VPCD punctuation-collapse bug was caused by `quantize`, not by `compile`, long free-run decode, or notebook control flow
- the new policy-constrained AIMET lane fixes that bounded correctness failure for the first `5` decoder steps

Current best result:

- local AIMET parity variant:
  - `w8a16 + min_max + local_quality_parity`
- AI Hub compile:
  - accepted
- local quantized teacher-forced:
  - `5/5` steps match FP32
- compiled-cloud teacher-forced:
  - `5/5` steps match FP32
- bounded hybrid:
  - generates the correct prefix `Hôm nay là buổi`
  - the remaining full-text mismatch under `max_decode_steps = 5` is an expected truncation artifact, not a punctuation collapse

Short version:

- `max_step = 5` fixed the runaway runtime cost
- the original bad lanes were already broken in the quantized model at teacher-forced step `2`
- the parity AIMET lane removes that early divergence in the bounded window by avoiding over-quantization of decoder-heavy regions

Update after the local-QDQ compile probe on `2026-05-18`:

- the current local ORT/QNN-flavored QDQ artifact does not reproduce the old step-`2` divergence when it is run locally with teacher-forced prefixes
- however, AI Hub compile rejects that same artifact before runtime because the graph still contains `com.microsoft:DequantizeLinear`
- this means the local-QDQ lane is currently semantically healthier than the AI Hub-quantized baseline for the bounded `5`-step check, but it is not yet a valid AI Hub compile input

Update after the official AIMET local-quantize probe on `2026-05-18`:

- the new Docker-backed `AIMET w8a8 + min_max` lane exported a valid `.aimet` package locally
- AI Hub compile accepted that `.aimet` package and produced target model `mn40gpyrq`
- however, the locally exported AIMET QDQ reference already diverged from FP32 at teacher-forced step `2`
- the compiled cloud target reproduced the same divergence pattern, so this AIMET variant solved compile compatibility but did not solve correctness
- bounded hybrid no longer collapsed into punctuation; instead it exited almost immediately with EOS and empty text

Update after the policy-constrained AIMET parity rerun on `2026-05-19`:

- the new parity variant kept the official AIMET route but changed the quantization policy to resemble the local `sd8g2_quality` lane:
  - `w8a16`
  - `min_max`
  - custom AIMET config `vpcd_matmul_only`
  - policy mode `local_quality_parity`
- AI Hub compile accepted the exported `.aimet` package and produced target model `mn1dlyevn`
- the local AIMET QDQ reference matched FP32 for teacher-forced steps `1..5`
- the compiled cloud target matched FP32 for teacher-forced steps `1..5`
- bounded hybrid no longer collapsed into punctuation or early EOS
- the bounded hybrid result is now limited only by the debug guardrail `max_decode_steps = 5`

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

Local AIMET probe run:

- executed notebook copy:
  - [On_device_Ai_option1_pilots.local_aimet.executed.ipynb](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/notebook_runs/On_device_Ai_option1_pilots.local_aimet.executed.ipynb)
- reduced notebook input used for the VPCD-only run:
  - [On_device_Ai_option1_pilots.local_aimet.input.ipynb](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/notebook_runs/On_device_Ai_option1_pilots.local_aimet.input.ipynb)
- probe log:
  - [local_aimet.log](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/notebook_runs/local_aimet.log)

Local AIMET parity rerun:

- executed notebook copy:
  - [On_device_Ai_option1_pilots.local_aimet_quality_parity.executed.ipynb](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/notebook_runs/On_device_Ai_option1_pilots.local_aimet_quality_parity.executed.ipynb)
- reduced notebook input used for the VPCD-only rerun:
  - [On_device_Ai_option1_pilots.local_aimet_quality_parity.input.ipynb](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/notebook_runs/On_device_Ai_option1_pilots.local_aimet_quality_parity.input.ipynb)
- probe log:
  - [local_aimet_quality_parity.log](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/notebook_runs/local_aimet_quality_parity.log)
- final-compare-only notebook rerun after the bounded-truncation reporting fix:
  - [On_device_Ai_option1_pilots.local_aimet_quality_parity.compare_only.executed.ipynb](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/notebook_runs/On_device_Ai_option1_pilots.local_aimet_quality_parity.compare_only.executed.ipynb)
  - [local_aimet_quality_parity.compare_only.log](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/notebook_runs/local_aimet_quality_parity.compare_only.log)

Docker-backed AIMET prep used for that run:

- Dockerfile:
  - [docker/aimet-onnx-ubuntu2204/Dockerfile](/D:/DS-AI/BKMeeting-Research/python-model-test/docker/aimet-onnx-ubuntu2204/Dockerfile)
- reusable image tag:
  - `bkmeeting-vpcd-aimet:ubuntu22.04-py310`
- host-side calibration artifact root:
  - `build/aihub/vpcd_option1_local_aimet/`
- export command used:

```bash
docker --config build/docker-config run --rm \
  -v D:/DS-AI/BKMeeting-Research/python-model-test:/workspace \
  -w /workspace \
  -e PYTHONPATH=/workspace/src \
  bkmeeting-vpcd-aimet:ubuntu22.04-py310 \
  python3 -m quantize.aimet export \
    --fp32-onnx /workspace/build/aihub/vpcd_option1_local_aimet/model.fp32.fixed.onnx \
    --calibration-dir /workspace/build/aihub/vpcd_option1_local_aimet/calibration \
    --package-dir /workspace/build/aihub/vpcd_option1_local_aimet/model.option1.aimet \
    --qdq-reference-model /workspace/build/aihub/vpcd_option1_local_aimet/model.option1.qdq.onnx \
    --report-path /workspace/build/aihub/vpcd_option1_local_aimet/model.option1.aimet.report.json
```

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

Official AIMET local-quantize probe:

- prepared artifact record:
  - [prepared-artifact-20260518-aimet-local-w8a8-minmax.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/records/vpcd_option1_local_aimet/prepared-artifact-20260518-aimet-local-w8a8-minmax.json)
- compile record:
  - [compile-run-20260518-aimet-local-w8a8-minmax.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/records/vpcd_option1_local_aimet/compile-run-20260518-aimet-local-w8a8-minmax.json)
- live run record:
  - [live-run-20260518-aimet-local-w8a8-minmax.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/records/vpcd_option1_local_aimet/live-run-20260518-aimet-local-w8a8-minmax.json)
- local quantized teacher-forced record:
  - [hybrid-run-20260518-aimet-local-w8a8-minmax.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/records/vpcd_quantized_teacher_forced_option1/hybrid-run-20260518-aimet-local-w8a8-minmax.json)
- compiled-cloud teacher-forced record:
  - [hybrid-run-20260518-aimet-local-w8a8-minmax.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/records/vpcd_teacher_forced_option1/hybrid-run-20260518-aimet-local-w8a8-minmax.json)
- bounded hybrid record:
  - [hybrid-run-20260518-aimet-local-w8a8-minmax.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/records/vpcd_hybrid_option1/hybrid-run-20260518-aimet-local-w8a8-minmax.json)

Policy-constrained AIMET parity rerun:

- prepared artifact record:
  - [prepared-artifact-20260519-aimet-local-quality-parity-notebook.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/records/vpcd_option1_local_aimet/prepared-artifact-20260519-aimet-local-quality-parity-notebook.json)
- compile record:
  - [compile-run-20260519-aimet-local-quality-parity-notebook.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/records/vpcd_option1_local_aimet/compile-run-20260519-aimet-local-quality-parity-notebook.json)
- live run record:
  - [live-run-20260519-aimet-local-quality-parity-notebook.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/records/vpcd_option1_local_aimet/live-run-20260519-aimet-local-quality-parity-notebook.json)
- local quantized teacher-forced record:
  - [hybrid-run-20260519-aimet-local-quality-parity-notebook.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/records/vpcd_quantized_teacher_forced_option1/hybrid-run-20260519-aimet-local-quality-parity-notebook.json)
- compiled-cloud teacher-forced record:
  - [hybrid-run-20260519-aimet-local-quality-parity-notebook.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/records/vpcd_teacher_forced_option1/hybrid-run-20260519-aimet-local-quality-parity-notebook.json)
- bounded hybrid record:
  - [hybrid-run-20260519-aimet-local-quality-parity-notebook.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/records/vpcd_hybrid_option1/hybrid-run-20260519-aimet-local-quality-parity-notebook.json)

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

### Approach 6: Official AIMET Local Quantize With Docker

What changed:

- added a reusable Docker-backed AIMET export lane
- kept the same `24`-sample autoregressive calibration fingerprint
- exported:
  - a local `.aimet` package for AI Hub compile
  - a local QDQ reference model for bounded ONNX Runtime CPU diagnostics
- ran:
  - local quantized teacher-forced
  - compiled-cloud teacher-forced
  - bounded hybrid free-run

Observed packaging result:

- `.aimet` package contract satisfied:
  - one `.onnx`
  - one `.encodings`
- no `.data` file was needed inside the `.aimet` package
- the local QDQ reference model was emitted separately with external data:
  - `model.option1.qdq.onnx`
  - `model.option1.qdq.onnx.data`

Observed correctness result:

- local quantized teacher-forced:
  - step `1` matched FP32 argmax
  - step `2` diverged immediately
  - FP32 argmax: `2232`
  - AIMET local QDQ argmax: `2`
- compiled-cloud teacher-forced:
  - step `1` matched FP32 argmax
  - step `2` diverged with the same EOS-heavy pattern
  - compiled cloud argmax: `2`
- bounded hybrid:
  - sample `0`: empty string
  - sample `1`: empty string
  - generated ids:
    - `[0, 2]`

Compile compatibility result:

- AI Hub compile succeeded
- compile target model id:
  - `mn40gpyrq`

Conclusion:

- this is the first official local-quantize path here that is clearly AI Hub compile-compatible
- however, the current default AIMET variant `w8a8 + min_max` is not semantically healthy enough for VPCD
- because the local AIMET QDQ reference already fails at step `2`, the fault is still attributable to quantization, not to AI Hub compile

### Approach 7: Policy-Constrained AIMET Parity Variant

What changed:

- kept the official Docker-backed AIMET export route
- kept the same calibration fingerprint
- changed the quantization policy to track the proven local `sd8g2_quality` intent much more closely:
  - `w8a16`
  - `min_max`
  - custom AIMET config `vpcd_matmul_only`
  - policy mode `local_quality_parity`
- explicitly disabled a large decoder-heavy region and kept `lm_head` conservative instead of broad default quantization

Observed correctness result:

- local quantized teacher-forced:
  - steps `1..5` all matched FP32
  - step `2` correctly stayed at token `2232`
- compiled-cloud teacher-forced:
  - steps `1..5` all matched FP32
  - compiled cloud no longer reproduced the old step-`2` failure
- bounded hybrid:
  - sample `0` generated ids:
    - `[0, 2232, 177, 9, 847]`
  - bounded text:
    - `Hôm nay là buổi`
  - this is the correct prefix for the chosen `5`-step debug window

Observed reporting nuance:

- with `max_decode_steps = 5`, the hybrid text is intentionally shorter than the full gold sentence
- that row should be treated as:
  - `comparison unavailable because the debug step limit truncated the run before EOS`
- it should not be treated as a real punctuation mismatch
- the notebook final compare cell was patched accordingly:
  - real mismatches now require `matches_expected is False`
  - bounded truncation now prints:
    - `vpcd full-text comparison unavailable`

Conclusion:

- the earlier AIMET failure was not evidence that "AIMET cannot work for VPCD"
- the actual issue was over-quantization relative to the proven local policy
- once the policy became decoder-conservative, the official AIMET lane passed the bounded correctness gates

## Matrix Status

The bounded follow-up matrix is only partially exercised.

Completed:

- `A`: `w8a16 + auto + calibration giữ nguyên`
  - historical quantize job `jp8wwq1op`: fails at local quantized step `2`
  - current quantize job `jp0ekj665`: fails at local quantized step `2`
- official local AIMET:
  - `w8a8 + min_max + same calibration fingerprint`
  - compiles on AI Hub
  - still fails at local quantized step `2`
- policy-constrained local AIMET:
  - `w8a16 + min_max + local_quality_parity + same calibration fingerprint`
  - compiles on AI Hub
  - local quantized teacher-forced passes `5/5`
  - compiled-cloud teacher-forced passes `5/5`
  - bounded hybrid produces the correct prefix and no longer collapses

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
- the repo now has a reusable Docker image and command path for AIMET export
- the repo now has an official `.aimet -> AI Hub compile` lane that is proven compile-compatible
- the repo now has a policy-constrained AIMET lane that matches FP32 for the bounded `5`-step teacher-forced window both locally and on compiled cloud
- the old punctuation-collapse behavior is no longer reproduced by the policy-constrained AIMET lane in the bounded hybrid check
- hybrid records now distinguish a real mismatch from a `decode_step_limit` truncation in bounded debug runs

## What Is Not Fixed Yet

- full-length VPCD free-run correctness on AI Hub is not re-proven yet beyond the bounded `5`-step debug window
- baseline `A` remains bad even with the current calibration fingerprint
- the current local QDQ artifact still cannot be compiled by AI Hub in its present ORT/QNN-specific format
- the current official AIMET variant `w8a8 + min_max` still diverges at local teacher-forced step `2`
- the current parity proof only covers the bounded `5`-step window
- a longer free-run validation is still needed before switching notebook defaults fully to the AIMET lane

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

1. keep the official `.aimet` packaging route
2. keep the new parity lane as the leading candidate:
   - `w8a16 + min_max + local_quality_parity`
3. increase the bounded decode window gradually before changing defaults:
   - `5 -> 10 -> 20`
4. if the longer window regresses, run the remaining official AIMET follow-ups:
   - `w8a8 + tf_enhanced`
5. only after exhausting official AIMET variants, revisit standard main-domain QDQ with `main` opset `21+`

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

For the current official AIMET probe, the verdict is now:

- AI Hub compile compatibility: `accepted`
- local quantized step-`2` correctness: `failed`
- compiled-cloud step-`2` correctness: `failed`
- switch-default decision: `do not switch yet`

For the current policy-constrained AIMET parity rerun, the verdict is now:

- AI Hub compile compatibility: `accepted`
- local quantized step-`2` correctness: `passed`
- compiled-cloud step-`2` correctness: `passed`
- bounded hybrid result: `correct prefix, truncated by debug step limit`
- switch-default decision:
  - `leading candidate`
  - keep as non-default until a longer free-run window is validated
