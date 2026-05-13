# VPCD Option 1 Debug Results

Date: `2026-05-13`

This note records the outcome of the VPCD Option 1 containment and debug plan after implementation, verification, and a real notebook rerun on Qualcomm AI Hub.

## Scope Executed

- enforced the shared VPCD notebook lane to stay on:
  - `FP32 fixed-shape prepare -> AI Hub quantize -> AI Hub compile`
- kept the bounded free-run guardrail:
  - `VPCD_HYBRID_MAX_STEPS = 5`
- added a teacher-forced diagnostic path before free-run hybrid
- reran the VPCD notebook path and saved outputs back into:
  - [On_device_Ai_option1_pilots.ipynb](/D:/DS-AI/BKMeeting-Research/python-model-test/On_device_Ai_option1_pilots.ipynb)

## Records Produced

- prepared artifact:
  - [prepared-artifact-20260513-1am.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/records/vpcd_option1/prepared-artifact-20260513-1am.json)
- compile record reused:
  - [compile-run-20260513-1am.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/records/vpcd_option1/compile-run-20260513-1am.json)
- live run:
  - [live-run-20260513-1am.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/records/vpcd_option1/live-run-20260513-1am.json)
- teacher-forced diagnostic:
  - [hybrid-run-20260513-1am.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/records/vpcd_teacher_forced_option1/hybrid-run-20260513-1am.json)
- bounded free-run hybrid:
  - [hybrid-run-20260513-1am.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/records/vpcd_hybrid_option1/hybrid-run-20260513-1am.json)

## Approaches And Results

### Approach 1: Bound The Free-Run Hybrid Loop

What changed:

- kept sample count bounded by the fixture set
- capped per-sample decode to `5` steps

Result:

- the old free-run record on `2026-05-13` had average cloud inference time around `12508.406844` seconds per sample
- the bounded rerun dropped that to `1258.532508` seconds per sample
- this is roughly a `10x` reduction in cloud time
- the run now stops after `5` decode steps instead of running out toward `48` and `55` steps

Observed output:

- sample 0 text: `",,,,"`
- sample 1 text: `",,,,"`
- generated ids for both samples collapsed to:
  - `[0, 4, 4, 4, 4]`

Conclusion:

- the cost-containment goal succeeded
- the correctness problem still remained

### Approach 2: Add Teacher-Forced Diagnostics Before Free-Run

What changed:

- each cloud step used the gold decoder prefix instead of the model's previous generated token
- each step recorded:
  - expected next token
  - local CPU top tokens
  - cloud top tokens
  - CPU argmax
  - cloud argmax

Result on sample `0`:

- step `1` matched:
  - expected next token `0`
  - CPU argmax `0`
  - cloud argmax `0`
- step `2` diverged immediately:
  - expected next token `2232`
  - CPU argmax `2232`
  - cloud argmax `4`
- steps `3`, `4`, and `5` stayed divergent:
  - CPU argmax followed the gold path `177 -> 9 -> 847`
  - cloud argmax kept collapsing to token `4`

Important token pattern:

- cloud top-k after divergence repeatedly favored punctuation-like ids:
  - `4`
  - `382`
  - sometimes `135`

Conclusion:

- this is not primarily a late free-run drift problem
- the compiled cloud target is already wrong by teacher-forced step `2`
- the repeated punctuation string is a downstream symptom of that earlier divergence

### Approach 3: Run The Notebook End-To-End Without Requiring Manual Reruns

What happened:

- a single foreground notebook execution hit a `1` hour tool timeout
- the notebook was then rerun using a background per-cell execution flow with per-cell saves

Cells executed for the VPCD path:

- AI Hub auth and setup
- imports and config
- VPCD prepare
- VPCD compile-only cell
- resolve existing compiled target
- VPCD live run
- VPCD teacher-forced diagnostics
- VPCD hybrid free-run
- VPCD final compare
- summary

Result:

- the notebook finished the VPCD path successfully
- outputs were written back into the notebook file
- no manual rerun is required for the completed `20260513-1am` path

## Root Cause Conclusion

Current best-supported conclusion:

- the VPCD output failure on AI Hub is not caused by the old missing decode-step cap alone
- the `max_step = 5` change fixed the runtime/debug-loop cost problem, but it did not fix token correctness
- the strongest current suspect is the compiled target produced by:
  - `FP32 fixed-shape prepare -> AI Hub quantize -> AI Hub compile`
- evidence for that conclusion is that:
  - local CPU reference stays aligned with the expected next token through the bounded teacher-forced window
  - the cloud target diverges at teacher-forced step `2`
  - once divergence starts, the cloud target repeatedly prefers punctuation-like token ids, especially `4`

Short version:

- the punctuation loop is a symptom
- the earlier failure is next-token divergence in the compiled AI Hub lane

## What Is Fixed Now

- the notebook no longer needs an unbounded VPCD hybrid rerun
- VPCD hybrid runs are capped at `5` steps
- teacher-forced diagnostics now exist and run before free-run hybrid
- the VPCD notebook path was executed successfully and recorded
- the team now has a concrete failure signature for the compiled target

## What Is Not Fixed Yet

- final VPCD correctness on the current AI Hub compiled target is still failing
- the compiled target still emits punctuation-collapse behavior after the first generated token

## Recommended Next Fixes

Run these in order:

1. Keep teacher-forced diagnostics as the first gate for any new compile attempt.
2. Re-run the `FP32 -> AI Hub quantize -> compile` lane with a new compile label and a calibration variant that is more aggressive about decoder-prefix coverage.
3. Compare at least one alternate quantize setting against the current preset, because the failure appears after quantization/compile rather than after a long free-run decode.
4. Do not spend more time extending hybrid decode length until teacher-forced step `2` stops diverging.

## Practical Decision Rule

- if a new compile still fails at teacher-forced step `2`, keep treating quantize/compile as the primary target
- if teacher-forced steps become correct but free-run still collapses later, then shift focus to stopping behavior and autoregressive drift
