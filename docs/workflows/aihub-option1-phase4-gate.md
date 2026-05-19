# AI Hub Option 1 Phase 4 Gate Workflow

This document describes the dedicated `Phase 4` quality and performance gate notebook for `Option 1`.

Scope:

- keep the existing Phase 2 compile records
- keep the existing Phase 3 hybrid pipeline
- rerun hybrid evaluation without recompiling
- write one deterministic gate record per pilot

## What Phase 4 Consumes

Phase 4 depends on the records already written by earlier phases:

- `prepared-artifact-<RUN_LABEL>.json`
- `compile-run-<RUN_LABEL>.json`
- `live-run-<RUN_LABEL>.json`
- `hybrid-run-<RUN_LABEL>.json`

The notebook writes the new gate records to:

- `build/aihub/records/zipformer_phase4_option1/phase4-gate-<RUN_LABEL>.json`
- `build/aihub/records/vpcd_phase4_option1/phase4-gate-<RUN_LABEL>.json`

Current notebook defaults are intentionally per-pilot:

- Zipformer:
  - `RUN_LABEL = 20260513-1am`
- VPCD:
  - `RUN_LABEL = 20260519-aimet-local-quality-parity-notebook`
  - `phase2 compile pilot override = vpcd_option1_local_aimet`
  - `max_decode_steps = 5`

## Notebook Sections

Phase 4 runs inside:

- [On_device_Ai_option1_phase4_gate.ipynb](/D:/DS-AI/BKMeeting-Research/python-model-test/On_device_Ai_option1_phase4_gate.ipynb)

Run these sections after Phase 3 hybrid evidence exists:

1. `## Phase 4 Config`
2. `### Zipformer Phase 4 Benchmark And Gate`
3. `### VPCD Phase 4 Benchmark And Gate`
4. `## Phase 4 Recommendation Summary`

## What The Gate Does

Each Phase 4 run:

- reruns the existing hybrid pipeline for the configured iteration count
- keeps compile skipping intact
- classifies each final sample into:
  - `exact_match`
  - `minor_text_drift`
  - `major_text_drift`
  - `catastrophic_decode_failure`
  - `comparison_unavailable`
- records benchmark summaries:
  - warmup timing
  - steady-state timing
  - per-iteration totals
- records footprint observations:
  - prepared model size
  - output tensor footprint
  - generated token footprint
- produces one recommendation:
  - `GO`
  - `WARN`
  - `NO_GO`

## Operator Rules

- Do not rerun compile just to run Phase 4.
- Do not use this notebook for normal pilot compile or run-and-compare work.
- Keep the per-pilot `RUN_LABEL` stable when you want to gate the same compiled target again.
- If you already know the target model id, you may keep using:
  - `ZIPFORMER_TARGET_MODEL_ID`
  - `VPCD_TARGET_MODEL_ID`
- Treat `catastrophic_decode_failure` as an immediate `NO_GO` until the lane changes.
- For VPCD bounded debug reruns, preserve:
  - `compile_pilot_name = vpcd_option1_local_aimet`
  - `max_decode_steps = 5`

## Recommendation Meanings

- `GO`: correctness is exact and timing is within the configured `GO` threshold.
- `WARN`: the lane is usable for continued work, but there is visible drift or timing risk that must stay attached to the evidence.
- `NO_GO`: the lane is not promotable for downstream deployment work.

## 2026-05-19 Run Result

Execution evidence:

- executed notebook:
  - [On_device_Ai_option1_phase4_gate.executed.ipynb](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/notebook_runs/On_device_Ai_option1_phase4_gate.executed.ipynb)
- execution log:
  - [option1_phase4_gate.log](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/notebook_runs/option1_phase4_gate.log)

Persisted gate records:

- Zipformer:
  - [phase4-gate-20260513-1am.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/records/zipformer_phase4_option1/phase4-gate-20260513-1am.json)
  - current verdict: `WARN`
  - reason: `minor_text_drift_present` with latency still under the configured `WARN` threshold
- VPCD:
  - [phase4-gate-20260519-aimet-local-quality-parity-notebook.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/records/vpcd_phase4_option1/phase4-gate-20260519-aimet-local-quality-parity-notebook.json)
  - current verdict: `NO_GO`
  - final reason: `comparison_unavailable`

Important note for VPCD:

- the first Phase 4 notebook run exposed a helper bug where `ResolvedCompiledTarget` was not JSON-serializable
- after fixing that helper, the VPCD gate logic was also corrected to treat `decode_step_limit_reached_before_eos` as `comparison_unavailable` instead of `major_text_drift`
- to avoid repeating another multi-hour cloud rerun, the final VPCD Phase 4 record was refreshed from the notebook-produced hybrid evidence using the corrected classifier and the existing benchmark summary

## Evidence To Preserve

After a Phase 4 run, keep:

- the latest Phase 4 gate record
- the Phase 3 hybrid record used by the gate
- the matching Phase 2 compile and live records
- the final notebook output that shows the recommendation summary
