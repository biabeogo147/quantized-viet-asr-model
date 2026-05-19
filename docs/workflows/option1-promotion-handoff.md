# Option 1 Promotion And Handoff

Use this after a fresh `Option 1` rerun has already produced the retained `Phase 2 + Phase 3` records.

This doc covers:

- `Phase 4` gate generation
- `Phase 5` contract packaging
- the boundary between Python-side evidence and BKMeeting-side runtime proof

## Inputs from the rerun

The promotion path expects fresh retained-lane records from `docs/workflows/option1-rerun.md`:

- `prepared-artifact-<RUN_LABEL>.json`
- `compile-run-<RUN_LABEL>.json`
- `live-run-<RUN_LABEL>.json`
- `hybrid-run-<RUN_LABEL>.json`

Current clean rerun label:

- `20260519-option1-final-rerun`

Lane decisions and current proof limits live in:

- `docs/qnn/option1-retained-lanes.md`
- `docs/qnn/model-quantization.md`

## Phase 4 Gate

Run in:

- `On_device_Ai_option1_phase4_gate.ipynb`

Phase 4 rereads the retained-lane evidence and writes one gate record per pilot:

- `build/aihub/records/zipformer_phase4_option1/phase4-gate-<RUN_LABEL>.json`
- `build/aihub/records/vpcd_phase4_option1/phase4-gate-<RUN_LABEL>.json`

Recommendation meanings:

- `GO`
  - exact enough and fast enough for downstream promotion work
- `WARN`
  - usable evidence, but keep the caveat attached
- `NO_GO`
  - do not promote the lane yet

Important VPCD rule:

- if the retained run stops only because of the bounded `5`-step limit, treat that as `comparison_unavailable`, not punctuation collapse

## Phase 5 Contract Packaging

Run in:

- `On_device_Ai_option1_phase5_contract.ipynb`

Phase 5 consumes the retained `Phase 2 + Phase 3 + Phase 4` evidence and writes one contract package per pilot:

- `build/aihub/contracts/option1/zipformer/<RUN_LABEL>/`
- `build/aihub/contracts/option1/vpcd/<RUN_LABEL>/`

Each package contains:

- `contract_manifest.json`
- `io_contract.json`
- `contract_summary.md`
- `evidence/`

Promotion mapping:

- `GO` -> `deployment_candidate`
- `WARN` -> `deployment_candidate`
- `NO_GO` -> `research_only`

Phase 5 does not override the Phase 4 verdict.

## Handoff boundary

`python-model-test` proves:

- artifact generation
- AI Hub compile and bounded runtime evidence
- gate classification
- contract packaging

BKMeeting still must prove:

- Android asset sync
- ORT/QNN runtime packaging
- device-side NPU execution
- final promotion on Snapdragon hardware

## Next step in BKMeeting

After `Phase 5`:

1. stage the packaged artifacts into BKMeeting
2. run the BKMeeting-side runtime checks
3. keep the retained-lane caveats attached during promotion discussion

For the generic sync CLI and non-Option-1 handoff commands, use:

- `docs/workflows/android-handoff.md`

For the current active BKMeeting follow-up plans, use:

- `docs/plans/active/2026-05-19-bkmeeting-android-option1-export-plan.md`
- `docs/plans/active/2026-05-19-option1-phase6-contract-sync-plan.md`
