# AI Hub Option 1 Phase 5 Contract Packaging Workflow

This document describes the dedicated `Phase 5` packaging notebook for `Option 1`.

Scope:

- consume Phase 2, Phase 3, and Phase 4 evidence
- package every pilot regardless of verdict
- preserve the latest recommendation inside the package

## What Phase 5 Consumes

Phase 5 expects these upstream records for the selected `RUN_LABEL`:

- `prepared-artifact-<RUN_LABEL>.json`
- `compile-run-<RUN_LABEL>.json`
- `live-run-<RUN_LABEL>.json`
- `hybrid-run-<RUN_LABEL>.json`
- `phase4-gate-<RUN_LABEL>.json`

Current notebook defaults are intentionally per-pilot:

- Zipformer:
  - `RUN_LABEL = 20260513-1am`
- VPCD:
  - `RUN_LABEL = 20260519-aimet-local-quality-parity-notebook`
  - `phase2 compile pilot override = vpcd_option1_local_aimet`

## Package Layout

Packages are written under:

- `build/aihub/contracts/option1/zipformer/<RUN_LABEL>/`
- `build/aihub/contracts/option1/vpcd/<RUN_LABEL>/`

Each package contains:

- `contract_manifest.json`
- `io_contract.json`
- `contract_summary.md`
- `evidence/`

## Notebook Sections

Phase 5 runs inside:

- [On_device_Ai_option1_phase5_contract.ipynb](/D:/DS-AI/BKMeeting-Research/python-model-test/On_device_Ai_option1_phase5_contract.ipynb)

Run these sections after Phase 4 records exist:

1. `## Phase 5 Config`
2. `### Package Zipformer Phase 5 Contract`
3. `### Package VPCD Phase 5 Contract`
4. `## Phase 5 Packaging Summary`

## Promotion Status Rules

Phase 5 does not override Phase 4.

It only maps the Phase 4 recommendation into package status:

- `GO` -> `deployment_candidate`
- `WARN` -> `deployment_candidate`
- `NO_GO` -> `research_only`

## What Downstream Consumers Should Read First

Start in this order:

1. `contract_summary.md`
2. `contract_manifest.json`
3. `io_contract.json`
4. the copied evidence under `evidence/`

## Important Notes

- Package creation is not proof of deployment readiness.
- `research_only` packages are for debugging and follow-up experiments, not for Android promotion.
- Phase 5 keeps compile skipping intact because it only reads records that already exist.
- This notebook is package-only:
  - no compile
  - no profiling
  - no hybrid rerun

## 2026-05-19 Run Result

Execution evidence:

- executed notebook:
  - [On_device_Ai_option1_phase5_contract.executed.ipynb](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/notebook_runs/On_device_Ai_option1_phase5_contract.executed.ipynb)
- execution log:
  - [option1_phase5_contract.log](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/notebook_runs/option1_phase5_contract.log)

Generated packages:

- Zipformer:
  - [contract_manifest.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/contracts/option1/zipformer/20260513-1am/contract_manifest.json)
  - promotion status: `deployment_candidate`
  - inherited Phase 4 verdict: `WARN`
- VPCD:
  - [contract_manifest.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/contracts/option1/vpcd/20260519-aimet-local-quality-parity-notebook/contract_manifest.json)
  - promotion status: `research_only`
  - inherited Phase 4 verdict: `NO_GO`

The VPCD package is still useful for downstream contract work:

- it packages the correct `vpcd_option1_local_aimet` evidence lane
- it records that the current bounded `5`-step gate is a `comparison_unavailable` research lane, not a proven deployment candidate
