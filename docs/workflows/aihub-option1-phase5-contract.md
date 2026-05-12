# AI Hub Option 1 Phase 5 Contract Packaging Workflow

This document describes the `Phase 5` packaging flow for `Option 1`.

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

- [On_device_Ai_option1_pilots.ipynb](/D:/DS-AI/BKMeeting-Research/python-model-test/On_device_Ai_option1_pilots.ipynb)

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
