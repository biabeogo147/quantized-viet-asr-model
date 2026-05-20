# AI Hub Deployment Packaging

Use this doc after the retained AI Hub notebook already produced compile and live-run evidence.

Run in:

- [On_device_Ai_option1_pilots.ipynb](/D:/DS-AI/BKMeeting-Research/python-model-test/On_device_Ai_option1_pilots.ipynb)
  - first, to generate or reuse the retained compile target and live-run records
- `python -m aihub.deployment`
  - second, to download the compiled `precompiled_qnn_onnx` artifact and materialize the deployment package

## Purpose

This is the post-notebook deployment bridge between:

- notebook proof:
  - prepare
  - compile
  - live run
  - hybrid compare
- downstream handoff work:
  - artifact packaging
  - I/O contract freeze
  - Android-facing review notes

This flow does not rerun compile by default.
It consumes the retained records for one `RUN_LABEL` and downloads the deployable compiled target from AI Hub.

## Inputs

The deployment packager expects these retained records to already exist for the same `RUN_LABEL`.
The record-group names below are legacy evidence keys that stay unchanged so the new `aihub.*` package can resolve past runs without migration.

### Zipformer

- `build/aihub/records/zipformer_encoder_option1/prepared-artifact-<RUN_LABEL>.json`
- `build/aihub/records/zipformer_encoder_option1/compile-run-<RUN_LABEL>.json`
- `build/aihub/records/zipformer_encoder_option1/live-run-<RUN_LABEL>.json`
- optional:
  - `build/aihub/records/zipformer_hybrid_option1/hybrid-run-<RUN_LABEL>.json`

### VPCD

- `build/aihub/records/vpcd_option1_local_aimet/prepared-artifact-<RUN_LABEL>.json`
- `build/aihub/records/vpcd_option1_local_aimet/compile-run-<RUN_LABEL>.json`
- `build/aihub/records/vpcd_option1_local_aimet/live-run-<RUN_LABEL>.json`
- optional:
  - `build/aihub/records/vpcd_hybrid_option1/hybrid-run-<RUN_LABEL>.json`

## Commands

### Dry run

Use this first when you want to confirm record resolution and target model ids without downloading.

```bash
python -m aihub.deployment \
  --project all \
  --run-label 20260519-6pm \
  --device-name "Samsung Galaxy S24 (Family)" \
  --qairt-version 2.46.0 \
  --dry-run
```

### Real deployment package build

```bash
python -m aihub.deployment \
  --project all \
  --run-label 20260519-6pm \
  --device-name "Samsung Galaxy S24 (Family)" \
  --qairt-version 2.46.0
```

You can also package one project at a time:

```bash
python -m aihub.deployment \
  --project zipformer \
  --run-label 20260519-6pm \
  --device-name "Samsung Galaxy S24 (Family)" \
  --qairt-version 2.46.0
```

## Output Layout

Per project, deployment packaging writes:

- `build/aihub/deploy/zipformer/<RUN_LABEL>/`
- `build/aihub/deploy/vpcd/<RUN_LABEL>/`

Each package contains:

- `deployment_manifest.json`
- `io_contract.json`
- `deploy_notes.md`
- `download/`
  - the downloaded compiled artifact from AI Hub
- `evidence/`
  - copied prepared, compile, live, and optional hybrid records
  - copied deployment-download record

## What To Inspect First

1. `deployment_manifest.json`
   - package summary
   - target model id
   - compile options
   - source bundle manifest
2. `io_contract.json`
   - compiled input dtypes
   - compiled output shapes
   - special handling like `truncate_64bit_io`
3. `deploy_notes.md`
   - runtime split notes for CPU-side and compiled components

## Current Runtime Split Notes

### Zipformer

- compiled target applies to the encoder artifact
- decoder and joiner stay CPU-side

### VPCD

- compiled target applies to the model session artifact
- tokenizer encode and tokenizer decode stay CPU-side

## Boundary

This deployment packager proves:

- retained records are sufficient to resolve one deployable target
- the compiled artifact can be downloaded from AI Hub
- a deterministic package can be materialized with evidence and an I/O contract

It does not prove:

- BKMeeting asset sync
- ONNX Runtime / QNN session creation on Android
- physical Snapdragon validation inside the app
