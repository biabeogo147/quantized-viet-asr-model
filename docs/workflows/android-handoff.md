# Android Handoff

This document explains how verified Python bundles are handed off into BKMeeting Android.

Use this file after:

- export and verification pass
- candidate-bundle checks pass
- Python-side QNN preflight passes where applicable

If you are working on the retained AI Hub path, read this after:

- `docs/workflows/aihub-overview.md`
- `docs/workflows/aihub-rerun.md`

## Responsibility split

### What `python-model-test` proves before handoff

- the bundle contract is valid
- the bundle can be consumed through Python manifest-mode smoke runners
- candidate bundles have passed Python-side comparison gates
- QNN-oriented metadata is present where the flow supports it

### What BKMeeting must still prove after handoff

- Android asset packaging is correct
- ONNX Runtime / QNN runtime packaging is correct
- provider selection on Android is correct
- strict-device validation on Snapdragon succeeds
- benchmark and promotion decisions are justified

## Canonical sync CLI

Use:

```bash
python -m tools.sync_android_bundle
```

This CLI:

- copies a verified bundle into the correct BKMeeting asset-pack path
- rewrites handoff fields in `bundle_manifest.json` when needed
- keeps variant-family paths aligned with Android naming

Current scope is bundle sync only. Retained AI Hub rerun evidence stays under `build/aihub/records/` and should be referenced separately in BKMeeting handoff notes.

The source bundle typically comes from one of these owners:

- `python -m tools.bundle_export ...` for manual baseline bundles
- `python -m quantize --project zipformer ...` for the retained quantized Zipformer candidate
- `python -m quantize --project vpcd ...` plus `python -m tools.prepare_vpcd_qnn_candidate ...` for the retained VPCD candidate path

For the retained AI Hub path, start the handoff review from:

- `build/aihub/deploy/zipformer/<RUN_LABEL>/`
- `build/aihub/deploy/vpcd/<RUN_LABEL>/`

Those deployment packages currently freeze:

- the downloaded compiled artifact
- retained evidence copies
- `io_contract.json`
- deploy notes for the CPU / compiled runtime split

The current sync CLI still handles bundle sync only.
It does not yet consume deployment packages directly or replace the future Android-ready AI Hub bundle lane.

## Supported handoff targets

### VPCD

- `vpcd_balanced`
  - `models/punctuation/vpcd/vpcd_balanced`
- `qnn_fixed_1024x128`
  - `models/punctuation/vpcd/qnn_fixed_1024x128`

### Zipformer

- `fp32`
  - `models/asr/zipformer/fp32`
- `qnn_u16u8`
  - `models/asr/zipformer/qnn_u16u8`

## Common handoff commands

### Sync VPCD baseline

```bash
python -m tools.sync_android_bundle \
  --project vpcd \
  --variant vpcd_balanced \
  --bkmeeting-root <BKMEETING_ROOT> \
  --overwrite
```

### Sync VPCD fixed-shape candidate

```bash
python -m tools.sync_android_bundle \
  --project vpcd \
  --variant qnn_fixed_1024x128 \
  --bkmeeting-root <BKMEETING_ROOT> \
  --overwrite
```

### Sync Zipformer FP32 reference

```bash
python -m tools.sync_android_bundle \
  --project zipformer \
  --variant fp32 \
  --bkmeeting-root <BKMEETING_ROOT> \
  --overwrite
```

### Sync Zipformer quantized candidate

```bash
python -m tools.sync_android_bundle \
  --project zipformer \
  --variant qnn_u16u8 \
  --bkmeeting-root <BKMEETING_ROOT> \
  --overwrite
```

## Why the handoff step exists

The Python repo and the Android repo are deliberately separated.

That means:

- Python can iterate on export and quantization without touching Android code
- Android can consume one canonical bundle layout instead of one-off copied files

The handoff step is the bridge between those responsibilities.

## What to record at handoff time

When handing off a candidate, record:

- project
- variant
- source bundle path
- deployment package path when the handoff comes from the retained AI Hub path
- destination asset namespace
- whether the bundle is baseline or QNN-oriented
- any QNN preflight report path and candidate report files

## Current Android-facing interpretations

### VPCD

- `vpcd_balanced` remains the CPU-safe baseline
- `qnn_fixed_1024x128` is the intended first Android QNN candidate
- tokenizer graphs stay CPU-only in the first slice

### Zipformer

- `fp32` is the reference Android bundle
- `qnn_u16u8` is the quantized Android candidate
- fixed-shape encoder metadata is part of the handoff contract

## What not to claim after handoff

Do not claim NPU support is complete just because the sync succeeded.

Sync only proves:

- the Python-side artifacts were transferred
- Android now has the bundle assets to attempt runtime integration
- the retained deployment package was available as the Python-side deployment input if this was an AI Hub handoff

It does not prove:

- QNN runtime packaging
- HTP session creation
- NPU execution

## Related docs

- `docs/workflows/export-verify-smoke.md`
- `docs/workflows/aihub-overview.md`
- `docs/workflows/aihub-rerun.md`
- `docs/qnn/model-quantization.md`
- `docs/qnn/preflight.md`
- `src/tools/README.md`
