# Android Handoff

This document explains how verified Python bundles are handed off into BKMeeting Android.

Use this file after:

- export and verification pass
- candidate-bundle checks pass
- Python-side QNN preflight passes where applicable

If you are working on the retained `Option 1` AI Hub path, read this after:

- `docs/workflows/option1-promotion-handoff.md`

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

The next planned extension is contract-aware `Option 1` sync, where the same CLI will also stage packaged `Phase 5` evidence under the Android asset namespace.

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

It does not prove:

- QNN runtime packaging
- HTP session creation
- NPU execution

## Related docs

- `docs/workflows/export-verify-smoke.md`
- `docs/workflows/option1-overview.md`
- `docs/workflows/option1-promotion-handoff.md`
- `docs/qnn/model-quantization.md`
- `docs/qnn/preflight.md`
- `src/tools/README.md`
