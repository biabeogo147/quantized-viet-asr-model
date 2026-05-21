# Android Handoff

This document explains how verified Python bundles are handed off into BKMeeting Android.

Use this file after:

- export and verification pass
- candidate-bundle checks pass
- Python-side QNN preflight passes where applicable
- AI Hub deployment packaging already exists when the handoff starts from retained compile output

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

Current scope stays intentionally narrow: sync receives one Android-ready bundle and copies it into the right BKMeeting asset namespace.

The source bundle typically comes from one of these owners:

- `python -m tools.bundle_export ...` for manual baseline bundles
- `python -m quantize --project zipformer ...` for the retained quantized Zipformer candidate
- `python -m quantize --project vpcd ...` plus `python -m tools.prepare_vpcd_qnn_candidate ...` for the retained VPCD candidate path
- `python -m aihub.android_bundle ...` for the retained AI Hub precompiled ONNX lane

For the retained AI Hub path, start the handoff review from the Step 5 deployment package:

- `build/aihub/deploy/zipformer/<RUN_LABEL>/`
- `build/aihub/deploy/vpcd/<RUN_LABEL>/`

Those deployment packages currently freeze:

- the downloaded compiled artifact
- retained evidence copies
- `io_contract.json`
- deploy notes for the CPU / compiled runtime split

Then materialize the Android-ready bundle:

```bash
python -m aihub.android_bundle \
  --project all \
  --run-label 20260519-6pm \
  --repo-root . \
  --device-name "Samsung Galaxy S24 (Family)" \
  --qairt-version 2.46.0 \
  --overwrite
```

This Step 6 synthesis creates:

- `build/aihub/android_bundle/zipformer/<RUN_LABEL>/`
- `build/aihub/android_bundle/vpcd/<RUN_LABEL>/`

Those bundle roots are the direct inputs to `tools.sync_android_bundle`.

## Supported handoff targets

### VPCD

- `vpcd_balanced`
  - `models/punctuation/vpcd/vpcd_balanced`
- `qnn_fixed_1024x128`
  - `models/punctuation/vpcd/qnn_fixed_1024x128`
- `precompiled_qnn_onnx`
  - `models/punctuation/vpcd/precompiled_qnn_onnx`

### Zipformer

- `fp32`
  - `models/asr/zipformer/fp32`
- `qnn_u16u8`
  - `models/asr/zipformer/qnn_u16u8`
- `precompiled_qnn_onnx`
  - `models/asr/zipformer/precompiled_qnn_onnx`

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

### Materialize AI Hub Android bundles

```bash
python -m aihub.android_bundle \
  --project all \
  --run-label 20260519-6pm \
  --repo-root . \
  --device-name "Samsung Galaxy S24 (Family)" \
  --qairt-version 2.46.0 \
  --overwrite
```

### Sync Zipformer precompiled ONNX lane

```bash
python -m tools.sync_android_bundle \
  --project zipformer \
  --variant precompiled_qnn_onnx \
  --source-bundle build/aihub/android_bundle/zipformer/20260519-6pm \
  --bkmeeting-root <BKMEETING_ROOT> \
  --overwrite
```

### Sync VPCD precompiled ONNX lane

```bash
python -m tools.sync_android_bundle \
  --project vpcd \
  --variant precompiled_qnn_onnx \
  --source-bundle build/aihub/android_bundle/vpcd/20260519-6pm \
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
- Android-ready bundle path when the handoff comes from `python -m aihub.android_bundle`
- destination asset namespace
- whether the bundle is baseline or QNN-oriented
- any QNN preflight report path and candidate report files
- whether the bundle carries `model.bin` external data and `io_contract.json`

## Current Android-facing interpretations

### VPCD

- `vpcd_balanced` remains the CPU-safe baseline
- `qnn_fixed_1024x128` is the intended first Android QNN candidate
- tokenizer graphs stay CPU-only in the first slice
- `precompiled_qnn_onnx` stages the compiled model-session lane from AI Hub
- `model.mobile.onnx` must stay beside `model.bin`
- Android runtime reads `io_contract.json` before narrowing integer tensors to `int32`

### Zipformer

- `fp32` is the reference Android bundle
- `qnn_u16u8` is the quantized Android candidate
- fixed-shape encoder metadata is part of the handoff contract
- `precompiled_qnn_onnx` stages the compiled encoder lane from AI Hub
- `encoder.onnx` must stay beside `model.bin`
- Android runtime reads `io_contract.json` before narrowing `x_lens` to `int32`

## What not to claim after handoff

Do not claim NPU support is complete just because the sync succeeded.

Sync only proves:

- the Python-side artifacts were transferred
- Android now has the bundle assets to attempt runtime integration
- the retained deployment package was available as the Python-side deployment input if this was an AI Hub handoff
- the Android-ready bundle now includes any required external data and I/O contract files

It does not prove:

- QNN runtime packaging
- HTP session creation
- NPU execution

## Related docs

- `docs/workflows/export-verify-smoke.md`
- `docs/workflows/aihub-overview.md`
- `docs/workflows/aihub-rerun.md`
- `docs/workflows/aihub-deployment.md`
- `docs/qnn/model-quantization.md`
- `docs/qnn/preflight.md`
- `src/tools/README.md`
