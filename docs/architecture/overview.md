# Architecture Overview

`python-model-test/` is the Python workspace for exporting, verifying, quantizing, and smoke-testing the model bundles that are later handed off to BKMeeting Android.

## Supported model families

The current repo supports two model families:

- `vpcd`
  - punctuation / capitalization / denormalization
- `zipformer`
  - RNNT acoustic model

## What this repo owns

This repo owns the Python-side model lifecycle up to Android handoff:

- export shared model bundles through project-owned helpers
- verify bundle correctness against a Python reference runtime
- prepare calibration data
- quantize supported models
- build candidate bundles for QNN-oriented Android experiments
- run smoke tests in bundle-manifest mode
- sync verified bundles into BKMeeting asset packs

## What this repo does not own

This repo does not own physical-device runtime proof.

BKMeeting owns:

- Android asset packaging
- ONNX Runtime / QNN runtime integration
- provider selection
- QNN strict-device validation
- Snapdragon benchmark and promotion decisions

## Source-of-truth split

### `python-model-test` is the source of truth for

- bundle export
- bundle verification
- fixture generation
- calibration-subset extraction
- model quantization
- fixed-shape candidate preparation
- Python-side QNN preflight checks
- Android bundle sync handoff

### `BKMeeting` is the source of truth for

- manifest-driven Android loading
- staged runtime packaging on Android
- session/provider policy on Android
- strict HTP validation on physical Snapdragon devices

## Repository layout

```text
python-model-test/
  assets/                # source models, audio/text fixtures
  build/                 # generated artifacts from export and quantize flows
  docs/                  # canonical docs and historical plans
  src/
    aihub/              # AI Hub compile, evaluation, and deployment helpers
    model_bundle/       # thin manifest, fixture, and runtime keep-set
    quantize/           # quantization framework
    verify/             # verification CLIs
    tools/              # helper CLIs and Android sync scripts
  test/                  # smoke runners and pytest coverage
```

## Module map

- `src/aihub/`
  - retained AI Hub session, evaluation, and deployment helpers
- `src/model_bundle/`
  - shared manifest contract, fixtures, and the small runtime keep-set still used by AI Hub
- `src/quantize/`
  - quantization framework plus project-owned bundle export helpers
- `src/verify/`
  - bundle verification and QNN preflight ownership
- `src/tools/`
  - helper CLIs for manual bundle export, source ONNX refresh, calibration, candidate prep, and Android sync
- `test/`
  - smoke runners and contract-focused pytest suite

## Current bundle variants

### VPCD

- CPU-safe baseline:
  - `build/model_bundle/vpcd/vpcd_balanced`
- fixed-shape QNN candidate:
  - `build/model_bundle/vpcd/qnn_fixed_1024x128`

### Zipformer

- FP32 reference bundle:
  - `build/model_bundle/zipformer/fp32`
- quantized QNN-oriented candidate bundle:
  - `build/model_bundle/zipformer/qnn_u16u8`

## How to think about the repo

The repo has one central contract:

- a shared bundle manifest plus artifact layout that both Python and Android understand

Everything else serves that contract:

- tools or project-owned bundle helpers produce it
- verify checks it
- quantize prepares improved candidate artifacts for it
- smoke tests run through it
- Android handoff syncs it into BKMeeting

## Related docs

- `docs/architecture/bundle-contract.md`
- `docs/workflows/README.md`
- `docs/workflows/export-verify-smoke.md`
- `docs/workflows/quantize-qnn-candidates.md`
- `docs/workflows/aihub-overview.md`
- `docs/workflows/android-handoff.md`
