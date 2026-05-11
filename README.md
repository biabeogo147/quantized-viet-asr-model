# Python Model Test Repo

`python-model-test/` is the Python workspace for exporting, verifying, quantizing, and smoke-testing the model bundles that are later handed off to BKMeeting Android.

## What this repo does

The repo currently supports two model families:

- `vpcd`
  - punctuation / capitalization / denormalization
- `zipformer`
  - RNNT acoustic model

Current responsibilities:

- export shared model bundles
- verify those bundles against Python reference runtimes
- prepare calibration inputs
- quantize supported models
- build QNN-oriented candidate bundles
- smoke-test bundle-manifest consumption
- sync verified bundles into BKMeeting Android assets

## Repository layout

```text
python-model-test/
  assets/
  build/
  docs/
  src/
  test/
```

## Quick start

Install the repo in editable mode:

```bash
python -m pip install -e .
```

Run commands from `python-model-test/`.

## Read this next

If you are new to the repo, start here:

1. `docs/README.md`
2. `docs/architecture/overview.md`
3. `docs/workflows/export-verify-smoke.md`
4. `docs/workflows/quantize-qnn-candidates.md`
5. `docs/workflows/android-handoff.md`

## Module docs

- `src/export/README.md`
- `src/model_bundle/README.md`
- `src/model_bundle/projects/README.md`
- `src/quantize/README.md`
- `src/quantize/projects/README.md`
- `src/verify/README.md`
- `src/tools/README.md`
- `test/README.md`
