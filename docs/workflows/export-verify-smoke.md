# Export, Verify, And Smoke-Test Workflows

This document is the canonical current workflow for:

- exporting bundles
- verifying them
- smoke-testing them in Python

Run commands from the `python-model-test/` repo root.

## Quick setup

Install the repo in editable mode first:

```bash
python -m pip install -e .
```

## Flow A: Export and verify a VPCD bundle

### Step 1. Export the bundle

```bash
python -m export.model_bundle \
  --project vpcd \
  --model-dir assets/vietnamese-punc-cap-denorm-v1 \
  --output-dir build/model_bundle/vpcd/vpcd_balanced \
  --asset-namespace models/punctuation/vpcd/vpcd_balanced \
  --model-variant vpcd_balanced
```

Why this step exists:

- it creates the canonical punctuation bundle layout
- later verify, smoke, and Android sync flows all consume that same bundle

### Step 2. Verify the bundle against the source model

```bash
python -m verify.model_bundle \
  --project vpcd \
  --model-dir assets/vietnamese-punc-cap-denorm-v1 \
  --bundle-dir build/model_bundle/vpcd/vpcd_balanced
```

Why this step exists:

- it checks tokenizer encode/decode parity against the reference runtime
- it proves the bundle contract is internally coherent before any Android handoff

### Step 3. Smoke-test the bundle in manifest mode

```bash
python -m test.test_punctuation_model_onnx \
  --bundle-manifest build/model_bundle/vpcd/vpcd_balanced/bundle_manifest.json \
  --text "hom nay la buoi nham chuc cua toi phuoc thanh"
```

Why this step exists:

- it exercises the manifest-driven runtime path directly
- this is the closest Python-side analog to how Android consumes the bundle

## Flow B: Export and verify a Zipformer FP32 bundle

### Step 1. Export the FP32 reference bundle

```bash
python -m export.model_bundle \
  --project zipformer \
  --model-dir assets/zipformer \
  --output-dir build/model_bundle/zipformer/fp32 \
  --asset-namespace models/asr/zipformer/fp32 \
  --model-variant fp32
```

Why this step exists:

- it creates the reference bundle that later quantized candidates are compared against

### Step 2. Verify the FP32 bundle against the source runtime

```bash
python -m verify.model_bundle \
  --project zipformer \
  --model-dir assets/zipformer \
  --bundle-dir build/model_bundle/zipformer/fp32
```

Why this step exists:

- it checks transcript parity between the source runtime and the exported bundle runtime

### Step 3. Smoke-test the FP32 bundle in manifest mode

```bash
python -m test.test_acoustic_model_onnx \
  --bundle-manifest build/model_bundle/zipformer/fp32/bundle_manifest.json \
  --audio-file assets/speech/sample-2.wav
```

Why this step exists:

- it confirms the bundle can be consumed through the same manifest-oriented path Android expects

## Flow C: Verify a candidate bundle against a reference bundle

### Zipformer candidate vs FP32 reference

```bash
python -m verify.model_bundle \
  --project zipformer \
  --reference-bundle build/model_bundle/zipformer/fp32 \
  --candidate-bundle build/model_bundle/zipformer/qnn_u16u8
```

Why this step exists:

- candidate bundles should be compared against a known reference bundle, not only against raw source assets

### VPCD fixed-shape candidate vs dynamic reference

```bash
python -m verify.model_bundle \
  --project vpcd \
  --reference-bundle build/model_bundle/vpcd/vpcd_balanced \
  --candidate-bundle build/model_bundle/vpcd/qnn_fixed_1024x128
```

Why this step exists:

- the fixed-shape VPCD candidate changes packaging and shape assumptions
- this check proves the candidate still matches the intended punctuation behavior

## Rebuild source ONNX for punctuation when needed

If the source punctuation ONNX needs to be refreshed before bundle export:

```bash
python -m export.punctuation_onnx \
  --model-dir assets/vietnamese-punc-cap-denorm-v1 \
  --output-dir assets/vietnamese-punc-cap-denorm-v1/onnx
```

Use this only when the source ONNX itself needs regeneration.
Bundle export stays on `python -m export.model_bundle`.

## Full test suite

Run the full pytest suite with:

```bash
python -m pytest test -q -p no:cacheprovider
```

Why this step exists:

- the smoke runners prove real runtime behavior
- the pytest suite locks down the shared contract so refactors stay safe

## Related docs

- `docs/architecture/bundle-contract.md`
- `docs/workflows/quantize-qnn-candidates.md`
- `docs/workflows/android-handoff.md`
- `test/README.md`
