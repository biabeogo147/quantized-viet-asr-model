# Python Model Test Repo

`python-model-test/` is the Python workspace for:

- exporting ONNX from source models
- packaging model bundles against a shared contract
- verifying parity between reference runtimes and bundle runtimes
- quantizing models for later deployment
- smoke-testing artifacts before they are handed off to Android

## What this repo is for

The repo currently focuses on two main model families:

- `vpcd`
  - punctuation / capitalization / denormalization seq2seq model
  - output is restored, formatted text
- `zipformer`
  - RNNT acoustic model
  - output is a transcript

Instead of maintaining one-off scripts per model, the repo is organized around four shared blocks:

- `export/`
  - entrypoints that create artifacts
- `verify/`
  - entrypoints that check parity and report mismatches
- `model_bundle/`
  - the shared bundle contract
- `quantize/`
  - the shared multi-project quantization framework

## Repository layout

```text
python-model-test/
  assets/                # source models, audio/text fixtures
  build/                 # generated artifacts from export/quantize
  docs/                  # implementation plans and design notes
  export/                # export CLIs
  model_bundle/          # shared bundle contract
  quantize/              # quantization framework
  test/                  # smoke runners + pytest suites
  verify/                # verification CLIs
  convert_bpe2token.py   # helper that generates tokens.txt from bpe.model
```

## What each module does

### `export/`

Creates input artifacts:
- shared bundles for `vpcd` and `zipformer`
- source ONNX for punctuation

See `export/README.md` for details.

### `verify/`

Checks bundles:
- punctuation encode/decode parity
- Zipformer reference-vs-bundle parity
- Zipformer reference-vs-candidate parity

See `verify/README.md` for details.

### `model_bundle/`

Shared core for:
- manifests
- fixtures
- generic exporting and verification
- project adapters

See:
- `model_bundle/README.md`
- `model_bundle/projects/README.md`

### `quantize/`

Shared quantization framework:
- calibration
- presets
- QNN PTQ + QDQ helpers
- reports
- project adapters for `vpcd` and `zipformer`

See:
- `quantize/README.md`
- `quantize/projects/README.md`

### `test/`

Contains:
- canonical smoke runners
- pytest suites that lock the contract
- legacy reference scripts used for behavior comparison

See `test/README.md` for details.

## End-to-end pipeline

### 1. Export source artifacts

Punctuation:
- if you need to rebuild source ONNX, run `python -m export.punctuation_onnx`

Bundles:
- run `python -m export.model_bundle --project <project>`

Outputs:
- `build/model_bundle/vpcd/...`
- `build/model_bundle/zipformer/...`

### 2. Verify reference bundles

Punctuation:
- `python -m verify.model_bundle --project vpcd --model-dir ... --bundle-dir ...`

Zipformer FP32:
- `python -m verify.model_bundle --project zipformer --model-dir ... --bundle-dir ...`

Goal:
- the bundle runtime must match the behavior of the reference runtime

### 3. Quantize a candidate bundle

For Zipformer:
- `python -m quantize --project zipformer ...`

The internal pipeline:
- collect calibration audio
- freeze fixed shapes
- run QNN PTQ + QDQ per component
- export the `qnn_u16u8` candidate bundle
- verify the candidate against the reference bundle
- write `quantization_report.json` and `evaluation_report.json`

### 4. Smoke-test artifacts

Punctuation:
- `python -m test.test_punctuation_model_onnx --bundle-manifest ...`

Zipformer:
- `python -m test.test_acoustic_model_onnx --bundle-manifest ...`

Goal:
- confirm the artifact really runs end to end

### 5. Hand off to Android

Once a reference or candidate bundle is ready:
- sync the bundle into `bkmeeting/modelassets`
- Android consumes the same `bundle_manifest.json` and exported layout produced by Python

## Concrete pipelines

### VPCD

`model-dir`
-> Hugging Face tokenizer + ONNX model
-> exported tokenizer ONNX + bridge maps
-> `vpcd/fp32` bundle
-> encode/decode parity verification
-> bundle-only smoke run

### Zipformer

`model-dir`
-> exported FP32 reference bundle
-> bundle-vs-model-dir verification
-> component-wise quantization into `qnn_u16u8`
-> exported candidate bundle
-> candidate-vs-reference verification
-> smoke run of the quantized bundle

## Important build artifacts

### Punctuation

```text
build/model_bundle/vpcd/fp32/
  bundle_manifest.json
  model.mobile.onnx
  tokenizer.encode.onnx
  tokenizer.decode.onnx
  tokenizer.to_model_id_map.json
  tokenizer.from_model_id_map.json
  golden_samples.jsonl
```

### Zipformer reference

```text
build/model_bundle/zipformer/fp32/
  bundle_manifest.json
  encoder.onnx
  decoder.onnx
  joiner.onnx
  tokens.txt
  sample_manifest.jsonl
  expected_outputs.jsonl
```

### Zipformer quantized candidate

```text
build/model_bundle/zipformer/qnn_u16u8/
  bundle_manifest.json
  encoder.onnx
  decoder.onnx
  joiner.onnx
  tokens.txt
  sample_manifest.jsonl
  expected_outputs.jsonl
  quantization_report.json
  evaluation_report.json
```

## Small helper at repo root

### `convert_bpe2token.py`

Role:
- reads `assets/zipformer/bpe.model`
- generates `assets/zipformer/tokens.txt`

Use it when:
- the token table is missing
- you need to regenerate `tokens.txt` from the SentencePiece model

## Current status

- the shared bundle contract is now used by both `vpcd` and `zipformer`
- quantization supports both projects
- the `zipformer/qnn_u16u8` candidate bundle is runnable
- that candidate still does not exact-match the FP32 reference on the current sample set, so more tuning is still needed if strict parity is the goal
