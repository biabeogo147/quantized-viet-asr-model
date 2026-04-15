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

Instead of maintaining one-off scripts per model, the repo is organized around five shared blocks:

- `src/export/`
  - entrypoints that create artifacts
- `src/verify/`
  - entrypoints that check parity and report mismatches
- `src/model_bundle/`
  - the shared bundle contract
- `src/quantize/`
  - the shared multi-project quantization framework
- `src/tools/`
  - small reusable CLI helpers and shared path utilities

## Repository layout

```text
python-model-test/
  assets/                # source models, audio/text fixtures
  build/                 # generated artifacts from export/quantize
  docs/                  # implementation plans and design notes
  src/
    export/             # export CLIs
    model_bundle/       # shared bundle contract
    quantize/           # quantization framework
    verify/             # verification CLIs
    tools/              # small helper scripts
  test/                  # smoke runners + pytest suites
```

## What each module does

### `src/export/`

Creates input artifacts:
- shared bundles for `vpcd` and `zipformer`
- source ONNX for punctuation

See `src/export/README.md` for details.

### `src/verify/`

Checks bundles:
- punctuation encode/decode parity
- Zipformer reference-vs-bundle parity
- Zipformer reference-vs-candidate parity

See `src/verify/README.md` for details.

### `src/model_bundle/`

Shared core for:
- manifests
- fixtures
- generic exporting and verification
- project adapters

See:
- `src/model_bundle/README.md`
- `src/model_bundle/projects/README.md`

### `src/quantize/`

Shared quantization framework:
- calibration
- presets
- QNN PTQ + QDQ helpers
- reports
- project adapters for `vpcd` and `zipformer`

See:
- `src/quantize/README.md`
- `src/quantize/projects/README.md`

### `src/tools/`

Contains small helper scripts that are useful across model workflows.

Important shared helper:
- `src/tools/paths.py`
  - resolves the repo root from any module under `src/`
  - converts repo-relative fixture paths such as `assets/speech/sample-1.mp3` into stable absolute paths
  - avoids fragile `Path(__file__).parents[...]` assumptions after refactors
- `src/tools/extract_vlsp2020_calibration_subset.py`
  - reads VLSP 2020 parquet shards
  - emits a deterministic audio subset for Zipformer and a matching transcription subset for VPCD

See `src/tools/README.md` for details.

### `test/`

Contains:
- canonical smoke runners
- pytest suites that lock the contract

See `test/README.md` for details.

## End-to-end pipeline

Command setup:
- `pytest` works from `python-model-test/` because `test/conftest.py` prepends `src/`.
- CLI and smoke-runner examples below assume you run commands from `python-model-test/`.

Why this matters:
- code under `src/` now resolves repo-relative assets through the shared helper in `src/tools/paths.py`
- moving packages around inside `src/` should no longer break audio/text fixture resolution

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

For VPCD:
- `python -m quantize --project vpcd ...`

Recommended shared calibration prep:

```bash
python -m tools.extract_vlsp2020_calibration_subset \
  --dataset-root <vlsp_dataset_root> \
  --max-samples 24 \
  --output-dir build/calibration/vlsp2020
```

This produces:
- `build/calibration/vlsp2020/zipformer_audio_manifest.txt`
- `build/calibration/vlsp2020/vpcd_transcriptions.txt`
- `build/calibration/vlsp2020/subset_manifest.json`

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

Tiny smoke runs already exercised in this repo:
- `vpcd`
  - quantized with `1` calibration text sample
  - quantized ONNX output matched the FP32 output on the smoke text
- `zipformer`
  - quantized with `1` calibration audio sample
  - exported candidate bundle ran successfully in `--bundle-manifest` mode
  - candidate bundle exact-matched the FP32 reference on the smoke audio

### 5. Hand off to Android

`vpcd` and `zipformer` do not hand off in the same way today:

- `vpcd`
  - Android already consumes the shared Python bundle format
  - export or refresh the punctuation bundle in `python-model-test`
  - copy the bundle files into `bkmeeting/modelassets/src/main/assets/models/punctuation/vpcd`
- `zipformer`
  - Python can export a bundle for verification and quantization work
  - the current Android ASR runtime still consumes raw `encoder` / `decoder` / `joiner` / `tokens.txt` assets, not `bundle_manifest.json`
  - sync those raw component files into `bkmeeting/modelassets/src/main/assets/models/asr/zipformer/<variant>`

Canonical punctuation handoff:

```bash
python -m export.model_bundle \
  --project vpcd \
  --model-dir assets/vietnamese-punc-cap-denorm-v1 \
  --output-dir build/model_bundle/vpcd/fp32 \
  --asset-namespace models/punctuation/vpcd \
  --model-variant vpcd_balanced

cp -R build/model_bundle/vpcd/fp32/. \
  ../bkmeeting/modelassets/src/main/assets/models/punctuation/vpcd/
```

After the copy:
- `bkmeeting` reads `models/punctuation/vpcd/bundle_manifest.json`
- the Android runtime copies the files into app-local storage and loads them through the manifest contract
- `bkmeeting/modelassets/README.md` is the canonical Android-side handoff document

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
-> optional Android handoff of raw ASR component files

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

## Small helper in `src/tools`

### `src/tools/convert_bpe2token.py`

Role:
- reads `assets/zipformer/bpe.model`
- generates `assets/zipformer/tokens.txt`

Use it when:
- the token table is missing
- you need to regenerate `tokens.txt` from the SentencePiece model

## Current status

- the shared bundle contract is now used by both `vpcd` and `zipformer`
- quantization supports both projects
- repo-relative asset resolution no longer depends on hardcoded parent-directory depth inside `src/`
- a tiny smoke quantize run has been verified for both `vpcd` and `zipformer`
- broader quality and parity still need larger calibration and evaluation sets than the current smoke run
