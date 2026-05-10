# Python Model Test Repo

`python-model-test/` is the Python workspace for exporting, verifying, quantizing, and smoke-testing the ONNX models that are later handed off to BKMeeting.

## What this repo does today

The repo currently supports two model families:

- `vpcd`
  - punctuation / capitalization / denormalization
  - bundle-backed runtime already used by BKMeeting Android
- `zipformer`
  - RNNT acoustic model
  - FP32 and quantized bundle flows are implemented
  - bundle-backed runtime is already used by BKMeeting Android
  - Qualcomm QNN / NPU offload is not wired yet

What is already working in this repo:

- export shared model bundles for `vpcd` and `zipformer`
- verify bundle behavior against the Python reference runtime
- quantize `vpcd`
- quantize `zipformer` into a `qnn_u16u8` candidate bundle
- smoke-test both projects in `--bundle-manifest` mode
- hand off bundles into BKMeeting Android asset packs

Current Android-facing status:

- `vpcd` bundle handoff is working
- `zipformer/fp32` bundle handoff is working
- `zipformer/qnn_u16u8` bundle handoff is working
- the Android runtime already honors fixed-shape metadata for `zipformer/qnn_u16u8`
- actual Snapdragon HTP / NPU execution is a separate next phase

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
    tools/              # helper scripts
  test/                  # smoke runners + pytest suites
```

## Quick start

This repo uses a `src/` layout, so install it in editable mode before running the CLIs:

```bash
python -m pip install -e .
```

Run commands from `python-model-test/`.

## Main modules

- `src/export/`
  - exports source ONNX and shared model bundles
- `src/model_bundle/`
  - defines the shared manifest + artifact contract
- `src/verify/`
  - verifies bundle parity
- `src/quantize/`
  - quantizes `vpcd` and `zipformer`
- `src/tools/`
  - helper CLIs such as VLSP calibration-subset extraction
- `test/`
  - canonical smoke runners and pytest coverage

See the per-module READMEs under `src/` and `test/` for file-by-file details.

## Bundle contract

Both supported projects use the same high-level bundle idea:

- `bundle_manifest.json`
- runtime artifacts listed under `artifacts`
- optional fixtures listed under `fixtures`

### VPCD bundle layout

```text
build/model_bundle/vpcd/<variant>/
  bundle_manifest.json
  model.mobile.onnx
  tokenizer.encode.onnx
  tokenizer.decode.onnx
  tokenizer.to_model_id_map.json
  tokenizer.from_model_id_map.json
  golden_samples.jsonl
```

### Zipformer bundle layout

```text
build/model_bundle/zipformer/<variant>/
  bundle_manifest.json
  encoder.onnx
  decoder.onnx
  joiner.onnx
  tokens.txt
  sample_manifest.jsonl
  expected_outputs.jsonl
```

The quantized Zipformer candidate bundle also includes:

```text
quantization_report.json
evaluation_report.json
```

## How to export models

### 1. Export a VPCD bundle

This is the canonical shared bundle export path for punctuation. The current default `vpcd_balanced` variant is the QNN-targeted PTQ/QDQ artifact, not FP32.

```bash
python -m export.model_bundle \
  --project vpcd \
  --model-dir assets/vietnamese-punc-cap-denorm-v1 \
  --output-dir build/model_bundle/vpcd/vpcd_balanced \
  --asset-namespace models/punctuation/vpcd \
  --model-variant vpcd_balanced
```

This produces:

- `bundle_manifest.json`
- `model.mobile.onnx`
- tokenizer ONNX graphs
- tokenizer ID bridge maps
- `golden_samples.jsonl`

For `vpcd_balanced`, the manifest also records:

- `metadata.quantization.format = "QDQ"`
- `metadata.quantization.activation_type = "quint16"`
- `metadata.quantization.weight_type = "quint8"`
- `metadata.quantization.fixed_shapes = false`
- `metadata.qnn_readiness.tokenizer_policy = "cpu_only_first_slice"`

`fixed_shapes = false` is intentional: the artifact is quantized QDQ, but VPCD still needs a fixed-shape export or session policy before strict QNN HTP validation.

To prepare the fixed-shape VPCD candidate for Android QNN preflight:

```bash
python -m tools.prepare_vpcd_qnn_candidate \
  --source-bundle build/model_bundle/vpcd/vpcd_balanced \
  --output-dir build/model_bundle/vpcd/qnn_fixed_1024x128 \
  --encoder-sequence 1024 \
  --decoder-sequence 128
```

Then run the QNN preflight checker:

```bash
python -m verify.qnn_preflight \
  --project vpcd \
  --bundle-dir build/model_bundle/vpcd/qnn_fixed_1024x128 \
  --output build/model_bundle/vpcd/qnn_fixed_1024x128/qnn_preflight_report.json
```

When this passes, the candidate has Python-verified QDQ metadata, fixed input shapes, and tokenizer CPU policy. It still does not prove physical Snapdragon HTP execution; that validation happens in BKMeeting's Android QNN runtime phase.

If you need to rebuild the source punctuation ONNX before bundling:

```bash
python -m export.punctuation_onnx \
  --model-dir assets/vietnamese-punc-cap-denorm-v1 \
  --output-dir assets/vietnamese-punc-cap-denorm-v1/onnx
```

### 2. Export a Zipformer FP32 bundle

This is the canonical FP32 bundle export path for the acoustic model:

```bash
python -m export.model_bundle \
  --project zipformer \
  --model-dir assets/zipformer \
  --output-dir build/model_bundle/zipformer/fp32 \
  --asset-namespace models/asr/zipformer/fp32 \
  --model-variant fp32
```

This produces:

- `bundle_manifest.json`
- `encoder.onnx`
- `decoder.onnx`
- `joiner.onnx`
- `tokens.txt`
- `sample_manifest.jsonl`
- `expected_outputs.jsonl`

## How to verify bundles

### Verify a VPCD bundle against the source model

```bash
python -m verify.model_bundle \
  --project vpcd \
  --model-dir assets/vietnamese-punc-cap-denorm-v1 \
  --bundle-dir build/model_bundle/vpcd/vpcd_balanced
```

What this checks:

- tokenizer encode parity
- tokenizer decode parity
- bundle contract correctness

### Verify a Zipformer FP32 bundle against the source model

```bash
python -m verify.model_bundle \
  --project zipformer \
  --model-dir assets/zipformer \
  --bundle-dir build/model_bundle/zipformer/fp32
```

What this checks:

- reference transcript from `model-dir`
- transcript from the exported bundle
- per-sample mismatches if they diverge

### Verify a Zipformer candidate bundle against the FP32 reference bundle

```bash
python -m verify.model_bundle \
  --project zipformer \
  --reference-bundle build/model_bundle/zipformer/fp32 \
  --candidate-bundle build/model_bundle/zipformer/qnn_u16u8
```

### Verify a fixed-shape VPCD candidate against the dynamic VPCD reference bundle

```bash
python -m verify.model_bundle \
  --project vpcd \
  --reference-bundle build/model_bundle/vpcd/vpcd_balanced \
  --candidate-bundle build/model_bundle/vpcd/qnn_fixed_1024x128
```

## How to quantize models

### 1. Prepare a shared calibration subset

If you want one external dataset to feed both `vpcd` and `zipformer`:

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

### 2. Quantize VPCD

The current VPCD quantize flow produces the quantized ONNX artifact used by the active `vpcd_balanced` bundle:

```bash
python -m quantize \
  --project vpcd \
  --preset sd8g2_balanced \
  --calibration-text build/calibration/vlsp2020/vpcd_transcriptions.txt \
  --max-calibration-samples 24 \
  --output build/vpcd/vpcd_balanced.onnx
```

Current balanced outputs:

- `build/vpcd/vpcd_balanced.onnx`
- `build/vpcd/fp32_vs_balanced_report.json`

After refreshing the artifact, rerun the VPCD bundle export command so `build/model_bundle/vpcd/vpcd_balanced/bundle_manifest.json` carries the QDQ and QNN-readiness metadata.

VPCD quantization and fixed-shape QNN preflight are separate steps:

1. `quantize --project vpcd` refreshes the QDQ ONNX artifact.
2. `export.model_bundle --project vpcd` packages the QDQ model with tokenizer artifacts.
3. `tools.prepare_vpcd_qnn_candidate` freezes model input shapes and updates manifest metadata.
4. `verify.qnn_preflight` confirms the candidate is ready for an Android QNN attempt.

### 3. Quantize Zipformer

The current Zipformer quantize flow produces a quantized candidate bundle directly:

```bash
python -m quantize \
  --project zipformer \
  --preset zipformer_sd8g2_balanced \
  --audio-manifest build/calibration/vlsp2020/zipformer_audio_manifest.txt \
  --output-root build/quantize/zipformer/qnn_u16u8 \
  --bundle-output-dir build/model_bundle/zipformer/qnn_u16u8 \
  --reference-bundle-dir build/model_bundle/zipformer/fp32 \
  --calibration-chunk-size 4
```

What this flow does:

- loads calibration audio
- traces encoder / decoder / joiner calibration records
- freezes fixed input shapes
- runs QNN-style PTQ + QDQ per component
- exports a `qnn_u16u8` bundle
- verifies that candidate bundle against the FP32 reference bundle
- writes quantization and evaluation reports

## How to use a bundle

There are two main ways to consume the bundles generated by this repo.

### 1. Use the bundle directly in Python

The canonical smoke runners both support `--bundle-manifest`.

#### VPCD bundle smoke test

```bash
python -m test.test_punctuation_model_onnx \
  --bundle-manifest build/model_bundle/vpcd/vpcd_balanced/bundle_manifest.json \
  --text "hom nay la buoi nham chuc cua toi phuoc thanh"
```

#### Zipformer bundle smoke test

```bash
python -m test.test_acoustic_model_onnx \
  --bundle-manifest build/model_bundle/zipformer/qnn_u16u8/bundle_manifest.json \
  --audio-file assets/speech/sample-2.wav
```

In `--bundle-manifest` mode, the smoke tests exercise the same manifest-driven behavior that Android is expected to consume.

### 2. Hand the bundle off to BKMeeting Android

#### VPCD Android handoff

```bash
python -m tools.sync_android_bundle \
  --project vpcd \
  --variant vpcd_balanced \
  --bkmeeting-root ../BKMeeting \
  --overwrite
```

For the fixed-shape QNN preflight candidate, use the optional validation asset pack:

```bash
python -m tools.sync_android_bundle \
  --project vpcd \
  --variant qnn_fixed_1024x128 \
  --bkmeeting-root ../BKMeeting \
  --qnn-validation-assets \
  --overwrite
```

This keeps the 800MB fixed-shape candidate out of the baseline `modelassets` pack. Tokenizer graphs still run on CPU in the first QNN slice.

#### Zipformer FP32 Android handoff

```bash
python -m tools.sync_android_bundle \
  --project zipformer \
  --variant fp32 \
  --bkmeeting-root ../BKMeeting \
  --overwrite
```

#### Zipformer QNN candidate Android handoff

```bash
python -m tools.sync_android_bundle \
  --project zipformer \
  --variant qnn_u16u8 \
  --bkmeeting-root ../BKMeeting \
  --overwrite
```

After the sync:

- BKMeeting reads `bundle_manifest.json`
- Android stages bundle files into local app storage
- the runtime opens local staged artifact paths from the manifest

Important current note:

- `zipformer/qnn_u16u8` already runs as a manifest-driven Android bundle
- Android already honors its fixed-shape encoder metadata
- this does not yet mean the model is running on Snapdragon NPU
- real QNN / HTP / NPU execution is still a separate integration step

## Recommended end-to-end flows

### Flow A: Export and verify a fresh FP32 bundle

1. export the bundle
2. verify the bundle against the source model
3. run a smoke test in `--bundle-manifest` mode
4. sync the bundle into BKMeeting with `tools.sync_android_bundle`

### Flow B: Build a quantized Zipformer candidate

1. prepare calibration data
2. export or refresh the FP32 reference bundle
3. run `python -m quantize --project zipformer`
4. verify the candidate bundle
5. smoke-test the candidate bundle
6. sync the candidate bundle into BKMeeting with `tools.sync_android_bundle`

## Tests

### Run the smoke runners

```bash
python -m test.test_punctuation_model_onnx --help
```

```bash
python -m test.test_acoustic_model_onnx --help
```

### Run the full pytest suite

```bash
python -m pytest test -q -p no:cacheprovider
```

## Important outputs

### Export outputs

- `build/model_bundle/vpcd/<variant>/...`
- `build/model_bundle/zipformer/fp32/...`

### Quantize outputs

- `build/vpcd/vpcd_balanced.onnx`
- `build/vpcd/fp32_vs_balanced_report.json`
- `build/model_bundle/zipformer/qnn_u16u8/...`

### Android handoff targets

- `../BKMeeting/modelassets/src/main/assets/models/punctuation/vpcd`
- `../BKMeeting/qnnvalidationassets/src/main/assets/models/punctuation/vpcd/qnn_fixed_1024x128`
- `../BKMeeting/modelassets/src/main/assets/models/asr/zipformer/fp32`
- `../BKMeeting/modelassets/src/main/assets/models/asr/zipformer/qnn_u16u8`

## Related READMEs

- `src/export/README.md`
- `src/model_bundle/README.md`
- `src/quantize/README.md`
- `src/verify/README.md`
- `src/tools/README.md`
- `test/README.md`
