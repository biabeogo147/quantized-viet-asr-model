# Quantize And Prepare QNN Candidates

This document is the canonical current workflow for:

- preparing shared calibration data
- quantizing VPCD
- building the fixed-shape VPCD candidate
- quantizing Zipformer into the `qnn_u16u8` candidate bundle

Run commands from the `python-model-test/` repo root.

## Step 1. Prepare a shared calibration subset

```bash
python -m tools.extract_vlsp2020_calibration_subset \
  --dataset-root <VLSP_DATASET_ROOT> \
  --max-samples 24 \
  --output-dir build/calibration/vlsp2020
```

Outputs:

- `build/calibration/vlsp2020/zipformer_audio_manifest.txt`
- `build/calibration/vlsp2020/vpcd_transcriptions.txt`
- `build/calibration/vlsp2020/subset_manifest.json`

Why this step exists:

- both supported projects can reuse the same deterministic external dataset slice
- it keeps calibration inputs reproducible across runs

## Flow A: Build the VPCD QNN-oriented candidate

### Step 2. Quantize VPCD into the balanced QDQ artifact

```bash
python -m quantize \
  --project vpcd \
  --preset sd8g2_balanced \
  --calibration-text build/calibration/vlsp2020/vpcd_transcriptions.txt \
  --max-calibration-samples 24 \
  --output build/vpcd/vpcd_balanced.onnx
```

Why this step exists:

- this is the QDQ artifact used by the active `vpcd_balanced` bundle
- quantization and fixed-shape freezing are intentionally separate concerns

### Step 3. Re-export the VPCD bundle

```bash
python -m export.model_bundle \
  --project vpcd \
  --model-dir assets/vietnamese-punc-cap-denorm-v1 \
  --output-dir build/model_bundle/vpcd/vpcd_balanced \
  --asset-namespace models/punctuation/vpcd/vpcd_balanced \
  --model-variant vpcd_balanced
```

Why this step exists:

- the bundle manifest must carry the updated quantization metadata, not just the raw ONNX file

### Step 4. Build the fixed-shape VPCD candidate

```bash
python -m tools.prepare_vpcd_qnn_candidate \
  --source-bundle build/model_bundle/vpcd/vpcd_balanced \
  --output-dir build/model_bundle/vpcd/qnn_fixed_1024x128 \
  --encoder-sequence 1024 \
  --decoder-sequence 128
```

Why this step exists:

- the balanced VPCD bundle still uses dynamic shapes
- the first Android QNN slice for VPCD needs a fixed-shape candidate for `model.mobile.onnx`

### Step 5. Verify the fixed-shape candidate against the dynamic reference bundle

```bash
python -m verify.model_bundle \
  --project vpcd \
  --reference-bundle build/model_bundle/vpcd/vpcd_balanced \
  --candidate-bundle build/model_bundle/vpcd/qnn_fixed_1024x128
```

Why this step exists:

- fixed-shape rewriting should not silently change punctuation behavior

### Step 6. Run QNN preflight on the fixed-shape candidate

```bash
python -m verify.qnn_preflight \
  --project vpcd \
  --bundle-dir build/model_bundle/vpcd/qnn_fixed_1024x128 \
  --output build/model_bundle/vpcd/qnn_fixed_1024x128/qnn_preflight_report.json
```

Why this step exists:

- it confirms the candidate bundle has the expected QDQ metadata, fixed shapes, and tokenizer CPU policy
- it is the Python-side gate before Android QNN handoff

## Flow B: Build the Zipformer QNN-oriented candidate

### Step 2. Export or refresh the FP32 reference bundle first

```bash
python -m export.model_bundle \
  --project zipformer \
  --model-dir assets/zipformer \
  --output-dir build/model_bundle/zipformer/fp32 \
  --asset-namespace models/asr/zipformer/fp32 \
  --model-variant fp32
```

Why this step exists:

- the quantized candidate is evaluated against a reference bundle, not only against raw model files

### Step 3. Quantize Zipformer into the candidate bundle

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

Why this step exists:

- this flow collects audio calibration data
- freezes shapes per component
- performs QNN-oriented PTQ + QDQ
- exports the candidate bundle directly
- writes quantization and evaluation reports

### Step 4. Verify the candidate against the FP32 reference

```bash
python -m verify.model_bundle \
  --project zipformer \
  --reference-bundle build/model_bundle/zipformer/fp32 \
  --candidate-bundle build/model_bundle/zipformer/qnn_u16u8
```

Why this step exists:

- quantized candidate quality should be checked against the reference bundle before Android handoff

### Step 5. Smoke-test the quantized bundle

```bash
python -m test.test_acoustic_model_onnx \
  --bundle-manifest build/model_bundle/zipformer/qnn_u16u8/bundle_manifest.json \
  --audio-file assets/speech/sample-2.wav
```

Why this step exists:

- it proves the quantized candidate is runnable through the same manifest-driven path Android will consume

## Acceptance notes

### VPCD

The fixed-shape VPCD candidate is ready for Android QNN handoff only when:

- bundle verification passes
- QNN preflight passes
- tokenizer policy remains CPU-only for the first slice

### Zipformer

The `qnn_u16u8` candidate bundle is acceptable for Android experimentation when:

- the bundle is runnable
- it avoids `Decode error:`
- it produces usable, non-empty transcripts

Exact FP32 transcript parity is not the strict gate for the quantized runtime candidate.

## What these flows do not prove

These Python flows do not prove:

- physical Snapdragon HTP execution
- strict Android QNN provider ownership
- production benchmark wins

Those are BKMeeting-side validation steps.

## Related docs

- `docs/qnn/preflight.md`
- `docs/workflows/android-handoff.md`
- `src/quantize/README.md`
- `src/quantize/projects/README.md`
