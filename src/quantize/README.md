# Quantize Module

`src/quantize/` now keeps only the two retained quantization lanes used by this repo:

- `zipformer`: local QNN-oriented PTQ/QDQ bundle generation
- `vpcd`: local AIMET service-backed quantize producer

Everything tied only to retired VPCD lanes has been removed.

## Entry point

Use:

```bash
python -m quantize --project <zipformer|vpcd> ...
```

`quantize/cli.py` resolves the project adapter and hands control to:

- [projects/zipformer.py](/D:/DS-AI/BKMeeting-Research/python-model-test/src/quantize/projects/zipformer.py)
- [projects/vpcd.py](/D:/DS-AI/BKMeeting-Research/python-model-test/src/quantize/projects/vpcd.py)

## Retained files

- `aimet.py`
  - AIMET service helpers, policy helpers, calibration batch IO, and package export internals
- `aimet_service.py`
  - HTTP service used by VPCD local AIMET quantization
- `calibration.py`
  - VPCD autoregressive calibration record generation
- `evaluate.py`
  - Zipformer candidate-vs-reference verification bridge
- `fixed_shapes.py`
  - ONNX input freezing helpers
- `model_introspection.py`
  - named-node loading used by Zipformer and VPCD policy summaries
- `qnn.py`
  - Zipformer QNN static quantization wrapper
- `reports.py`
  - Zipformer quantization report schema
- `runner.py`
  - shared retained helper `file_size_mb(...)`
- `runtime.py`
  - Windows-safe tempdir helpers used by QNN quantization
- `types.py`
  - shared dataclasses for retained flows

## Retained commands

### Zipformer

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

### VPCD

Start the AIMET service first:

```bash
docker build -t bkmeeting/aimet-onnx-service docker/aimet-onnx-ubuntu2204
docker run --rm -d --name bkmeeting-aimet-service -p 18080:8080 -v <python-model-test-root>:/workspace bkmeeting/aimet-onnx-service
```

Then quantize:

```bash
python -m quantize \
  --project vpcd \
  --model-dir assets/vietnamese-punc-cap-denorm-v1 \
  --fp32-onnx assets/vietnamese-punc-cap-denorm-v1/onnx/model.fp32.onnx \
  --calibration-text build/calibration/vlsp2020/vpcd_transcriptions.txt \
  --output-root build/quantize/vpcd/local_aimet \
  --aimet-param-type int8 \
  --aimet-activation-type int16 \
  --aimet-quant-scheme min_max \
  --aimet-config-file vpcd_matmul_only \
  --aimet-policy-mode local_quality_parity \
  --aimet-service-url http://127.0.0.1:18080
```

The retained canonical VPCD artifact lives under:

- `build/quantize/vpcd/local_aimet/wint8_aint16_min_max_local_quality_parity/`

## Rule of thumb

- If the work is about reusable local quantize artifacts, it belongs in `src/quantize/`.
- If the work is about AI Hub compile/run evidence, it belongs in `src/tools/` and `build/aihub/`.
