# Quantize Project Adapters

`src/quantize/projects/` holds the two retained `python -m quantize` adapters:

- [zipformer.py](/D:/DS-AI/BKMeeting-Research/python-model-test/src/quantize/projects/zipformer.py)
- [vpcd.py](/D:/DS-AI/BKMeeting-Research/python-model-test/src/quantize/projects/vpcd.py)

## `zipformer.py`

Role:

- build the retained `qnn_u16u8` bundle from local audio calibration data
- freeze fixed shapes for encoder, decoder, and joiner
- quantize each component with the retained preset `zipformer_sd8g2_balanced`
- export the candidate bundle and verify it against the FP32 reference bundle

Important outputs:

- `build/quantize/zipformer/qnn_u16u8/`
- `build/model_bundle/zipformer/qnn_u16u8/`

## `vpcd.py`

Role:

- build the retained local AIMET quantize artifact for `Option 1`
- freeze the FP32 staging model to the fixed bundle input shapes
- generate autoregressive calibration batches
- apply the retained parity policy:
  - `int8` weights
  - `int16` activations
  - `min_max`
  - `vpcd_matmul_only`
  - `local_quality_parity`
- call the local AIMET service over HTTP
- write the reusable `.aimet` package, local QDQ diagnostic model, and `quantize_report.json`

Important output:

- `build/quantize/vpcd/local_aimet/wint8_aint16_min_max_local_quality_parity/`

## What is gone

These are intentionally no longer part of the active adapters:

- VPCD legacy local QDQ lane
- VPCD AI Hub quantize baseline lane
- Zipformer legacy preset aliases
