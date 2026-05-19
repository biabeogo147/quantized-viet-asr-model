# Option 1 Rerun

Use this doc when you want one fresh retained-lane rerun before Phase 4 and Phase 5.

Run in:

- [On_device_Ai_option1_pilots.ipynb](/D:/DS-AI/BKMeeting-Research/python-model-test/On_device_Ai_option1_pilots.ipynb)

## Prerequisites

1. install the Python env used by this repo
2. have a valid `QAI_HUB_API_TOKEN`
3. build the retained local artifacts first:
   - `python -m quantize --project zipformer ...`
   - `python -m quantize --project vpcd ...`
4. for VPCD, keep the AIMET service alive on `http://127.0.0.1:18080` while producing the local artifact

The notebook itself starts after those producer steps. It does not run local VPCD quantization.

## Retained defaults

- `Zipformer`
  - compile pilot: `zipformer_encoder_option1`
- `VPCD`
  - source strategy: `local_aimet_compile_candidate`
  - compile pilot: `vpcd_option1_local_aimet`
- bounded VPCD guardrails:
  - `VPCD_HYBRID_MAX_SAMPLES = 2`
  - `VPCD_HYBRID_MAX_STEPS = 5`
  - `VPCD_TEACHER_FORCED_SAMPLE_INDEX = 0`

## Run order

### Zipformer

1. prepare encoder upload artifact
2. compile on AI Hub
3. run compiled target
4. run hybrid transcript comparison

### VPCD

1. resolve the prebuilt local AIMET artifact
2. compile that artifact on AI Hub
3. run local quantized teacher-forced diagnostics
4. run compiled teacher-forced diagnostics
5. run bounded hybrid
6. run final compare

## Expected record roots

- `build/aihub/records/zipformer_encoder_option1/`
- `build/aihub/records/zipformer_hybrid_option1/`
- `build/aihub/records/vpcd_option1_local_aimet/`
- `build/aihub/records/vpcd_quantized_teacher_forced_option1/`
- `build/aihub/records/vpcd_teacher_forced_option1/`
- `build/aihub/records/vpcd_hybrid_option1/`

## Stop conditions

Stop and investigate if:

- the notebook asks for a retired VPCD lane
- the retained local AIMET artifact is missing
- VPCD diverges before the retained `5`-step window
- records are written under stale VPCD pilot names such as `vpcd_option1`
