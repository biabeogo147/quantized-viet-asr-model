# AI Hub Rerun

Use this doc when you want one fresh retained AI Hub rerun.

Run in:

- [On_device_Ai_option1_pilots.ipynb](/D:/DS-AI/BKMeeting-Research/python-model-test/On_device_Ai_option1_pilots.ipynb)

This is now the full retained AI Hub notebook flow in this repo.
The notebook filename and record-group names still carry legacy `option1` slugs because they are part of the retained evidence lookup surface.

## Prerequisites

1. install the Python env used by this repo
2. have a valid `QAI_HUB_API_TOKEN`
3. build the retained local artifacts first:
   - `python -m quantize --project zipformer ...`
   - `python -m quantize --project vpcd ...`
4. for VPCD, keep the AIMET service alive on `http://127.0.0.1:18080` while producing the local artifact

The notebook itself starts after those producer steps. It does not run local VPCD quantization.
It also no longer prepares the Zipformer encoder for AI Hub inside the notebook.

## Retained defaults

- `Zipformer`
  - legacy compile record group: `zipformer_encoder_option1`
- `VPCD`
  - source strategy: `local_aimet_compile_candidate`
  - legacy compile record group: `vpcd_option1_local_aimet`
- bounded VPCD guardrails:
  - `VPCD_HYBRID_MAX_SAMPLES = 2`
  - `VPCD_HYBRID_MAX_STEPS = 5`
  - `VPCD_TEACHER_FORCED_SAMPLE_INDEX = 0`

## Run order

### Zipformer

1. read the prebuilt AI Hub-ready encoder from `build/quantize/zipformer/qnn_u16u8/aihub_compile/encoder.aihub.option1.onnx`
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

These retained record-group names are intentionally unchanged:

- `build/aihub/records/zipformer_encoder_option1/`
- `build/aihub/records/zipformer_hybrid_option1/`
- `build/aihub/records/vpcd_option1_local_aimet/`
- `build/aihub/records/vpcd_quantized_teacher_forced_option1/`
- `build/aihub/records/vpcd_teacher_forced_option1/`
- `build/aihub/records/vpcd_hybrid_option1/`

## Deployment Packaging After The Notebook

After the retained notebook already wrote the records for one `RUN_LABEL`, run deployment packaging separately.

Dry run first:

```bash
python -m aihub.deployment \
  --project all \
  --run-label 20260519-6pm \
  --device-name "Samsung Galaxy S24 (Family)" \
  --qairt-version 2.46.0 \
  --dry-run
```

Then package the retained deployable artifacts:

```bash
python -m aihub.deployment \
  --project all \
  --run-label 20260519-6pm \
  --device-name "Samsung Galaxy S24 (Family)" \
  --qairt-version 2.46.0
```

Deployment output root:

- `build/aihub/deploy/zipformer/<RUN_LABEL>/`
- `build/aihub/deploy/vpcd/<RUN_LABEL>/`

## Stop conditions

Stop and investigate if:

- the retained Zipformer AI Hub-ready encoder is missing
- the notebook asks for a retired VPCD lane
- the retained local AIMET artifact is missing
- VPCD diverges before the retained `5`-step window
- records are written outside the expected retained record groups above
- deployment dry-run cannot resolve the compile or live-run records for the same `RUN_LABEL`
