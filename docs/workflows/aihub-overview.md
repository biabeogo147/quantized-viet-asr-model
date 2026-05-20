# AI Hub Overview

This is the shortest current map for the retained AI Hub workflow.

## Retained lanes

- `Zipformer`
  - legacy record group: `zipformer_encoder_option1`
  - producer: local quantize bundle under `build/quantize/zipformer/qnn_u16u8/`
  - retained compile input: `build/quantize/zipformer/qnn_u16u8/aihub_compile/encoder.aihub.option1.onnx`
  - notebook proof: encoder-first AI Hub compile/run plus hybrid comparison
- `VPCD`
  - retained lane: `local_aimet_compile_candidate`
  - legacy record group: `vpcd_option1_local_aimet`
  - producer: local AIMET package under `build/quantize/vpcd/local_aimet/wint8_aint16_min_max_local_quality_parity/`
  - notebook proof: local teacher-forced, compiled teacher-forced, bounded hybrid

Lane history and retirement rationale live in:

- [aihub-retained-lanes.md](/D:/DS-AI/BKMeeting-Research/python-model-test/docs/qnn/aihub-retained-lanes.md)

## Producer then notebook

The notebook does not quantize locally anymore.

Do this first:

1. run `python -m quantize --project zipformer ...`
2. run `python -m quantize --project vpcd ...`

Then run the retained AI Hub notebook.

## Current notebook surface

1. `On_device_Ai_option1_pilots.ipynb`
   - retained notebook filename kept for evidence continuity
   - retained `Phase 2` and `Phase 3` evidence
   - `Zipformer` compile/run plus hybrid comparison
   - `VPCD` local and compiled teacher-forced checks plus bounded hybrid

## Current deployment surface

1. `python -m aihub.deployment`
   - resolves retained compile/live evidence for one `RUN_LABEL`
   - downloads the AI Hub `precompiled_qnn_onnx` artifact
   - materializes:
     - `deployment_manifest.json`
     - `io_contract.json`
     - `deploy_notes.md`
2. operator doc:
   - [aihub-deployment.md](/D:/DS-AI/BKMeeting-Research/python-model-test/docs/workflows/aihub-deployment.md)

## Current proof boundary

`python-model-test` proves:

- compile-ready artifacts
- AI Hub compile/run evidence
- bounded hybrid evidence
- deployment package tooling for retained deployable artifact download

It does not prove:

- BKMeeting asset integration
- physical Snapdragon NPU behavior on the final app build
