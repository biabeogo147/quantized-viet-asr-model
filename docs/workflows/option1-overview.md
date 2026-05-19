# Option 1 Overview

This is the shortest current map for `Option 1`.

## Retained lanes

- `Zipformer`
  - retained lane: `zipformer_encoder_option1`
  - producer: local quantize bundle under `build/quantize/zipformer/qnn_u16u8/`
  - notebook proof: encoder-first AI Hub compile/run plus hybrid comparison
- `VPCD`
  - retained lane: `local_aimet_compile_candidate`
  - compile pilot: `vpcd_option1_local_aimet`
  - producer: local AIMET package under `build/quantize/vpcd/local_aimet/wint8_aint16_min_max_local_quality_parity/`
  - notebook proof: local teacher-forced, compiled teacher-forced, bounded hybrid

Lane history and retirement rationale live in:

- [option1-retained-lanes.md](/D:/DS-AI/BKMeeting-Research/python-model-test/docs/qnn/option1-retained-lanes.md)

## Producer then notebook

The notebook does not quantize locally anymore.

Do this first:

1. run `python -m quantize --project zipformer ...`
2. run `python -m quantize --project vpcd ...`

Then run the AI Hub notebooks.

## Notebook phases

1. `On_device_Ai_option1_pilots.ipynb`
   - Phase 2 and Phase 3
2. `On_device_Ai_option1_phase4_gate.ipynb`
   - Phase 4 gate verdicts
3. `On_device_Ai_option1_phase5_contract.ipynb`
   - Phase 5 contract packaging

## Current proof boundary

`python-model-test` proves:

- compile-ready artifacts
- AI Hub compile/run evidence
- bounded hybrid evidence
- Phase 5 contract packaging

It does not prove:

- BKMeeting asset integration
- physical Snapdragon NPU behavior on the final app build
