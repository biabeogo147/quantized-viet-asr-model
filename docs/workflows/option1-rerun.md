# Option 1 Rerun Workflow

This is the canonical current `Phase 2 + Phase 3` operator workflow for `Option 1`.

Run it in:

- `On_device_Ai_option1_pilots.ipynb`

Use this doc after the local bundle and quantization flows are already in place and you want one fresh AI Hub rerun before Android handoff work.

## Prerequisites

Prepare a Python environment that can already run the local bundle logic:

```bash
python -m pip install -e .
python -m pip install qai-hub "qai-hub[torch]"
```

You also need:

- a valid `QAI_HUB_API_TOKEN`
- access to at least one Snapdragon device family in AI Hub
- a compatible local `torch` and `torchaudio` setup for the Zipformer path

## Retained lanes

Use only the retained lanes:

- `Zipformer`
  - compile pilot: `zipformer_encoder_option1`
- `VPCD`
  - source strategy: `local_aimet_compile_candidate`
  - compile pilot: `vpcd_option1_local_aimet`

Lane history and rationale live in:

- `docs/qnn/option1-retained-lanes.md`

## Current clean rerun baseline

Keep these defaults aligned with the current retained proof:

- `RUN_LABEL = "20260519-option1-final-rerun"`
- `ENABLE_ZIPFORMER = True`
- `ENABLE_VPCD = True`
- `VPCD_HYBRID_MAX_SAMPLES = 2`
- `VPCD_HYBRID_MAX_STEPS = 5`
- `VPCD_TEACHER_FORCED_SAMPLE_INDEX = 0`

Do not raise the bounded VPCD limits until the retained lane is clean again.

## Notebook run order

### Zipformer

Run the retained encoder-first path in this order:

1. prepare the encoder upload artifact
2. compile on AI Hub
3. run the compiled target
4. run the hybrid transcript comparison

### VPCD

Run the retained AIMET parity path in this order:

1. freeze the fixed-shape FP32 source
2. export the local AIMET package in Docker
3. compile that package on AI Hub
4. run quantized-local teacher-forced diagnostics
5. run compiled-cloud teacher-forced diagnostics
6. run the bounded hybrid flow
7. run the final compare cell

That ordering matters:

- local quantized teacher-forced checks whether the retained local artifact is semantically healthy
- compiled-cloud teacher-forced checks whether compile changed that behavior
- the bounded hybrid run is the final end-to-end proof for the current retained window

## Expected fresh records

After a clean rerun, expect these record roots:

- `build/aihub/records/zipformer_encoder_option1/`
- `build/aihub/records/zipformer_hybrid_option1/`
- `build/aihub/records/vpcd_option1_local_aimet/`
- `build/aihub/records/vpcd_quantized_teacher_forced_option1/`
- `build/aihub/records/vpcd_teacher_forced_option1/`
- `build/aihub/records/vpcd_hybrid_option1/`

Those records are the inputs for `Phase 4` and `Phase 5`.

## When to stop and investigate

Stop the rerun and investigate if:

- the notebook falls back to a retired VPCD lane
- the VPCD teacher-forced path diverges before the retained `5`-step window is complete
- the rerun writes records under stale pilot names such as `vpcd_option1`
- the notebook no longer produces one fresh record set per retained pilot

## Related docs

- `docs/workflows/option1-overview.md`
- `docs/workflows/option1-promotion-handoff.md`
- `docs/qnn/option1-retained-lanes.md`
- `docs/qnn/model-quantization.md`
