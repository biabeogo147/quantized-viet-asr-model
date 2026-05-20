# Model Quantization

This is the quick current-state summary for the Android-facing BKMeeting model families in `python-model-test`.

Use it when you need a fast answer to:

- what is already quantized
- what still stays on CPU
- which retained lane currently has the strongest NPU evidence
- what proof is still missing

Status date:

- `2026-05-19`

For full lane history and the retained-lane rationale, read:

- `docs/qnn/option1-retained-lanes.md`

## Current best lanes

### Zipformer

- retained lane: `zipformer_encoder_option1`
- read: encoder-first NPU proof only

### VPCD

- retained lane: `local_aimet_compile_candidate`
- Phase 2 compile pilot: `vpcd_option1_local_aimet`
- read: bounded AIMET parity proof

## What is quantized

### Zipformer

- local `qnn_u16u8` candidate bundle exists
- encoder, decoder, and joiner are quantized in that bundle
- the bundle is still more ambitious than the retained AI Hub proof lane

### VPCD

- the retained AIMET parity lane quantizes the fixed-shape seq2seq model graph
- the local bundle remains the `qnn_fixed_1024x128` candidate

## What is still CPU-side

### Zipformer

- feature extraction
- decoder and joiner in the retained AI Hub proof lane

### VPCD

- tokenizer encode
- tokenizer decode
- autoregressive decode loop

## What remains unproven

### Zipformer

- full end-to-end NPU parity for the full quantized bundle

### VPCD

- free-run parity beyond the current bounded `5`-step window

## Practical read

For Android planning:

- `Zipformer` is ready only with the explicit caveat that the current NPU proof is encoder-first
- `VPCD` is ready only with the explicit caveat that the retained proof is still bounded and CPU-assisted

## Related docs

- `docs/qnn/option1-retained-lanes.md`
- `docs/qnn/preflight.md`
- `docs/workflows/option1-overview.md`
- `docs/workflows/option1-rerun.md`
