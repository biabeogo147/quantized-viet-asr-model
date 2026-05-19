# Model Quantization Status

This document captures the current quantization state for the two Android-facing BKMeeting model families in `python-model-test`.

Use it when you need a fast answer to:

- what is already quantized
- what still stays on CPU or in FP32
- which artifact is currently the strongest NPU candidate
- which results are already proven and which are still research-only

Status date: `2026-05-19`

## Zipformer

### Current quantized bundle

- Bundle manifest:
  - `build/model_bundle/zipformer/qnn_u16u8/bundle_manifest.json`
- Quantization report:
  - `build/model_bundle/zipformer/qnn_u16u8/quantization_report.json`
- Evaluation report:
  - `build/model_bundle/zipformer/qnn_u16u8/evaluation_report.json`

### What is quantized

- `encoder`
- `decoder`
- `joiner`

All three ONNX graphs are exported in fixed-shape `QDQ` form with:

- weights: `quint8`
- activations: `quint16`
- preset: `zipformer_sd8g2_balanced`

### What is not quantized

- host-side feature extraction
- host-side greedy RNNT loop
- Android-side runtime integration for the full end-to-end quantized bundle

### Current result

- The local `qnn_u16u8` candidate bundle exists and is structurally complete.
- The current Python evaluation report is still not clean:
  - `evaluation_report.json` reports `passed = false`
  - one of the checked samples still mismatches the expected transcript
- The current Qualcomm AI Hub proof is narrower than the bundle:
  - the verified cloud-NPU lane is the prepared encoder slice
  - it compiles and runs as `precompiled_qnn_onnx`
- The current AI Hub quantize lane is not the preferred Zipformer path:
  - quantizing the prepared Zipformer graph on AI Hub still collides with QAIRT conversion around control-flow outputs

### Practical interpretation

- `qnn_u16u8` is a real quantized candidate bundle.
- It is useful for Android experimentation and bundle handoff work.
- It is not yet a fully proven end-to-end NPU-ready Zipformer deployment lane.
- The strongest NPU proof for Zipformer today is still `encoder-first`, not full end-to-end bundle parity.

## VPCD

### Current quantized bundle

- Bundle manifest:
  - `build/model_bundle/vpcd/qnn_fixed_1024x128/bundle_manifest.json`
- Prepared local AIMET parity records:
  - `build/aihub/records/vpcd_option1_local_aimet/prepared-artifact-20260519-aimet-local-quality-parity-notebook.json`
  - `build/aihub/records/vpcd_option1_local_aimet/compile-run-20260519-aimet-local-quality-parity-notebook.json`

### What is quantized

- the fixed-shape seq2seq model graph

The local bundle `qnn_fixed_1024x128` carries:

- format: `QDQ`
- weights: `quint8`
- activations: `quint16`
- fixed model inputs:
  - `input_ids [1, 1024]`
  - `attention_mask [1, 1024]`
  - `decoder_input_ids [1, 128]`
  - `decoder_attention_mask [1, 128]`

The current leading AI Hub-compatible local quantize lane is:

- `local_aimet_compile_candidate`
- policy: `w8a16 + min_max + local_quality_parity`
- custom AIMET config: `vpcd_matmul_only`

### What is not quantized

- tokenizer encode graph
- tokenizer decode graph
- host-side autoregressive decode loop

Those remain CPU-side by design in the current first-slice architecture.

### Current result by lane

#### AI Hub quantize baseline

- Lane:
  - `prefer_fp32_fixed -> AI Hub quantize -> AI Hub compile`
- Result:
  - fails teacher-forced at decode step `2`
  - historically collapsed to punctuation-heavy output such as `0, 4, 4, 4, 4`

#### Historical local-QDQ compile probe

- Lane:
  - bundled local QDQ artifact uploaded directly to AI Hub compile
- Result:
  - local semantics looked healthier than AI Hub quantize
  - AI Hub compile rejected the graph because of `com.microsoft:DequantizeLinear`
- Interpretation:
  - useful as historical evidence
  - no longer an active supported notebook strategy

#### Broad AIMET default probe

- Lane:
  - `w8a8 + min_max`
- Result:
  - AI Hub compile accepted the `.aimet` package
  - local teacher-forced already diverged at step `2`
  - compiled cloud repeated that divergence
- Interpretation:
  - compile compatibility alone was not enough
  - the broad quantization policy was too aggressive for VPCD

#### AIMET parity lane

- Lane:
  - `w8a16 + min_max + local_quality_parity`
- Result:
  - local quantized teacher-forced matches FP32 for bounded `5` steps
  - compiled-cloud teacher-forced also matches FP32 for bounded `5` steps
  - bounded hybrid no longer collapses to punctuation or early-EOS failure
  - the recorded hybrid prefix now matches the expected prefix for the bounded window

### Practical interpretation

- VPCD now has a local quantize lane that is both:
  - official enough for AI Hub compile
  - semantically healthy in the current bounded proof window
- The strongest current VPCD NPU candidate is:
  - `local AIMET parity`
- The remaining limitation is scope of proof:
  - the current correctness proof is still bounded to `max_decode_steps = 5`
  - full free-run behavior beyond that bounded window still needs broader validation before default promotion

## Summary

### Ready enough to continue Android export planning

- `Zipformer`
  - yes, but only with the explicit caveat that end-to-end NPU proof is still encoder-first
- `VPCD`
  - yes, with the AIMET parity lane as the leading NPU candidate

### Not ready to claim yet

- full end-to-end NPU parity for Zipformer
- unbounded free-run VPCD parity beyond the current bounded decode window

## Related Docs

- `docs/workflows/aihub-option1-npu-pilots.md`
- `docs/workflows/aihub-option1-hybrid-pipeline.md`
- `docs/workflows/aihub-option1-phase5-contract.md`
- `docs/workflows/android-handoff.md`
- `docs/plans/active/2026-05-13-vpcd-option1-debug-results.md`
