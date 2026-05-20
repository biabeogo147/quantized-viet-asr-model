# Option 1 Retained Lanes

This is the canonical lane-history and lane-decision doc for `Option 1`.

Use it when you need to answer:

- which lanes were tried for `Zipformer` and `VPCD`
- what each lane actually did
- what result each lane produced
- why `On_device_Ai_option1_pilots.ipynb` now defaults to the retained lanes

Status date:

- `2026-05-20`

## Final decision

### Zipformer

Best retained NPU lane:

- `zipformer_encoder_option1`
- source: `build/quantize/zipformer/qnn_u16u8/aihub_compile/encoder.aihub.option1.onnx`
- compile path: direct AI Hub compile to `precompiled_qnn_onnx`

Why:

- this is the strongest proven AI Hub NPU path today
- it matches the first Android NPU slice we can realistically promote

### VPCD

Best retained NPU lane:

- source strategy: `local_aimet_compile_candidate`
- Phase 2 compile pilot: `vpcd_option1_local_aimet`
- policy: `w8a16 + min_max + local_quality_parity`
- AIMET config: `vpcd_matmul_only`

Why:

- bounded local quantized teacher-forced matches FP32
- bounded compiled-cloud teacher-forced matches FP32
- bounded hybrid no longer collapses to punctuation or early EOS

## Notebook decision

`On_device_Ai_option1_pilots.ipynb` should use:

- `Zipformer`: `zipformer_encoder_option1`
- `VPCD`: `local_aimet_compile_candidate`

It should not use:

- VPCD AI Hub quantize
- VPCD local QDQ direct compile probe
- broad AIMET `w8a8 + min_max`

## Zipformer lane history

| Lane | Quantize / Source | Compile path | Result | Decision |
| --- | --- | --- | --- | --- |
| FP32 reference bundle | Local FP32 export only | None | Good reference bundle for Python verification and Android baseline, but not an NPU lane | Keep as reference only |
| Local quantized bundle `qnn_u16u8` | Local QDQ candidate bundle with `quint8` weights and `quint16` activations | None in this lane | Structurally valid and useful for Android experimentation, but current Python evaluation is still not clean end-to-end | Keep as candidate bundle, not as the main AI Hub proof lane |
| Prepared encoder direct compile | `python -m quantize --project zipformer` now emits the retained AI Hub-ready encoder under `build/quantize/.../aihub_compile/` | Direct AI Hub compile | Compile and cloud inference work; the `2026-05-20` post-quantize-only rerun matched the prior retained hybrid output exactly | Keep |
| AI Hub quantize on prepared encoder | AI Hub quantize attempt on prepared encoder graph | AI Hub quantize -> compile | Not the active path; QAIRT conversion still collides with control-flow outputs on this graph family | Retire from active operator flow |
| AIMET local quantize | Not used | Not used | No active AIMET-based Zipformer lane exists in this repo | Not applicable |

### Zipformer conclusion

Choose:

- `zipformer_encoder_option1`

Do not claim yet:

- full end-to-end NPU parity for the full quantized Zipformer bundle

## VPCD lane history

Notes on naming:

- "local PDP" in earlier discussion maps to the local ORT/QNN-flavored QDQ probe in code
- code name: `local_qdq_compile_candidate`

| Lane | Quantize / Source | Compile path | Result | Decision |
| --- | --- | --- | --- | --- |
| FP32 reference | Fixed-shape FP32 ONNX | None | Correct local reference; not itself an NPU lane | Keep as reference only |
| AI Hub quantize baseline | `prefer_fp32_fixed -> AI Hub quantize` | AI Hub quantize -> AI Hub compile | Teacher-forced diverged at step `2`; hybrid historically collapsed into punctuation-heavy output like `0, 4, 4, 4, 4` | Retire |
| Local PDP/QDQ probe | Local ORT/QNN-flavored QDQ export, code: `local_qdq_compile_candidate` | Direct AI Hub compile probe | Local semantics looked healthier, but AI Hub compile rejected `com.microsoft:DequantizeLinear` | Retire from active operator flow |
| Broad AIMET probe | Official local AIMET `w8a8 + min_max` | AIMET `.aimet` package -> AI Hub compile | Compile-compatible, but local teacher-forced already diverged at step `2` | Retire |
| AIMET parity lane | Official local AIMET `w8a16 + min_max + local_quality_parity` with `vpcd_matmul_only` | AIMET `.aimet` package -> AI Hub compile | Bounded local teacher-forced matches FP32; bounded compiled-cloud teacher-forced matches FP32; bounded hybrid no longer collapses | Keep |

### VPCD conclusion

Choose:

- `local_aimet_compile_candidate`
- `vpcd_option1_local_aimet`

Why this beat the older lanes:

- AI Hub quantize failed on correctness
- local PDP/QDQ looked better semantically but failed compile compatibility
- broad AIMET solved compile compatibility but still over-quantized
- AIMET parity is the first lane that satisfies both bounded correctness and AI Hub compile compatibility

## What is quantized right now

### Zipformer

Quantized:

- encoder
- decoder
- joiner
- bundle format: fixed-shape `QDQ`
- weights: `quint8`
- activations: `quint16`
- retained AI Hub compile input: `build/quantize/zipformer/qnn_u16u8/aihub_compile/encoder.aihub.option1.onnx`

Still CPU-side in the retained AI Hub lane:

- feature extraction
- decoder and joiner in the hybrid AI Hub proof lane
- Android end-to-end runtime proof

### VPCD

Quantized in the retained lane:

- fixed-shape seq2seq model graph
- policy: `w8a16 + min_max + local_quality_parity`

Still CPU-side:

- tokenizer encode
- tokenizer decode
- autoregressive decode loop

## Operator defaults

Current clean rerun defaults:

- `RUN_LABEL = "20260519-option1-final-rerun"`
- `VPCD_SOURCE_STRATEGY = "local_aimet_compile_candidate"`
- `VPCD_HYBRID_MAX_SAMPLES = 2`
- `VPCD_HYBRID_MAX_STEPS = 5`
- `VPCD_TEACHER_FORCED_SAMPLE_INDEX = 0`

These defaults are the current decision, not just a convenience.

## What remains unproven

Zipformer:

- full end-to-end NPU parity for the full quantized bundle

VPCD:

- free-run parity beyond the current bounded `5`-step proof window

## Related docs

- `docs/workflows/option1-overview.md`
- `docs/workflows/option1-rerun.md`
- `docs/qnn/model-quantization.md`
- `docs/plans/archive/2026-05-13-vpcd-option1-debug-results.md`
- `docs/plans/archive/2026-05-11-aihub-option1-npu-pilots.md`
