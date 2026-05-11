# QNN Validation Log

This file records dated Python-side QNN checkpoints and handoff notes.

For the maintained QNN preflight rules, use `docs/qnn/preflight.md`.

## 2026-05-09: VPCD fixed-shape candidate preflight

- Source bundle:
  - `build/model_bundle/vpcd/vpcd_balanced`
- Candidate bundle:
  - `build/model_bundle/vpcd/qnn_fixed_1024x128`
- Model variant:
  - `vpcd_balanced_fixed_1024x128`
- Quantization:
  - QDQ
  - activations: `QUInt16`
  - weights: `QUInt8`
- Candidate model SHA256:
  - `3A54567924281D472C8B271E0D5FCCB59652DF0C05A7A6D9CC586E17AB9888CA`
- Candidate manifest SHA256:
  - `DC4ADE5F18CD9474148B50BC83A27DD5C79962FBF2D31FE1281DCA41E5FBB561`
- QNN preflight report SHA256:
  - `33D8C15FEF86AFA12CE263B8BFE2DAF3276E9A6D7797F669B9E69CA1F7095A70`
- Candidate model size:
  - `792,197,445` bytes

### Fixed shapes

- `input_ids`: `[1, 1024]`
- `attention_mask`: `[1, 1024]`
- `decoder_input_ids`: `[1, 128]`
- `decoder_attention_mask`: `[1, 128]`

### Validation status

- tokenizer policy:
  - CPU-only first slice
- tokenizer bundle verification:
  - passed
- reference-vs-candidate parity:
  - passed
- QNN preflight:
  - passed
- Python fixed-shape smoke:
  - passed on CPU
- HTP execution:
  - not tested in Python

### Candidate files

```text
build/model_bundle/vpcd/qnn_fixed_1024x128/
  bundle_manifest.json
  model.mobile.onnx
  tokenizer.encode.onnx
  tokenizer.decode.onnx
  tokenizer.to_model_id_map.json
  tokenizer.from_model_id_map.json
  golden_samples.jsonl
  qnn_preflight_report.json
```

### Android handoff note

- sync the candidate into BKMeeting under the VPCD family folder
- use the fixed-shape `model.mobile.onnx` candidate as the first Android QNN target
- keep tokenizer sessions on CPU in the first Android QNN slice
