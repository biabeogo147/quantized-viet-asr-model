# QNN Preflight Results

## VPCD fixed shape 1024x128

- Date: 2026-05-09
- Source bundle: `build/model_bundle/vpcd/vpcd_balanced`
- Candidate bundle: `build/model_bundle/vpcd/qnn_fixed_1024x128`
- Model variant: `vpcd_balanced_fixed_1024x128`
- Quantization: QDQ, `QUInt16` activations, `QUInt8` weights
- Candidate model SHA256: `3A54567924281D472C8B271E0D5FCCB59652DF0C05A7A6D9CC586E17AB9888CA`
- Candidate manifest SHA256: `DC4ADE5F18CD9474148B50BC83A27DD5C79962FBF2D31FE1281DCA41E5FBB561`
- QNN preflight report SHA256: `33D8C15FEF86AFA12CE263B8BFE2DAF3276E9A6D7797F669B9E69CA1F7095A70`
- Candidate model size: `792,197,445` bytes
- Fixed shapes:
  - `input_ids`: `[1, 1024]`
  - `attention_mask`: `[1, 1024]`
  - `decoder_input_ids`: `[1, 128]`
  - `decoder_attention_mask`: `[1, 128]`
- Tokenizer policy: CPU-only first slice
- Tokenizer bundle verification: passed, 2 encode samples and 2 decode samples
- Reference-vs-candidate parity: passed, 2 checked samples
- QNN preflight: passed
- Python fixed-shape smoke: passed on CPU, one sample latency was about 285 seconds
- HTP execution: not tested in Python

## Candidate files

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

## Android handoff

- Sync candidate bundle contents into BKMeeting `modelassets` under the VPCD family folder.
- Recommended first Android QNN target: the fixed-shape `model.mobile.onnx` from `qnn_fixed_1024x128`, not the dynamic-shape `vpcd_balanced` reference.
- Keep the current dynamic `models/punctuation/vpcd/vpcd_balanced` namespace as the CPU-safe baseline until the Android branch deliberately promotes a fixed-shape production namespace.
- Keep VPCD tokenizer sessions on CPU.
- In strict QNN mode, disable ORT CPU fallback for the VPCD model session.
- If HTP rejects the graph, preserve CPU fallback and attach the HTP error to `BKMeeting/docs/qnn-device-validation.md`.

Suggested sync command:

```powershell
python -m tools.sync_android_bundle `
  --project vpcd `
  --variant qnn_fixed_1024x128 `
  --bkmeeting-root ../BKMeeting `
  --overwrite
```

After syncing, rerun `verify.qnn_preflight` in `python-model-test` and the Android bundle tests in `BKMeeting` before enabling `QNN_HTP_STRICT`.
