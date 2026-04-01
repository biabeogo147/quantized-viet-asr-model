# Android Bundle Contract

This folder contains the Python-side bundle contract for the Android punctuation runtime of `vietnamese-punc-cap-denorm-v1`.

## What this step changed

- Added `runtime.py` with a real bundle-only runtime: `BundleOnnxRuntime`.
- Refactored `test/test_punctuation_onnx.py` so it now has two clean modes instead of one hybrid mode.
- Kept `ModelDirOnnxRuntime` for the original Hugging Face tokenizer flow.
- Removed the old behavior where `--bundle-manifest` still depended on `--model-dir`.
- Added tests that lock the new CLI/runtime separation.

## Files in this folder

```text
python-model-test/
  android_bundle/
    __init__.py
    exporter.py
    golden_samples.py
    manifest.py
    runtime.py
    tokenizer_bridge.py
    verifier.py
    README.md
```

## Bundle layout

After export, the Android bundle must contain:

```text
build/android_bundle/vpcd/
  bundle_manifest.json
  model.mobile.onnx
  tokenizer.encode.onnx
  tokenizer.decode.onnx
  tokenizer.to_model_id_map.json
  tokenizer.from_model_id_map.json
  golden_samples.jsonl
```

## What each module does

- `manifest.py`
  - Defines `AndroidBundleManifest`
  - Normalizes bundle file names
  - Resolves bundle-relative file paths
- `exporter.py`
  - Copies the selected ONNX model variant into `model.mobile.onnx`
  - Exports `tokenizer.encode.onnx` and `tokenizer.decode.onnx`
  - Writes bridge JSON files
  - Writes `bundle_manifest.json`
  - Writes `golden_samples.jsonl`
- `tokenizer_bridge.py`
  - Builds the dense ID maps between ORT tokenizer IDs and model IDs
- `verifier.py`
  - Verifies encode parity between the exported tokenizer bundle and Hugging Face
  - Verifies decode parity between the exported tokenizer bundle and Hugging Face
- `runtime.py`
  - Runs the bundle directly without using `AutoTokenizer`
  - Uses only `bundle_manifest.json` and the exported bundle files

## How to use

### 1. Run the contract tests

```powershell
& D:\Anaconda\Scripts\pytest.exe D:\DS-AI\BKMeeting-Research\python-model-test\test\test_android_punctuation_bundle.py -q
```

Current verified result:

```text
9 passed
```

### 2. Export the Android bundle

```powershell
& D:\Anaconda\envs\speech2text\python.exe D:\DS-AI\BKMeeting-Research\python-model-test\export_android_punctuation_bundle.py `
  --model-dir D:\DS-AI\BKMeeting-Research\python-model-test\assets\vietnamese-punc-cap-denorm-v1 `
  --output-dir D:\DS-AI\BKMeeting-Research\python-model-test\build\android_bundle\vpcd `
  --model-variant vpcd_balanced `
  --asset-namespace models/punctuation/vpcd
```

### 3. Verify tokenizer parity for the exported bundle

```powershell
& D:\Anaconda\envs\speech2text\python.exe D:\DS-AI\BKMeeting-Research\python-model-test\verify_android_punctuation_bundle.py `
  --model-dir D:\DS-AI\BKMeeting-Research\python-model-test\assets\vietnamese-punc-cap-denorm-v1 `
  --bundle-dir D:\DS-AI\BKMeeting-Research\python-model-test\build\android_bundle\vpcd
```

Current verified result:

```text
Tokenizer bundle verification passed.
Bundle dir      : D:\DS-AI\BKMeeting-Research\python-model-test\build\android_bundle\vpcd
Encode samples  : 2
Decode samples  : 2
```

### 4. Run smoke inference

The smoke runner now supports two modes. See:

- `D:\DS-AI\BKMeeting-Research\python-model-test\test\README.md`

## Technical notes

- ORT Extensions does not recognize `BartphoTokenizer` directly by class name during export, so export currently uses a thin alias to `XLMRobertaTokenizer` while generating tokenizer graphs.
- The exported tokenizer graphs and the punctuation model do not share the same ID space, so the bundle must carry:
  - `tokenizer.to_model_id_map.json`
  - `tokenizer.from_model_id_map.json`
- `BundleOnnxRuntime` exists to exercise the exact same bundle contract that Android consumes:
  - encode graph
  - tokenizer-to-model bridge
  - model inference
  - model-to-tokenizer bridge
  - decode graph

## Verified for this step

- Contract tests: PASS
- Tokenizer bundle verifier: PASS
- Model-dir smoke runner: PASS
- Bundle-only smoke runner: PASS
