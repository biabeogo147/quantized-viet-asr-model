# Punctuation Smoke Runner

This folder contains the local smoke runner for the punctuation ONNX model.

## What this step changed

- `test_punctuation_onnx.py` now has two explicit runtime modes.
- The old hybrid behavior was removed.
- `--bundle-manifest` no longer depends on `--model-dir`.
- Added a real bundle-only runtime path through `android_bundle.runtime.BundleOnnxRuntime`.

## Runtime modes

### 1. Model-dir mode

Use this mode when you want the original Python reference flow:

- Hugging Face `AutoTokenizer`
- config files from the model directory
- ONNX model from `<model-dir>/onnx/<model-variant>.onnx`

Command:

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m test.test_punctuation_onnx `
  --model-dir .\assets\vietnamese-punc-cap-denorm-v1 `
  --model-variant vpcd_balanced `
  --text "hom nay la buoi nham chuc cua toi phuoc thanh"
```

### 2. Bundle-only mode

Use this mode when you want to run exactly from the exported Android bundle:

- `bundle_manifest.json`
- `model.mobile.onnx`
- `tokenizer.encode.onnx`
- `tokenizer.decode.onnx`
- `tokenizer.to_model_id_map.json`
- `tokenizer.from_model_id_map.json`

No `--model-dir` is needed in this mode.

Command:

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m test.test_punctuation_onnx `
  --bundle-manifest .\build\android_bundle\vpcd\bundle_manifest.json `
  --text "hom nay la buoi nham chuc cua toi phuoc thanh"
```

## CLI summary

```text
python -m test.test_punctuation_onnx [--model-dir <dir> | --bundle-manifest <manifest>] [--model-variant <variant>] [--text <text> | --text-file <file>] [--max-length <n>] [--provider <provider>]
```

Rules:

- `--model-dir` and `--bundle-manifest` are mutually exclusive.
- If neither is provided, the runner defaults to model-dir mode with `assets/vietnamese-punc-cap-denorm-v1`.
- `--model-variant` is used for model-dir mode only.
- `--provider` works for both modes.

## Files involved

- `test_punctuation_onnx.py`
  - CLI entrypoint
  - defines `ModelDirOnnxRuntime`
  - selects runtime mode
- `..\android_bundle\runtime.py`
  - defines `BundleOnnxRuntime`

## Verified for this step

### Contract tests

```powershell
& D:\Anaconda\Scripts\pytest.exe D:\DS-AI\BKMeeting-Research\python-model-test\test\test_android_punctuation_bundle.py -q
```

Result:

```text
9 passed
```

### Model-dir smoke run

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m test.test_punctuation_onnx `
  --model-dir .\assets\vietnamese-punc-cap-denorm-v1 `
  --text "hom nay la buoi nham chuc cua toi phuoc thanh"
```

Observed output:

```text
Input   : hom nay la buoi nham chuc cua toi phuoc thanh
Output  : hom nay la buoi nham chuc cua toi phuoc thanh.
```

### Bundle-only smoke run

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m test.test_punctuation_onnx `
  --bundle-manifest .\build\android_bundle\vpcd\bundle_manifest.json `
  --text "hom nay la buoi nham chuc cua toi phuoc thanh"
```

Observed output:

```text
Input   : hom nay la buoi nham chuc cua toi phuoc thanh
Output  : hom nay la buoi nham chuc cua toi phuoc thanh.
```
