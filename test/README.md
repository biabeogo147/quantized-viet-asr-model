# Test Module

`test/` contains three different groups of scripts:

- smoke runners that execute real models
- pytest suites that lock down the bundle and quantization contracts
- a few legacy reference scripts used for behavior comparison

## Import setup

- `pytest` from `python-model-test/` works without extra shell setup because `test/conftest.py` prepends `src/` to `sys.path`.
- Direct smoke-runner commands still need one of these:
  - editable install, or
  - `PYTHONPATH=src`

Example from `python-model-test/`:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
```

## Main file map

```text
python-model-test/test/
  test_punctuation_model_onnx.py
  test_acoustic_model_onnx.py
  test_punctuation_model.py
  test_vietasr.py
  test_model_bundle_core.py
  test_vpcd_bundle.py
  test_zipformer_bundle.py
  test_quantize_projects.py
  test_zipformer_quantize.py
  test_export_verify_modules.py
  README.md
```

## Smoke runners

### `test_punctuation_model_onnx.py`

Role:
- canonical smoke runner for punctuation ONNX
- supports two modes:
  - `--model-dir`
  - `--bundle-manifest`

Main classes and functions:
- `PunctuationRuntime`
  - protocol for punctuation runtimes
- `ModelDirOnnxRuntime`
  - reference runtime built from the Hugging Face tokenizer + ONNX model
- `build_argument_parser()`
- `load_inputs(args)`
- `create_runtime(args)`
  - chooses `ModelDirOnnxRuntime` or `BundleOnnxRuntime`
- `main()`

### `test_acoustic_model_onnx.py`

Role:
- canonical smoke runner for the Zipformer acoustic model
- supports two modes:
  - `--model-dir`
  - `--bundle-manifest`

Main classes and functions:
- `AcousticRuntime`
  - protocol for acoustic runtimes
- `build_argument_parser()`
- `load_inputs(args)`
- `create_runtime(args)`
  - chooses `ModelDirAcousticRuntime` or `BundleAcousticRuntime`
- `main()`

## Legacy reference scripts that still have value

### `test_punctuation_model.py`

Role:
- compares two punctuation/capitalization models at the reference level
- not part of the newer shared bundle pipeline

Main classes and functions:
- `VietnamesePuncCapDenormModel`
- `VibertCapuModel`
- `patch_vibert_capu_runtime(model_dir)`
- `load_inputs(args)`
- `print_result_block(...)`
- `main()`

### `test_vietasr.py`

Role:
- older ASR reference script for Zipformer/VietASR
- includes its own beam-search implementation
- still useful for quick benchmarking outside the shared bundle pipeline

Main classes and functions:
- `ZipformerASR`
- `ZipformerASRWithBeamSearch`
- `main()`

## Pytest suite

### `test_model_bundle_core.py`

Locks generic assumptions:
- manifest round-trip behavior
- project registry behavior

### `test_vpcd_bundle.py`

Locks the punctuation bundle contract:
- manifest schema
- fixture JSONL
- runtime selection
- tokenizer bridge behavior
- bundle runtime restoration
- export layout

### `test_zipformer_bundle.py`

Locks the acoustic bundle contract:
- runtime selection
- manifest schema
- audio fixtures
- export layout
- bundle runtime path resolution
- bundle verification

### `test_quantize_projects.py`

Locks quantize CLI dispatch by `--project`.

### `test_zipformer_quantize.py`

Locks the fixed-shape helper and Zipformer fixed-encoder-frames metadata.

### `test_export_verify_modules.py`

Locks the two package entrypoints:
- `export.model_bundle`
- `verify.model_bundle`

## How to use it

### Run the punctuation smoke test

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m test.test_punctuation_model_onnx `
  --bundle-manifest D:\DS-AI\BKMeeting-Research\python-model-test\build\model_bundle\vpcd\fp32\bundle_manifest.json `
  --text "hom nay la buoi nham chuc cua toi phuoc thanh"
```

### Run the Zipformer quantized-bundle smoke test

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m test.test_acoustic_model_onnx `
  --bundle-manifest D:\DS-AI\BKMeeting-Research\python-model-test\build\model_bundle\zipformer\qnn_u16u8\bundle_manifest.json `
  --audio-file D:\DS-AI\BKMeeting-Research\python-model-test\assets\speech\sample-2.wav
```

### Run the core pytest suite

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m pytest `
  D:\DS-AI\BKMeeting-Research\python-model-test\test\test_model_bundle_core.py `
  D:\DS-AI\BKMeeting-Research\python-model-test\test\test_vpcd_bundle.py `
  D:\DS-AI\BKMeeting-Research\python-model-test\test\test_zipformer_bundle.py `
  D:\DS-AI\BKMeeting-Research\python-model-test\test\test_quantize_projects.py `
  D:\DS-AI\BKMeeting-Research\python-model-test\test\test_zipformer_quantize.py `
  D:\DS-AI\BKMeeting-Research\python-model-test\test\test_export_verify_modules.py -q
```

## How to think about `test/`

- if you want to run a real model, use the smoke runners
- if you want to lock the contract and refactor safely, use the pytest suite
- if you want to inspect older behavior, read `test_punctuation_model.py` and `test_vietasr.py`
