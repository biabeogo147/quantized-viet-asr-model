# Test Module

`test/` now contains only two things:

- canonical smoke runners that execute real models
- pytest suites that lock down the shared `src/` contracts

Legacy standalone comparison scripts have been removed so the repo keeps a single canonical path.

## Import setup

- `pytest` from `python-model-test/` works without extra shell setup because `test/conftest.py` prepends `src/` to `sys.path`.
- Examples below assume you run commands from `python-model-test/`.

## Main file map

```text
python-model-test/test/
  test_punctuation_model_onnx.py
  test_acoustic_model_onnx.py
  test_model_bundle_core.py
  test_vpcd_bundle.py
  test_zipformer_bundle.py
  test_quantize_projects.py
  test_zipformer_quantize.py
  test_extract_vlsp2020_calibration_subset.py
  test_export_verify_modules.py
  test_src_layout_bootstrap.py
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

### `test_extract_vlsp2020_calibration_subset.py`

Locks the deterministic VLSP calibration-subset extractor:
- source-order preservation
- emitted manifest/file layout

### `test_export_verify_modules.py`

Locks the two package entrypoints:
- `export.model_bundle`
- `verify.model_bundle`

### `test_src_layout_bootstrap.py`

Locks the `src/` layout assumptions:
- imports resolve from `src/`
- legacy repo-root wrappers are gone
- deleted reference scripts are not reintroduced by accident

## How to use it

### Run the punctuation smoke test

```bash
python -m test.test_punctuation_model_onnx \
  --bundle-manifest build/model_bundle/vpcd/fp32/bundle_manifest.json \
  --text "hom nay la buoi nham chuc cua toi phuoc thanh"
```

### Run the Zipformer quantized-bundle smoke test

```bash
python -m test.test_acoustic_model_onnx \
  --bundle-manifest build/model_bundle/zipformer/qnn_u16u8/bundle_manifest.json \
  --audio-file assets/speech/sample-2.wav
```

### Run the core pytest suite

```bash
python -m pytest test -q -p no:cacheprovider
```

## How to think about `test/`

- if you want to run a real model, use the smoke runners
- if you want to lock the contract and refactor safely, use the pytest suite
- if you need a new experiment, build it on top of the canonical `src/` runtimes instead of adding a parallel legacy script
