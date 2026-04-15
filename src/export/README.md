# Export Module

`src/export/` contains the CLIs used to create input artifacts for verification, smoke tests, quantization, and Android sync.

## Goals

- provide clear entrypoints for exporting bundles through the shared contract
- keep punctuation ONNX export separate from the generic bundle flow
- keep the command-line surface stable for developers and for the root README

## File map

```text
python-model-test/src/export/
  __init__.py
  model_bundle.py
  punctuation_onnx.py
  README.md
```

## Command setup

Examples below assume you run commands from `python-model-test/`.

## What each script is responsible for

### `model_bundle.py`

This is the canonical CLI for exporting bundles by `project`.

Problems it solves:
- avoids one custom export script per model
- routes into the shared `model_bundle` core
- selects the correct adapter for `vpcd` or `zipformer`

Main functions:
- `build_argument_parser()`
  - defines the shared CLI surface
  - supports `--project`, `--model-dir`, `--output-dir`, `--asset-namespace`, `--model-variant`, `--provider`, and `--max-decode-length`
- `main(argv=None)`
  - parses arguments
  - calls `resolve_bundle_project(...)`
  - calls `export_model_bundle(...)`
  - prints the output directory, manifest path, and model name

This file does not implement a runtime. It is a command wrapper around the shared bundle layer.

### `punctuation_onnx.py`

This is a dedicated helper for exporting source ONNX from a local Hugging Face punctuation model.

Problems it solves:
- creates `model.fp32.onnx` from a local checkpoint
- can optionally produce `model.int8.onnx`
- lets you rebuild source ONNX without going through the bundle contract

Main functions:
- `has_local_transformers_onnx()`
  - checks whether the current interpreter has `transformers.onnx`
- `can_run_transformers_onnx(python_exe)`
  - probes a specific interpreter to see whether export is possible
- `resolve_export_python(preferred_python)`
  - auto-selects a suitable Python interpreter for export
- `build_command(export_python, model_dir, output_dir, opset, atol)`
  - creates the `python -m transformers.onnx ...` command
- `build_argument_parser()`
  - parses FP32 / INT8 export options
- `main(argv=None)`
  - runs the export
  - renames `model.onnx` to `model.fp32.onnx`
  - optionally runs `quantize_dynamic(...)` to create `model.int8.onnx`

## When to use this module

- use `python -m export.model_bundle` when your goal is to create artifacts for verification, testing, or Android sync
- use `python -m export.punctuation_onnx` when you need to rebuild source ONNX for punctuation before bundling

## Common commands

### Export a punctuation bundle

```bash
python -m export.model_bundle \
  --project vpcd \
  --model-dir assets/vietnamese-punc-cap-denorm-v1 \
  --output-dir build/model_bundle/vpcd/fp32
```

### Export a Zipformer FP32 reference bundle

```bash
python -m export.model_bundle \
  --project zipformer \
  --model-dir assets/zipformer \
  --output-dir build/model_bundle/zipformer/fp32
```

### Export source ONNX for punctuation

```bash
python -m export.punctuation_onnx \
  --model-dir assets/vietnamese-punc-cap-denorm-v1 \
  --output-dir assets/vietnamese-punc-cap-denorm-v1/onnx
```

## Relationship to other modules

- `export/model_bundle.py` calls the shared core in `model_bundle/exporter.py`
- `export/punctuation_onnx.py` is a standalone helper and is not part of the shared bundle contract

## Android handoff status

- `vpcd`
  - the exported bundle layout is already consumed by `bkmeeting`
  - after export, copy the bundle files into `bkmeeting/modelassets/src/main/assets/models/punctuation/vpcd`
- `zipformer`
  - the exported bundle is the canonical Python-side verification artifact
  - the current Android ASR runtime still consumes raw component files instead of `bundle_manifest.json`
