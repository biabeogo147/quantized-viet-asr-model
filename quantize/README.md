# Quantize Module

`quantize/` contains the shared multi-project quantization framework. It currently serves two use cases:

- quantizing the punctuation model `vpcd`
- quantizing Zipformer to produce the `qnn_u16u8` candidate bundle

## Goals

- clearly separate generic logic from project-specific logic
- keep calibration, presets, runners, QNN helpers, and reports in one place
- support the Zipformer path from fixed-shape preparation -> PTQ + QDQ -> candidate bundle export

## File map

```text
python-model-test/quantize/
  cli.py
  calibration.py
  config.py
  evaluate.py
  fixed_shapes.py
  model_introspection.py
  presets.py
  qnn.py
  reports.py
  runner.py
  runtime.py
  types.py
  projects/
    __init__.py
    vpcd.py
    zipformer.py
  README.md
```

## What each script is responsible for

### `cli.py`

Role:
- shared entrypoint for the quantize module
- parses `--project` first
- routes the parser and the runner to the correct project adapter

Main functions:
- `_build_project_probe_parser()`
- `parse_args(argv=None)`
- `main(argv=None)`

### `types.py`

Role:
- declares dataclasses used throughout the module

Main classes:
- `CalibrationSample`
- `PresetSpec`
- `QuantizationPlan`

### `config.py`

Role:
- stores default paths and numeric defaults for the older punctuation flow
- currently used by project `vpcd`

### `calibration.py`

Role:
- creates calibration records for static quantization
- normalizes provider selection and sample padding

Main classes and functions:
- `ListCalibrationDataReader`
  - adapter between a list of `CalibrationSample` values and the ORT quantization API
- `resolve_ort_providers(...)`
- `iter_calibration_texts(...)`
- `iter_calibration_files(...)`
- `make_calibration_records(...)`
- `pad_calibration_samples(...)`
- `load_decoder_start_token_id(...)`
- `greedy_decode_ids(...)`
- `build_calibration_records(...)`

### `presets.py`

Role:
- contains presets for punctuation static/dynamic quantization
- maps exclusion patterns into `QuantizationPlan`

Main functions:
- `list_supported_presets()`
- `get_preset_spec(preset)`
- `build_quantization_plan(node_names, preset, extra_exclude_patterns=None)`

### `runner.py`

Role:
- contains the generic quantization runners for static and dynamic paths

Main functions:
- `resolve_calibration_method(...)`
- `run_static_quantization(...)`
- `_run_static_quantization_chunked(...)`
- `run_dynamic_quantization(...)`
- `file_size_mb(...)`
- `build_size_budget_message(...)`
- `recommend_next_steps(...)`

### `qnn.py`

Role:
- helper module for QNN-targeted static quantization
- wraps `qnn_preprocess_model` and `get_qnn_qdq_config`

Main functions:
- `resolve_quant_type(...)`
- `resolve_safe_stride(...)`
- `run_qnn_static_quantization(...)`

### `fixed_shapes.py`

Role:
- freezes ONNX input shapes before quantization
- especially important for Zipformer because NPU/QNN flows prefer fixed shapes over dynamic ones

Main function:
- `freeze_model_inputs(model_path, output_path, input_shapes)`

### `evaluate.py`

Role:
- thin bridge between quantization and `model_bundle.verifier`
- lets the quantize phase call verification without duplicating bundle logic

Main functions:
- `evaluate_bundle_against_model_dir(...)`
- `evaluate_candidate_bundle(...)`

### `reports.py`

Role:
- defines the JSON report schema used to persist quantization results

Main classes:
- `ComponentQuantizationReport`
- `QuantizationReport`

### `model_introspection.py`

Role:
- reads named nodes from ONNX graphs
- generates dry-run summaries

Main functions:
- `load_model_node_names(path)`
- `summarize_quantization_plan(plan, node_names)`

### `runtime.py`

Role:
- works around temporary-directory and hardlink/copy issues on Windows
- helps ORT quantization run more reliably in the current workspace

Main classes and functions:
- `ManualTemporaryDirectory`
- `isolated_model_input(...)`
- `temporary_workspace_tempdir(...)`

## Project adapters

Details are documented in `quantize/projects/README.md`, but in short:

- `projects/vpcd.py`
  - routes punctuation quantization through presets
- `projects/zipformer.py`
  - collects audio calibration data
  - freezes shapes for encoder / decoder / joiner
  - quantizes each component
  - exports the candidate bundle
  - writes `quantization_report.json` and `evaluation_report.json`

## High-level quantization flows

### VPCD

`python -m quantize --project vpcd`
-> load FP32 ONNX
-> build the preset plan
-> run text calibration or dynamic quantization
-> quantize
-> print size-budget guidance and next steps

### Zipformer

`python -m quantize --project zipformer`
-> load audio fixtures
-> trace encoder / decoder / joiner calibration records
-> freeze fixed shapes
-> run QNN PTQ + QDQ on each component
-> export the `qnn_u16u8` candidate bundle
-> verify the candidate against the reference bundle
-> write reports

## Honest status note

- the `zipformer/qnn_u16u8` candidate bundle is runnable today
- it still does not exact-match the FP32 reference on the current sample set
