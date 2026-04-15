# Quantize Project Adapters

`src/quantize/projects/` contains project-specific logic for the `python -m quantize` CLI.

## File map

```text
python-model-test/src/quantize/projects/
  __init__.py
  vpcd.py
  zipformer.py
  README.md
```

## `__init__.py`

Role:
- project registry for quantization

Main functions:
- `resolve_quantize_project(name)`
- `list_quantize_projects()`

## `vpcd.py`

Role:
- project adapter for punctuation quantization
- solves the problem of quantizing a single seq2seq model

Main functions:
- `apply_default_arguments(parser)`
  - adds CLI options for `vpcd`
- `validate_args(args)`
  - validates the selected preset and chunk size
- `_resolve_output_path(args)`
  - chooses the output ONNX path from the preset
- `run(args)`
  - loads node names
  - builds the quantization plan
  - performs a dry run or real quantization
  - prints size-budget guidance and next steps

This adapter calls generic helpers from:
- `quantize.presets`
- `quantize.calibration`
- `quantize.runner`
- `quantize.qnn`

## `zipformer.py`

Role:
- project adapter for Zipformer fixed-shape PTQ + QDQ
- solves the component-wise quantization workflow

Main functions:
- `apply_default_arguments(parser)`
  - adds options for model dir, output root, bundle output dir, reference bundle dir, provider, and audio manifest
- `validate_args(args)`
  - validates preset support
- `_load_audio_fixtures(manifest_path)`
  - reads the audio set used for calibration and evaluation
  - accepts UTF-8 BOM manifests
- `_collect_component_records(runtime, fixtures)`
  - traces calibration records for:
    - encoder
    - decoder
    - joiner
  - resolves repo-relative audio paths through `tools.paths.resolve_repo_path(...)`
- `_fixed_shape_paths(model_dir, output_root, stats)`
  - creates fixed-shape ONNX files
- `_fixed_input_shapes(stats)`
  - generates fixed-shape metadata for the bundle manifest
- `_build_component_plan(component_model, preset)`
  - creates a dedicated `QuantizationPlan` per component
- `_load_reference_expected_outputs(reference_bundle_dir)`
  - loads the reference transcripts exported earlier
- `run(args)`
  - full pipeline:
    - collect records
    - freeze shapes
    - quantize each component
    - export the candidate bundle
    - evaluate the candidate
    - write reports

## How to think about the two adapters

- `vpcd.py`
  - single-model quantization
  - text-based calibration
- `zipformer.py`
  - component-wise quantization
  - audio-based calibration
  - fixed-shape preparation and candidate-bundle staging
  - stable repo-root fixture resolution after the `src/` refactor
