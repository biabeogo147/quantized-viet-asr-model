# Quantize Project Adapters

`quantize/projects/` chua logic dac thu tung model cho CLI `python -m quantize`.

## File map

```text
python-model-test/quantize/projects/
  __init__.py
  vpcd.py
  zipformer.py
  README.md
```

## `__init__.py`

Vai tro:
- registry project cho quantize

Ham chinh:
- `resolve_quantize_project(name)`
- `list_quantize_projects()`

## `vpcd.py`

Vai tro:
- project adapter cho punctuation quantization
- giai bai toan quantize mot model seq2seq duy nhat

Ham chinh:
- `apply_default_arguments(parser)`
  - them cac option CLI cho `vpcd`
- `validate_args(args)`
  - validate preset va chunk size
- `_resolve_output_path(args)`
  - chon output onnx theo preset
- `run(args)`
  - load node names
  - build quantization plan
  - dry-run hoac quantize that su
  - in size budget va next steps

No goi cac helper generic trong:
- `quantize.presets`
- `quantize.calibration`
- `quantize.runner`
- `quantize.qnn`

## `zipformer.py`

Vai tro:
- project adapter cho Zipformer fixed-shape PTQ + QDQ
- giai bai toan component-wise quantization

Ham chinh:
- `apply_default_arguments(parser)`
  - them option cho model dir, output root, bundle output dir, reference bundle dir, provider, audio manifest
- `validate_args(args)`
  - validate preset support
- `_load_audio_fixtures(manifest_path)`
  - doc bo audio dung cho calibration/evaluation
- `_collect_component_records(runtime, fixtures)`
  - trace calibration record cho:
    - encoder
    - decoder
    - joiner
- `_fixed_shape_paths(model_dir, output_root, stats)`
  - tao fixed-shape ONNX file
- `_fixed_input_shapes(stats)`
  - sinh metadata fixed-shape de ghi vao bundle manifest
- `_build_component_plan(component_model, preset)`
  - tao `QuantizationPlan` rieng cho tung component
- `_load_reference_expected_outputs(reference_bundle_dir)`
  - lay transcript reference da export truoc do
- `run(args)`
  - pipeline tong:
    - collect records
    - freeze shape
    - quantize tung component
    - export candidate bundle
    - evaluate candidate
    - ghi report

## Cach nghi ve hai adapter

- `vpcd.py`
  - single-model quantization
  - text calibration
- `zipformer.py`
  - component-wise quantization
  - audio calibration
  - fixed-shape va candidate bundle staging
