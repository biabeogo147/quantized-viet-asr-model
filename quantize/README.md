# Quantize Module

`quantize/` chua framework quantization multi-project. Hien tai no phuc vu 2 bai toan:

- quantize punctuation model `vpcd`
- quantize Zipformer de tao candidate bundle `qnn_u16u8`

## Muc tieu

- chia ro phan generic va phan project-specific
- gom calibration, preset, runner, QNN helper, va report vao mot noi
- de `zipformer` co the di tu fixed-shape -> PTQ + QDQ -> bundle candidate

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

## Tung script giai quyet van de gi

### `cli.py`

Vai tro:
- entrypoint chung cua module quantize
- parse `--project` truoc
- route parser va runner sang project adapter dung

Ham chinh:
- `_build_project_probe_parser()`
- `parse_args(argv=None)`
- `main(argv=None)`

### `types.py`

Vai tro:
- khai bao dataclass dung xuyen suot module

Class chinh:
- `CalibrationSample`
- `PresetSpec`
- `QuantizationPlan`

### `config.py`

Vai tro:
- chua default path va default numeric config cho flow punctuation cu
- dac biet dung boi project `vpcd`

### `calibration.py`

Vai tro:
- tao calibration records cho static quantization
- chuan hoa provider selection va sample padding

Class/ham chinh:
- `ListCalibrationDataReader`
  - adapter giua list `CalibrationSample` va ORT quantization API
- `resolve_ort_providers(...)`
- `iter_calibration_texts(...)`
- `iter_calibration_files(...)`
- `make_calibration_records(...)`
- `pad_calibration_samples(...)`
- `load_decoder_start_token_id(...)`
- `greedy_decode_ids(...)`
- `build_calibration_records(...)`

### `presets.py`

Vai tro:
- chua preset cho punctuation static/dynamic quantization
- map pattern exclusion -> `QuantizationPlan`

Ham chinh:
- `list_supported_presets()`
- `get_preset_spec(preset)`
- `build_quantization_plan(node_names, preset, extra_exclude_patterns=None)`

### `runner.py`

Vai tro:
- chua generic quantization runner cho static va dynamic path

Ham chinh:
- `resolve_calibration_method(...)`
- `run_static_quantization(...)`
- `_run_static_quantization_chunked(...)`
- `run_dynamic_quantization(...)`
- `file_size_mb(...)`
- `build_size_budget_message(...)`
- `recommend_next_steps(...)`

### `qnn.py`

Vai tro:
- helper rieng cho QNN-targeted static quantization
- dung `qnn_preprocess_model` va `get_qnn_qdq_config`

Ham chinh:
- `resolve_quant_type(...)`
- `resolve_safe_stride(...)`
- `run_qnn_static_quantization(...)`

### `fixed_shapes.py`

Vai tro:
- dong bang input shape cua ONNX model truoc khi quantize
- rat quan trong cho Zipformer vi NPU/QNN khong thich dynamic shape

Ham chinh:
- `freeze_model_inputs(model_path, output_path, input_shapes)`

### `evaluate.py`

Vai tro:
- bridge nho giua quantize va `model_bundle.verifier`
- de quantize phase co the goi verify ma khong duplicate logic

Ham chinh:
- `evaluate_bundle_against_model_dir(...)`
- `evaluate_candidate_bundle(...)`

### `reports.py`

Vai tro:
- dinh nghia report schema de ghi ket qua quantization ra JSON

Class chinh:
- `ComponentQuantizationReport`
- `QuantizationReport`

### `model_introspection.py`

Vai tro:
- doc named nodes tu ONNX graph
- sinh preview text cho dry-run

Ham chinh:
- `load_model_node_names(path)`
- `summarize_quantization_plan(plan, node_names)`

### `runtime.py`

Vai tro:
- workaround cho temp directory va hardlink/copy model input tren Windows
- giup ORT quantization chay on dinh hon trong workspace hien tai

Class/ham chinh:
- `ManualTemporaryDirectory`
- `isolated_model_input(...)`
- `temporary_workspace_tempdir(...)`

## Project adapters

Chi tiet nam o `quantize/projects/README.md`, nhung tom tat:

- `projects/vpcd.py`
  - route punctuation quantization theo preset
- `projects/zipformer.py`
  - collect audio calibration
  - freeze shape cho encoder/decoder/joiner
  - quantize tung component
  - export candidate bundle
  - ghi `quantization_report.json` va `evaluation_report.json`

## Flow tong quat cua quantize

### VPCD

`python -m quantize --project vpcd`
-> load FP32 ONNX
-> build preset plan
-> calibration text / dynamic path
-> quantize
-> in size budget + goi y

### Zipformer

`python -m quantize --project zipformer`
-> load audio fixtures
-> trace encoder/decoder/joiner records
-> freeze fixed shapes
-> QNN PTQ + QDQ tung component
-> export candidate bundle `qnn_u16u8`
-> verify candidate voi reference bundle
-> ghi report

## Ghi chu trung thuc

- candidate `zipformer/qnn_u16u8` hien da runnable
- no chua exact-match 100% voi FP32 reference tren bo sample hien tai
