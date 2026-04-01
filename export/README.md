# Export Module

`export/` chua cac CLI dung de tao artifact dau vao cho cac pha verify, smoke test, quantize, va Android sync.

## Muc tieu

- cung cap entrypoint ro rang de export bundle theo shared contract
- tach helper export punctuation ONNX khoi logic bundle chung
- giu command line on dinh cho dev va cho README root

## File map

```text
python-model-test/export/
  __init__.py
  model_bundle.py
  punctuation_onnx.py
  README.md
```

## Tung script giai quyet van de gi

### `model_bundle.py`

Day la CLI canonical de export bundle theo `project`.

Van de no giai quyet:
- tranh viec moi model co mot script export rieng
- route len shared core trong `model_bundle`
- chon adapter phu hop cho `vpcd` hoac `zipformer`

Ham chinh:
- `build_argument_parser()`
  - dinh nghia CLI chung cho bundle export
  - cho phep chon `--project`, `--model-dir`, `--output-dir`, `--asset-namespace`, `--model-variant`, `--provider`, `--max-decode-length`
- `main(argv=None)`
  - parse args
  - goi `resolve_bundle_project(...)`
  - goi `export_model_bundle(...)`
  - in ra output dir, manifest, model name

No khong chua class runtime; no chi la command wrapper.

### `punctuation_onnx.py`

Day la helper rieng cho viec export ONNX goc tu source Hugging Face local cua punctuation model.

Van de no giai quyet:
- tao `model.fp32.onnx` tu checkpoint local
- optional tao them `model.int8.onnx`
- khong can di qua bundle contract khi chi muon tai tao ONNX source

Ham chinh:
- `has_local_transformers_onnx()`
  - kiem tra interpreter hien tai co `transformers.onnx` hay khong
- `can_run_transformers_onnx(python_exe)`
  - probe mot interpreter cu the de biet co the export hay khong
- `resolve_export_python(preferred_python)`
  - auto-chon Python interpreter phu hop cho export
- `build_command(export_python, model_dir, output_dir, opset, atol)`
  - tao command `python -m transformers.onnx ...`
- `build_argument_parser()`
  - parse cac option export FP32/INT8
- `main(argv=None)`
  - chay export
  - doi ten `model.onnx` thanh `model.fp32.onnx`
  - optional chay `quantize_dynamic(...)` de tao `model.int8.onnx`

## Khi nao dung module nay

- dung `python -m export.model_bundle` khi muc tieu la tao artifact de verify, test, hoac sync sang Android
- dung `python -m export.punctuation_onnx` khi ban can tai tao ONNX source cho punctuation truoc khi bundle hoa

## Lenh hay dung

### Export bundle punctuation

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m export.model_bundle `
  --project vpcd `
  --model-dir D:\DS-AI\BKMeeting-Research\python-model-test\assets\vietnamese-punc-cap-denorm-v1 `
  --output-dir D:\DS-AI\BKMeeting-Research\python-model-test\build\model_bundle\vpcd\fp32
```

### Export bundle zipformer FP32 reference

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m export.model_bundle `
  --project zipformer `
  --model-dir D:\DS-AI\BKMeeting-Research\python-model-test\assets\zipformer `
  --output-dir D:\DS-AI\BKMeeting-Research\python-model-test\build\model_bundle\zipformer\fp32
```

### Export ONNX source cho punctuation

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m export.punctuation_onnx `
  --model-dir D:\DS-AI\BKMeeting-Research\python-model-test\assets\vietnamese-punc-cap-denorm-v1 `
  --output-dir D:\DS-AI\BKMeeting-Research\python-model-test\assets\vietnamese-punc-cap-denorm-v1\onnx
```

## Quan he voi module khac

- `export/model_bundle.py` goi shared core o `model_bundle/exporter.py`
- `export/punctuation_onnx.py` la helper doc lap, khong phai mot phan cua shared bundle contract
