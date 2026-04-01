# Test Module

`test/` chua 3 nhom script khac nhau:

- smoke runner de chay model that su
- pytest suite de khoa contract cua bundle va quantize
- mot vai reference script cu de so sanh behavior

## File map chinh

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

## Smoke runner

### `test_punctuation_model_onnx.py`

Vai tro:
- smoke runner canonical cho punctuation ONNX
- ho tro 2 mode:
  - `--model-dir`
  - `--bundle-manifest`

Class/ham chinh:
- `PunctuationRuntime`
  - protocol cho runtime punctuation
- `ModelDirOnnxRuntime`
  - runtime reference dung Hugging Face tokenizer + ONNX model
- `build_argument_parser()`
- `load_inputs(args)`
- `create_runtime(args)`
  - chon `ModelDirOnnxRuntime` hoac `BundleOnnxRuntime`
- `main()`

### `test_acoustic_model_onnx.py`

Vai tro:
- smoke runner canonical cho Zipformer acoustic model
- ho tro 2 mode:
  - `--model-dir`
  - `--bundle-manifest`

Class/ham chinh:
- `AcousticRuntime`
  - protocol cho runtime acoustic
- `build_argument_parser()`
- `load_inputs(args)`
- `create_runtime(args)`
  - chon `ModelDirAcousticRuntime` hoac `BundleAcousticRuntime`
- `main()`

## Reference script cu van con gia tri

### `test_punctuation_model.py`

Vai tro:
- so sanh 2 model punctuation/capitalization o muc reference
- khong thuoc shared bundle pipeline moi

Class/ham chinh:
- `VietnamesePuncCapDenormModel`
- `VibertCapuModel`
- `patch_vibert_capu_runtime(model_dir)`
- `load_inputs(args)`
- `print_result_block(...)`
- `main()`

### `test_vietasr.py`

Vai tro:
- reference script ASR cu cho Zipformer/VietASR
- chua beam search implementation rieng
- huu ich de benchmark nhanh ngoai shared bundle pipeline

Class/ham chinh:
- `ZipformerASR`
- `ZipformerASRWithBeamSearch`
- `main()`

## Pytest suite

### `test_model_bundle_core.py`

Khoa cac assumption generic:
- manifest round-trip
- project registry

### `test_vpcd_bundle.py`

Khoa punctuation bundle contract:
- manifest schema
- fixture JSONL
- runtime selection
- tokenizer bridge
- bundle runtime restore
- export layout

### `test_zipformer_bundle.py`

Khoa acoustic bundle contract:
- runtime selection
- manifest schema
- audio fixtures
- export layout
- bundle runtime path resolution
- verify bundle

### `test_quantize_projects.py`

Khoa quantize CLI dispatch theo `--project`.

### `test_zipformer_quantize.py`

Khoa fixed-shape helper va metadata fixed encoder frames cua Zipformer.

### `test_export_verify_modules.py`

Khoa 2 package entrypoint:
- `export.model_bundle`
- `verify.model_bundle`

## Cach su dung

### Chay smoke punctuation

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m test.test_punctuation_model_onnx `
  --bundle-manifest D:\DS-AI\BKMeeting-Research\python-model-test\build\model_bundle\vpcd\fp32\bundle_manifest.json `
  --text "hom nay la buoi nham chuc cua toi phuoc thanh"
```

### Chay smoke Zipformer quantized bundle

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m test.test_acoustic_model_onnx `
  --bundle-manifest D:\DS-AI\BKMeeting-Research\python-model-test\build\model_bundle\zipformer\qnn_u16u8\bundle_manifest.json `
  --audio-file D:\DS-AI\BKMeeting-Research\python-model-test\assets\speech\sample-2.wav
```

### Chay pytest core suite

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m pytest `
  D:\DS-AI\BKMeeting-Research\python-model-test\test\test_model_bundle_core.py `
  D:\DS-AI\BKMeeting-Research\python-model-test\test\test_vpcd_bundle.py `
  D:\DS-AI\BKMeeting-Research\python-model-test\test\test_zipformer_bundle.py `
  D:\DS-AI\BKMeeting-Research\python-model-test\test\test_quantize_projects.py `
  D:\DS-AI\BKMeeting-Research\python-model-test\test\test_zipformer_quantize.py `
  D:\DS-AI\BKMeeting-Research\python-model-test\test\test_export_verify_modules.py -q
```

## Cach nghi ve `test/`

- muon chay model that su -> dung smoke runner
- muon khoa contract va refactor an toan -> dung pytest suite
- muon tham khao behavior cu -> xem `test_punctuation_model.py` va `test_vietasr.py`
