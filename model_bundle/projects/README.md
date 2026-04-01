# Model Bundle Project Adapters

Thu muc `model_bundle/projects/` chua logic dac thu tung model family. Shared core khong biet chi tiet tokenizer hay RNNT; no chi biet adapter.

## File map

```text
python-model-test/model_bundle/projects/
  __init__.py
  vpcd.py
  zipformer.py
  _vpcd_support.py
  README.md
```

## `__init__.py`

Vai tro:
- registry project adapter
- noi `model_bundle` core voi adapter cu the

Ham chinh:
- `resolve_bundle_project(name)`
- `list_bundle_projects()`

## `vpcd.py`

Vai tro:
- adapter punctuation model `tourmii/vietnamese-punc-cap-denorm-v1`
- export punctuation bundle
- verify tokenizer parity encode/decode

Ham chinh:
- `export_bundle(...)`
  - copy `model.mobile.onnx`
  - goi tokenizer exporter
  - sinh `golden_samples.jsonl`
  - ghi manifest
- `iter_golden_samples(bundle_dir)`
  - doc fixture punctuation
- `verify_bundle(model_dir, bundle_dir)`
  - load tokenizer ONNX graphs
  - load 2 bridge json
  - compare encode/decode voi Hugging Face tokenizer

Object quan trong:
- `ADAPTER`
  - `BundleProjectAdapter` cho project `vpcd`

## `_vpcd_support.py`

Vai tro:
- chua toan bo helper chi danh rieng cho punctuation
- tach khoi `vpcd.py` de file adapter chinh gon hon

Class chinh:
- `TokenizerExportArtifacts`
  - ten file tokenizer artifacts sau khi export
- `TokenizerIdBridge`
  - chua 2 dense id map:
    - tokenizer -> model
    - model -> tokenizer
  - method `write_files(...)` de ghi json
- `BundleOnnxRuntime`
  - runtime bundle-only cho punctuation
  - dung trong smoke runner `test_punctuation_model_onnx.py`

Ham chinh:
- `ensure_local_vendor_path()`
  - dua `_vendor` vao `sys.path` neu can
- `resolve_variant_onnx_path(model_dir, model_variant)`
  - tim file ONNX variant
- `bartpho_tokenizer_ortx_alias(tokenizer)`
  - context manager de ORT Extensions chap nhan tokenizer alias
- `build_ort_tokenizer_id_bridge(tokenizer)`
  - sinh 2 dense id map giua tokenizer graph ids va model ids
- `default_tokenizer_exporter(model_dir, bundle_dir)`
  - export `tokenizer.encode.onnx`, `tokenizer.decode.onnx`, va 2 bridge json
- `default_golden_sample_builder(...)`
  - chay reference runtime de sinh fixture punctuation

## `zipformer.py`

Vai tro:
- adapter RNNT cho acoustic model Zipformer
- export bundle FP32 reference
- load bundle runtime tu manifest
- verify transcript giua model-dir mode, reference bundle, va candidate bundle

Class chinh:
- `ZipformerRuntimeBase`
  - helper chung cho loading features, token table, va decode token ids
- `ModelDirAcousticRuntime`
  - runtime doc truc tiep tu `assets/zipformer`
- `BundleAcousticRuntime`
  - runtime doc tu `bundle_manifest.json`

Ham chinh:
- `prepare_encoder_inputs(features, fixed_encoder_frames=None)`
  - pad/truncate input cho encoder
- `trim_encoder_frames(encoder_frames, encoder_out_lens)`
  - trim frame hop le sau encoder
- `resolve_fixed_encoder_frames(metadata)`
  - doc metadata fixed-shape trong manifest
- `export_bundle(...)`
  - copy `encoder/decoder/joiner/tokens`
  - sinh `sample_manifest.jsonl`
  - sinh `expected_outputs.jsonl`
  - ghi manifest
- `verify_bundle(...)`
  - mode 1: so `model_dir` voi `bundle_dir`
  - mode 2: so `reference_bundle` voi `candidate_bundle`

Object quan trong:
- `ADAPTER`
  - `BundleProjectAdapter` cho project `zipformer`

## Cach nghi ve adapter

- Shared core giai quyet:
  - registry
  - manifest
  - generic dispatch
- Adapter giai quyet:
  - artifact nao duoc xuat
  - runtime nao duoc tao
  - verify parity theo tieu chi nao
