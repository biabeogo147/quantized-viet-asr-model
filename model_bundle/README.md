# Model Bundle Module

`model_bundle/` la shared core cho toan bo flow export, verify, va consume bundle trong repo Python nay.

## Muc tieu

- co mot contract bundle chung cho nhieu model family
- tach core generic khoi logic dac thu tung model
- de Android va Python cung doc cung mot layout va manifest

## File map

```text
python-model-test/model_bundle/
  __init__.py
  contracts.py
  exporter.py
  fixtures.py
  layout.py
  manifest.py
  verifier.py
  projects/
    __init__.py
    vpcd.py
    zipformer.py
    _vpcd_support.py
  README.md
```

## Tung script giai quyet van de gi

### `contracts.py`

No dinh nghia cac kieu du lieu chung de cac module noi voi nhau.

Class/ham chinh:
- `BundleRuntimeProtocol`
  - protocol marker cho runtime duoc tao tu bundle
- `BundleVerificationReport`
  - dataclass tong hop ket qua verify
- `BundleProjectAdapter`
  - dataclass quan trong nhat cua registry
  - chua:
    - `name`
    - default paths
    - `export_bundle`
    - `verify_bundle`
    - `bundle_runtime_from_manifest`
- `normalize_path(value)`
  - helper chuyen string sang `Path`

### `manifest.py`

No la noi mo ta contract bundle tren dia va cach doc/ghi manifest.

Class/ham chinh:
- `ModelBundleManifest`
  - dataclass dai dien cho `bundle_manifest.json`
  - field quan trong:
    - `project`
    - `model_family`
    - `model_variant`
    - `asset_namespace`
    - `runtime_kind`
    - `artifacts`
    - `fixtures`
    - `metadata`
  - method quan trong:
    - `to_dict()`
    - `from_dict()`
    - `from_path()`
    - `write_json()`
    - `resolve_artifact_path()`
    - `resolve_fixture_path()`
- `_from_legacy_punctuation(...)`
- `_from_legacy_zipformer(...)`
  - compatibility hook de doc manifest cu neu can

### `fixtures.py`

No chuan hoa sample fixture de export va verify.

Class/ham chinh:
- `TextGoldenSample`
  - row cho punctuation bundle
- `AudioSampleFixture`
  - row input cho ASR bundle
- `AudioExpectedOutput`
  - row transcript expected cho ASR bundle
- `serialize_jsonl(items)`
  - ghi dataclass thanh JSONL
- `read_jsonl(path)`
  - doc JSONL ve list dict

### `layout.py`

No giai quyet mot viec duy nhat:
- `resolve_bundle_dir(project, variant)`
  - chuan hoa output dir thanh `build/model_bundle/<project>/<variant>`

### `exporter.py`

No la generic dispatcher cho export bundle.

Ham chinh:
- `_filter_kwargs(callable_obj, kwargs)`
  - chi truyen nhung argument ma adapter support
- `export_model_bundle(project, model_dir, output_dir, **kwargs)`
  - resolve adapter
  - route qua `adapter.export_bundle(...)`

### `verifier.py`

No la generic dispatcher cho verify bundle.

Ham chinh:
- `_filter_kwargs(callable_obj, kwargs)`
  - giong exporter
- `verify_model_bundle(project, **kwargs)`
  - normalize cac path-like kwargs
  - route qua `adapter.verify_bundle(...)`

### `projects/__init__.py`

No la registry cua shared core.

Ham chinh:
- `resolve_bundle_project(name)`
  - tra ve adapter cho `vpcd` hoac `zipformer`
- `list_bundle_projects()`
  - tra ve tuple project names duoc support

## Hai adapter chinh

- `projects/vpcd.py`
  - adapter cho punctuation seq2seq bundle
- `projects/zipformer.py`
  - adapter cho RNNT acoustic bundle

Chi tiet tung adapter duoc mo ta o `model_bundle/projects/README.md`.

## Shared flow

### Export flow

`export.model_bundle`
-> `model_bundle.exporter.export_model_bundle(...)`
-> `resolve_bundle_project(project)`
-> `adapter.export_bundle(...)`
-> ghi `bundle_manifest.json` + artifact + fixture

### Verify flow

`verify.model_bundle`
-> `model_bundle.verifier.verify_model_bundle(...)`
-> `resolve_bundle_project(project)`
-> `adapter.verify_bundle(...)`

## Bundle layout chuan

```text
build/model_bundle/<project>/<variant>/
  bundle_manifest.json
  ...artifacts...
  ...fixtures...
```

Vi du:

```text
build/model_bundle/vpcd/fp32/
  model.mobile.onnx
  tokenizer.encode.onnx
  tokenizer.decode.onnx
  tokenizer.to_model_id_map.json
  tokenizer.from_model_id_map.json
  golden_samples.jsonl

build/model_bundle/zipformer/qnn_u16u8/
  encoder.onnx
  decoder.onnx
  joiner.onnx
  tokens.txt
  sample_manifest.jsonl
  expected_outputs.jsonl
  quantization_report.json
  evaluation_report.json
```
