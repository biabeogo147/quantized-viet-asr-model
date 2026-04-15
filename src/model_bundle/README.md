# Model Bundle Module

`src/model_bundle/` is the shared core for the Python-side export, verification, and bundle-consumption flow.

## Goals

- define one shared bundle contract for multiple model families
- separate generic bundle mechanics from model-specific logic
- let Android and Python consume the same layout and manifest structure

## File map

```text
python-model-test/src/model_bundle/
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

## What each script is responsible for

### `contracts.py`

Defines the shared data types used by the bundle modules.

Main classes and functions:
- `BundleRuntimeProtocol`
  - protocol marker for runtimes constructed from bundles
- `BundleVerificationReport`
  - dataclass that summarizes verification results
- `BundleProjectAdapter`
  - the most important registry dataclass
  - contains:
    - `name`
    - default paths
    - `export_bundle`
    - `verify_bundle`
    - `bundle_runtime_from_manifest`
- `normalize_path(value)`
  - helper that converts strings into `Path`

### `manifest.py`

Describes the on-disk bundle contract and how manifests are read and written.

Main classes and functions:
- `ModelBundleManifest`
  - dataclass that represents `bundle_manifest.json`
  - important fields:
    - `project`
    - `model_family`
    - `model_variant`
    - `asset_namespace`
    - `runtime_kind`
    - `artifacts`
    - `fixtures`
    - `metadata`
  - important methods:
    - `to_dict()`
    - `from_dict()`
    - `from_path()`
    - `write_json()`
    - `resolve_artifact_path()`
    - `resolve_fixture_path()`
- `_from_legacy_punctuation(...)`
- `_from_legacy_zipformer(...)`
  - compatibility hooks for older manifest layouts when needed

### `fixtures.py`

Normalizes sample fixtures used for export and verification.

Main classes and functions:
- `TextGoldenSample`
  - row schema for punctuation bundles
- `AudioSampleFixture`
  - input row schema for ASR bundles
- `AudioExpectedOutput`
  - expected transcript row schema for ASR bundles
- `serialize_jsonl(items)`
  - writes dataclasses as JSONL
- `read_jsonl(path)`
  - reads JSONL into a list of dicts

### `layout.py`

Handles one job:
- `resolve_bundle_dir(project, variant)`
  - normalizes output directories into `build/model_bundle/<project>/<variant>`

### `exporter.py`

Generic dispatcher for bundle export.

Main functions:
- `_filter_kwargs(callable_obj, kwargs)`
  - passes only the arguments supported by the adapter
- `export_model_bundle(project, model_dir, output_dir, **kwargs)`
  - resolves the adapter
  - routes into `adapter.export_bundle(...)`

### `verifier.py`

Generic dispatcher for bundle verification.

Main functions:
- `_filter_kwargs(callable_obj, kwargs)`
  - same idea as in the exporter
- `verify_model_bundle(project, **kwargs)`
  - normalizes path-like kwargs
  - routes into `adapter.verify_bundle(...)`

### `projects/__init__.py`

Registry for the shared core.

Main functions:
- `resolve_bundle_project(name)`
  - returns the adapter for `vpcd` or `zipformer`
- `list_bundle_projects()`
  - returns the tuple of supported project names

## The two main adapters

- `projects/vpcd.py`
  - adapter for punctuation seq2seq bundles
- `projects/zipformer.py`
  - adapter for RNNT acoustic bundles

Adapter details are documented in `src/model_bundle/projects/README.md`.

## Shared flow

### Export flow

`export.model_bundle`
-> `model_bundle.exporter.export_model_bundle(...)`
-> `resolve_bundle_project(project)`
-> `adapter.export_bundle(...)`
-> write `bundle_manifest.json` + artifacts + fixtures

### Verify flow

`verify.model_bundle`
-> `model_bundle.verifier.verify_model_bundle(...)`
-> `resolve_bundle_project(project)`
-> `adapter.verify_bundle(...)`

## Shared dependency from `src/tools/`

Project adapters can keep fixture rows repo-relative, for example `assets/speech/sample-1.mp3`.

They resolve those rows through `tools.paths.resolve_repo_path(...)`, which means:
- bundle verification does not depend on fragile `Path(__file__).parents[...]` assumptions
- refactors inside `src/` do not break fixture lookup
- the same manifest rows stay portable across reference and candidate bundle flows

## Standard bundle layout

```text
build/model_bundle/<project>/<variant>/
  bundle_manifest.json
  ...artifacts...
  ...fixtures...
```

Examples:

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
