# Model Bundle Project Adapters

`src/model_bundle/projects/` contains model-family-specific logic. The shared core does not know tokenizer details or RNNT details. It only knows how to call adapters.

## File map

```text
python-model-test/src/model_bundle/projects/
  __init__.py
  vpcd.py
  zipformer.py
  _vpcd_support.py
  README.md
```

## `__init__.py`

Role:
- project adapter registry
- connection point between the `model_bundle` core and concrete adapters

Main functions:
- `resolve_bundle_project(name)`
- `list_bundle_projects()`

## `vpcd.py`

Role:
- adapter for the `tourmii/vietnamese-punc-cap-denorm-v1` punctuation model
- exports punctuation bundles
- verifies tokenizer encode/decode parity

Main functions:
- `export_bundle(...)`
  - copies `model.mobile.onnx`
  - calls the tokenizer exporter
  - generates `golden_samples.jsonl`
  - writes the manifest
- `iter_golden_samples(bundle_dir)`
  - reads punctuation fixtures
- `verify_bundle(model_dir, bundle_dir)`
  - loads tokenizer ONNX graphs
  - loads the two bridge JSON files
  - compares encode/decode behavior against the Hugging Face tokenizer

Important object:
- `ADAPTER`
  - the `BundleProjectAdapter` for project `vpcd`

## `_vpcd_support.py`

Role:
- contains helpers that are specific to the punctuation pipeline
- keeps `vpcd.py` smaller and easier to read

Main classes:
- `TokenizerExportArtifacts`
  - names the tokenizer export files
- `TokenizerIdBridge`
  - stores two dense ID maps:
    - tokenizer -> model
    - model -> tokenizer
  - method `write_files(...)` writes the JSON files
- `BundleOnnxRuntime`
  - bundle-only punctuation runtime
  - used by the smoke runner in `test_punctuation_model_onnx.py`

Main functions:
- `ensure_local_vendor_path()`
  - adds `_vendor` to `sys.path` when needed
- `resolve_variant_onnx_path(model_dir, model_variant)`
  - finds the ONNX file for the selected variant
- `bartpho_tokenizer_ortx_alias(tokenizer)`
  - context manager that makes ORT Extensions accept the tokenizer alias
- `build_ort_tokenizer_id_bridge(tokenizer)`
  - generates the two dense ID maps between tokenizer-graph IDs and model IDs
- `default_tokenizer_exporter(model_dir, bundle_dir)`
  - exports `tokenizer.encode.onnx`, `tokenizer.decode.onnx`, and the two bridge JSON files
- `default_golden_sample_builder(...)`
  - runs the reference runtime to build punctuation fixtures

Note:
- `_vpcd_support.py` now resolves optional repo-local dependencies through `tools.paths.resolve_repo_path(...)` instead of assuming a fixed directory depth under `src/`

## `zipformer.py`

Role:
- RNNT adapter for the Zipformer acoustic model
- exports the FP32 reference bundle
- constructs bundle runtimes from a manifest
- verifies transcripts across model-dir mode, reference bundles, and candidate bundles

Main classes:
- `ZipformerRuntimeBase`
  - shared helper for feature loading, token-table loading, and token decoding
- `ModelDirAcousticRuntime`
  - runtime that reads directly from `assets/zipformer`
- `BundleAcousticRuntime`
  - runtime that reads from `bundle_manifest.json`

Main functions:
- `prepare_encoder_inputs(features, fixed_encoder_frames=None)`
  - pads or trims encoder inputs
- `trim_encoder_frames(encoder_frames, encoder_out_lens)`
  - trims valid frames after the encoder stage
- `resolve_fixed_encoder_frames(metadata)`
  - reads fixed-shape metadata from the manifest
- `export_bundle(...)`
  - copies `encoder`, `decoder`, `joiner`, and `tokens`
  - generates `sample_manifest.jsonl`
  - generates `expected_outputs.jsonl`
  - writes the manifest
- `verify_bundle(...)`
  - mode 1: compare `model_dir` with `bundle_dir`
  - mode 2: compare `reference_bundle` with `candidate_bundle`

Important behavior:
- audio fixtures stay repo-relative in manifests, for example `assets/speech/sample-1.mp3`
- `zipformer.py` resolves those paths through `tools.paths.resolve_repo_path(...)`
- this keeps export and verify flows stable after the move to `src/`

Important object:
- `ADAPTER`
  - the `BundleProjectAdapter` for project `zipformer`

## How to think about adapters

- The shared core handles:
  - the registry
  - the manifest type
  - generic dispatch
- Each adapter handles:
  - which artifacts are exported
  - which runtime gets constructed
  - which parity criteria must hold during verification
