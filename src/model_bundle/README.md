# Model Bundle Module

`src/model_bundle/` is now a thin internal contract package.

It keeps only the bundle contract pieces that are still shared by AI Hub, verification, quantize-owned bundle helpers, and Android handoff.

It no longer owns the public export or verification CLIs.

For the canonical high-level bundle contract, use:

- `docs/architecture/bundle-contract.md`

## Goals

- keep one shared manifest contract for multiple model families
- keep fixtures and tiny runtime helpers close to that contract
- avoid keeping retired exporter, verifier, or adapter layers alive by accident

## File map

```text
python-model-test/src/model_bundle/
  __init__.py
  fixtures.py
  manifest.py
  vpcd_runtime.py
  vpcd_shapes.py
  zipformer_runtime.py
  README.md
```

## What each script is responsible for

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

### `zipformer_runtime.py`

Keeps only the retained Zipformer runtime helpers shared by AI Hub and bundle verification.

Main functions and classes:

- `prepare_encoder_inputs(...)`
- `trim_encoder_frames(...)`
- `resolve_fixed_encoder_frames(...)`
- `decode_encoder_frames_greedy(...)`
- `ModelDirAcousticRuntime`
- `BundleAcousticRuntime`

### `vpcd_shapes.py`

Keeps the fixed-shape helpers shared by AI Hub and retained VPCD flows.

Main functions:

- `resolve_vpcd_model_input_shapes(...)`
- `pad_token_row(...)`
- `attention_mask_for_length(...)`

### `vpcd_runtime.py`

Keeps the retained VPCD runtime helpers shared by AI Hub, bundle verification, and bundle export.

Main functions:

- `ensure_local_vendor_path(...)`
- `resolve_variant_onnx_path(...)`
- `ModelDirOnnxRuntime`
- `BundleOnnxRuntime`
- tokenizer bridge and export helper utilities

## Shared dependency from `src/tools/`

Project adapters can keep fixture rows repo-relative, for example `assets/speech/sample-1.mp3`.

They resolve those rows through `tools.paths.resolve_repo_path(...)`, which means:
- bundle verification does not depend on fragile `Path(__file__).parents[...]` assumptions
- refactors inside `src/` do not break fixture lookup
- the same manifest rows stay portable across reference and candidate bundle flows

## Public owners outside this package

- manual bundle export CLI:
  - `python -m tools.bundle_export ...`
- bundle verification CLI:
  - `python -m verify.model_bundle ...`
- QNN preflight CLI:
  - `python -m verify.qnn_preflight ...`
- source punctuation ONNX refresh helper:
  - `python -m tools.punctuation_onnx ...`

## Standard bundle layout

```text
build/model_bundle/<project>/<variant>/
  bundle_manifest.json
  ...artifacts...
  ...fixtures...
```

Examples:

```text
build/model_bundle/vpcd/vpcd_balanced/
  model.mobile.onnx
  tokenizer.encode.onnx
  tokenizer.decode.onnx
  tokenizer.to_model_id_map.json
  tokenizer.from_model_id_map.json
  golden_samples.jsonl

build/model_bundle/vpcd/qnn_fixed_1024x128/
  bundle_manifest.json
  model.mobile.onnx
  tokenizer.encode.onnx
  tokenizer.decode.onnx
  tokenizer.to_model_id_map.json
  tokenizer.from_model_id_map.json
  golden_samples.jsonl
  qnn_preflight_report.json

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

## VPCD-specific metadata

`vpcd` bundles carry runtime behavior and NPU-preflight metadata:

- `input_text_case: "lower"`
  - tells bundle consumers to lowercase incoming text before running the exported tokenizer graph
  - keeps Python bundle-manifest mode aligned with the Android runtime
  - prevents ASR-style uppercase transcripts from drifting away from the punctuation model's expected input distribution
- `quantization`
  - present for the default `vpcd_balanced` variant
  - declares `format: "QDQ"`, `activation_type: "quint16"`, `weight_type: "quint8"`, and `preset: "sd8g2_balanced"`
  - intentionally declares `fixed_shapes: false` until a fixed-shape VPCD export or session override is validated
- `qnn_readiness`
  - marks `model.mobile.onnx` as the first QNN HTP candidate
  - records that tokenizer graphs stay CPU-only in the first slice
  - records the current blocker: the VPCD model inputs still use symbolic sequence dimensions

The fixed-shape candidate `qnn_fixed_1024x128` changes that blocker into explicit fixed-shape metadata:

- `fixed_input_shapes.model.input_ids: [1, 1024]`
- `fixed_input_shapes.model.attention_mask: [1, 1024]`
- `fixed_input_shapes.model.decoder_input_ids: [1, 128]`
- `fixed_input_shapes.model.decoder_attention_mask: [1, 128]`
- `quantization.fixed_shapes: true`
- `qnn_readiness.fixed_shapes_ready: true`
