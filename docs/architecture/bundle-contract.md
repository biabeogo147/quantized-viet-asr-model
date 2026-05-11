# Bundle Contract

This repo uses one shared model-bundle contract for all supported model families.

The goal is to let:

- Python export and verification
- Python smoke tests
- Android bundle sync
- BKMeeting runtime loading

all speak the same manifest and artifact layout.

## Core bundle structure

Every bundle lives under:

```text
build/model_bundle/<project>/<variant>/
```

Every bundle contains:

- `bundle_manifest.json`
- model artifacts listed in `artifacts`
- optional fixtures listed in `fixtures`

## Shared manifest responsibilities

The manifest records the data Android and Python need to consume a bundle without hard-coding filenames in multiple places.

Important fields include:

- `project`
- `model_family`
- `model_variant`
- `asset_namespace`
- `runtime_kind`
- `artifacts`
- `fixtures`
- `metadata`

The shared Python implementation lives in:

- `src/model_bundle/manifest.py`

## VPCD bundle layout

Typical VPCD bundle:

```text
build/model_bundle/vpcd/<variant>/
  bundle_manifest.json
  model.mobile.onnx
  tokenizer.encode.onnx
  tokenizer.decode.onnx
  tokenizer.to_model_id_map.json
  tokenizer.from_model_id_map.json
  golden_samples.jsonl
```

The fixed-shape candidate also includes:

```text
qnn_preflight_report.json
```

## Zipformer bundle layout

Typical Zipformer bundle:

```text
build/model_bundle/zipformer/<variant>/
  bundle_manifest.json
  encoder.onnx
  decoder.onnx
  joiner.onnx
  tokens.txt
  sample_manifest.jsonl
  expected_outputs.jsonl
```

The quantized candidate bundle also includes:

```text
quantization_report.json
evaluation_report.json
```

## Fixture-path policy

Zipformer fixture manifests may keep repo-relative paths such as:

```text
assets/speech/sample-1.mp3
```

Those paths are resolved through:

- `src/tools/paths.py`

This keeps fixture rows portable across:

- reference runtime checks
- candidate-bundle verification
- smoke tests

## QNN-related metadata in the bundle contract

The bundle contract also carries metadata needed to reason about QNN-readiness before Android integration.

### VPCD metadata

VPCD bundles may record:

- `metadata.input_text_case`
- `metadata.quantization`
- `metadata.qnn_readiness`
- `metadata.fixed_input_shapes`

Important meanings:

- `input_text_case = "lower"`
  - tells consumers to lowercase text before punctuation-tokenizer processing
- `quantization`
  - records QDQ-related facts such as activation and weight types
- `qnn_readiness`
  - records the intended first-slice QNN target and tokenizer CPU policy
- `fixed_input_shapes`
  - records the frozen dimensions used by the fixed-shape candidate

### Zipformer metadata

Zipformer candidate bundles may record:

- fixed encoder-frame metadata
- quantization metadata
- evaluation and report artifacts

This metadata lets Android and Python agree on which shapes and artifacts belong to the quantized candidate.

## What the bundle contract guarantees

When a bundle is valid, Python-side tooling can:

- export it
- verify it
- run it through a bundle-manifest smoke path
- sync it into BKMeeting

## What the bundle contract does not guarantee

The bundle contract does not prove:

- physical Snapdragon HTP execution
- ORT QNN provider compatibility on Android
- benchmark wins on a real device

Those are Android-side concerns and are owned by BKMeeting.

## Related module docs

- `src/model_bundle/README.md`
- `src/model_bundle/projects/README.md`
- `docs/qnn/preflight.md`
