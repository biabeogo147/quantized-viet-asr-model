# Verify Module

`src/verify/` contains the CLIs used to check whether exported artifacts still match the expected contract and quality gates.

For the canonical repo-wide verification and QNN preflight flow, use:

- `docs/workflows/export-verify-smoke.md`
- `docs/workflows/quantize-qnn-candidates.md`
- `docs/qnn/preflight.md`

## Goals

- provide a single verification entrypoint per `project`
- keep `quantize` and smoke-test code from re-implementing comparison logic
- print clear mismatch reports when a candidate bundle diverges from the reference

## File map

```text
python-model-test/src/verify/
  __init__.py
  model_bundle.py
  qnn_preflight.py
  README.md
```

## Command setup

Examples below assume you run commands from `python-model-test/`.

## What each script is responsible for

### `model_bundle.py`

This is the canonical CLI for bundle verification.

Problems it solves:
- for `vpcd`, it verifies encode/decode parity between the bundle and the Hugging Face tokenizer
- for `vpcd`, it can also compare a fixed-shape candidate bundle against the dynamic `vpcd_balanced` reference bundle
- for `zipformer`, it verifies transcripts between the model-dir runtime and the bundle runtime, or between a reference bundle and a candidate bundle

Main functions:
- `build_argument_parser()`
  - parses `--project`
  - supports three input modes:
    - `--model-dir` + `--bundle-dir`
    - `--reference-bundle` + `--candidate-bundle`
    - or adapter defaults when applicable
- `main(argv=None)`
  - resolves the project adapter
  - builds valid kwargs for that project
  - calls `verify_model_bundle(...)`
  - prints the summary:
    - encode/decode sample counts for `vpcd`
    - checked samples, pass/fail, and mismatches for candidate comparisons

### `qnn_preflight.py`

This CLI checks whether a bundle is ready for an Android QNN HTP attempt. It does not run HTP.

For VPCD it checks:

- manifest project and `artifacts.model`
- `metadata.quantization` is QDQ with `quint16` activations and `quint8` weights
- `metadata.quantization.fixed_shapes = true`
- `metadata.qnn_readiness.fixed_shapes_ready = true`
- tokenizer policy remains `cpu_only_first_slice`
- ONNX graph has fixed input shapes matching `metadata.fixed_input_shapes.model`
- ONNX graph contains QDQ nodes and `UINT16` / `UINT8` initializers

## How to read the output

- If the output contains:
  - `Encode samples : ...`
  - `Decode samples : ...`
  then you are looking at tokenizer-bundle verification for `vpcd`.

- If the output contains:
  - `Checked samples: ...`
  - `Passed : True/False`
  - a `mismatches` list
  then you are looking at candidate bundle verification.

## Common commands

### Verify a punctuation bundle

```bash
python -m verify.model_bundle \
  --project vpcd \
  --model-dir assets/vietnamese-punc-cap-denorm-v1 \
  --bundle-dir build/model_bundle/vpcd/vpcd_balanced
```

### Verify a Zipformer FP32 bundle against `model-dir`

```bash
python -m verify.model_bundle \
  --project zipformer \
  --model-dir assets/zipformer \
  --bundle-dir build/model_bundle/zipformer/fp32
```

### Verify a Zipformer quantized candidate against the FP32 reference bundle

```bash
python -m verify.model_bundle \
  --project zipformer \
  --reference-bundle build/model_bundle/zipformer/fp32 \
  --candidate-bundle build/model_bundle/zipformer/qnn_u16u8
```

### Verify a fixed-shape VPCD candidate against the dynamic reference bundle

```bash
python -m verify.model_bundle \
  --project vpcd \
  --reference-bundle build/model_bundle/vpcd/vpcd_balanced \
  --candidate-bundle build/model_bundle/vpcd/qnn_fixed_1024x128
```

### Run QNN preflight for a fixed-shape VPCD candidate

```bash
python -m verify.qnn_preflight \
  --project vpcd \
  --bundle-dir build/model_bundle/vpcd/qnn_fixed_1024x128 \
  --output build/model_bundle/vpcd/qnn_fixed_1024x128/qnn_preflight_report.json
```

## Relationship to other modules

- this CLI is only a wrapper
- the actual generic verification logic lives in `model_bundle/verifier.py`
- project-specific comparison logic lives in:
  - `model_bundle/projects/vpcd.py`
  - `model_bundle/projects/zipformer.py`
