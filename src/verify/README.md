# Verify Module

`src/verify/` contains the CLIs used to check whether exported artifacts still match the expected contract and quality gates.

## Goals

- provide a single verification entrypoint per `project`
- keep `quantize` and smoke-test code from re-implementing comparison logic
- print clear mismatch reports when a candidate bundle diverges from the reference

## File map

```text
python-model-test/src/verify/
  __init__.py
  model_bundle.py
  README.md
```

## Command setup

Commands below assume one of these is true:

- the repo is installed in editable mode, or
- the current shell has `PYTHONPATH` pointing to `src/`

Example from `python-model-test/`:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
```

## What each script is responsible for

### `model_bundle.py`

This is the canonical CLI for bundle verification.

Problems it solves:
- for `vpcd`, it verifies encode/decode parity between the bundle and the Hugging Face tokenizer
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
    - checked samples, pass/fail, and mismatches for `zipformer`

## How to read the output

- If the output contains:
  - `Encode samples : ...`
  - `Decode samples : ...`
  then you are looking at tokenizer-bundle verification for `vpcd`.

- If the output contains:
  - `Checked samples: ...`
  - `Passed : True/False`
  - a `mismatches` list
  then you are looking at transcript verification for `zipformer`.

## Common commands

### Verify a punctuation bundle

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m verify.model_bundle `
  --project vpcd `
  --model-dir D:\DS-AI\BKMeeting-Research\python-model-test\assets\vietnamese-punc-cap-denorm-v1 `
  --bundle-dir D:\DS-AI\BKMeeting-Research\python-model-test\build\model_bundle\vpcd\fp32
```

### Verify a Zipformer FP32 bundle against `model-dir`

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m verify.model_bundle `
  --project zipformer `
  --model-dir D:\DS-AI\BKMeeting-Research\python-model-test\assets\zipformer `
  --bundle-dir D:\DS-AI\BKMeeting-Research\python-model-test\build\model_bundle\zipformer\fp32
```

### Verify a Zipformer quantized candidate against the FP32 reference bundle

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m verify.model_bundle `
  --project zipformer `
  --reference-bundle D:\DS-AI\BKMeeting-Research\python-model-test\build\model_bundle\zipformer\fp32 `
  --candidate-bundle D:\DS-AI\BKMeeting-Research\python-model-test\build\model_bundle\zipformer\qnn_u16u8
```

## Relationship to other modules

- this CLI is only a wrapper
- the actual generic verification logic lives in `model_bundle/verifier.py`
- project-specific comparison logic lives in:
  - `model_bundle/projects/vpcd.py`
  - `model_bundle/projects/zipformer.py`
