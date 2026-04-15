# Tools Module

`src/tools/` contains small reusable helper scripts that do not belong to the main export, bundle, quantize, or verify flows.

## File map

```text
python-model-test/src/tools/
  __init__.py
  convert_bpe2token.py
  paths.py
  README.md
```

## `paths.py`

Role:
- resolves the `python-model-test/` repo root from code running anywhere under `src/`
- converts repo-relative fixture paths into stable absolute paths
- prevents regressions when packages move deeper or shallower inside `src/`

Main functions:
- `find_repo_root(anchor)`
  - walks upward from a file or directory until it finds the repo root
- `resolve_repo_path(path_like, anchor=...)`
  - returns `<repo-root>/<path_like>`

Use it when:
- sample manifests store paths like `assets/speech/sample-1.mp3`
- calibration logic needs to open repo assets from inside `src/quantize/...`
- bundle verification needs to re-open fixtures from inside `src/model_bundle/...`

## `convert_bpe2token.py`

Role:
- reads a SentencePiece `bpe.model`
- generates a `tokens.txt` table in the format expected by the Zipformer pipeline

Main functions:
- `build_argument_parser()`
  - defines `--bpe-model` and `--output`
- `main(argv=None)`
  - loads the SentencePiece model
  - writes one token per line as `<piece> <id>`

## Command setup

Commands below assume one of these is true:

- the repo is installed in editable mode, or
- the current shell has `PYTHONPATH` pointing to `src/`

Example from `python-model-test/`:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
```

## Example command

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m tools.convert_bpe2token `
  --bpe-model D:\DS-AI\BKMeeting-Research\python-model-test\assets\zipformer\bpe.model `
  --output D:\DS-AI\BKMeeting-Research\python-model-test\assets\zipformer\tokens.txt
```
