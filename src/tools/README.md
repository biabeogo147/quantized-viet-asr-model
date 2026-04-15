# Tools Module

`src/tools/` contains small reusable helper scripts that do not belong to the main export, bundle, quantize, or verify flows.

## File map

```text
python-model-test/src/tools/
  __init__.py
  convert_bpe2token.py
  extract_vlsp2020_calibration_subset.py
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

## `extract_vlsp2020_calibration_subset.py`

Role:
- reads VLSP 2020 Hugging Face parquet shards from a user-provided dataset root
- selects a deterministic subset in lexical shard order and original row order
- materializes embedded audio bytes into local WAV files under `build/calibration/vlsp2020/audio/`
- emits:
  - `zipformer_audio_manifest.txt`
  - `vpcd_transcriptions.txt`
  - `subset_manifest.json`

Main classes and functions:
- `VlspCalibrationRow`
  - normalized in-memory row used by the extractor
- `iter_vlsp_parquet_rows(dataset_root, batch_size=32)`
  - streams parquet rows with `audio.bytes`, `audio.path`, and `transcription`
- `select_subset_rows(rows, max_samples)`
  - keeps deterministic order and truncates to the requested subset size
- `write_subset_outputs(rows, output_dir)`
  - writes audio files and the two calibration artifacts
- `extract_subset(dataset_root, output_dir, max_samples)`
  - end-to-end helper used by the CLI
- `main(argv=None)`
  - CLI entrypoint

Example command:

```bash
python -m tools.extract_vlsp2020_calibration_subset \
  --dataset-root <vlsp_dataset_root> \
  --max-samples 24 \
  --output-dir build/calibration/vlsp2020
```

The emitted artifacts can be fed directly into:
- `python -m quantize --project zipformer --audio-manifest ...`
- `python -m quantize --project vpcd --calibration-text ...`

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

Examples below assume you run commands from `python-model-test/`.

## Example command

```bash
python -m tools.convert_bpe2token \
  --bpe-model assets/zipformer/bpe.model \
  --output assets/zipformer/tokens.txt
```
