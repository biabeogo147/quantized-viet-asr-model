# VLSP Zipformer Balanced Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic calibration-subset workflow from `E:\data\vlsp2020_vinai_100h\data` so Zipformer can be quantized with a reproducible VLSP-based audio subset, and reuse the same subset's transcriptions to evaluate balanced quantization stability for both `zipformer` and `vpcd`.

**Architecture:** Introduce one small extraction tool that reads the VLSP parquet shards, picks a bounded and deterministic subset, and emits two artifacts into `build/`: an audio manifest for Zipformer and a text calibration file for VPCD. Keep the existing project adapters in `src/quantize/projects/` unchanged at the interface level so the new dataset-driven flow is only an input-preparation layer plus a documented evaluation loop.

**Tech Stack:** Python 3, local parquet reader (`pyarrow` or equivalent), ONNX Runtime, existing `src/quantize` and `src/model_bundle` modules, pytest

---

## Scope To Lock

- This plan is for `python-model-test` only.
- The VLSP source stays external at `E:\data\vlsp2020_vinai_100h\data`.
- The plan must not hardcode one-off manual manifests.
- Calibration subset generation must be deterministic:
  - same shard order
  - same row order
  - explicit max sample count
- The Zipformer quantize CLI should continue consuming a plain text audio manifest.
- The VPCD quantize CLI should continue consuming a plain text calibration file.
- The extraction layer should translate VLSP parquet into those two existing formats, rather than changing quantize CLI contracts.

## Dataset Assumptions

From the local dataset README:

- dataset format is Hugging Face style parquet shards
- features include `audio` and `transcription`
- split available here is `train`

Implementation must verify the exact parquet schema before wiring the extractor permanently, because the current session may not have a parquet reader installed.

## Planned File Map

### Extraction And Dataset Prep

- Create: `python-model-test/src/tools/extract_vlsp2020_calibration_subset.py`
  - reads parquet shards
  - selects a deterministic subset
  - writes:
    - `build/calibration/vlsp2020/zipformer_audio_manifest.txt`
    - `build/calibration/vlsp2020/vpcd_transcriptions.txt`
    - `build/calibration/vlsp2020/subset_manifest.json`
- Create: `python-model-test/test/test_extract_vlsp2020_calibration_subset.py`
  - unit tests for deterministic row selection and emitted file formats

### Quantize And Eval Documentation

- Modify: `python-model-test/src/quantize/README.md`
- Modify: `python-model-test/src/tools/README.md`
- Modify: `python-model-test/README.md`

## Deterministic Subset Policy

Use a small first-pass subset:

- target size for Zipformer audio calibration: `24` samples
- target size for VPCD text calibration: the same `24` transcriptions
- shard policy: lexical order
- row policy: original row order
- stop as soon as target size is reached

Reasoning:

- small enough to run on a developer machine
- larger than the current 1-2 sample smoke runs
- deterministic and easy to debug
- enough to expose obvious decoder collapse or punctuation instability without pretending to be a full benchmark

## Output Contract

### `zipformer_audio_manifest.txt`

One absolute audio path per line.

### `vpcd_transcriptions.txt`

One normalized transcription per line:

- strip whitespace
- drop empty rows
- preserve Vietnamese unicode

### `subset_manifest.json`

Record:

- source dataset root
- selected shard names
- selected row count
- emitted file paths
- optional row ids or file-relative indices for reproducibility

## Task 1: Add The VLSP Extraction Test Skeleton

**Files:**
- Create: `python-model-test/test/test_extract_vlsp2020_calibration_subset.py`

- [ ] **Step 1: Write the failing test for deterministic subset ordering**
- [ ] **Step 2: Write the failing test for emitted manifest formats**
- [ ] **Step 3: Run the tests and confirm they fail**

Run:

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m pytest .\python-model-test\test\test_extract_vlsp2020_calibration_subset.py -q
```

Expected: FAIL because the extraction tool does not exist yet.

## Task 2: Implement The Extraction Tool

**Files:**
- Create: `python-model-test/src/tools/extract_vlsp2020_calibration_subset.py`
- Modify: `python-model-test/src/tools/__init__.py`

- [ ] **Step 1: Implement a parquet row iterator**
- [ ] **Step 2: Implement deterministic row selection**
- [ ] **Step 3: Implement output writers**
- [ ] **Step 4: Re-run the new tests**

Expected: PASS

## Task 3: Wire A Real VLSP Prep Command

**Files:**
- Modify: `python-model-test/src/tools/README.md`
- Modify: `python-model-test/README.md`

- [ ] **Step 1: Add the documented command**

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
& D:\Anaconda\envs\speech2text\python.exe -m tools.extract_vlsp2020_calibration_subset `
  --dataset-root E:\data\vlsp2020_vinai_100h\data `
  --max-samples 24 `
  --output-dir .\build\calibration\vlsp2020
```

- [ ] **Step 2: Document how the emitted artifacts feed `zipformer` and `vpcd` quantization**

## Task 4: Run Balanced Quantization And Stability Comparison

**Files:**
- Modify: `python-model-test/src/quantize/README.md`
- Optionally create: `python-model-test/build/quantize_eval/<timestamp>/comparison_report.md`

- [ ] **Step 1: Export or refresh the FP32 Zipformer reference bundle**
- [ ] **Step 2: Run Zipformer balanced quantization on the VLSP subset**
- [ ] **Step 3: Run VPCD balanced quantization on the same subset's transcriptions**
- [ ] **Step 4: Compare quantized outputs against FP32 outputs**
- [ ] **Step 5: Summarize stability**

Minimum report fields:

- calibration subset size
- Zipformer exact-match count
- VPCD side-by-side outputs
- obvious regressions or a clear "stable enough / not stable enough" decision

## Task 5: Add Regression Tests For Preset Naming

**Files:**
- Modify: `python-model-test/test/test_zipformer_quantize.py`
- Modify: `python-model-test/src/quantize/projects/zipformer.py`

- [ ] **Step 1: Ensure the Zipformer project accepts `zipformer_sd8g2_balanced`**
- [ ] **Step 2: Keep backward compatibility for the existing preset alias if needed**
- [ ] **Step 3: Re-run targeted quantize tests**

## Exit Criteria

- a deterministic VLSP subset prep command exists
- both `zipformer` and `vpcd` can consume the generated artifacts without changing their CLI contracts
- balanced quantization has been run on a non-trivial VLSP-derived subset
- FP32 vs quantized comparisons are written down clearly enough to decide the next tuning step
