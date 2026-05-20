# AI Hub Deployment Packaging Results

> Historical note: this archive note preserves some original rollout terminology such as `Option 1` and `Step 5` where it was used to describe exact retained evidence keys. The current implementation now lives under `src/aihub/`, and the current operator docs live under `docs/workflows/aihub-*`.

Date: `2026-05-20`

This note records the implementation and real execution results for the retained AI Hub deployment-packaging flow in `python-model-test`, after the retained `zipformer` and `vpcd` notebook evidence had already completed Step 3 and Step 4.

## Executive Summary

Deployment packaging is now implemented as a Python-side post-notebook flow:

- it resolves retained compile plus live-run evidence by `RUN_LABEL`
- it downloads the AI Hub `precompiled_qnn_onnx` target artifact without recompiling
- it writes a deterministic deployment package per project under `build/aihub/deploy/...`
- it exports a normalized `io_contract.json`
- it copies the retained proof records into an `evidence/` folder for later handoff

Real package creation succeeded for both retained pilots on `2026-05-20`.

## What Was Implemented

### Shared helper surface

Current code locations:

- [session.py](/D:/DS-AI/BKMeeting-Research/python-model-test/src/aihub/session.py)
- [deployment.py](/D:/DS-AI/BKMeeting-Research/python-model-test/src/aihub/deployment.py)

Historical implementation details below preserve the original rollout wording for auditability.

Updated shared helper module:

- [aihub_option1_pilots.py](/D:/DS-AI/BKMeeting-Research/python-model-test/src/tools/aihub_option1_pilots.py)

New Step 5 helper behavior:

- `download_compiled_target_model(...)`
  - downloads the compiled target artifact from AI Hub
  - creates parent directories automatically
  - fails clearly if the resolved download path does not exist
- `write_deployable_download_record(...)`
  - writes a retained Step 5 download record beside the other pilot records
  - records downloaded artifact metadata for later packaging

Focused test coverage added in:

- [test_aihub_option1_pilots.py](/D:/DS-AI/BKMeeting-Research/python-model-test/test/test_aihub_option1_pilots.py)

### Step 5 package builder and CLI

Added new module:

- [aihub_option1_step5_artifacts.py](/D:/DS-AI/BKMeeting-Research/python-model-test/src/tools/aihub_option1_step5_artifacts.py)

This module now owns:

- deterministic pilot layout resolution for `zipformer` and `vpcd`
- required-record resolution from retained notebook outputs
- Step 5 package materialization
- `io_contract.json` generation from record-backed shapes and dtypes
- `deploy_notes.md` generation for the runtime split
- CLI execution through `python -m tools.aihub_option1_step5_artifacts`

Focused test coverage added in:

- [test_aihub_option1_step5_artifacts.py](/D:/DS-AI/BKMeeting-Research/python-model-test/test/test_aihub_option1_step5_artifacts.py)

### Workflow documentation refresh

Added operator doc:

- [option1-step5-download.md](/D:/DS-AI/BKMeeting-Research/python-model-test/docs/workflows/option1-step5-download.md)

Updated workflow/index docs:

- [README.md](/D:/DS-AI/BKMeeting-Research/python-model-test/docs/workflows/README.md)
- [option1-overview.md](/D:/DS-AI/BKMeeting-Research/python-model-test/docs/workflows/option1-overview.md)
- [option1-rerun.md](/D:/DS-AI/BKMeeting-Research/python-model-test/docs/workflows/option1-rerun.md)
- [android-handoff.md](/D:/DS-AI/BKMeeting-Research/python-model-test/docs/workflows/android-handoff.md)
- [README.md](/D:/DS-AI/BKMeeting-Research/python-model-test/src/tools/README.md)

## Verification

All verification below was run with:

- `D:\Anaconda\envs\speech2text\python.exe`

Reason:

- the repo-local environment used in this session already had the required test dependencies available

Commands and results:

```powershell
& 'D:\Anaconda\envs\speech2text\python.exe' -m pytest test/test_aihub_option1_pilots.py -k "download_compiled_target_model or write_deployable_download_record" -v -p no:cacheprovider
```

- result: `3 passed, 23 deselected`

```powershell
& 'D:\Anaconda\envs\speech2text\python.exe' -m pytest test/test_aihub_option1_step5_artifacts.py -v -p no:cacheprovider
```

- result: `3 passed`

```powershell
& 'D:\Anaconda\envs\speech2text\python.exe' -m pytest test/test_option1_notebook_layout.py -v -p no:cacheprovider
```

- result: `11 passed`
- meaning: [On_device_Ai_option1_pilots.ipynb](/D:/DS-AI/BKMeeting-Research/python-model-test/On_device_Ai_option1_pilots.ipynb) remains outside Step 5 packaging scope

```powershell
& 'D:\Anaconda\envs\speech2text\python.exe' -m compileall src
```

- result: passed

Dry-run verification:

```powershell
& 'D:\Anaconda\envs\speech2text\python.exe' -m tools.aihub_option1_step5_artifacts --pilot all --run-label 20260519-6pm --repo-root . --device-name "Samsung Galaxy S24 (Family)" --dry-run
```

- result: both pilots resolved successfully
- resolved target model ids:
  - `zipformer`: `mqero78kn`
  - `vpcd`: `mmxwpeyen`

## Real Package Build Result

Real Step 5 package build command:

```powershell
$envPath = 'D:\DS-AI\BKMeeting-Research\python-model-test\.env'
$tokenLine = Get-Content -Path $envPath | Where-Object { $_ -match '^QAI_HUB_API_TOKEN=' } | Select-Object -First 1
$env:QAI_HUB_API_TOKEN = $tokenLine.Substring('QAI_HUB_API_TOKEN='.Length)
& 'D:\Anaconda\envs\speech2text\python.exe' -m tools.aihub_option1_step5_artifacts --pilot all --run-label 20260519-6pm --repo-root . --device-name "Samsung Galaxy S24 (Family)"
```

- result: passed
- both retained pilots downloaded successfully from AI Hub

### Zipformer package

Package root:

- [zipformer Step 5 package](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/step5/option1/zipformer/20260519-6pm)

Key outputs:

- [step5_manifest.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/step5/option1/zipformer/20260519-6pm/step5_manifest.json)
- [io_contract.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/step5/option1/zipformer/20260519-6pm/io_contract.json)
- [deploy_notes.md](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/step5/option1/zipformer/20260519-6pm/deploy_notes.md)
- downloaded artifact:
  - [encoder.precompiled_qnn_onnx.onnx.onnx.zip](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/step5/option1/zipformer/20260519-6pm/download/encoder.precompiled_qnn_onnx.onnx.onnx.zip)

Observed metadata:

- target model id: `mqero78kn`
- artifact size: `50062413` bytes
- runtime split:
  - compiled target: encoder
  - CPU-side: decoder and joiner
- I/O handling:
  - `x_lens` was recorded as `source_dtype = int64`
  - deploy contract normalizes it to `dtype = int32` because compile used `--truncate_64bit_io`

### VPCD package

Package root:

- [vpcd Step 5 package](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/step5/option1/vpcd/20260519-6pm)

Key outputs:

- [step5_manifest.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/step5/option1/vpcd/20260519-6pm/step5_manifest.json)
- [io_contract.json](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/step5/option1/vpcd/20260519-6pm/io_contract.json)
- [deploy_notes.md](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/step5/option1/vpcd/20260519-6pm/deploy_notes.md)
- downloaded artifact:
  - [model.precompiled_qnn_onnx.onnx.onnx.zip](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/step5/option1/vpcd/20260519-6pm/download/model.precompiled_qnn_onnx.onnx.onnx.zip)

Observed metadata:

- target model id: `mmxwpeyen`
- artifact size: `641242813` bytes
- runtime split:
  - compiled target: VPCD model session
  - CPU-side: tokenizer encode and tokenizer decode
- I/O handling:
  - all four integer inputs were recorded as `source_dtype = int64`
  - deploy contract normalizes them to `dtype = int32` because compile used `--truncate_64bit_io`

## Notes And Follow-Ups

- The current package manifests record `qairt_version: null` because no explicit `--qairt-version` override was supplied during the real Step 5 build.
- AI Hub returned downloadable filenames that ended with `.onnx.onnx.zip`. The files were kept as-downloaded so the retained metadata matches the real artifact path exactly.
- The shared pilot notebook was intentionally left unchanged for Step 5 packaging, and the notebook-layout guard passed after the implementation.
- No repository commit was created in this session.

## Current Recommendation

For Python-side handoff, treat the Step 5 packages below as the retained deploy inputs:

- [zipformer Step 5 package](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/step5/option1/zipformer/20260519-6pm)
- [vpcd Step 5 package](/D:/DS-AI/BKMeeting-Research/python-model-test/build/aihub/step5/option1/vpcd/20260519-6pm)

Do not treat Step 5 package creation as proof of Android Snapdragon runtime success. It proves the retained AI Hub target can be downloaded and packaged deterministically on the Python side.
