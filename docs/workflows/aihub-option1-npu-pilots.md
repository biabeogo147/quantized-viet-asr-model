# AI Hub Option 1 NPU Pilots

This workflow is the Python-side entry point for validating the BKMeeting model candidates on Qualcomm AI Hub before any Android packaging work.

Use this workflow when you want to prove:

- the model can be quantized in the Qualcomm-official lane when needed,
- the model can be compiled to `precompiled_qnn_onnx`,
- the compiled artifact can be profiled on a Snapdragon cloud device with NPU requested,
- the compiled artifact can return valid inference tensors on device.

Use `On_device_Ai.ipynb` as the minimal Qualcomm sample reference.
Use `On_device_Ai_option1_pilots.ipynb` as the repo-specific notebook for BKMeeting pilots.
Use `docs/plans/active/2026-05-11-aihub-option1-npu-pilots.md` for the execution plan behind this workflow.

## Why This Workflow Exists

The current BKMeeting Android branch is blocked below ORT by hosted-device HTP device creation issues.

This workflow intentionally moves upstream:

- first prove Qualcomm AI Hub compile/profile/inference on real Snapdragon cloud devices,
- only later return to Android integration with stronger evidence about model-side viability.

## Pilot Order

1. `Zipformer encoder-first`
   - source: a prepared encoder artifact derived from the fixed-shape encoder ONNX
   - goal: prove the first-slice ASR graph can compile and run on NPU even though the raw fixed-shape source still needs graph preparation first
2. `VPCD model-session-first`
   - source: fixed-shape FP32 ONNX when available, otherwise the existing fixed-shape QDQ fallback
   - goal: prove the first-slice punctuation model session can compile and run on NPU

## Prerequisites

Prepare a Python environment that can already run the local `python-model-test` bundle logic.

Minimum practical requirements:

- `python-model-test` installed in editable mode
- `numpy`
- `torch`
- `torchaudio`
- `qai-hub`
- Jupyter support if you want to execute the notebook directly

Recommended setup flow:

```bash
python -m pip install -e .
python -m pip install qai-hub "qai-hub[torch]"
```

If the current environment does not already have `torchaudio`, install a version compatible with your local `torch` build before running the Zipformer pilot.

## Qualcomm AI Hub Authentication

You need:

- a valid Qualcomm AI Hub API token
- access to at least one Snapdragon device family visible in `qai-hub list-devices`

The notebook includes bootstrap cells for:

- `qai-hub configure`
- `qai-hub list-devices`

Recommended local secret setup:

- put your token in `python-model-test/.env`
- use the variable name `QAI_HUB_API_TOKEN`
- keep `.env.example` as the committed template
- do not place the token directly in notebook source cells

## Current Source Artifacts

### Zipformer

Current preferred source resolution:

- bundle metadata:
  - `build/zipformer/bundle_manifest.json`
- base source model:
  - `build/zipformer/artifacts/fixed_shapes/encoder.fixed.onnx`
- prepared AI Hub upload artifact:
  - `build/aihub/zipformer_encoder_option1/encoder.aihub.option1.onnx`

Why this source is preferred:

- it is fixed-shape already,
- the helper can now prepare it into an AI Hub-friendly graph by:
  - ORT graph optimization,
  - ORT symbolic-shape materialization,
  - cleanup of conflicting `value_info`,
  - rewriting the HTP-blocking boolean `Slice` path to use `uint8` slicing with cast-back to `bool`,
- it aligns with the first ASR slice in BKMeeting.

Current caveat:

- the verified Zipformer lane currently compiles the prepared source model directly
- AI Hub quantize on the prepared Zipformer graph still collides with control-flow outputs during QAIRT conversion, so `submit_quantize_job(...)` is not the default Zipformer notebook path today

### VPCD

Current preferred source resolution:

- fixed-shape candidate bundle:
  - `build/model_bundle/vpcd/qnn_fixed_1024x128/`
- preferred AI Hub source graph:
  - `assets/vietnamese-punc-cap-denorm-v1/onnx/model.fp32.onnx`
- prepared fixed-shape upload artifact:
  - `build/aihub/vpcd_fp32_fixed/model.fp32.fixed.onnx`
- fallback direct-compile graph:
  - `build/model_bundle/vpcd/qnn_fixed_1024x128/model.mobile.onnx`

Current caveat:

- the preferred FP32 source must be frozen to the bundle's `1024 x 128` input shapes before upload
- the fallback bundle graph is already QDQ and remains useful only as a backup experiment lane
- when compile uses `--truncate_64bit_io`, submit compiled-model inference tensors as `int32` for the integer inputs

## Canonical Run Outputs

Each successful Phase 2 run should leave behind a predictable set of local outputs.

### Zipformer

- prepared upload model:
  - `build/aihub/zipformer_encoder_option1/encoder.aihub.option1.onnx`
- prepared artifact record:
  - `build/aihub/records/zipformer_encoder_option1/prepared-artifact-latest.json`
- live run record:
  - `build/aihub/records/zipformer_encoder_option1/live-run-latest.json`

### VPCD

- prepared upload model:
  - `build/aihub/vpcd_option1/model.option1.onnx`
- prepared artifact record:
  - `build/aihub/records/vpcd_option1/prepared-artifact-latest.json`
- live run record:
  - `build/aihub/records/vpcd_option1/live-run-latest.json`

### What The Records Contain

The Phase 2 records are the minimum handoff artifacts for every pilot run.

- prepared artifact record:
  - source model path
  - prepared upload model path
  - file size
  - SHA256 hash
  - input specs
  - compile options
- live run record:
  - device name
  - QAIRT version
  - compile options
  - job options
  - compile/profile/inference job ids and URLs when available
  - output tensor names, shapes, and dtypes

## Notebook Outputs To Preserve

For each pilot, record:

- selected device name
- prepared upload model path
- prepared artifact record path
- compile job URL
- compile job status
- profile job URL
- inference job URL
- live run record path
- output tensor names and shapes
- downloaded target model path if you save it locally

## What Counts As Success

### Compile-only success

- `submit_compile_job(...)` returns successfully
- `get_target_model()` works
- optional download of the target model succeeds

### Profile-ready success

- profile job completes
- profile output exists
- NPU-targeted run does not fail immediately on unsupported model or invalid runtime format

### Inference-ready success

- inference job completes
- output tensors download successfully
- tensor names and shapes match expectations for the selected pilot graph

## Failure Classification

Use the following buckets when a run fails.

### Source-model preparation failure

Symptoms:

- local helper crashes before `submit_compile_job(...)`
- prepared upload model is not written
- prepared artifact record is missing

Typical examples:

- Zipformer graph-prep regression
- missing local source artifact
- FP32 VPCD export missing when the run expected it

### AI Hub compile failure

Symptoms:

- prepared artifact exists locally
- `submit_compile_job(...)` returns a failed job or `get_target_model()` is `None`
- no successful profile or inference jobs can run from that compile result

Typical examples:

- unsupported op
- internal compiler error
- device/runtime incompatibility

### AI Hub profile failure

Symptoms:

- compile succeeded
- profile job fails before returning metrics
- inference may or may not still be attempted depending on the experiment

Typical examples:

- device-side runtime limitation
- profile-only backend validation issue

### AI Hub inference failure

Symptoms:

- compile succeeded
- profile may succeed
- inference job fails or returns unusable tensors

Typical examples:

- bad input dtype after `truncate_64bit_io`
- tensor shape mismatch in submitted inputs
- device-side runtime issue after successful compile

## Minimum Evidence To Call A Pilot Hardened

Do not call a Phase 2 pilot hardened unless all of the following exist for the current run:

- prepared upload model exists in the canonical pilot folder
- prepared artifact record exists with file hashes and input specs
- compile/profile/inference job URLs are stored in the live run record
- live run record contains output tensor summaries
- the notebook cells print the exact local record paths for later review

## What This Workflow Does Not Prove

This workflow does not prove:

- Android ORT integration correctness
- strict ORT QNN partitioning on hosted devices
- end-to-end ASR transcript parity
- full punctuation decode parity
- production deployment readiness

## Related Files

- `On_device_Ai.ipynb`
- `On_device_Ai_option1_pilots.ipynb`
- `src/tools/aihub_option1_pilots.py`
- `test/test_aihub_option1_pilots.py`
- `BKMeeting/docs/qnn/qualcomm-official-pipeline-options.md`
