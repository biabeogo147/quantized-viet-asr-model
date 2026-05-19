# AI Hub Option 1 NPU Pilots

This workflow is the Python-side pilot entry point for validating the BKMeeting model candidates on Qualcomm AI Hub before any Android packaging work.

Use this workflow when you want to prove:

- the model can be quantized in the Qualcomm-official lane when needed,
- the model can be compiled to `precompiled_qnn_onnx`,
- the compiled artifact can be profiled on a Snapdragon cloud device with NPU requested,
- the compiled artifact can return valid inference tensors on device.

Use `On_device_Ai.ipynb` as the minimal Qualcomm sample reference.
Use `On_device_Ai_option1_pilots.ipynb` for the day-to-day `Phase 2 + Phase 3` pilot flow.
Use `On_device_Ai_option1_phase4_gate.ipynb` for benchmark and recommendation reruns.
Use `On_device_Ai_option1_phase5_contract.ipynb` for package creation only.
Use `docs/workflows/model-quantization-status.md` for the current per-model quantization summary.
Use `docs/plans/archive/2026-05-11-aihub-option1-npu-pilots.md` for the historical roadmap behind this workflow.
Use `docs/plans/active/2026-05-19-bkmeeting-android-option1-export-plan.md` for the next Android handoff phase.
Use `docs/plans/active/2026-05-19-option1-phase6-contract-sync-plan.md` for the next implementation pass that extends Android contract-aware sync.
Use `docs/plans/archive/2026-05-14-vpcd-quantize-vs-compile-isolation-plan.md` and `docs/plans/archive/2026-05-18-vpcd-aimet-local-quantize-aihub-compile-plan.md` only for historical VPCD attribution context.

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
   - source: fixed-shape FP32 ONNX prepared locally, then exported through the local AIMET parity lane by default
   - goal: prove the first-slice punctuation model session can compile and run on NPU
   - archived context: older AI Hub quantize probe variants were retired once the AIMET parity lane became the retained path

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

The pilot notebook resolves the API token from `.env` or the shell environment and uses the Python API directly.

Recommended local secret setup:

- put your token in `python-model-test/.env`
- use the variable name `QAI_HUB_API_TOKEN`
- keep `.env.example` as the committed template
- do not place the token directly in notebook source cells

## Current Source Artifacts

### Zipformer

Current preferred source resolution:

- bundle metadata:
  - `build/model_bundle/zipformer/qnn_u16u8/bundle_manifest.json`
- base source model:
  - `build/quantize/zipformer/qnn_u16u8/fixed_shapes/encoder.fixed.onnx`
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

- preferred AI Hub source graph:
  - `assets/vietnamese-punc-cap-denorm-v1/onnx/model.fp32.onnx`
- prepared fixed-shape upload artifact:
  - `build/aihub/vpcd_option1_local_aimet/model.fp32.fixed.onnx`

Current caveat:

- the preferred FP32 source must be frozen to the bundle's `1024 x 128` input shapes before upload
- the current leading notebook lane is:
  - fixed-shape FP32 prepare locally
  - autoregressive calibration build locally
  - local AIMET parity export
  - AI Hub compile
  - AI Hub inference
- when compile uses `--truncate_64bit_io`, submit compiled-model inference tensors as `int32` for the integer inputs
- older AI Hub quantize probe variants are archived and are no longer part of the active execution path

Current local-AIMET parity lane:

- strategy flag:
  - `VPCD_SOURCE_STRATEGY = "local_aimet_compile_candidate"`
- Dockerfile:
  - `docker/aimet-onnx-ubuntu2204/Dockerfile`
- reusable image tag:
  - `bkmeeting-vpcd-aimet:ubuntu22.04-py310`
- retained variant root:
  - `build/aihub/vpcd_option1_local_aimet/wint8_aint16_min_max_local_quality_parity/`
- exported compile input:
  - `build/aihub/vpcd_option1_local_aimet/wint8_aint16_min_max_local_quality_parity/model.option1.aimet/`
- exported local QDQ diagnostic model:
  - `build/aihub/vpcd_option1_local_aimet/wint8_aint16_min_max_local_quality_parity/model.option1.qdq.onnx`

Current status of the local-AIMET probe:

- AI Hub compile accepts the exported `.aimet` package
- the broad default official variant `w8a8 + min_max` already diverges at local teacher-forced step `2`
- a newer policy-constrained parity variant now exists:
  - `w8a16 + min_max + local_quality_parity`
  - custom AIMET config `vpcd_matmul_only`
- that parity variant matches FP32 for the bounded `5`-step teacher-forced window both:
  - locally on the exported QDQ reference
  - on the compiled AI Hub cloud target
- bounded hybrid also produces the correct `5`-step prefix instead of collapsing to punctuation or early EOS
- because the current proof window is still bounded to `5` decode steps, local AIMET is now the leading replacement candidate, but not yet the default notebook source

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
  - `build/aihub/vpcd_option1_local_aimet/model.fp32.fixed.onnx`
- exported AIMET package:
  - `build/aihub/vpcd_option1_local_aimet/wint8_aint16_min_max_local_quality_parity/model.option1.aimet/`
- exported local QDQ diagnostic model:
  - `build/aihub/vpcd_option1_local_aimet/wint8_aint16_min_max_local_quality_parity/model.option1.qdq.onnx`
- prepared artifact record:
  - `build/aihub/records/vpcd_option1_local_aimet/prepared-artifact-20260519-aimet-local-quality-parity-notebook.json`
- compile record:
  - `build/aihub/records/vpcd_option1_local_aimet/compile-run-20260519-aimet-local-quality-parity-notebook.json`
- live run record:
  - `build/aihub/records/vpcd_option1_local_aimet/live-run-20260519-aimet-local-quality-parity-notebook.json`

Historical AI Hub quantize probe outputs are intentionally not retained in `build/` after cleanup. If they ever need to be revisited, use the archived VPCD investigation plans rather than the active workflow.

Recent parity rerun outputs:

- executed notebook:
  - `build/aihub/notebook_runs/On_device_Ai_option1_pilots.local_aimet_quality_parity.executed.ipynb`
- log:
  - `build/aihub/notebook_runs/local_aimet_quality_parity.log`

### What The Records Contain

The Phase 2 records are the minimum handoff artifacts for every pilot run.

- prepared artifact record:
  - source model path
  - prepared upload model path
  - file size
  - SHA256 hash
  - input specs
  - compile options
- compile-only record:
  - compile job id and URL
  - resolved target model id and URL
  - compile options
- quantize-run record:
  - quantize job id and URL
  - quantized target model id and URL
  - downloaded quantized ONNX path
  - quantize dtype names
  - quantize options
  - calibration fingerprint and stats
- live run record:
  - device name
  - QAIRT version
  - compile options
  - job options
  - compile/profile/inference job ids and URLs when available
  - output tensor names, shapes, and dtypes

## Operator Flow

Run this workflow from:

- [On_device_Ai_option1_pilots.ipynb](/D:/DS-AI/BKMeeting-Research/python-model-test/On_device_Ai_option1_pilots.ipynb)

Two normal modes are supported:

### Compile From Scratch

Use this when no compile record exists yet for the chosen `RUN_LABEL`.

Run, per enabled pilot:

1. `Prepare`
2. `Compile Only`
3. `Resolve Existing Compiled Target`
4. `Run And Compare Against The Compiled Target`
5. optional `Output Inspection (Debug Only)`
6. `Quantized Local Teacher-Forced Diagnostics` for VPCD
7. `Teacher-Forced Diagnostics` for VPCD
8. `Hybrid E2E Run`
9. `Final Compare`

### Reuse An Existing Compiled Target

Use this when you only want to rerun inference and correctness checks.

Keep the same `RUN_LABEL`, or set an explicit `*_TARGET_MODEL_ID`, then run:

1. `Prepare`
2. `Resolve Existing Compiled Target`
3. `Run And Compare Against The Compiled Target`
4. optional `Output Inspection (Debug Only)`
5. `Quantized Local Teacher-Forced Diagnostics` for VPCD
6. `Teacher-Forced Diagnostics` for VPCD
7. `Hybrid E2E Run`
8. `Final Compare`

Recommended config defaults:

- `ENABLE_PROFILE_DURING_RUN = False`
- `ENABLE_DEBUG_OUTPUT_INSPECTION = False`

Operator note:

- environment setup belongs outside the normal notebook execution path
- the pilot notebook no longer includes the old mandatory `pip install` cell
- for the current VPCD failure, use quantized-local teacher-forced before cloud teacher-forced, and both before any free-run hybrid rerun
- if `VPCD_SOURCE_STRATEGY = "local_aimet_compile_candidate"`, the notebook reuses the local QDQ diagnostic model for quantized-local teacher-forced checks and now preserves the local AIMET compile pilot name across teacher-forced and hybrid records

## Notebook Outputs To Preserve

For each pilot, record:

- selected device name
- prepared upload model path
- prepared artifact record path
- compile-only record path
- compile job URL
- compile job status
- target model id
- profile job URL
- inference job URL
- live run record path
- output tensor names and shapes
- downloaded target model path if you save it locally
- hybrid record path

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
- local QDQ artifact still contains `com.microsoft` Q/DQ operators that AI Hub Workbench does not accept

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
