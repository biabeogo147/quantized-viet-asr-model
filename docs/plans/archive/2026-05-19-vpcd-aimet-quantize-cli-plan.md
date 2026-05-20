# VPCD AIMET Quantize CLI Reuse Plan

Archived on 2026-05-19 after the retained VPCD AIMET producer flow moved fully into `python -m quantize`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make VPCD local AIMET quantization a reusable `python -m quantize` producer like Zipformer, store the canonical quantized artifacts under `build/quantize/vpcd/...`, run AIMET from a reusable Docker service that exposes a mapped host port, and change `On_device_Ai_option1_pilots.ipynb` so it starts from an already-built local AIMET package instead of exporting AIMET inside the notebook.

**Architecture:** Move the heavy VPCD AIMET work out of the AI Hub notebook helper and into the `quantize` module, but do not let the quantize module shell out to Docker directly on every run. Instead:
- the Docker image becomes a long-lived AIMET export service
- the service exposes a mapped host port
- the quantize CLI acts as a client that prepares inputs locally and calls the service over HTTP or a similarly simple local port protocol

The quantize CLI becomes the only place that builds the fixed-shape FP32 staging model, calibration batches, AIMET config/policy, export request manifest, `.aimet` package metadata, local QDQ reference metadata, and quantize report. `build/quantize/vpcd/...` becomes the canonical reusable source tree; `build/aihub/...` becomes compile/run evidence only.

**Tech Stack:** Python, `python -m quantize`, AIMET ONNX service in Docker, local HTTP client, ONNX fixed-shape prep, Jupyter notebook JSON, Markdown docs, pytest

---

## Desired End State

- `python -m quantize --project vpcd` supports a reusable local AIMET candidate flow.
- `docker/aimet-onnx-ubuntu2204/Dockerfile` builds a runnable AIMET service image.
- the AIMET service can be started once and exposes a host-mapped port for reuse across multiple quantize runs.
- The canonical VPCD AIMET artifacts live under `build/quantize/vpcd/...`, not under `build/aihub/...`.
- `On_device_Ai_option1_pilots.ipynb` no longer runs AIMET export itself.
- The notebook fails fast with a clear message if the local AIMET package has not been built yet.
- `build/aihub` contains compile/runtime evidence only:
  - compile records
  - teacher-forced diagnostics
  - hybrid results
  - phase4 / phase5 evidence

## Proposed Artifact Contract

Use a Zipformer-like producer layout rooted at:

- `build/quantize/vpcd/local_aimet/`

Inside that root, use the AIMET variant name as the canonical leaf:

- `build/quantize/vpcd/local_aimet/wint8_aint16_min_max_local_quality_parity/`

Expected contents:

- `model.fp32.fixed.onnx`
  - fixed-shape FP32 ONNX used as the AIMET export input
- `calibration/`
  - serialized fixed-shape autoregressive calibration batches
- `aimet.config.json`
  - generated when the selected policy mode needs a custom config
- `aimet.policy.json`
  - explicit policy manifest describing the local-quality parity intent
- `model.option1.aimet/`
  - canonical `.aimet` package consumed by AI Hub compile
- `model.option1.qdq.onnx`
  - local QDQ diagnostic reference used by teacher-forced local checks
- `model.option1.aimet.report.json`
  - package inspection output
- `quantize_report.json`
  - top-level VPCD quantize report that ties together:
    - source model path
    - calibration fingerprint and stats
    - AIMET config
    - policy mode
    - AIMET service endpoint used for the export
    - exported package paths
    - QDQ reference path

## AIMET Service Contract

The Docker image should not be treated as an opaque one-shot runner. It must become a stable local service with a simple contract.

Recommended runtime model:

- container image built from:
  - `docker/aimet-onnx-ubuntu2204/Dockerfile`
- container listens on an internal port, for example:
  - `8080`
- operator maps that port to the host, for example:
  - `18080:8080`
- `python -m quantize` talks to:
  - `http://127.0.0.1:18080`

Recommended minimal endpoints:

- `GET /healthz`
  - returns ready/not-ready status
- `POST /export`
  - accepts an export request describing:
    - fixed-shape FP32 ONNX path
    - calibration directory path
    - output package path
    - output QDQ reference model path
    - config path
    - policy manifest path
    - param type
    - activation type
    - quant scheme
    - model prefix
- optional `GET /version`
  - returns service version and AIMET package version

The quantize module should remain responsible for local filesystem preparation. The service should remain responsible for AIMET-specific export work only.

## CLI Shape

Keep the existing `python -m quantize --project vpcd` entrypoint and extend the VPCD project adapter with an explicit pipeline switch instead of inventing a separate tool.

Recommended arguments:

- `--pipeline`
  - choices:
    - `legacy_qdq`
    - `local_aimet_candidate`
- `--output-root`
  - canonical default for AIMET:
    - `build/quantize/vpcd/local_aimet`
- `--aimet-param-type`
- `--aimet-activation-type`
- `--aimet-quant-scheme`
- `--aimet-config-file`
- `--aimet-policy-mode`
- `--aimet-service-url`
- `--aimet-health-timeout-seconds`

Recommended retained defaults for the active lane:

- `--aimet-param-type int8`
- `--aimet-activation-type int16`
- `--aimet-quant-scheme min_max`
- `--aimet-config-file vpcd_matmul_only`
- `--aimet-policy-mode local_quality_parity`
- `--aimet-service-url http://127.0.0.1:18080`

Example target command:

```bash
python -m quantize \
  --project vpcd \
  --model-dir assets/vietnamese-punc-cap-denorm-v1 \
  --fp32-onnx assets/vietnamese-punc-cap-denorm-v1/onnx/model.fp32.onnx \
  --calibration-text build/calibration/vlsp2020/vpcd_transcriptions.txt \
  --max-calibration-samples 24 \
  --max-generation-length 32 \
  --ort-provider cpu \
  --output-root build/quantize/vpcd/local_aimet \
  --aimet-param-type int8 \
  --aimet-activation-type int16 \
  --aimet-quant-scheme min_max \
  --aimet-config-file vpcd_matmul_only \
  --aimet-policy-mode local_quality_parity \
  --aimet-service-url http://127.0.0.1:18080
```

---

## Task 1: Make the AIMET Docker image runnable as a service

**Files:**
- Modify: `docker/aimet-onnx-ubuntu2204/Dockerfile`
- Add: service entrypoint module or script under `src/quantize/` or `docker/aimet-onnx-ubuntu2204/`
- Modify: `src/quantize/README.md`
- Modify: `docs/workflows/quantize-qnn-candidates.md`
- Test: add service-smoke coverage if practical

- [ ] **Step 1: Define the container runtime contract**

Cover:
- internal service port
- host mapping convention
- health endpoint
- export endpoint
- mounted workspace expectations

- [ ] **Step 2: Change the Docker image from interactive shell to runnable service**

The Dockerfile should no longer end as:
- `CMD ["bash"]`

Instead, it should start the AIMET export service automatically.

- [ ] **Step 3: Add a service healthcheck and operator run command**

Document one canonical way to run it, for example:

```bash
docker build -t bkmeeting/aimet-onnx-service docker/aimet-onnx-ubuntu2204
docker run --rm -p 18080:8080 -v <repo-root>:/workspace bkmeeting/aimet-onnx-service
```

- [ ] **Step 4: Verify the service is callable over the mapped port**

Minimum proof:
- health endpoint responds
- export endpoint accepts a dry-run or smoke request

- [ ] **Step 5: Commit**

```bash
git add docker/aimet-onnx-ubuntu2204/Dockerfile src/quantize/README.md docs/workflows/quantize-qnn-candidates.md
git commit -m "feat: expose aimet docker image as local export service"
```

## Task 2: Turn VPCD AIMET export into the retained quantize CLI lane

**Files:**
- Modify: `src/quantize/projects/vpcd.py`
- Modify: `src/quantize/aimet.py`
- Modify: `src/quantize/types.py`
- Modify: `src/quantize/README.md`
- Modify: `src/quantize/projects/README.md`
- Test: `test/test_vpcd_quantize_aihub.py`

- [ ] **Step 1: Add failing tests for the new CLI pipeline contract**

Cover:
- dry-run support for `python -m quantize --project vpcd`
- default AIMET output root under `build/quantize/vpcd/local_aimet`
- top-level report paths and variant-root layout
- no dependency on `build/aihub/...` in the producer path
- no direct `docker run` dependency in the quantize execution path
- service URL validation and healthcheck behavior

- [ ] **Step 2: Collapse the VPCD project adapter to the retained lane**

In `src/quantize/projects/vpcd.py`:
- add AIMET-specific output-root, config, and service URL arguments
- route `run(args)` directly to the retained local AIMET candidate flow

- [ ] **Step 3: Move the reusable AIMET export work under the quantize module and make it service-backed**

The new quantize pipeline should own:
- fixed-shape FP32 staging ONNX generation
- `build_vpcd_aimet_quantize_recipe(...)`
- calibration batch serialization
- config/policy manifest generation
- request-manifest assembly for the AIMET service
- service healthcheck before export
- service-backed AIMET export call
- package inspection
- `quantize_report.json`

Avoid duplicating logic that already exists in:
- `build_vpcd_aimet_quantize_recipe(...)`
- `write_calibration_batches(...)`
- `export_aimet_package(...)`
- AIMET package inspection helpers

Avoid this architecture:
- `python -m quantize` spawning `docker run ...` directly
- `python -m quantize` rebuilding the image on demand

- [ ] **Step 4: Define a reusable return/report object for the CLI lane**

If needed, extend `src/quantize/types.py` so the quantize lane can report:
- variant name
- package dir
- qdq reference model path
- fixed-shape source path
- calibration fingerprint
- local-quality policy summary
- AIMET service URL
- service response metadata

- [ ] **Step 5: Run targeted quantize tests**

Run:
- `pytest test/test_vpcd_quantize_aihub.py -k "aimet or vpcd" -v`

Expected:
- PASS

- [ ] **Step 6: Commit**

```bash
git add src/quantize/projects/vpcd.py src/quantize/aimet.py src/quantize/types.py src/quantize/README.md src/quantize/projects/README.md test/test_vpcd_quantize_aihub.py
git commit -m "feat: add reusable vpcd aimet quantize cli pipeline"
```

## Task 3: Refactor the AI Hub notebook helper to consume prebuilt AIMET artifacts only

**Files:**
- Modify: `src/tools/aihub_option1_pilots.py`
- Modify: `On_device_Ai_option1_pilots.ipynb`
- Test: `test/test_aihub_option1_pilots.py`
- Test: `test/test_option1_notebook_layout.py`

- [ ] **Step 1: Add failing tests for the new consumer-only behavior**

Cover:
- `prepare_vpcd_option1_source_model(...)` must stop generating calibration batches and calling AIMET export directly
- the helper must resolve a prebuilt canonical artifact from `build/quantize/vpcd/...`
- the notebook must no longer present local AIMET export as part of the VPCD prepare cell

- [ ] **Step 2: Split "build" from "resolve" in the VPCD prepare helper**

Refactor `src/tools/aihub_option1_pilots.py` so the active notebook path:
- resolves the fixed-shape source path from the canonical quantize output tree
- resolves the `.aimet` package path
- resolves the local QDQ diagnostic model path
- loads the stored quantize report
- fails fast with a clear operator message if the quantize lane has not been run yet

The notebook helper must stop doing these actions directly:
- `freeze_model_inputs(...)`
- `build_vpcd_aimet_quantize_recipe(...)`
- `write_calibration_batches(...)`
- any direct export call that assumes notebook-owned AIMET execution

- [ ] **Step 3: Reword the notebook so it starts from prebuilt local AIMET**

In `On_device_Ai_option1_pilots.ipynb`:
- keep `VPCD_SOURCE_STRATEGY = "local_aimet_compile_candidate"`
- add a small config block for the canonical local quantize root
- make the markdown explain:
  - run `python -m quantize --project vpcd ...` first
  - then run the notebook from compile onward
- update the VPCD "Model-Session-First" section so it resolves and prints:
  - package path
  - QDQ diagnostic path
  - quantize report path

- [ ] **Step 4: Run targeted notebook/helper tests**

Run:
- `pytest test/test_aihub_option1_pilots.py -k "vpcd and aimet" -v`
- `pytest test/test_option1_notebook_layout.py -k "vpcd" -v`

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add src/tools/aihub_option1_pilots.py On_device_Ai_option1_pilots.ipynb test/test_aihub_option1_pilots.py test/test_option1_notebook_layout.py
git commit -m "refactor: make option1 notebook consume prebuilt vpcd aimet artifacts"
```

## Task 4: Make the canonical output and operator workflow explicit

**Files:**
- Modify: `docs/workflows/quantize-qnn-candidates.md`
- Modify: `docs/workflows/aihub-option1-npu-pilots.md`
- Modify: `docs/workflows/aihub-option1-lane-history-and-decision.md`
- Modify: `docs/workflows/model-quantization-status.md`
- Modify: `src/tools/README.md`

- [ ] **Step 1: Update the canonical quantize workflow**

In `docs/workflows/quantize-qnn-candidates.md`:
- replace the old VPCD balanced QDQ path as the active `Option 1` lane
- document the new reusable AIMET quantize command
- document the required AIMET service startup command and port
- explain that `build/quantize/vpcd/...` is reusable across notebook reruns

- [ ] **Step 2: Update the notebook workflow docs**

In `docs/workflows/aihub-option1-npu-pilots.md`:
- make it explicit that VPCD Phase 2 starts from a prebuilt local AIMET package
- describe `build/aihub` as compile/runtime evidence only

- [ ] **Step 3: Update lane-history and status docs**

Document clearly:
- the old notebook-built AIMET lane is retired
- the canonical retained lane is now:
  - start the AIMET Docker service
  - `python -m quantize --project vpcd`
  - then `On_device_Ai_option1_pilots.ipynb`

- [ ] **Step 4: Verify docs stay aligned with runtime defaults**

Run:
- `pytest test/test_option1_notebook_layout.py -v`

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add docs/workflows/quantize-qnn-candidates.md docs/workflows/aihub-option1-npu-pilots.md docs/workflows/aihub-option1-lane-history-and-decision.md docs/workflows/model-quantization-status.md src/tools/README.md
git commit -m "docs: document reusable vpcd aimet quantize workflow"
```

## Task 5: Prune the old VPCD AIMET build location from the active path

**Files:**
- Modify: `src/tools/aihub_option1_pilots.py`
- Delete or stop reusing: `build/aihub/vpcd_option1_local_aimet/` as the canonical quantize source location
- Modify: docs only if a path changes

- [ ] **Step 1: Keep `build/aihub` for evidence only**

After the refactor:
- VPCD compile records should still be written under `build/aihub/records/...`
- the source quantize artifacts must no longer be regenerated into `build/aihub/vpcd_option1_local_aimet/...`

- [ ] **Step 2: Delete stale canonical-source assumptions**

Remove active assumptions that the notebook-owned AIMET source lives under:
- `build/aihub/vpcd_option1_local_aimet/...`

If compatibility shims are temporarily needed, keep them narrow and mark them transitional in code comments.

- [ ] **Step 3: Verify the helper payloads now point at `build/quantize/vpcd/...`**

Run:
- `pytest test/test_aihub_option1_pilots.py -k "vpcd and source_model" -v`

Expected:
- PASS

- [ ] **Step 4: Commit**

```bash
git add src/tools/aihub_option1_pilots.py test/test_aihub_option1_pilots.py
git commit -m "refactor: point vpcd option1 source resolution at quantize artifacts"
```

## Task 6: Operator handoff for the final rerun

**Files:**
- Modify: `On_device_Ai_option1_pilots.ipynb`
- Modify: `docs/workflows/aihub-option1-npu-pilots.md`
- Modify: `docs/workflows/quantize-qnn-candidates.md`

- [ ] **Step 1: Pre-fill the final operator command sequence**

Document the final sequence clearly:

1. build or pull the AIMET Docker service image
2. start the AIMET Docker service with the canonical port mapping
3. prepare calibration subset
4. run VPCD local AIMET quantize CLI
5. run `On_device_Ai_option1_pilots.ipynb`
6. run `On_device_Ai_option1_phase4_gate.ipynb`
7. run `On_device_Ai_option1_phase5_contract.ipynb`

- [ ] **Step 2: Keep the notebook guardrails unchanged**

Retain:
- `VPCD_HYBRID_MAX_SAMPLES = 2`
- `VPCD_HYBRID_MAX_STEPS = 5`
- `VPCD_TEACHER_FORCED_SAMPLE_INDEX = 0`

The goal of this plan is to move the quantize producer, not to widen the runtime gate.

- [ ] **Step 3: Add a clear failure message for missing prebuilt artifacts**

The notebook should tell the operator exactly:
- which Docker service command to start
- which `python -m quantize` command to run
when the local AIMET package is missing.

- [ ] **Step 4: Verify the operator flow is self-explanatory**

Run:
- `pytest test/test_option1_notebook_layout.py -v`

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add On_device_Ai_option1_pilots.ipynb docs/workflows/aihub-option1-npu-pilots.md docs/workflows/quantize-qnn-candidates.md
git commit -m "docs: prepare operator rerun flow for prebuilt vpcd aimet artifacts"
```

---

## Acceptance Criteria

- VPCD AIMET local quantization can be produced from `python -m quantize --project vpcd`.
- the AIMET Docker image is runnable as a reusable service over a mapped host port.
- The canonical reusable artifacts live under `build/quantize/vpcd/...`.
- The AI Hub pilot notebook no longer exports AIMET locally.
- The notebook starts from a prebuilt local AIMET package and local QDQ diagnostic model.
- Workflow docs explain the producer/consumer split clearly and concisely.

## Verification Checklist

- `pytest test/test_vpcd_quantize_aihub.py -k "aimet or vpcd" -v`
- `pytest test/test_aihub_option1_pilots.py -k "vpcd and aimet" -v`
- `pytest test/test_option1_notebook_layout.py -k "vpcd" -v`

## Out of Scope

- widening VPCD hybrid decode steps beyond the current debug guardrail
- Phase 6 Android sync implementation
- deleting the legacy VPCD quantize CLI path in the same pass as the new AIMET producer
  - that can happen later after the new lane is fully proven
