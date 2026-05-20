# Option 1 Step 5 Deployable Artifact Download Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic Step 5 flow in `python-model-test` that downloads the AI Hub `precompiled_qnn_onnx` artifact for the retained `zipformer` and `vpcd` pilots, records the exact download metadata, and materializes a Python-side package that later handoff work can consume without reopening AI Hub.

**Architecture:** Keep `On_device_Ai_option1_pilots.ipynb` limited to the current prepare / compile / run / hybrid evidence loop. Add a separate Step 5 helper + CLI that resolves existing compile and live-run records by `RUN_LABEL`, downloads the compiled target model from AI Hub, exports a normalized `io_contract.json`, and writes one per-pilot Step 5 package under `build/aihub/step5/option1/...`. Do not sync to BKMeeting and do not change Android runtime code in this phase.

**Tech Stack:** Python, `qai_hub`, existing Option 1 record helpers in `src/tools/aihub_option1_pilots.py`, `ModelBundleManifest`, JSON manifests, Markdown workflow docs, `pytest`

---

## Scope And Assumptions

- This plan is for **Qualcomm official Option 1 Step 5** from [bkmeeting/docs/qnn/qualcomm-official-pipeline-options.md](/D:/DS-AI/BKMeeting-Research/bkmeeting/docs/qnn/qualcomm-official-pipeline-options.md), not the older archived meaning of "Phase 5 contract packaging".
- Current verified input state already exists from `On_device_Ai_option1_pilots.ipynb`:
  - `RUN_LABEL = 20260519-6pm`
  - `zipformer` compile record:
    - `build/aihub/records/zipformer_encoder_option1/compile-run-20260519-6pm.json`
    - target model id: `mqero78kn`
  - `vpcd` compile record:
    - `build/aihub/records/vpcd_option1_local_aimet/compile-run-20260519-6pm.json`
    - target model id: `mmxwpeyen`
- The shared pilot notebook intentionally excludes downstream Step 5 logic today, and that behavior is locked by [test/test_option1_notebook_layout.py](/D:/DS-AI/BKMeeting-Research/python-model-test/test/test_option1_notebook_layout.py).
- Step 5 must be runnable **without recompiling** and without moving AI Hub logic into the shared pilot notebook.

## Non-Goals

- Android asset sync
- Android runtime/provider changes
- new Phase 4/Phase 5 notebook sprawl inside the pilot notebook
- rerunning compile, profile, or inference unless a required record is missing
- redefining bundle contracts for CPU baseline bundles

## File Structure

**Shared download helpers**

- Modify: `src/tools/aihub_option1_pilots.py`
  - add compiled-target download helper(s)
  - add deterministic Step 5 record writer(s)
  - keep shared path and record logic centralized
- Modify: `test/test_aihub_option1_pilots.py`
  - lock the shared download helper contract

**Step 5 package builder**

- Create: `src/tools/aihub_option1_step5_artifacts.py`
  - Step 5 pilot layout
  - record resolution
  - package materialization
  - CLI entrypoint
- Create: `test/test_aihub_option1_step5_artifacts.py`
  - locks parser behavior, record resolution, package layout, manifest contents, and I/O contract export

**Workflow docs**

- Create: `docs/workflows/option1-step5-download.md`
  - operator instructions for post-notebook artifact download
- Modify: `docs/workflows/option1-overview.md`
  - show Step 5 as the next action after notebook proof
- Modify: `docs/workflows/option1-rerun.md`
  - document exact post-run Step 5 command
- Modify: `docs/workflows/android-handoff.md`
  - clarify that Step 5 package becomes the Python-side handoff input, while Android proof remains out of scope
- Modify: `src/tools/README.md`
  - add the new CLI to the tools index

**Files that should remain unchanged but must stay green**

- Verify only: `On_device_Ai_option1_pilots.ipynb`
- Verify only: `test/test_option1_notebook_layout.py`

## Target Output Layout

Per pilot, Step 5 should create one deterministic package directory:

- `build/aihub/step5/option1/zipformer/20260519-6pm/`
- `build/aihub/step5/option1/vpcd/20260519-6pm/`

First version package contents:

- `step5_manifest.json`
- `io_contract.json`
- `deploy_notes.md`
- `download/`
  - downloaded compiled artifact from AI Hub
- `evidence/`
  - copied `prepared-artifact` record
  - copied `compile-run` record
  - copied `live-run` record
  - copied `hybrid-run` record when available
  - copied `deployable-download` record

Required manifest fields in `step5_manifest.json`:

- `project`
- `pilot_name`
- `run_label`
- `target_model_id`
- `target_runtime`
- `device_name`
- `qairt_version`
- `compile_options`
- `downloaded_artifact`
- `source_bundle_manifest`
- `evidence`
- `special_handling`

Required `io_contract.json` shape:

```json
{
  "target_runtime": "precompiled_qnn_onnx",
  "inputs": [
    {
      "name": "input_tensor_name",
      "shape": [1, 1024],
      "dtype": "int32",
      "source_dtype": "int64"
    }
  ],
  "outputs": [
    {
      "name": "output_0",
      "shape": [1, 1024, 50265],
      "dtype": "float32"
    }
  ],
  "special_handling": [
    "truncate_64bit_io required"
  ],
  "deployment_notes": [
    "zipformer keeps decoder and joiner on CPU",
    "vpcd keeps tokenizer encode and decode on CPU"
  ]
}
```

## Current State Snapshot

- `zipformer`
  - prepared record exists:
    - `build/aihub/records/zipformer_encoder_option1/prepared-artifact-20260519-6pm.json`
  - compile record exists:
    - `build/aihub/records/zipformer_encoder_option1/compile-run-20260519-6pm.json`
  - live-run record exists:
    - `build/aihub/records/zipformer_encoder_option1/live-run-20260519-6pm.json`
  - hybrid record exists:
    - `build/aihub/records/zipformer_hybrid_option1/hybrid-run-20260519-6pm.json`
- `vpcd`
  - prepared record exists:
    - `build/aihub/records/vpcd_option1_local_aimet/prepared-artifact-20260519-6pm.json`
  - compile record exists:
    - `build/aihub/records/vpcd_option1_local_aimet/compile-run-20260519-6pm.json`
  - live-run record exists:
    - `build/aihub/records/vpcd_option1_local_aimet/live-run-20260519-6pm.json`
  - hybrid record exists:
    - `build/aihub/records/vpcd_hybrid_option1/hybrid-run-20260519-6pm.json`
- current gap:
  - there is no helper today for downloading compiled target models from compile records
  - there is no Step 5 package layout
  - current sync tooling still assumes CPU/QNN bundle manifests, not AI Hub Step 5 packages

## Task 1: Add Shared Compiled-Target Download Helpers And Download Records

**Files:**

- Modify: `src/tools/aihub_option1_pilots.py`
- Modify: `test/test_aihub_option1_pilots.py`

- [ ] **Step 1: Write failing tests for compiled-target download helpers**

Cover:

- `download_compiled_target_model(...)` downloads through a fake AI Hub target model object that exposes `.download(...)`
- helper creates parent directories automatically
- helper fails clearly if AI Hub returns a path that does not exist
- `write_deployable_download_record(...)` writes:
  - pilot name
  - run label
  - compile record path
  - target model metadata
  - downloaded file metadata
  - device and QAIRT info

- [ ] **Step 2: Run the focused helper tests to confirm they fail**

Run: `pytest test/test_aihub_option1_pilots.py -k "compiled_target_download or deployable_download_record" -v`

Expected: FAIL because the Step 5 helpers do not exist yet.

- [ ] **Step 3: Implement the shared helpers in `src/tools/aihub_option1_pilots.py`**

Add minimal shared helpers:

```python
def download_compiled_target_model(*, target_model: object, output_path: str | Path) -> Path: ...

def write_deployable_download_record(
    *,
    pilot_name: str,
    runtime_config: Option1RuntimeConfig,
    compile_record_path: str | Path,
    target_model: object,
    downloaded_artifact_path: str | Path,
    run_label: str | None = None,
) -> Path: ...
```

Rules:

- store download records beside other pilot records under:
  - `build/aihub/records/<pilot>/deployable-download-<RUN_LABEL>.json`
- record SHA256, file size, and resolved absolute path for the downloaded artifact
- keep all file metadata generation on the existing helper path already used for prepared records

- [ ] **Step 4: Re-run the focused helper tests**

Run: `pytest test/test_aihub_option1_pilots.py -k "compiled_target_download or deployable_download_record" -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tools/aihub_option1_pilots.py test/test_aihub_option1_pilots.py
git commit -m "feat: add option1 compiled target download helpers"
```

## Task 2: Create The Step 5 Resolver And Package Layout Module

**Files:**

- Create: `src/tools/aihub_option1_step5_artifacts.py`
- Create: `test/test_aihub_option1_step5_artifacts.py`

- [ ] **Step 1: Write failing tests for deterministic Step 5 input resolution**

Cover:

- `zipformer` resolves Phase 2 records from:
  - `zipformer_encoder_option1`
- `vpcd` resolves Phase 2 records from:
  - `vpcd_option1_local_aimet`
- required inputs:
  - prepared-artifact record
  - compile-run record
  - live-run record
- optional inputs:
  - hybrid-run record
- explicit target model override stays possible, but default behavior must come from compile records
- missing required records raise actionable errors that include the exact missing path

- [ ] **Step 2: Run the resolver tests to confirm they fail**

Run: `pytest test/test_aihub_option1_step5_artifacts.py -k "resolve or layout" -v`

Expected: FAIL because the Step 5 module does not exist yet.

- [ ] **Step 3: Implement the pilot layout and record resolver**

In `src/tools/aihub_option1_step5_artifacts.py`, add:

- one small per-pilot layout table
- one resolver for deterministic upstream record paths
- one function that maps compile pilot names to Step 5 output directories:
  - `zipformer_encoder_option1 -> build/aihub/step5/option1/zipformer/<RUN_LABEL>/`
  - `vpcd_option1_local_aimet -> build/aihub/step5/option1/vpcd/<RUN_LABEL>/`

Keep the module separate from the pilot notebook so the notebook remains Phase 2 plus Phase 3 only.

- [ ] **Step 4: Re-run the resolver tests**

Run: `pytest test/test_aihub_option1_step5_artifacts.py -k "resolve or layout" -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tools/aihub_option1_step5_artifacts.py test/test_aihub_option1_step5_artifacts.py
git commit -m "feat: add option1 step5 record resolver"
```

## Task 3: Materialize Per-Pilot Step 5 Packages

**Files:**

- Modify: `src/tools/aihub_option1_step5_artifacts.py`
- Modify: `test/test_aihub_option1_step5_artifacts.py`

- [ ] **Step 1: Write failing tests for package materialization**

Cover:

- package writer creates the deterministic output directory
- downloaded compiled artifact lands under `download/`
- copied evidence lands under `evidence/`
- `step5_manifest.json` records:
  - target model id
  - compile options
  - compile record path
  - live-run record path
  - source bundle manifest path
  - downloaded artifact hash and size
- `io_contract.json` is generated from existing record data, not handwritten constants
- `deploy_notes.md` includes the pilot-specific runtime split:
  - Zipformer: encoder on compiled target, decoder and joiner remain CPU
  - VPCD: model session on compiled target, tokenizers remain CPU

- [ ] **Step 2: Run the package tests to confirm they fail**

Run: `pytest test/test_aihub_option1_step5_artifacts.py -k "materialize or manifest or io_contract" -v`

Expected: FAIL because the package writer does not exist yet.

- [ ] **Step 3: Implement the Step 5 package writer**

Implementation rules:

- use the shared helper from `aihub_option1_pilots.py` to download the target model
- copy JSON evidence into the package instead of only referencing it
- keep large source bundle files out of scope for Step 5 package v1
- derive `special_handling` from compile options:
  - `--truncate_64bit_io` must become a structured note in `io_contract.json`
- prefer record-backed shapes and dtypes over hand-maintained tables

- [ ] **Step 4: Re-run the package tests**

Run: `pytest test/test_aihub_option1_step5_artifacts.py -k "materialize or manifest or io_contract" -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tools/aihub_option1_step5_artifacts.py test/test_aihub_option1_step5_artifacts.py
git commit -m "feat: materialize option1 step5 packages"
```

## Task 4: Add A CLI Operator Surface And Refresh Workflow Docs

**Files:**

- Modify: `src/tools/aihub_option1_step5_artifacts.py`
- Modify: `test/test_aihub_option1_step5_artifacts.py`
- Create: `docs/workflows/option1-step5-download.md`
- Modify: `docs/workflows/option1-overview.md`
- Modify: `docs/workflows/option1-rerun.md`
- Modify: `docs/workflows/android-handoff.md`
- Modify: `src/tools/README.md`

- [ ] **Step 1: Write failing tests for the CLI parser and dry-run mode**

Cover:

- `--pilot zipformer`
- `--pilot vpcd`
- `--pilot all`
- `--run-label 20260519-6pm`
- `--dry-run`
- clear error if the user requests an unsupported pilot label

- [ ] **Step 2: Run the focused CLI tests to confirm they fail**

Run: `pytest test/test_aihub_option1_step5_artifacts.py -k "parser or cli or dry_run" -v`

Expected: FAIL because no CLI exists yet.

- [ ] **Step 3: Implement the CLI**

Required command surface:

```bash
python -m tools.aihub_option1_step5_artifacts --pilot all --run-label 20260519-6pm
python -m tools.aihub_option1_step5_artifacts --pilot zipformer --run-label 20260519-6pm --dry-run
```

Behavior:

- resolve the required records first
- in `--dry-run`, print the planned package path and required source records without downloading
- in normal mode, download the compiled model and write the full Step 5 package

- [ ] **Step 4: Write the new Step 5 workflow doc**

Document:

- exact prerequisite notebook outputs
- exact Step 5 CLI commands
- output directories
- what to inspect first:
  - `step5_manifest.json`
  - `io_contract.json`
  - `deploy_notes.md`
- explicit boundary:
  - Step 5 proves download and package materialization
  - it does not prove Android runtime success

- [ ] **Step 5: Refresh the short workflow docs**

Update:

- `docs/workflows/option1-overview.md`
  - add Step 5 as the immediate next action after notebook evidence
- `docs/workflows/option1-rerun.md`
  - append one post-notebook Step 5 command block
- `docs/workflows/android-handoff.md`
  - change the Option 1 note so Android handoff starts from the Step 5 package instead of loose `build/aihub/records/` files
- `src/tools/README.md`
  - list the new CLI

- [ ] **Step 6: Verify the pilot notebook stays out of Step 5 scope**

Run: `pytest test/test_option1_notebook_layout.py -v`

Expected: PASS because `On_device_Ai_option1_pilots.ipynb` still excludes downstream Step 5 logic.

- [ ] **Step 7: Commit**

```bash
git add src/tools/aihub_option1_step5_artifacts.py test/test_aihub_option1_step5_artifacts.py docs/workflows/option1-step5-download.md docs/workflows/option1-overview.md docs/workflows/option1-rerun.md docs/workflows/android-handoff.md src/tools/README.md
git commit -m "docs: add option1 step5 artifact download workflow"
```

## Task 5: Run Full Verification And One Real-Step Dry Run

**Files:**

- Verify only

- [ ] **Step 1: Run the focused Step 5 and shared-helper tests**

Run:

- `pytest test/test_aihub_option1_pilots.py -k "compiled_target_download or deployable_download_record" -v`
- `pytest test/test_aihub_option1_step5_artifacts.py -v`

Expected: PASS

- [ ] **Step 2: Re-run the notebook layout guard**

Run: `pytest test/test_option1_notebook_layout.py -v`

Expected: PASS

- [ ] **Step 3: Run Python compile verification**

Run: `python -m compileall src`

Expected: PASS

- [ ] **Step 4: Run one CLI dry-run against the current retained evidence**

Run: `python -m tools.aihub_option1_step5_artifacts --pilot all --run-label 20260519-6pm --dry-run`

Expected:

- both pilots resolve successfully
- dry-run prints:
  - compile record path
  - live-run record path
  - target model id
  - planned Step 5 package path

- [ ] **Step 5: Run one real Step 5 package build when credentials are available**

Run: `python -m tools.aihub_option1_step5_artifacts --pilot all --run-label 20260519-6pm`

Expected:

- `zipformer` package exists under `build/aihub/step5/option1/zipformer/20260519-6pm/`
- `vpcd` package exists under `build/aihub/step5/option1/vpcd/20260519-6pm/`
- each package contains:
  - `step5_manifest.json`
  - `io_contract.json`
  - `deploy_notes.md`
  - downloaded compiled artifact
  - copied evidence records

## Acceptance Criteria

- Step 5 can run after compile plus live-run evidence already exists; it does not force recompilation
- both retained pilots download their compiled `precompiled_qnn_onnx` artifacts from AI Hub through one shared code path
- each pilot gets one deterministic Step 5 package under `build/aihub/step5/option1/...`
- each package records device family, QAIRT version, compile options, target model id, file hashes, and special I/O handling notes
- workflow docs tell the operator exactly what to run after `On_device_Ai_option1_pilots.ipynb`
- the shared pilot notebook remains limited to Phase 2 plus Phase 3 responsibilities

## Decision Gates After Step 5

1. If Step 5 package download fails for a pilot, do not start Android handoff for that pilot.
2. If `io_contract.json` cannot be derived cleanly from the current records, fix the Python-side record contract before touching BKMeeting.
3. If Step 5 succeeds for only one pilot, move only that pilot forward; do not block the stronger lane on the weaker one.
4. Treat Step 5 package creation as a Python-side deployment input, not as proof of working Snapdragon runtime behavior.

## Recommended Execution Order

1. Implement Task 1 first so target-model download and Step 5 records have one shared contract.
2. Implement Task 2 next to lock deterministic record resolution and output paths.
3. Implement Task 3 to materialize packages only after the resolver contract is stable.
4. Implement Task 4 to expose the CLI and document the operator workflow after the core package writer is working.
5. Finish with Task 5 verification before calling Step 5 complete.
