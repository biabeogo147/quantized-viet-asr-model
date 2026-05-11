# VPCD Fixed Shape QNN Preflight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a Python-verified, fixed-shape VPCD QDQ bundle that is ready to hand off to BKMeeting's Android QNN HTP deployment phase, or produce an explicit blocker report if VPCD cannot be made fixed-shape safely.

**Architecture:** Keep the existing `vpcd_balanced` QDQ bundle as the reference bundle. Add a fixed-shape candidate path that freezes the VPCD model inputs, makes the bundle runtime pad encoder and decoder inputs correctly, compares the fixed-shape candidate against the reference bundle on golden samples, and emits a machine-readable QNN preflight report. Do not attempt actual HTP execution in this plan; that belongs to the Android/QNN runtime phase.

**Tech Stack:** Python 3, ONNX, ONNX Runtime CPUExecutionProvider, ONNX Runtime Extensions tokenizer graphs, NumPy, pytest, existing `model_bundle`, `quantize.fixed_shapes`, `verify.model_bundle`, and `test.test_punctuation_model_onnx` modules.

---

## Implementation Status

Updated 2026-05-09:

- Tasks 1-5 implemented in `python-model-test`.
- Tasks 6-8 verification and handoff were completed, except repository commit steps which remain intentionally deferred until branch integration.
- Real candidate generated at `build/model_bundle/vpcd/qnn_fixed_1024x128`.
- Tokenizer bundle verification passed for the fixed-shape candidate.
- Reference-vs-candidate VPCD parity passed for 2 golden samples.
- QNN preflight passed and wrote `build/model_bundle/vpcd/qnn_fixed_1024x128/qnn_preflight_report.json`.
- Python CPU smoke ran successfully, but one fixed-shape sample took about 285 seconds. This remains a Python CPU cost note, not an HTP blocker.
- The Android deploy input is now the fixed-shape `qnn_fixed_1024x128` candidate. The dynamic `vpcd_balanced` bundle remains the CPU/reference baseline.
- HTP/NPU execution remains untested until the BKMeeting Android QNN runtime phase.

## Android NPU Deploy Handoff

Use this packet when BKMeeting starts the Android QNN branch:

- Source candidate: `build/model_bundle/vpcd/qnn_fixed_1024x128`
- Model variant: `vpcd_balanced_fixed_1024x128`
- Model SHA256: `3A54567924281D472C8B271E0D5FCCB59652DF0C05A7A6D9CC586E17AB9888CA`
- Manifest SHA256: `DC4ADE5F18CD9474148B50BC83A27DD5C79962FBF2D31FE1281DCA41E5FBB561`
- QNN preflight report SHA256: `33D8C15FEF86AFA12CE263B8BFE2DAF3276E9A6D7797F669B9E69CA1F7095A70`
- Fixed model inputs: `input_ids[1,1024]`, `attention_mask[1,1024]`, `decoder_input_ids[1,128]`, `decoder_attention_mask[1,128]`
- QNN target: VPCD `model.mobile.onnx` session only
- CPU-only first slice: `tokenizer.encode.onnx`, `tokenizer.decode.onnx`, tokenizer id maps, and text pre/post processing

Do not treat the dynamic `vpcd_balanced` reference as an HTP target anymore. For Android validation, either copy the fixed-shape candidate into the chosen production namespace on the QNN branch or stage it under a separate namespace and point the punctuation runtime at that namespace for `QNN_HTP_STRICT`.

## Current State

- `vpcd_balanced` is already a QNN-targeted PTQ + QDQ artifact.
- `build/model_bundle/vpcd/vpcd_balanced/model.mobile.onnx` is byte-identical to the active BKMeeting VPCD model asset.
- The VPCD graph has QDQ nodes and `UINT16` / `UINT8` initializers.
- The VPCD bundle manifest now declares:
  - `metadata.quantization.format = "QDQ"`
  - `metadata.quantization.activation_type = "quint16"`
  - `metadata.quantization.weight_type = "quint8"`
  - `metadata.quantization.fixed_shapes = false`
  - `metadata.qnn_readiness.fixed_shapes_ready = false`
  - `metadata.qnn_readiness.tokenizer_policy = "cpu_only_first_slice"`
- The remaining Python-side blocker is fixed-shape readiness.
- Current ONNX inputs are symbolic:
  - `input_ids`: `[batch, encoder_sequence]`
  - `attention_mask`: `[batch, encoder_sequence]`
  - `decoder_input_ids`: `[batch, decoder_sequence]`
  - `decoder_attention_mask`: `[batch, decoder_sequence]`

## Target Candidate Shape

Use the conservative first candidate:

```text
batch = 1
encoder_sequence = 1024
decoder_sequence = 128
```

Rationale:

- `max_source_length` is already `1024`.
- `max_decode_length` is already `128`.
- The Android first slice keeps tokenizer graphs on CPU, so only `model.mobile.onnx` needs fixed-shape QNN readiness.
- Decoder generation currently feeds growing dynamic decoder inputs. Fixed-shape runtime must pad decoder inputs to 128 and read logits at the current decoder position, not blindly at `-1`.

## Planned File Map

### Runtime Shape Handling

- Create: `src/model_bundle/projects/vpcd_shapes.py`
  - Shared VPCD shape constants.
  - Reads `metadata.fixed_input_shapes.model`.
  - Pads `input_ids`, `attention_mask`, `decoder_input_ids`, and `decoder_attention_mask`.
  - Computes the logits index for the current decoder step.
- Modify: `src/model_bundle/projects/_vpcd_support.py`
  - Make `BundleOnnxRuntime.restore(...)` fixed-shape aware.
  - Keep dynamic-shape behavior unchanged when no fixed-shape metadata exists.
- Test: `test/test_vpcd_bundle.py`
  - Add fixed-shape runtime tests with fake sessions.

### Candidate Bundle Generation

- Create: `src/tools/prepare_vpcd_qnn_candidate.py`
  - Copies an existing VPCD bundle.
  - Freezes `model.mobile.onnx` input shapes.
  - Updates manifest metadata to mark fixed-shape readiness.
- Test: `test/test_vpcd_qnn_candidate.py`
  - Unit-test manifest update and fixed-shape call boundaries.

### QNN Preflight Verification

- Create: `src/model_bundle/qnn_preflight.py`
  - Reusable preflight checks for manifest and ONNX graph.
- Create: `src/verify/qnn_preflight.py`
  - CLI entry point for QNN preflight reports.
- Test: `test/test_qnn_preflight.py`
  - Synthetic ONNX graph tests and manifest validation tests.
- Modify: `src/verify/README.md`
  - Document the new preflight command.

### VPCD Reference-vs-Candidate Parity

- Modify: `src/model_bundle/projects/vpcd.py`
  - Extend `verify_bundle(...)` to support `reference_bundle` + `candidate_bundle`.
- Modify: `src/verify/model_bundle.py`
  - No parser change should be needed because it already accepts `--reference-bundle` and `--candidate-bundle`.
- Test: `test/test_vpcd_bundle.py`
  - Add candidate parity tests using fake runtimes.

### Docs And Handoff

- Modify: `README.md`
- Modify: `src/model_bundle/README.md`
- Modify: `src/quantize/README.md`
- Modify: `docs/plans/2026-05-09-vpcd-fixed-shape-qnn-preflight.md` as tasks complete.

## Task 1: Add VPCD Fixed-Shape Utilities

**Files:**

- Create: `src/model_bundle/projects/vpcd_shapes.py`
- Test: `test/test_vpcd_bundle.py`

Tracking note: implementation, real-bundle verification, and documentation steps in this plan have been executed on branch `codex-vpcd-qnn-preflight`. The explicit `git commit` checklist items are left open because branch integration has not been requested yet.

- [ ] **Step 1: Write failing tests for fixed-shape metadata parsing**

Add tests similar to:

```python
from model_bundle.projects.vpcd_shapes import resolve_vpcd_model_input_shapes


def test_resolve_vpcd_model_input_shapes_reads_manifest_metadata():
    metadata = {
        "fixed_input_shapes": {
            "model": {
                "input_ids": [1, 1024],
                "attention_mask": [1, 1024],
                "decoder_input_ids": [1, 128],
                "decoder_attention_mask": [1, 128],
            }
        }
    }

    shapes = resolve_vpcd_model_input_shapes(metadata)

    assert shapes.input_ids == (1, 1024)
    assert shapes.attention_mask == (1, 1024)
    assert shapes.decoder_input_ids == (1, 128)
    assert shapes.decoder_attention_mask == (1, 128)
```

- [ ] **Step 2: Run the focused failing test**

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m pytest test\test_vpcd_bundle.py -q
```

Expected: fails because `model_bundle.projects.vpcd_shapes` does not exist yet.

- [ ] **Step 3: Implement the shape dataclass and parser**

Implement:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VpcdModelInputShapes:
    input_ids: tuple[int, int]
    attention_mask: tuple[int, int]
    decoder_input_ids: tuple[int, int]
    decoder_attention_mask: tuple[int, int]

    @property
    def encoder_sequence(self) -> int:
        return self.input_ids[1]

    @property
    def decoder_sequence(self) -> int:
        return self.decoder_input_ids[1]


def resolve_vpcd_model_input_shapes(metadata: dict[str, Any]) -> VpcdModelInputShapes | None:
    fixed_input_shapes = metadata.get("fixed_input_shapes")
    if not isinstance(fixed_input_shapes, dict):
        return None
    model = fixed_input_shapes.get("model")
    if not isinstance(model, dict):
        return None

    def shape2(name: str) -> tuple[int, int]:
        value = model.get(name)
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"VPCD fixed shape for {name} must be [batch, sequence]")
        batch, sequence = int(value[0]), int(value[1])
        if batch != 1 or sequence <= 0:
            raise ValueError(f"Unsupported VPCD fixed shape for {name}: {value}")
        return batch, sequence

    return VpcdModelInputShapes(
        input_ids=shape2("input_ids"),
        attention_mask=shape2("attention_mask"),
        decoder_input_ids=shape2("decoder_input_ids"),
        decoder_attention_mask=shape2("decoder_attention_mask"),
    )
```

- [ ] **Step 4: Add token padding helpers**

Add tests for:

```python
from model_bundle.projects.vpcd_shapes import pad_token_row, attention_mask_for_length


def test_pad_token_row_pads_to_fixed_length():
    assert pad_token_row([7, 8, 2], target_length=5, pad_value=1).tolist() == [[7, 8, 2, 1, 1]]


def test_attention_mask_for_length_marks_padding_as_zero():
    assert attention_mask_for_length(actual_length=3, target_length=5).tolist() == [[1, 1, 1, 0, 0]]
```

Implementation should:

- return 2D `np.int64` arrays with shape `[1, target_length]`
- raise `ValueError` if actual length exceeds target length
- keep dynamic runtime path separate from fixed-shape runtime path

- [ ] **Step 5: Run tests**

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m pytest test\test_vpcd_bundle.py -q
```

Expected: all VPCD bundle tests pass.

- [ ] **Step 6: Commit**

```powershell
git add src/model_bundle/projects/vpcd_shapes.py test/test_vpcd_bundle.py
git commit -m "feat: add vpcd fixed shape helpers"
```

## Task 2: Make VPCD Bundle Runtime Fixed-Shape Aware

**Files:**

- Modify: `src/model_bundle/projects/_vpcd_support.py`
- Test: `test/test_vpcd_bundle.py`

- [ ] **Step 1: Write a failing test that fixed encoder inputs are padded**

Use fake sessions and a manifest with:

```python
metadata = {
    "pad_token_id": 1,
    "eos_token_id": 2,
    "decoder_start_token_id": 2,
    "max_source_length": 1024,
    "max_decode_length": 128,
    "fixed_input_shapes": {
        "model": {
            "input_ids": [1, 1024],
            "attention_mask": [1, 1024],
            "decoder_input_ids": [1, 128],
            "decoder_attention_mask": [1, 128],
        }
    },
}
```

Assert the first model feed has:

```python
assert feeds["input_ids"].shape == (1, 1024)
assert feeds["attention_mask"].shape == (1, 1024)
assert feeds["input_ids"][0, :4].tolist() == [0, 11, 12, 2]
assert feeds["input_ids"][0, 4] == 1
assert feeds["attention_mask"][0, :4].tolist() == [1, 1, 1, 1]
assert feeds["attention_mask"][0, 4] == 0
```

- [ ] **Step 2: Write a failing test that fixed decoder reads logits at the active position**

Construct fake logits with the desired next token only at decoder position 0, and a bad token at decoder position 127:

```python
logits = np.zeros((1, 128, 7), dtype=np.float32)
logits[0, 0, 5] = 9.0
logits[0, 127, 6] = 99.0
```

Expected: runtime chooses token `5`, proving it uses `current_decoder_length - 1` instead of `-1`.

- [ ] **Step 3: Run focused tests to verify they fail**

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m pytest test\test_vpcd_bundle.py -q
```

Expected: fails until `BundleOnnxRuntime.restore(...)` supports fixed-shape feeds.

- [ ] **Step 4: Implement fixed-shape feed preparation**

In `BundleOnnxRuntime.__init__`, resolve fixed shapes once:

```python
from model_bundle.projects.vpcd_shapes import resolve_vpcd_model_input_shapes

self.fixed_input_shapes = resolve_vpcd_model_input_shapes(self.metadata)
```

In `restore(...)`:

- keep the current dynamic path when `self.fixed_input_shapes is None`
- for fixed shape:
  - encode raw model IDs without fixed padding first
  - pad `input_ids` to `encoder_sequence` with `pad_token_id`
  - build `attention_mask` with 1 for real tokens and 0 for padding
  - at every decoder step, pad `decoder_input_ids` to `decoder_sequence` with `pad_token_id`
  - build `decoder_attention_mask` with 1 for active decoder tokens and 0 for padding
  - get logits at `active_decoder_length - 1`

- [ ] **Step 5: Add a helper for active-position argmax**

Replace or extend `_argmax_last_token(...)` with:

```python
@staticmethod
def _argmax_token_at(logits: object, position: int | None = None) -> int:
    array = np.asarray(logits)
    if array.ndim == 3:
        index = -1 if position is None else int(position)
        return int(np.argmax(array[:, index, :], axis=-1)[0])
    if array.ndim == 2:
        index = -1 if position is None else int(position)
        return int(np.argmax(array[index]))
    if array.ndim == 1:
        return int(np.argmax(array))
    raise ValueError(f"Unsupported logits shape: {array.shape}")
```

- [ ] **Step 6: Run VPCD tests**

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m pytest test\test_vpcd_bundle.py -q
```

Expected: all VPCD bundle tests pass.

- [ ] **Step 7: Commit**

```powershell
git add src/model_bundle/projects/_vpcd_support.py test/test_vpcd_bundle.py
git commit -m "feat: support fixed shape vpcd bundle runtime"
```

## Task 3: Generate A Fixed-Shape VPCD QDQ Candidate Bundle

**Files:**

- Create: `src/tools/prepare_vpcd_qnn_candidate.py`
- Test: `test/test_vpcd_qnn_candidate.py`
- Reuse: `src/quantize/fixed_shapes.py`

- [ ] **Step 1: Write a failing CLI test for candidate bundle preparation**

Test behavior with monkeypatched `freeze_model_inputs(...)`:

```python
def test_prepare_vpcd_qnn_candidate_updates_manifest(tmp_path, monkeypatch):
    source = tmp_path / "source"
    output = tmp_path / "candidate"
    # create source bundle files and source manifest

    frozen_calls = []
    monkeypatch.setattr(
        "tools.prepare_vpcd_qnn_candidate.freeze_model_inputs",
        lambda model_path, output_path, input_shapes: frozen_calls.append((model_path, output_path, input_shapes)) or output_path,
    )

    main([
        "--source-bundle", str(source),
        "--output-dir", str(output),
        "--encoder-sequence", "1024",
        "--decoder-sequence", "128",
    ])

    manifest = ModelBundleManifest.from_path(output / "bundle_manifest.json")
    assert manifest.metadata["quantization"]["fixed_shapes"] is True
    assert manifest.metadata["qnn_readiness"]["fixed_shapes_ready"] is True
    assert manifest.metadata["fixed_input_shapes"]["model"]["input_ids"] == [1, 1024]
```

- [ ] **Step 2: Run the failing test**

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m pytest test\test_vpcd_qnn_candidate.py -q
```

Expected: fails because the CLI does not exist yet.

- [ ] **Step 3: Implement the CLI**

Behavior:

- require `--source-bundle`
- require `--output-dir`
- default `--encoder-sequence 1024`
- default `--decoder-sequence 128`
- optional `--model-variant vpcd_balanced_fixed_1024x128`
- copy all files from source bundle to output bundle
- run `freeze_model_inputs(...)` for `model.mobile.onnx`
- update manifest:

```python
fixed_input_shapes = {
    "model": {
        "input_ids": [1, encoder_sequence],
        "attention_mask": [1, encoder_sequence],
        "decoder_input_ids": [1, decoder_sequence],
        "decoder_attention_mask": [1, decoder_sequence],
    }
}
metadata["fixed_input_shapes"] = fixed_input_shapes
metadata["quantization"]["fixed_shapes"] = True
metadata["qnn_readiness"]["fixed_shapes_ready"] = True
metadata["qnn_readiness"].pop("fixed_shape_blocker", None)
```

Keep tokenizer artifacts unchanged.

- [ ] **Step 4: Add manifest guardrails**

Fail with clear errors if:

- source bundle is not `project == "vpcd"`
- source manifest has no `artifacts.model`
- source `metadata.quantization.format` is not `QDQ`
- source `metadata.quantization.activation_type` is not `quint16`
- source `metadata.quantization.weight_type` is not `quint8`
- output directory is the same as source directory

- [ ] **Step 5: Run tests**

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m pytest test\test_vpcd_qnn_candidate.py -q
```

Expected: candidate preparation unit tests pass.

- [ ] **Step 6: Generate the real candidate**

Run from repo root:

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m tools.prepare_vpcd_qnn_candidate `
  --source-bundle build/model_bundle/vpcd/vpcd_balanced `
  --output-dir build/model_bundle/vpcd/qnn_fixed_1024x128 `
  --encoder-sequence 1024 `
  --decoder-sequence 128
```

Expected:

- `build/model_bundle/vpcd/qnn_fixed_1024x128/bundle_manifest.json`
- `build/model_bundle/vpcd/qnn_fixed_1024x128/model.mobile.onnx`
- tokenizer graphs and ID maps copied unchanged
- manifest has `fixed_input_shapes.model`
- manifest has `metadata.quantization.fixed_shapes = true`
- manifest has `metadata.qnn_readiness.fixed_shapes_ready = true`

- [ ] **Step 7: Commit**

```powershell
git add src/tools/prepare_vpcd_qnn_candidate.py test/test_vpcd_qnn_candidate.py build/model_bundle/vpcd/qnn_fixed_1024x128/bundle_manifest.json
git commit -m "feat: prepare fixed shape vpcd qnn candidate"
```

If the binary candidate is too large or should not be committed, commit the CLI and test only, then record the generated artifact path in `docs/qnn-preflight-results.md`.

## Task 4: Add QNN Preflight Checker

**Files:**

- Create: `src/model_bundle/qnn_preflight.py`
- Create: `src/verify/qnn_preflight.py`
- Test: `test/test_qnn_preflight.py`
- Modify: `src/verify/README.md`

- [ ] **Step 1: Write failing preflight tests for VPCD fixed-shape readiness**

Test a passing manifest with:

```python
metadata = {
    "fixed_input_shapes": {
        "model": {
            "input_ids": [1, 1024],
            "attention_mask": [1, 1024],
            "decoder_input_ids": [1, 128],
            "decoder_attention_mask": [1, 128],
        }
    },
    "quantization": {
        "format": "QDQ",
        "activation_type": "quint16",
        "weight_type": "quint8",
        "fixed_shapes": True,
    },
    "qnn_readiness": {
        "target_backend": "qnn_htp",
        "model_session_candidate": True,
        "tokenizer_policy": "cpu_only_first_slice",
        "requires_fixed_shapes": True,
        "fixed_shapes_ready": True,
    },
}
```

Expected report:

```python
assert report["passed"] is True
assert report["checks"]["manifest_quantization"]["passed"] is True
assert report["checks"]["fixed_input_shapes"]["passed"] is True
assert report["checks"]["onnx_qdq_graph"]["passed"] is True
```

- [ ] **Step 2: Write failing tests for expected blockers**

Cover:

- manifest missing `metadata.quantization`
- `fixed_shapes = false`
- graph input still has `dim_param`
- ONNX graph has no `QuantizeLinear` or `DequantizeLinear`
- ONNX initializers do not include `UINT16` and `UINT8`

- [ ] **Step 3: Run tests to verify failure**

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m pytest test\test_qnn_preflight.py -q
```

Expected: fails because preflight code does not exist yet.

- [ ] **Step 4: Implement ONNX graph inspection**

In `src/model_bundle/qnn_preflight.py`, implement:

```python
def inspect_onnx_for_qnn_qdq(model_path: Path) -> dict:
    model = onnx.load(str(model_path), load_external_data=False)
    # count op types
    # count initializer dtypes
    # collect graph input shapes and symbolic dims
```

Return at least:

```python
{
    "op_counts": {"QuantizeLinear": 329, "DequantizeLinear": 513},
    "initializer_dtypes": {"UINT16": 329, "UINT8": 368},
    "inputs": {
        "input_ids": [1, 1024],
        "attention_mask": [1, 1024],
        "decoder_input_ids": [1, 128],
        "decoder_attention_mask": [1, 128],
    },
    "symbolic_inputs": [],
}
```

- [ ] **Step 5: Implement manifest checks**

Validate:

- `project == "vpcd"`
- artifact `model` exists
- `quantization.format == "QDQ"`
- `quantization.activation_type == "quint16"`
- `quantization.weight_type == "quint8"`
- `quantization.fixed_shapes is True`
- `qnn_readiness.target_backend == "qnn_htp"`
- `qnn_readiness.model_session_candidate is True`
- `qnn_readiness.tokenizer_policy == "cpu_only_first_slice"`
- `qnn_readiness.fixed_shapes_ready is True`
- `fixed_input_shapes.model` exists and matches graph input shapes

- [ ] **Step 6: Implement CLI**

Command:

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m verify.qnn_preflight `
  --project vpcd `
  --bundle-dir build/model_bundle/vpcd/qnn_fixed_1024x128 `
  --output build/model_bundle/vpcd/qnn_fixed_1024x128/qnn_preflight_report.json
```

CLI output:

```text
QNN preflight complete.
Project : vpcd
Bundle  : build/model_bundle/vpcd/qnn_fixed_1024x128
Passed  : True
Report  : build/model_bundle/vpcd/qnn_fixed_1024x128/qnn_preflight_report.json
```

- [ ] **Step 7: Run tests**

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m pytest test\test_qnn_preflight.py -q
```

Expected: all preflight tests pass.

- [ ] **Step 8: Commit**

```powershell
git add src/model_bundle/qnn_preflight.py src/verify/qnn_preflight.py test/test_qnn_preflight.py src/verify/README.md
git commit -m "feat: add qnn preflight verifier"
```

## Task 5: Add VPCD Reference-vs-Candidate Bundle Parity

**Files:**

- Modify: `src/model_bundle/projects/vpcd.py`
- Test: `test/test_vpcd_bundle.py`

- [ ] **Step 1: Write a failing test for candidate comparison mode**

Use fake runtimes:

```python
def test_verify_vpcd_candidate_bundle_matches_reference(monkeypatch, tmp_case_dir):
    reference_bundle = tmp_case_dir / "reference"
    candidate_bundle = tmp_case_dir / "candidate"
    # write candidate golden_samples.jsonl

    class FakeRuntime:
        def __init__(self, label):
            self.label = label

        @classmethod
        def from_manifest_path(cls, manifest_path, provider="CPUExecutionProvider"):
            return cls("candidate" if "candidate" in str(manifest_path) else "reference")

        def restore(self, text, max_length=128):
            return "Xin chao."

    monkeypatch.setattr("model_bundle.projects.vpcd.BundleOnnxRuntime", FakeRuntime)

    report = verify_bundle(reference_bundle=reference_bundle, candidate_bundle=candidate_bundle)

    assert report["passed"] is True
    assert report["checked_samples"] == 1
```

- [ ] **Step 2: Run failing test**

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m pytest test\test_vpcd_bundle.py -q
```

Expected: fails because `verify_bundle(...)` only supports `model_dir` + `bundle_dir` today.

- [ ] **Step 3: Implement VPCD candidate parity**

Extend signature:

```python
def verify_bundle(
    *,
    model_dir: Path | None = None,
    bundle_dir: Path | None = None,
    reference_bundle: Path | None = None,
    candidate_bundle: Path | None = None,
    provider: str = "CPUExecutionProvider",
) -> tuple[int, int] | dict:
```

Behavior:

- if `reference_bundle` and `candidate_bundle` are provided:
  - load both with `BundleOnnxRuntime.from_manifest_path(...)`
  - read golden samples from candidate bundle
  - run `restore(raw_text)` on both
  - compare strings exactly after `.strip()`
  - return:

```python
{
    "checked_samples": checked,
    "passed": not mismatches,
    "mismatches": mismatches,
}
```

- preserve current tokenizer verification behavior for `model_dir` + `bundle_dir`

- [ ] **Step 4: Run candidate parity on real bundles**

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m verify.model_bundle `
  --project vpcd `
  --reference-bundle build/model_bundle/vpcd/vpcd_balanced `
  --candidate-bundle build/model_bundle/vpcd/qnn_fixed_1024x128
```

Expected:

```text
Verification complete.
Project        : vpcd
Checked samples: 2
Passed         : True
```

If this fails, inspect whether the fixed decoder path is reading logits from the wrong position or whether fixed-shape ONNX changed model behavior.

- [ ] **Step 5: Commit**

```powershell
git add src/model_bundle/projects/vpcd.py test/test_vpcd_bundle.py
git commit -m "test: compare vpcd fixed shape candidate with reference"
```

## Task 6: Run Real Candidate Smoke And Preflight

**Files:**

- Generated: `build/model_bundle/vpcd/qnn_fixed_1024x128/qnn_preflight_report.json`
- Optional create: `docs/qnn-preflight-results.md`

- [ ] **Step 1: Smoke-test bundle-manifest runtime**

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m test.test_punctuation_model_onnx `
  --bundle-manifest build/model_bundle/vpcd/qnn_fixed_1024x128/bundle_manifest.json `
  --text "hom nay la buoi nham chuc cua toi phuoc thanh"
```

Expected:

- no ONNX Runtime shape error
- output text is produced
- runtime feeds fixed `[1, 1024]` encoder tensors and `[1, 128]` decoder tensors internally

- [ ] **Step 2: Run tokenizer verification**

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m verify.model_bundle `
  --project vpcd `
  --model-dir assets/vietnamese-punc-cap-denorm-v1 `
  --bundle-dir build/model_bundle/vpcd/qnn_fixed_1024x128
```

Expected:

```text
Verification complete.
Project        : vpcd
Encode samples : 2
Decode samples : 2
```

- [ ] **Step 3: Run reference-vs-candidate parity**

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m verify.model_bundle `
  --project vpcd `
  --reference-bundle build/model_bundle/vpcd/vpcd_balanced `
  --candidate-bundle build/model_bundle/vpcd/qnn_fixed_1024x128
```

Expected:

```text
Checked samples: 2
Passed         : True
```

- [ ] **Step 4: Run QNN preflight**

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m verify.qnn_preflight `
  --project vpcd `
  --bundle-dir build/model_bundle/vpcd/qnn_fixed_1024x128 `
  --output build/model_bundle/vpcd/qnn_fixed_1024x128/qnn_preflight_report.json
```

Expected:

```text
Passed  : True
```

- [ ] **Step 5: Record the result**

If all checks pass, create or update `docs/qnn-preflight-results.md`:

```markdown
# QNN Preflight Results

## VPCD fixed shape 1024x128

- Source bundle: `build/model_bundle/vpcd/vpcd_balanced`
- Candidate bundle: `build/model_bundle/vpcd/qnn_fixed_1024x128`
- Quantization: QDQ, QUInt16 activations, QUInt8 weights
- Fixed shapes: input/attention `[1, 1024]`, decoder inputs `[1, 128]`
- Tokenizer policy: CPU-only first slice
- Bundle parity: Passed
- QNN preflight: Passed
- HTP execution: Not tested in Python
```

If any check fails, write the blocker instead:

```markdown
## VPCD fixed shape 1024x128

Decision: blocked before Android QNN deploy.
Reason: <exact error or mismatch>
Next action: <shape/export/runtime fix needed>
```

- [ ] **Step 6: Commit**

```powershell
git add docs/qnn-preflight-results.md build/model_bundle/vpcd/qnn_fixed_1024x128/qnn_preflight_report.json
git commit -m "docs: record vpcd qnn preflight result"
```

## Task 7: Update Developer Documentation

**Files:**

- Modify: `README.md`
- Modify: `src/model_bundle/README.md`
- Modify: `src/quantize/README.md`
- Modify: `src/verify/README.md`

- [ ] **Step 1: Update root README VPCD handoff instructions**

Document:

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m tools.prepare_vpcd_qnn_candidate `
  --source-bundle build/model_bundle/vpcd/vpcd_balanced `
  --output-dir build/model_bundle/vpcd/qnn_fixed_1024x128

& D:\Anaconda\envs\speech2text\python.exe -m verify.qnn_preflight `
  --project vpcd `
  --bundle-dir build/model_bundle/vpcd/qnn_fixed_1024x128
```

State clearly:

- QDQ readiness is Python-verified.
- Fixed-shape readiness is Python-verified only if `qnn_preflight_report.json` passes.
- Actual HTP execution still requires Android QNN EP validation.

- [ ] **Step 2: Update model bundle README**

Add a VPCD fixed-shape candidate section:

```text
build/model_bundle/vpcd/qnn_fixed_1024x128/
  bundle_manifest.json
  model.mobile.onnx
  tokenizer.encode.onnx
  tokenizer.decode.onnx
  tokenizer.to_model_id_map.json
  tokenizer.from_model_id_map.json
  golden_samples.jsonl
  qnn_preflight_report.json
```

- [ ] **Step 3: Update quantize README**

Explain that VPCD quantization and VPCD fixed-shape candidate preparation are separate steps:

```text
quantize vpcd -> produces/refreshes QDQ ONNX
export vpcd bundle -> packages QDQ ONNX with tokenizer artifacts
prepare_vpcd_qnn_candidate -> freezes model input shapes and updates manifest
verify.qnn_preflight -> confirms package is ready for Android QNN attempt
```

- [ ] **Step 4: Update verify README**

Document:

```powershell
python -m verify.qnn_preflight --project vpcd --bundle-dir build/model_bundle/vpcd/qnn_fixed_1024x128
```

- [ ] **Step 5: Run doc grep checks**

```powershell
rg -n "qnn_fixed_1024x128|verify.qnn_preflight|prepare_vpcd_qnn_candidate|fixed_shapes_ready" README.md src docs -S
```

Expected: docs consistently point to the same candidate path and command names.

- [ ] **Step 6: Commit**

```powershell
git add README.md src/model_bundle/README.md src/quantize/README.md src/verify/README.md
git commit -m "docs: document vpcd fixed shape qnn preflight"
```

## Task 8: Final Verification Gate Before Android NPU Deploy

**Files:**

- Read: `build/model_bundle/vpcd/qnn_fixed_1024x128/bundle_manifest.json`
- Read: `build/model_bundle/vpcd/qnn_fixed_1024x128/qnn_preflight_report.json`
- Read: `docs/qnn-preflight-results.md`

- [x] **Step 1: Run focused unit tests**

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m pytest `
  test\test_vpcd_bundle.py `
  test\test_vpcd_qnn_candidate.py `
  test\test_qnn_preflight.py `
  -q
```

Expected: all selected tests pass.

- [x] **Step 2: Run relevant broader tests**

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m pytest test -q
```

Expected: all tests pass. If the full suite is too slow because it touches heavyweight ONNX assets, run and record the focused suite plus the real bundle commands from Task 6.

- [x] **Step 3: Run final real-bundle commands**

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m verify.model_bundle --project vpcd --model-dir assets/vietnamese-punc-cap-denorm-v1 --bundle-dir build/model_bundle/vpcd/qnn_fixed_1024x128

& D:\Anaconda\envs\speech2text\python.exe -m verify.model_bundle --project vpcd --reference-bundle build/model_bundle/vpcd/vpcd_balanced --candidate-bundle build/model_bundle/vpcd/qnn_fixed_1024x128

& D:\Anaconda\envs\speech2text\python.exe -m verify.qnn_preflight --project vpcd --bundle-dir build/model_bundle/vpcd/qnn_fixed_1024x128 --output build/model_bundle/vpcd/qnn_fixed_1024x128/qnn_preflight_report.json
```

Expected:

- tokenizer verification succeeds
- candidate parity passes
- QNN preflight report has `"passed": true`

- [x] **Step 4: Confirm manifest handoff fields**

Inspect:

```powershell
Get-Content build\model_bundle\vpcd\qnn_fixed_1024x128\bundle_manifest.json
```

Required fields:

```json
"fixed_input_shapes": {
  "model": {
    "input_ids": [1, 1024],
    "attention_mask": [1, 1024],
    "decoder_input_ids": [1, 128],
    "decoder_attention_mask": [1, 128]
  }
}
```

```json
"quantization": {
  "format": "QDQ",
  "activation_type": "quint16",
  "weight_type": "quint8",
  "preset": "sd8g2_balanced",
  "fixed_shapes": true
}
```

```json
"qnn_readiness": {
  "target_backend": "qnn_htp",
  "model_session_candidate": true,
  "tokenizer_policy": "cpu_only_first_slice",
  "requires_fixed_shapes": true,
  "fixed_shapes_ready": true
}
```

- [x] **Step 5: Write Android handoff note**

In `docs/qnn-preflight-results.md`, include:

```text
Android handoff:
- Copy candidate bundle contents into BKMeeting modelassets only after Android QNN branch chooses the production namespace.
- Keep VPCD tokenizer sessions on CPU.
- In strict QNN mode, disable ORT CPU fallback for the model session.
- If HTP rejects the graph, preserve CPU fallback and attach the HTP error to BKMeeting/docs/qnn-device-validation.md.
```

- [ ] **Step 6: Final commit**

```powershell
git add docs/qnn-preflight-results.md build/model_bundle/vpcd/qnn_fixed_1024x128/qnn_preflight_report.json
git commit -m "chore: finalize vpcd qnn preflight handoff"
```

## Acceptance Criteria

- `vpcd_balanced` remains the dynamic-shape QDQ reference bundle.
- A fixed-shape candidate bundle exists at `build/model_bundle/vpcd/qnn_fixed_1024x128`.
- Candidate manifest declares fixed shapes for all four VPCD model inputs.
- Candidate manifest declares QDQ `quint16` / `quint8` quantization with `fixed_shapes = true`.
- Candidate manifest declares QNN HTP readiness with tokenizer policy `cpu_only_first_slice`.
- `BundleOnnxRuntime` still works for dynamic VPCD bundles.
- `BundleOnnxRuntime` works for fixed-shape VPCD bundles by padding inputs and reading logits at the active decoder position.
- `verify.model_bundle --project vpcd --reference-bundle ... --candidate-bundle ...` passes.
- `verify.qnn_preflight --project vpcd --bundle-dir ...` passes.
- Docs clearly state that Python preflight does not prove physical HTP execution.

## Risk Register

- Fixed decoder padding may alter generated output.
  - Mitigation: active-position logits test plus real reference-vs-candidate parity.
- Freezing shapes after QDQ may not be accepted by ONNX Runtime or later QNN.
  - Mitigation: preflight detects graph metadata, Android strict validation remains the final authority.
- Freezing FP32 before quantization may be required instead of freezing QDQ after quantization.
  - Mitigation: if Task 6 fails, add a follow-up path to freeze FP32 first, rerun `sd8g2_balanced`, and compare parity.
- Candidate bundle may be too large to commit.
  - Mitigation: commit code and reports, document the reproducible generation command, and leave binary artifact untracked if needed.
- Python CPUExecutionProvider passing does not imply HTP support.
  - Mitigation: final docs and reports must say "Python QNN preflight passed", not "NPU deploy passed".

## Recommended Implementation Order

1. Add VPCD fixed-shape utility functions.
2. Make bundle runtime fixed-shape aware and prove logits indexing is correct.
3. Generate the fixed-shape candidate bundle from `vpcd_balanced`.
4. Add QNN preflight verifier.
5. Add reference-vs-candidate VPCD parity.
6. Run real candidate smoke, parity, and preflight commands.
7. Update docs and handoff notes.
8. Only then start BKMeeting Android QNN runtime deployment.
