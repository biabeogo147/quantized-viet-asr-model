# Zipformer Python-First Bundle And QNN Plan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python-first Zipformer workflow that can run `test_acoustic_model_onnx.py` in both model-dir mode and bundle-manifest mode, verify parity between the two, and only then produce QNN-oriented fixed-shape quantized artifacts for Snapdragon 8 Gen 2.

**Architecture:** Treat `python-model-test` as the source of truth for the acoustic model contract before Android consumes anything. First extract the current RNNT inference logic from `test/test_vietasr.py` into reusable runtimes, define an `acoustic_bundle` contract analogous to punctuation but specialized for encoder/decoder/joiner plus `tokens.txt`, and add a clean smoke runner `test_acoustic_model_onnx.py` with two mutually-exclusive modes. After bundle parity is locked, extend the existing `quantize` package to target Zipformer with fixed-shape preprocessing, `QUInt16/QUInt8` QDQ presets, and offline evaluation scripts.

**Tech Stack:** Python 3, ONNX Runtime, ONNX, NumPy, Torchaudio, pytest, existing `quantize` package in `python-model-test`

---

## Scope To Lock

- This plan is **Python-only**. Do not touch `bkmeeting` implementation while executing it.
- This plan covers only `assets/zipformer`.
- `test/test_vietasr.py` remains the legacy exploratory script until the new runner is complete.
- The first success milestone is **not** quantization. It is a clean `test_acoustic_model_onnx.py` that works like the punctuation runner:
  - `--model-dir` mode uses only the source model directory.
  - `--bundle-manifest` mode uses only the exported acoustic bundle.
- QNN quantization starts only after bundle parity exists and is tested.

## Assumptions

- The reference source model directory is `python-model-test/assets/zipformer`.
- The first acoustic bundle targets the current FP32 model set from that directory.
- Bundle layout should be future-proof for Android, but this plan must be fully useful even if Android work is postponed.
- Default quantization target for Snapdragon 8 Gen 2 remains `QUInt16` activations + `QUInt8` weights.
- `QUInt8/QUInt8` stays experimental and must not be the default preset.

## Planned File Map

### Acoustic Bundle Contract

- Create: `python-model-test/acoustic_bundle/__init__.py` - package boundary for acoustic bundle helpers.
- Create: `python-model-test/acoustic_bundle/manifest.py` - manifest schema and path resolution helpers.
- Create: `python-model-test/acoustic_bundle/runtime.py` - bundle-backed RNNT runtime for encoder/decoder/joiner.
- Create: `python-model-test/acoustic_bundle/exporter.py` - export FP32 acoustic bundle from a model directory.
- Create: `python-model-test/acoustic_bundle/verifier.py` - compare model-dir runtime against bundle runtime on audio fixtures.
- Create: `python-model-test/acoustic_bundle/eval_fixtures.py` - deterministic sample manifest and transcript fixture helpers.
- Create: `python-model-test/export_acoustic_bundle.py` - CLI entry point for bundle export.
- Create: `python-model-test/verify_acoustic_bundle.py` - CLI entry point for parity verification.

### Acoustic Runtimes And Smoke Tests

- Create: `python-model-test/test/test_acoustic_model_onnx.py` - new smoke runner with model-dir mode and bundle-manifest mode.
- Create: `python-model-test/test/test_acoustic_bundle.py` - tests for bundle manifest, exporter, and runtime selection.
- Modify: `python-model-test/test/test_vietasr.py` - extract reusable Zipformer RNNT helpers and reduce duplicated ad-hoc logic.
- Modify: `python-model-test/test/README.md` - document the new acoustic smoke runner.

### Zipformer Quantization Phase

- Create: `python-model-test/quantize/projects/__init__.py` - project registry for quantization targets.
- Create: `python-model-test/quantize/projects/zipformer.py` - Zipformer-specific quantization metadata and path discovery.
- Create: `python-model-test/quantize/fixed_shapes.py` - helper to make encoder/decoder/joiner inputs fixed for QNN.
- Create: `python-model-test/quantize/evaluate_zipformer.py` - offline evaluation and quality reporting.
- Modify: `python-model-test/quantize/config.py` - remove punctuation-only assumptions from defaults.
- Modify: `python-model-test/quantize/types.py` - add project/component/fixed-shape metadata.
- Modify: `python-model-test/quantize/presets.py` - add Zipformer-specific QNN presets.
- Modify: `python-model-test/quantize/cli.py` - add `--project zipformer` and component-level execution.
- Modify: `python-model-test/quantize/qnn.py` - support Zipformer fixed-shape QDQ generation.
- Modify: `python-model-test/quantize/README.md` - document the new Zipformer flow separately from punctuation.

### Plan And Repo Docs

- Create: `python-model-test/plans/README.md` - index for repo-level plans.
- Create: `python-model-test/plans/2026-04-01-zipformer-python-first-bundle-qnn.md` - this plan.

## Bundle Layout To Standardize

The first exported acoustic bundle should look like:

```text
python-model-test/build/acoustic_bundle/zipformer/fp32/
  bundle_manifest.json
  encoder.onnx
  decoder.onnx
  joiner.onnx
  tokens.txt
  sample_manifest.jsonl
  expected_outputs.jsonl
```

Minimum manifest fields:

```json
{
  "bundle_version": 1,
  "model_family": "zipformer-rnnt",
  "model_name": "zipformer/fp32",
  "asset_namespace": "models/asr/zipformer/fp32",
  "encoder_file": "encoder.onnx",
  "decoder_file": "decoder.onnx",
  "joiner_file": "joiner.onnx",
  "tokens_file": "tokens.txt",
  "sample_manifest_file": "sample_manifest.jsonl",
  "expected_outputs_file": "expected_outputs.jsonl",
  "sample_rate": 16000,
  "feature_dim": 80,
  "blank_id": 0,
  "context_size": 2
}
```

## Success Gates

Do not call this phase complete until all of these are true:

- `test/test_acoustic_model_onnx.py` exists and supports `--model-dir` and `--bundle-manifest` modes.
- `BundleAcousticRuntime` does not read anything from `model-dir` once `--bundle-manifest` is chosen.
- `verify_acoustic_bundle.py` passes on the default sample set.
- The default exported FP32 bundle can be replayed end-to-end using only the bundle manifest.
- Zipformer quantization presets exist in the `quantize` package.
- A fixed-shape QNN-ready artifact set can be produced for Zipformer and evaluated offline.

## Task 1: Create The Clean Acoustic Smoke Runner

**Files:**
- Create: `python-model-test/test/test_acoustic_model_onnx.py`
- Modify: `python-model-test/test/test_vietasr.py`
- Modify: `python-model-test/test/README.md`

- [ ] **Step 1: Write a failing test for runtime selection**

```python
def test_create_runtime_uses_bundle_runtime_when_bundle_manifest_is_provided():
    args = parser.parse_args(["--bundle-manifest", "bundle_manifest.json"])
    runtime = create_runtime(args)
    assert runtime.__class__.__name__ == "BundleAcousticRuntime"
```

- [ ] **Step 2: Write a failing test for model-dir mode**

```python
def test_create_runtime_uses_model_dir_runtime_by_default():
    args = parser.parse_args([])
    runtime = create_runtime(args)
    assert runtime.__class__.__name__ == "ModelDirAcousticRuntime"
```

- [ ] **Step 3: Run the tests and confirm they fail**

Run:

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m pytest python-model-test/test/test_acoustic_bundle.py -q
```

Expected: FAIL because `test_acoustic_model_onnx.py` and the new runtime classes do not exist yet.

- [ ] **Step 4: Extract shared RNNT inference helpers from `test/test_vietasr.py`**
- [ ] **Step 5: Implement `ModelDirAcousticRuntime` in the new runner**
- [ ] **Step 6: Re-run the tests**

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add python-model-test/test/test_acoustic_model_onnx.py python-model-test/test/test_acoustic_bundle.py python-model-test/test/test_vietasr.py python-model-test/test/README.md
git commit -m "feat: add clean zipformer acoustic smoke runner"
```

## Task 2: Define The Acoustic Bundle Contract

**Files:**
- Create: `python-model-test/acoustic_bundle/__init__.py`
- Create: `python-model-test/acoustic_bundle/manifest.py`
- Create: `python-model-test/acoustic_bundle/eval_fixtures.py`
- Create: `python-model-test/test/test_acoustic_bundle.py`

- [ ] **Step 1: Write a failing test for bundle manifest serialization**

```python
def test_acoustic_bundle_manifest_round_trips():
    manifest = AcousticBundleManifest(...)
    payload = manifest.to_dict()
    restored = AcousticBundleManifest.from_dict(payload)
    assert restored.encoder_file == "encoder.onnx"
    assert restored.context_size == 2
```

- [ ] **Step 2: Write a failing test for fixture serialization**

```python
def test_expected_output_fixture_round_trips():
    fixture = ExpectedOutputFixture(audio_path="sample.wav", text="xin chao")
    payload = fixture.to_dict()
    assert payload["text"] == "xin chao"
```

- [ ] **Step 3: Run the tests and verify failure**
- [ ] **Step 4: Implement the manifest dataclass and path helpers**
- [ ] **Step 5: Implement fixture helpers for sample manifests and expected outputs**
- [ ] **Step 6: Re-run the tests**

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add python-model-test/acoustic_bundle python-model-test/test/test_acoustic_bundle.py
git commit -m "feat: add acoustic bundle manifest and fixtures"
```

## Task 3: Export And Replay A FP32 Acoustic Bundle

**Files:**
- Create: `python-model-test/acoustic_bundle/exporter.py`
- Create: `python-model-test/acoustic_bundle/runtime.py`
- Create: `python-model-test/export_acoustic_bundle.py`
- Modify: `python-model-test/test/test_acoustic_model_onnx.py`
- Test: `python-model-test/test/test_acoustic_bundle.py`

- [ ] **Step 1: Write a failing test that exporter writes all required files**

```python
def test_export_acoustic_bundle_writes_manifest_and_model_files(tmp_path):
    output_dir = tmp_path / "bundle"
    export_acoustic_bundle(..., output_dir=output_dir)
    assert (output_dir / "bundle_manifest.json").exists()
    assert (output_dir / "encoder.onnx").exists()
    assert (output_dir / "decoder.onnx").exists()
    assert (output_dir / "joiner.onnx").exists()
    assert (output_dir / "tokens.txt").exists()
```

- [ ] **Step 2: Write a failing test that bundle runtime loads without `model-dir`**

```python
def test_bundle_runtime_uses_only_manifest_paths(tmp_path):
    runtime = BundleAcousticRuntime.from_manifest_path(tmp_path / "bundle_manifest.json")
    assert runtime is not None
```

- [ ] **Step 3: Run the tests and verify failure**
- [ ] **Step 4: Implement the exporter**
- [ ] **Step 5: Implement the bundle runtime**
- [ ] **Step 6: Wire `--bundle-manifest` into `test_acoustic_model_onnx.py`**
- [ ] **Step 7: Re-run the tests**

Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add python-model-test/acoustic_bundle python-model-test/export_acoustic_bundle.py python-model-test/test/test_acoustic_model_onnx.py python-model-test/test/test_acoustic_bundle.py
git commit -m "feat: add fp32 acoustic bundle export and runtime"
```

## Task 4: Verify Bundle Parity Against Model-Dir Mode

**Files:**
- Create: `python-model-test/acoustic_bundle/verifier.py`
- Create: `python-model-test/verify_acoustic_bundle.py`
- Modify: `python-model-test/test/test_acoustic_bundle.py`
- Modify: `python-model-test/test/README.md`

- [ ] **Step 1: Write a failing test for transcript parity verification**

```python
def test_verify_exported_bundle_matches_model_dir_mode(tmp_path):
    report = verify_exported_acoustic_bundle(...)
    assert report["passed"] is True
```

- [ ] **Step 2: Run the tests and verify failure**
- [ ] **Step 3: Implement `verify_exported_acoustic_bundle()`**
- [ ] **Step 4: Implement the CLI wrapper**
- [ ] **Step 5: Re-run the tests**

Expected: PASS

- [ ] **Step 6: Smoke-run both modes manually**

Run:

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m test.test_acoustic_model_onnx --model-dir python-model-testssets\zipformer --audio-file python-model-testssets\speech\sample-2.wav
```

```powershell
& D:\Anaconda\envs\speech2text\python.exe python-model-test\export_acoustic_bundle.py --model-dir python-model-testssets\zipformer --output-dir python-model-testuildcoustic_bundle\zipformerp32 --asset-namespace models/asr/zipformer/fp32
```

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m test.test_acoustic_model_onnx --bundle-manifest python-model-testuildcoustic_bundle\zipformerp32undle_manifest.json --audio-file python-model-testssets\speech\sample-2.wav
```

Expected: the bundle mode transcript matches model-dir mode on the sample set.

- [ ] **Step 7: Commit**

```bash
git add python-model-test/acoustic_bundle python-model-test/verify_acoustic_bundle.py python-model-test/test/test_acoustic_bundle.py python-model-test/test/README.md
git commit -m "feat: verify acoustic bundle parity against model-dir runtime"
```

## Task 5: Make The Quantize Package Project-Aware For Zipformer

**Files:**
- Create: `python-model-test/quantize/projects/__init__.py`
- Create: `python-model-test/quantize/projects/zipformer.py`
- Modify: `python-model-test/quantize/config.py`
- Modify: `python-model-test/quantize/types.py`
- Modify: `python-model-test/quantize/cli.py`
- Modify: `python-model-test/quantize/README.md`

- [ ] **Step 1: Write a failing test for Zipformer quantization project discovery**
- [ ] **Step 2: Run the test and verify failure**
- [ ] **Step 3: Add project registry and project-specific defaults**
- [ ] **Step 4: Add `--project zipformer` and component-aware CLI options**
- [ ] **Step 5: Re-run the tests**

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add python-model-test/quantize python-model-test/test/test_acoustic_bundle.py
git commit -m "feat: generalize quantize package for zipformer"
```

## Task 6: Add Fixed-Shape Preprocessing Before QNN

**Files:**
- Create: `python-model-test/quantize/fixed_shapes.py`
- Create: `python-model-test/quantize/evaluate_zipformer.py`
- Modify: `python-model-test/quantize/qnn.py`
- Modify: `python-model-test/quantize/presets.py`
- Test: `python-model-test/test/test_acoustic_bundle.py`

- [ ] **Step 1: Write a failing test for fixed encoder shape conversion**

```python
def test_make_zipformer_encoder_shape_fixed():
    fixed_model = make_zipformer_encoder_shape_fixed(..., frames=256)
    assert read_input_shape(fixed_model, "x") == [1, 256, 80]
```

- [ ] **Step 2: Write a failing test for fixed decoder and joiner shapes**
- [ ] **Step 3: Run the tests and verify failure**
- [ ] **Step 4: Implement fixed-shape helpers**
- [ ] **Step 5: Re-run the tests**

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add python-model-test/quantize python-model-test/test/test_acoustic_bundle.py
git commit -m "feat: add fixed-shape preprocessing for zipformer qnn"
```

## Task 7: Add Zipformer QNN Presets And Artifact Evaluation

**Files:**
- Modify: `python-model-test/quantize/presets.py`
- Modify: `python-model-test/quantize/qnn.py`
- Modify: `python-model-test/quantize/cli.py`
- Create: `python-model-test/quantize/evaluate_zipformer.py`
- Modify: `python-model-test/quantize/README.md`

- [ ] **Step 1: Write a failing test for the default Zipformer QNN preset**

```python
def test_zipformer_default_qnn_preset_uses_u16_u8():
    plan = build_quantization_plan(...)
    assert plan.activation_type == "quint16"
    assert plan.weight_type == "quint8"
```

- [ ] **Step 2: Write a failing test for an experimental `u8/u8` preset**
- [ ] **Step 3: Run the tests and verify failure**
- [ ] **Step 4: Implement presets**
  - `sd8g2_zipformer_u16u8_matmul_first`
  - `sd8g2_zipformer_u16u8_matmul_conv`
  - `sd8g2_zipformer_u8u8_experimental`
- [ ] **Step 5: Add offline evaluation command that reports transcript quality and model support**
- [ ] **Step 6: Re-run the tests**

Expected: PASS

- [ ] **Step 7: Run a dry-run and a real smoke quantization**

Run:

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m quantize --project zipformer --component encoder --preset sd8g2_zipformer_u16u8_matmul_first --model-dir python-model-testssets\zipformer --fixed-encoder-frames 256 --dry-run
```

Expected: plan summary prints `qnn_static`, `quint16`, `quint8`, and fixed-shape settings.

- [ ] **Step 8: Commit**

```bash
git add python-model-test/quantize
git commit -m "feat: add zipformer qnn presets and evaluation"
```

## Task 8: Write Docs For The Python-First Acoustic Phase

**Files:**
- Create: `python-model-test/acoustic_bundle/README.md`
- Modify: `python-model-test/quantize/README.md`
- Modify: `python-model-test/test/README.md`
- Modify: `python-model-test/plans/README.md`

- [ ] **Step 1: Document the acoustic bundle contract and CLI usage**
- [ ] **Step 2: Document the two smoke-runner modes**
- [ ] **Step 3: Document the Zipformer quantization flow and its gates**
- [ ] **Step 4: Re-run the smoke commands from the README files**

Expected: every documented command has been validated at least once.

- [ ] **Step 5: Commit**

```bash
git add python-model-test/acoustic_bundle/README.md python-model-test/quantize/README.md python-model-test/test/README.md python-model-test/plans/README.md
git commit -m "docs: document python-first zipformer bundle and qnn flow"
```

## Deliverables For The Android Phase

The next repo must not start until this repo can provide all of the following:

- `python-model-test/test/test_acoustic_model_onnx.py`
- `python-model-test/build/acoustic_bundle/zipformer/fp32/bundle_manifest.json`
- `python-model-test/verify_acoustic_bundle.py`
- a tested fixed-shape Zipformer artifact set for QNN evaluation
- README docs that explain the bundle contract and smoke commands
