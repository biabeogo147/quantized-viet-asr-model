# Phase 5.5 Export Retirement And Model Bundle Prune Results

## Outcome

Phase 5.5 is complete.

The repo no longer keeps `src/export/` as a maintained package, and `src/model_bundle/` has been reduced to the manifest, fixture, shape, and runtime helpers that are still shared by AI Hub, verification, quantize-owned bundle helpers, and Android handoff.

## What changed

### Retired modules

- deleted `src/export/`
- deleted `src/model_bundle/contracts.py`
- deleted `src/model_bundle/exporter.py`
- deleted `src/model_bundle/verifier.py`
- deleted `src/model_bundle/qnn_preflight.py`
- deleted `src/model_bundle/projects/`

### New or re-homed owners

- manual bundle export now lives at `src/tools/bundle_export.py`
- punctuation source ONNX refresh now lives at `src/tools/punctuation_onnx.py`
- verify-owned dispatch now lives at:
  - `src/verify/bundle_projects.py`
  - `src/verify/bundle_runtime.py`
  - `src/verify/qnn_preflight_core.py`
- quantize-owned bundle production now lives at:
  - `src/quantize/zipformer_bundle.py`
  - `src/quantize/vpcd_bundle.py`
- retained `model_bundle` keep-set now lives at:
  - `src/model_bundle/manifest.py`
  - `src/model_bundle/fixtures.py`
  - `src/model_bundle/zipformer_runtime.py`
  - `src/model_bundle/vpcd_runtime.py`
  - `src/model_bundle/vpcd_shapes.py`
- bundle path routing moved to `src/tools/bundle_paths.py`

### Updated maintained command surface

- manual bundle export:
  - `python -m tools.bundle_export ...`
- bundle verification:
  - `python -m verify.model_bundle ...`
- QNN preflight:
  - `python -m verify.qnn_preflight ...`
- punctuation source ONNX refresh:
  - `python -m tools.punctuation_onnx ...`

## Documentation refresh

The maintained docs were updated to reflect the new ownership boundaries and command surface:

- `README.md`
- `docs/architecture/overview.md`
- `docs/architecture/bundle-contract.md`
- `docs/workflows/export-verify-smoke.md`
- `docs/workflows/android-handoff.md`
- `src/model_bundle/README.md`
- `src/verify/README.md`
- `src/tools/README.md`

## Verification

All verification below was run with:

- `D:\Anaconda\envs\speech2text\python.exe`

### Boundary and command-surface checks

```powershell
& 'D:\Anaconda\envs\speech2text\python.exe' -m pytest test/test_phase55_import_boundaries.py test/test_export_verify_modules.py test/test_src_layout_bootstrap.py -v
```

Result:

- `14 passed`

### Verify, bundle, AI Hub, and Android-sync checks

```powershell
& 'D:\Anaconda\envs\speech2text\python.exe' -m pytest test/test_qnn_preflight.py test/test_vpcd_bundle.py test/test_zipformer_bundle.py test/test_zipformer_quantize.py test/test_model_bundle_core.py test/test_sync_android_bundle.py test/test_aihub_session.py test/test_aihub_evaluation.py -v
```

Result:

- `85 passed`

### Deployment package regression checks

```powershell
& 'D:\Anaconda\envs\speech2text\python.exe' -m pytest test/test_aihub_deployment.py -v
```

Result:

- `3 passed`

### Compile sweep

```powershell
& 'D:\Anaconda\envs\speech2text\python.exe' -m compileall src
```

Result:

- `pass`

### CLI smoke

```powershell
& 'D:\Anaconda\envs\speech2text\python.exe' -m tools.bundle_export --help
& 'D:\Anaconda\envs\speech2text\python.exe' -m verify.model_bundle --help
& 'D:\Anaconda\envs\speech2text\python.exe' -m verify.qnn_preflight --help
```

Result:

- all three entrypoints resolved and printed help successfully

## Notes

- `pytest` emitted a cache warning because `.pytest_cache` could not write under the current workspace permissions. This did not affect test results.
- The public migration is intentionally conservative: `model_bundle` still exists, but only as the reduced keep-set required by current AI Hub and Android-handoff flows.
