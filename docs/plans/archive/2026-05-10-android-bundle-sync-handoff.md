# Android Bundle Sync Handoff Plan

**Goal:** Make `python-model-test` the source of truth for Android bundle handoff into BKMeeting, with normalized bundle output paths and no manual copy or manifest patch steps.

**Status as of 2026-05-10:** Implemented. `tools.sync_android_bundle` now maps verified Python bundles into BKMeeting's `modelassets` pack and rewrites Android-facing manifest fields during sync.

## Canonical Bundle Paths

| Project | Variant | Python bundle path | BKMeeting target |
| --- | --- | --- | --- |
| `zipformer` | `fp32` | `build/model_bundle/zipformer/fp32` | `modelassets/src/main/assets/models/asr/zipformer/fp32` |
| `zipformer` | `qnn_u16u8` | `build/model_bundle/zipformer/qnn_u16u8` | `modelassets/src/main/assets/models/asr/zipformer/qnn_u16u8` |
| `vpcd` | `vpcd_balanced` | `build/model_bundle/vpcd/vpcd_balanced` | `modelassets/src/main/assets/models/punctuation/vpcd/vpcd_balanced` |
| `vpcd` | `qnn_fixed_1024x128` | `build/model_bundle/vpcd/qnn_fixed_1024x128` | `modelassets/src/main/assets/models/punctuation/vpcd/qnn_fixed_1024x128` |

## Completed Work

- [x] Add a Python sync command at `src/tools/sync_android_bundle.py`.
- [x] Patch Zipformer FP32 handoff manifest fields to `model_name = zipformer/fp32`, `model_variant = fp32`, and `asset_namespace = models/asr/zipformer/fp32`.
- [x] Route VPCD fixed-shape QNN sync to `modelassets/src/main/assets/models/punctuation/vpcd/qnn_fixed_1024x128`.
- [x] Add pytest coverage for Zipformer FP32 sync, VPCD fixed-shape sync gating, and VPCD fixed-shape sibling-variant output.
- [x] Update README and module docs to replace manual copy commands with `python -m tools.sync_android_bundle`.

## Verification

```powershell
& 'D:\Anaconda\envs\speech2text\python.exe' -m pytest test\test_sync_android_bundle.py test\test_vpcd_bundle.py -q
```

Expected: pass.

Latest local run: full `test` suite passed.

## Next Decision Point

After BKMeeting strict QNN HTP validation and benchmark data exist, decide whether to promote `models/punctuation/vpcd/qnn_fixed_1024x128` into the production `models/punctuation/vpcd/vpcd_balanced` namespace or keep the current variant split:

- keep alias split if CPU fallback, APK size, or HTP compatibility still needs clear separation;
- promote only if strict QNN succeeds, benchmarks justify it, and Android packaging can absorb the fixed-shape candidate intentionally.
