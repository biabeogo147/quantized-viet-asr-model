# Tools Module

`src/tools/` keeps small repo utilities that sit around the main export, quantize, verify, and Android handoff flows.

## Retained Option 1 rule

For `VPCD`, local quantization is no longer built in the notebook helpers.

Run the producer first:

```bash
python -m quantize --project vpcd ...
```

Then the AI Hub notebooks only consume the prebuilt artifact from:

- `build/quantize/vpcd/local_aimet/wint8_aint16_min_max_local_quality_parity/`

## Key helpers

- `extract_vlsp2020_calibration_subset.py`
  - emits a shared calibration subset for `zipformer` and `vpcd`
- `prepare_vpcd_qnn_candidate.py`
  - freezes VPCD bundle input shapes for the Android/NPU candidate bundle
- `sync_android_bundle.py`
  - copies verified bundles or contract packages into BKMeeting asset namespaces
- `paths.py`
  - stable repo-root path resolution used from inside `src/`

## Related workflow docs

- [option1-overview.md](/D:/DS-AI/BKMeeting-Research/python-model-test/docs/workflows/option1-overview.md)
- [option1-rerun.md](/D:/DS-AI/BKMeeting-Research/python-model-test/docs/workflows/option1-rerun.md)
- [android-handoff.md](/D:/DS-AI/BKMeeting-Research/python-model-test/docs/workflows/android-handoff.md)
