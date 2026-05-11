# QNN Preflight

This document defines what "QNN-ready" means on the Python side of this repo.

Use it before handing a candidate bundle to BKMeeting for Android QNN validation.

## What Python-side QNN preflight is for

Python preflight answers a narrow question:

- is this bundle structurally ready for an Android QNN attempt?

It does not attempt to prove:

- physical Snapdragon HTP execution
- ONNX Runtime QNN provider success on Android
- benchmark wins on a device

## Current supported Python-side preflight target

The current explicit preflight path is:

- VPCD fixed-shape candidate:
  - `build/model_bundle/vpcd/qnn_fixed_1024x128`

## The canonical preflight CLI

```bash
python -m verify.qnn_preflight \
  --project vpcd \
  --bundle-dir build/model_bundle/vpcd/qnn_fixed_1024x128 \
  --output build/model_bundle/vpcd/qnn_fixed_1024x128/qnn_preflight_report.json
```

## What the VPCD preflight check validates

For VPCD, preflight checks that:

- manifest project and `artifacts.model` are correct
- `metadata.quantization` records the expected QDQ format
- activations are `quint16`
- weights are `quint8`
- `metadata.quantization.fixed_shapes = true`
- `metadata.qnn_readiness.fixed_shapes_ready = true`
- tokenizer policy stays `cpu_only_first_slice`
- the ONNX graph has fixed input shapes matching the manifest metadata
- the ONNX graph contains QDQ nodes and expected quantized initializers

## What "QNN-ready" means in this repo

For the current Python flow, a candidate is QNN-ready only when all of these are true:

- the candidate bundle exports cleanly
- candidate verification against the reference bundle passes
- Python manifest-mode smoke passes
- the preflight CLI passes for the supported candidate type
- the intended Android handoff metadata is present

## What preflight still does not prove

Even after preflight passes, the following are still unknown until BKMeeting validates on Android:

- whether ORT QNN runtime packaging is correct
- whether HTP device creation works on the real target
- whether graph partitioning stays on QNN in strict mode
- whether performance is good enough to promote

## How preflight fits the full pipeline

The intended sequence is:

1. export or refresh the reference bundle
2. prepare calibration data
3. quantize or build the candidate bundle
4. verify the candidate against a reference bundle
5. run Python smoke tests in manifest mode
6. run QNN preflight
7. sync the candidate into BKMeeting
8. let BKMeeting attempt Android runtime validation

## Current honest status

- VPCD has an explicit fixed-shape QNN preflight path
- Zipformer has a quantized candidate-bundle flow and fixed-shape metadata, but physical Snapdragon proof still belongs to BKMeeting

## Related docs

- `docs/qnn/validation-log.md`
- `docs/workflows/quantize-qnn-candidates.md`
- `docs/workflows/android-handoff.md`
- `src/verify/README.md`
