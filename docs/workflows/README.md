# Workflow Docs

This folder holds operator-facing how-to docs.

If you are choosing what to run next, start here.

## Read order

1. `docs/workflows/export-verify-smoke.md`
   - export and verify the baseline bundles
2. `docs/workflows/quantize-qnn-candidates.md`
   - build the QNN-oriented candidate bundles
3. `docs/workflows/android-handoff.md`
   - sync verified bundles into BKMeeting
4. `docs/workflows/option1-overview.md`
   - understand the current `Option 1` AI Hub flow
5. `docs/workflows/option1-rerun.md`
   - run the retained `Phase 2 + Phase 3` notebook flow
6. `docs/workflows/option1-promotion-handoff.md`
   - package `Phase 4 + Phase 5` evidence and hand it toward BKMeeting

## Scope split

Use the workflow docs for step-by-step execution.

Use the QNN docs when you need status or decisions:

- `docs/qnn/option1-retained-lanes.md`
- `docs/qnn/model-quantization.md`
- `docs/qnn/preflight.md`
- `docs/qnn/validation-log.md`

## Compatibility aliases

Some older Option 1 filenames remain in this folder as short aliases so existing plan docs and notes still resolve cleanly.

They are not the canonical docs anymore.
