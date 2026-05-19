# Option 1 Overview

This is the canonical reader guide for the current `Option 1` AI Hub workflow.

Use it when you need to understand:

- which retained lanes are active right now
- which notebook runs which phase
- where the current proof stops before BKMeeting takes over

## Current retained lanes

Read the QNN-side decision docs first:

- `docs/qnn/option1-retained-lanes.md`
- `docs/qnn/model-quantization.md`

Short version:

- `Zipformer`
  - retained lane: `zipformer_encoder_option1`
  - read: encoder-first NPU proof only
- `VPCD`
  - retained lane: `local_aimet_compile_candidate`
  - compile pilot: `vpcd_option1_local_aimet`
  - read: bounded AIMET parity proof

## Evidence chain

The current `Option 1` flow is:

1. `Phase 2`
   - prepare the retained source artifact
   - compile on AI Hub
   - run the compiled target
2. `Phase 3`
   - run the hybrid host-plus-NPU flow
   - keep the bounded VPCD diagnostics
3. `Phase 4`
   - turn the rerun evidence into a gate verdict
4. `Phase 5`
   - package that evidence into contract directories
5. `BKMeeting`
   - sync assets
   - prove device-side runtime behavior

## Which doc to open next

If you need to rerun the active notebook path:

- `docs/workflows/option1-rerun.md`

If you already have fresh rerun evidence and need promotion packaging plus handoff:

- `docs/workflows/option1-promotion-handoff.md`

If you need generic BKMeeting sync commands outside the `Option 1` packaging flow:

- `docs/workflows/android-handoff.md`

## What this workflow does not prove

`python-model-test` can prove:

- compile-ready artifacts
- AI Hub compile success
- bounded hybrid evidence
- packaging into Phase 5 contracts

It does not prove:

- BKMeeting runtime packaging
- physical Snapdragon HTP execution
- final device promotion
