# Python Model Test Docs

This repo's docs are split into four layers:

- architecture docs for stable repo boundaries and contracts
- workflow docs for operator-facing how-to flows
- QNN docs for retained lane decisions, readiness, and validation logs
- dated plans that explain how the repo changed over time

If you are new to the repo, read the docs in this order:

1. `docs/architecture/overview.md`
2. `docs/architecture/bundle-contract.md`
3. `docs/workflows/README.md`
4. `docs/workflows/export-verify-smoke.md`
5. `docs/workflows/quantize-qnn-candidates.md`
6. `docs/workflows/aihub-overview.md`
7. `docs/qnn/preflight.md`
8. `docs/qnn/aihub-retained-lanes.md`
9. `docs/qnn/model-quantization.md`
10. `docs/qnn/validation-log.md`
11. `docs/plans/README.md`

## Canonical doc map

### Architecture

- `docs/architecture/overview.md`
  - current repo purpose, supported model families, and source-of-truth boundaries
- `docs/architecture/bundle-contract.md`
  - shared bundle manifest contract, artifact layout, and QNN-related metadata

### Workflows

- `docs/workflows/README.md`
  - entrypoint and reading order for operator-facing workflows
- `docs/workflows/export-verify-smoke.md`
  - canonical export, verify, and smoke-test flows
- `docs/workflows/quantize-qnn-candidates.md`
  - calibration, quantization, candidate-bundle generation, and acceptance gates
- `docs/workflows/aihub-overview.md`
  - current retained AI Hub reader guide and evidence chain
- `docs/workflows/aihub-rerun.md`
  - current `Phase 2 + Phase 3` rerun flow in `On_device_Ai_option1_pilots.ipynb`
- `docs/workflows/android-handoff.md`
  - generic bundle sync handoff into BKMeeting and the Python-vs-Android responsibility split

### QNN

- `docs/qnn/preflight.md`
  - what "QNN-ready" means in this repo and what Python preflight does and does not prove
- `docs/qnn/aihub-retained-lanes.md`
  - retained-lane decisions, lane history, and the current notebook defaults
- `docs/qnn/model-quantization.md`
  - quick current-state summary of quantized assets, CPU-side pieces, and remaining proof gaps
- `docs/qnn/validation-log.md`
  - dated QNN checkpoints, preflight results, and handoff notes

### Plans

- `docs/plans/README.md`
  - explains active vs archived plans
- `docs/plans/active/`
  - plans still being executed
- `docs/plans/archive/`
  - historical plans retained for design context

## Ground rules

- Canonical docs explain the current state only.
- Canonical workflow docs should answer "what do I run next?"
- QNN docs should answer "what lane did we keep and how strong is the proof?"
- Historical execution detail belongs in `docs/plans/archive/`.
- Stable docs should not use date-based filenames.
- Canonical docs should prefer repo-relative paths and placeholders such as `<BKMEETING_ROOT>`, never machine-specific absolute paths.
- If two docs say the same thing, keep one as canonical and archive or trim the duplicate.
- The canonical retained AI Hub workflow entrypoints are `aihub-overview.md`, `aihub-rerun.md`, and `android-handoff.md`.
- Archived plans may still mention superseded filenames, old interpreter commands, or machine-specific paths because they record historical execution context rather than the current canonical workflow.
