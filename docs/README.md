# Python Model Test Docs

This repo's docs are split into two layers:

- canonical docs for the current Python model workflow
- dated plans that explain how the repo changed over time

If you are new to the repo, read the docs in this order:

1. `docs/architecture/overview.md`
2. `docs/architecture/bundle-contract.md`
3. `docs/workflows/export-verify-smoke.md`
4. `docs/workflows/quantize-qnn-candidates.md`
5. `docs/workflows/android-handoff.md`
6. `docs/qnn/preflight.md`
7. `docs/qnn/validation-log.md`
8. `docs/plans/README.md`

## Canonical doc map

### Architecture

- `docs/architecture/overview.md`
  - current repo purpose, supported model families, and source-of-truth boundaries
- `docs/architecture/bundle-contract.md`
  - shared bundle manifest contract, artifact layout, and QNN-related metadata

### Workflows

- `docs/workflows/export-verify-smoke.md`
  - canonical export, verify, and smoke-test flows
- `docs/workflows/quantize-qnn-candidates.md`
  - calibration, quantization, candidate-bundle generation, and acceptance gates
- `docs/workflows/android-handoff.md`
  - bundle sync handoff into BKMeeting and the Python-vs-Android responsibility split

### QNN

- `docs/qnn/preflight.md`
  - what "QNN-ready" means in this repo and what Python preflight does and does not prove
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
- Historical execution detail belongs in `docs/plans/archive/`.
- Stable docs should not use date-based filenames.
- Canonical docs should prefer repo-relative paths and placeholders such as `<BKMEETING_ROOT>`, never machine-specific absolute paths.
- If two docs say the same thing, keep one as canonical and archive or trim the duplicate.
- Archived plans may still mention superseded filenames, old interpreter commands, or machine-specific paths because they record historical execution context rather than the current canonical workflow.
