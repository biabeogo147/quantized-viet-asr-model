# Agent Working Agreement Implementation Plan

**Status:** Completed

**Goal:** Establish one repository working agreement, a traceable active-to-completed plan lifecycle, and complete Google-style docstrings for all handwritten Python functions.

**Scope:** Only `python-model-test`; preserve the existing dirty worktree and do not modify `BKMeeting`.

## Options Considered

| Approach | Advantages | Trade-offs | Decision |
|---|---|---|---|
| Handbook, plan template, and one-time docstring migration | Gives developers and agents one policy, one reusable workflow, and a consistent documented codebase | Touches every handwritten Python function | Selected |
| One large `AGENTS.md` only | Adds fewer files | Plan quality can drift because no reusable template or lifecycle documentation exists | Rejected |
| Split policy between `AGENTS.md` and `CONTRIBUTING.md` | Separates agent and developer audiences | Duplicates policy and creates an unnecessary documentation boundary | Rejected |

## Selected Approach

Use a root `AGENTS.md` as the policy source, `docs/plans/TEMPLATE.md` as the operational contract, and repository READMEs for discoverability. Migrate all handwritten functions in `src/model_pipeline/**` and `test/**` to Google-style English docstrings without adding a permanent lint dependency.

The single-handbook approach was not selected because it cannot keep plan records structurally consistent. A separate `CONTRIBUTING.md` was not selected because the same rules would need to be maintained in two places.

## Canonical Docs To Update

- `README.md`: link the handbook and plan workflow.
- `docs/architecture.md`: record the repository development and documentation contract.

Plan files do not count as canonical documentation updates.

## Tasks

- [x] Create the handbook, plan template, lifecycle READMEs, and onboarding links.
- [x] Add docstrings to core, CLI, pipeline, and runtime functions.
- [x] Add docstrings to Zipformer/VPCD model functions and graph transforms.
- [x] Add docstrings to AI Hub, Android integration, and dataset functions.
- [x] Add docstrings to all handwritten test functions and helpers.
- [x] Run the AST audit, full tests, compileall, four CLI dry-runs, and repository hygiene gates.
- [x] Record closure evidence and move this plan to `docs/plans/completed/`.

## Verification Gates

- A non-tracked AST audit reports no missing function docstrings, argument documentation, or return/yield documentation.
- `conda run -n speech2text python -m pytest -q` passes.
- `conda run -n speech2text python -m compileall -q src` passes.
- All four `zipformer|vpcd x fp32|production` CLI dry-runs pass.
- `git diff --check` passes and repository scans find no stale rollout names, machine-local paths, or tracked secrets.
- README workflow links resolve and no completed plan remains in `docs/plans/active/`.

## Progress Log

- 2026-07-15: Created the active plan before any other tracked change for this task.
- 2026-07-15: Added `AGENTS.md`, the reusable plan lifecycle/template, and verified all README/onboarding link targets exist.
- 2026-07-15: Documented core/CLI/pipeline/runtime functions; the AST batch audit passed across 8 files and focused pytest passed 15 tests.
- 2026-07-15: Documented model protocols, ONNX helpers, Zipformer, and VPCD functions; the AST batch audit passed across 15 files and focused model-contract pytest passed 12 tests.
- 2026-07-15: Documented AI Hub, Android bundle/sync, and dataset functions; the AST batch audit passed across 15 files and focused integration/dataset pytest passed 9 tests.
- 2026-07-15: Documented all handwritten test functions, nested callbacks, and fake adapters; the AST batch audit passed across 13 test files.
- 2026-07-15: Full AST audit passed across 53 source/test files; removed the non-tracked audit helper after use.
- 2026-07-15: Full pytest passed 37 tests, compileall passed, all four CLI dry-runs passed, and diff/stale-name/path/secret scans were clean.
- 2026-07-15: Added closure evidence and moved the completed plan out of `active/`.

## Completion

**Completion Status:** Completed

**Verification Evidence:**

- AST docstring audit — `PASS: 53 files`.
- Focused core pytest — `15 passed`.
- Focused model-contract pytest — `12 passed`.
- Focused integration/dataset pytest — `9 passed`.
- Full pytest — `37 passed in 3.73s`.
- `python -m compileall -q src` — exit code 0.
- Four CLI dry-runs — Zipformer/VPCD control and production combinations all exited 0 with canonical artifact IDs.
- `git diff --check` — exit code 0.
- Stale rollout-name, machine-local path, and secret-material scans — no matches.

**Canonical Docs Updated:**

- `README.md` now links the handbook, plan template, active plans, and completed plans.
- `docs/architecture.md` now records the development, plan-lifecycle, documentation, and function-docstring contract.

**Repository Update Notes:**

- `AGENTS.md` is the maintained working agreement for the entire repository.
- `docs/plans/TEMPLATE.md` is the required starting structure for every tracked change.
- `docs/plans/active/` contains only unfinished work; verified plans move to `docs/plans/completed/`.
- Every handwritten function in `src/model_pipeline/**` and `test/**` now has a Google-style English docstring describing purpose, inputs, and outputs or yields.
- No permanent lint dependency or docstring contract test was added.
