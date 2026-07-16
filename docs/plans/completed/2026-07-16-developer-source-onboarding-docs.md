# Developer Source Onboarding Documentation Implementation Plan

**Status:** Completed

**Goal:** Create a complete, verified learning path that lets a new developer understand, run, trace, and safely change the Zipformer/VPCD model pipeline.

**Scope:** Documentation-only changes in `quantized-viet-asr-model`; include the BKMeeting handoff boundary but do not change BKMeeting, public CLI behavior, model code, or cloud artifacts.

## Options Considered

| Approach | Advantages | Trade-offs | Decision |
|---|---|---|---|
| Modular onboarding index, practical guide, architecture, and source tour | Separates learning, system design, and code navigation while keeping each document maintainable | Requires careful cross-linking and consistency checks | Selected |
| Expand only `docs/architecture.md` | Keeps one obvious document | Produces a large mixed-purpose document that is difficult to scan and maintain | Rejected |
| Generate API reference from source | Automatically inventories symbols | Does not explain intent, data flow, operating prerequisites, or safe change boundaries | Rejected |

## Selected Approach

Use `docs/README.md` as the documentation entrypoint, `docs/getting-started.md` as the reproducible local lab, `docs/architecture.md` as the contract-level system view, and `docs/source-code-guide.md` as the guided call-chain tour. Keep model details in the existing recipe documents and link to BKMeeting only at the Android integration boundary. This gives all developers a shared foundation before splitting into model, core/pipeline, or integration learning paths.

## Expected Repository Changes

- `docs/README.md`: canonical documentation index, role-based learning paths, task-to-module map, and completion criteria.
- `docs/getting-started.md`: asset/environment prerequisites and full Zipformer/VPCD local validation walkthrough.
- `docs/architecture.md`: system context, dependency/stage flows, artifact lifecycle, model data flows, and truth boundaries.
- `docs/source-code-guide.md`: entrypoint-to-integration source tour, ownership map, tests, extension points, and debugging order.
- Root and model/operations documentation: synchronized onboarding links, clean-clone caveats, and Android handoff truth.

## Canonical Docs To Update

- `README.md`: make the new documentation index the start-here path and state external asset prerequisites accurately.
- `AGENTS.md`: replace the short onboarding order with the maintained learning path.
- `docs/architecture.md`: own system boundaries and data-flow contracts.
- `docs/zipformer-recipe.md` and `docs/vpcd-recipe.md`: connect model-specific code and lab routes.
- `docs/aihub-android-operations.md`: separate local onboarding from cloud operations and document the current sync limitation.

New canonical documents are necessary because no existing document owns onboarding execution, documentation navigation, or guided source reading. `docs/README.md` will index all three additions.

## Tasks

- [x] Create the documentation index and role-based learning paths; verify every initial link target exists.
- [x] Write and verify the environment, asset, VLSP, Docker AIMET, Zipformer, and VPCD local walkthrough.
- [x] Expand the architecture document with system, dependency, stage, artifact, and model-flow contracts.
- [x] Write the source-code guide with the canonical call chain, ownership map, tests, and debugging sequence.
- [x] Synchronize root onboarding, working agreement, recipes, and AI Hub-to-Android operations documentation.
- [x] Run the full local walkthrough and every repository/documentation verification gate.
- [x] Record closure evidence and move this plan unchanged to `docs/plans/completed/`.

Check a task only after its output and verification pass. Update this file immediately after each task; never batch progress updates at the end.

## Verification Gates

- Full Zipformer and VPCD AIMET walkthrough through `validate` using a fresh onboarding build root — expected: both validations report `passed` without AI Hub credentials.
- `python -m pytest -q --basetemp build/pytest-onboarding-docs` — expected: all tests pass.
- `python -m compileall -q src` — expected: exit zero.
- Every documented `--dry-run` command — expected: exit zero and the documented model/configuration/stage contract.
- Markdown link/anchor/file-reference audit — expected: no stale local links or missing code paths.
- `git diff --check` — expected: no whitespace errors.
- Secret, absolute-path, and binary tracking scan — expected: no secret or machine-local path in canonical docs; no model/VLSP/build binary added to Git.

## Progress Log

- 2026-07-16: Plan created as the first tracked change after read-only review of the current docs, package tree, entrypoints, model adapters, dependency boundaries, assets, and BKMeeting integration contract.
- 2026-07-16: Created `docs/README.md` with the common and role-based learning paths, task-to-owner map, onboarding completion criteria, and Python-to-BKMeeting boundary. A targeted local-link audit resolved every Markdown target successfully.
- 2026-07-16: Replaced the architecture summary with system/dependency/stage/artifact/model-flow diagrams and explicit local/cloud/device truth. Added the guided source tour from `__main__` through adapters, stage runner, AI Hub, and Android boundaries. Targeted link and required-contract scans passed for both documents.
- 2026-07-16: Synchronized root onboarding, `AGENTS.md`, both model recipes, and the cloud/Android operations guide. The docs now distinguish clean-clone prerequisites, AIMET versus post-compile packages, bounded hosted evidence, and the current destructive-risk boundary of direct live-namespace Android sync. Targeted link and content checks passed.
- 2026-07-16: Verified the full walkthrough with no AI Hub token or QAIRT environment value. Fresh Zipformer AIMET validation passed with 278/0/0 MatMul and its policy contract; fresh VPCD AIMET validation passed after 776 seconds with 96/168/1 MatMul, encoder policy/encoding contracts, and CPU tokenizer/loop targets. A second run resumed `source`, `prepare`, `quantize`, and `validate` for both models and still returned `passed`.
- 2026-07-16: Final verification passed: 77 pytest tests, compileall, six configuration dry-runs, 56 Markdown links/anchors across 11 files, 95 code-file references, changed-file whitespace/path/secret checks, binary/build tracking checks, and `git diff --check`. The plan was closed only after all canonical docs and walkthrough evidence were complete.

## Completion

Fill this section only after every task and required gate passes.

**Completion Status:** Completed

**Verification Evidence:**

- Fresh Zipformer AIMET run through `validate` — `validation=passed`, no resumed stage, no AI Hub credentials.
- Fresh VPCD AIMET run through `validate` — `validation=passed` after 776 seconds, no resumed stage, no AI Hub credentials.
- Second local run — both models resumed `source`, `prepare`, `quantize`, and `validate` and remained `passed`.
- `python -m pytest -q --basetemp build/pytest-onboarding-docs-final` — `77 passed in 5.39s`.
- `python -m compileall -q src` — exit zero.
- Six supported model/configuration dry-runs through `validate` — all exited zero with matching JSON contracts.
- Markdown link/anchor audit — 56 links across 11 files passed; code-file reference audit passed 95 references.
- Absolute-path, secret, untracked binary/build, whitespace, and `git diff --check` gates — passed.

**Canonical Docs Updated:**

- `docs/README.md` — canonical index, common/role-based paths, task map, and completion criteria.
- `docs/getting-started.md` — portable prerequisites and verified two-model local walkthrough.
- `docs/architecture.md` — system, dependency, stage, artifact, model, evidence, and Android boundaries.
- `docs/source-code-guide.md` — canonical call chain, ownership, tests, extension points, and debugging order.
- `README.md`, `AGENTS.md`, both model recipes, and `docs/aihub-android-operations.md` — synchronized onboarding and deployment truth.

**Repository Update Notes:**

- Dev onboarding now starts at `docs/README.md` and requires the local walkthrough before role-specific reading.
- Clean clones require team-provided model/VLSP assets outside Git; no downloader or new source-code behavior was added.
- Python documentation stops at the Android handoff boundary and links to BKMeeting canonical docs for runtime details.
- Direct sync into a live BKMeeting namespace remains explicitly unsafe until the Android bundle adapter preserves `bundle_manifest.json`, `io_contract.json`, and fixtures.
