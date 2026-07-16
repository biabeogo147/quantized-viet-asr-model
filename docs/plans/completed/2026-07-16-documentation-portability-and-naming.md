# Documentation Portability And Naming Implementation Plan

**Status:** Completed

**Goal:** Make all project documentation concise, Bash-oriented, and consistently branded as `quantized-viet-asr-model`.

**Scope:** Documentation and documentation indexes only; preserve source code, runtime behavior, model artifacts, BKMeeting, and the uncommitted onboarding work already present.

## Options Considered

| Approach | Advantages | Trade-offs | Decision |
|---|---|---|---|
| Rewrite onboarding around concepts and project-specific Bash commands | Portable, readable, and useful to experienced developers | Removes procedural hand-holding | Selected |
| Mechanically translate every platform-specific line to Bash | Small diff | Preserves excessive basic instructions and platform noise | Rejected |
| Keep platform-specific appendices | Covers more environments | Conflicts with a single Bash documentation contract | Rejected |

## Selected Approach

Keep only project-specific prerequisites, contracts, and commands. Use Bash syntax for every command example, refer to the repository only as `quantized-viet-asr-model`, and remove environment activation/interpreter-discovery guidance. Normalize historical plan references where necessary so repository-wide documentation scans enforce one public name.

## Expected Repository Changes

- Getting-started and operations docs: shorter conceptual guidance and Bash commands.
- Root/index/architecture/working-agreement docs: consistent repository name and shell contract.
- Completed plan records: normalize obsolete repository-name references and environment-command examples without changing recorded outcomes.

## Canonical Docs To Update

- `README.md`, `AGENTS.md`, and `docs/README.md`: public repository identity and onboarding contract.
- `docs/getting-started.md`: concise Bash-oriented onboarding.
- `docs/aihub-android-operations.md`: Bash cloud/Android commands.
- Existing architecture, recipes, evidence, and plan records where the old identity or non-Bash command guidance appears.

## Tasks

- [x] Rewrite getting-started around concepts and essential Bash commands.
- [x] Normalize the repository identity across all documentation.
- [x] Convert remaining command blocks/examples to Bash and remove environment activation guidance.
- [x] Run documentation, naming, shell, link, secret/path, and repository gates.
- [x] Record closure evidence and move this plan to `docs/plans/completed/`.

## Verification Gates

- Repository-wide documentation scan — no legacy repository identity, non-Bash command block, platform-specific shell cmdlet, Gradle `.bat`, Conda activation, or venv-activation guidance.
- Bash syntax check for extracted shell blocks — all checked blocks parse successfully.
- Markdown link/anchor and code-file reference audits — no missing target.
- Full pytest, compileall, and `git diff --check` — all pass.
- Secret, machine-local path, and binary/build tracking scans — no violations.

## Progress Log

- 2026-07-16: Created this active plan before making any tracked change for the documentation portability request; existing uncommitted onboarding docs were preserved.
- 2026-07-16: Rewrote getting-started as a concise concept/contract guide, keeping only project-specific Docker and pipeline commands. Removed interpreter discovery, environment activation, basic installation, and procedural filesystem inspection guidance.
- 2026-07-16: Standardized the public repository identity as `quantized-viet-asr-model` across current docs and completed records. Converted all command fences to Bash, normalized Android Gradle examples to `./gradlew`, and verified all eight Bash blocks with Git Bash syntax parsing.
- 2026-07-16: Final verification passed: repository-wide naming/shell scan, eight Bash syntax checks, 56 Markdown links/anchors, 95 code-file references, 77 pytest tests, compileall, changed-file path/secret/binary/whitespace hygiene, and `git diff --check`.

## Completion

Fill this section only after every task and required gate passes.

**Completion Status:** Completed

**Verification Evidence:**

- Documentation portability scan — no legacy repository identity, non-Bash shell guidance, environment activation command, or Gradle `.bat` example.
- Git Bash syntax audit — all eight `bash` blocks parsed successfully.
- Markdown link/anchor audit — 56 links across 17 Markdown files passed.
- Code-file reference audit — 95 references passed.
- `python -m pytest -q --basetemp build/pytest-doc-portability` — `77 passed in 5.52s`.
- `python -m compileall -q src` and `git diff --check` — exit zero.
- Changed-file path, secret, binary/build, and whitespace hygiene — passed for 13 files.

**Canonical Docs Updated:**

- `README.md`, `AGENTS.md`, and `docs/README.md` — public identity and concise onboarding contract.
- `docs/getting-started.md` — concept-first local guide with Bash-only project commands.
- `docs/aihub-android-operations.md` — Bash cloud/Gradle examples and portable workspace identity.
- Architecture and completed plan records — normalized public repository identity and command examples.

**Repository Update Notes:**

- Documentation refers to the repository only as `quantized-viet-asr-model`.
- Command blocks use Bash; Python source examples and non-command text/diagram blocks retain their appropriate language tags.
- Environment activation, interpreter discovery, and basic tool usage are intentionally outside the onboarding boundary.
- No source code, model artifact, BKMeeting file, or runtime behavior changed.
