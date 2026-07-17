# BKMeeting Documentation Link Migration Plan

**Status:** Active

**Goal:** Update canonical cross-repository links to the rebuilt BKMeeting documentation without changing the model pipeline.

**Scope:** `docs/README.md`, `docs/aihub-android-operations.md`, plan lifecycle records, and link verification in `quantized-viet-asr-model`. Exclude source code, model artifacts, quantization recipes, and deployment defaults.

**Counterpart plan:** `BKMeeting/docs/plans/active/2026-07-17-documentation-and-runtime-naming-refactor.md`.

## Options Considered

| Approach | Advantages | Trade-offs | Decision |
|---|---|---|---|
| Keep links to Git history | Preserves access to every historical file | Gives new developers no maintained operational route | Rejected |
| Copy Android guidance into this repository | Links remain local | Duplicates BKMeeting ownership and will drift | Rejected |
| Link canonical BKMeeting guides at the integration boundary | Preserves clear ownership and one source of truth | Requires coordinated path verification | Selected |

## Selected Approach

Keep model preparation and artifact provenance in this repository, then hand readers to the new BKMeeting architecture, Qualcomm NPU operations, testing, and QDC benchmark guides. Verify each cross-repository target against the sibling checkout before closing the plan.

## Expected Repository Changes

- `docs/README.md`: update the Android continuation of the learning path.
- `docs/aihub-android-operations.md`: update handoff, packaging, runtime validation, and benchmark references.
- No Python source, tests, model bytes, manifests, or generated outputs change.

## Canonical Docs To Update

- `docs/README.md`: canonical documentation index.
- `docs/aihub-android-operations.md`: canonical AI Hub to Android handoff.

## Tasks

- [ ] Update canonical BKMeeting links after the counterpart paths exist.
- [ ] Audit cross-repository links and repository terminology.
- [ ] Run repository verification and close this plan.

Check a task only after its output and verification pass. Update this file immediately after each task.

## Verification Gates

- `pytest` — model pipeline tests pass unchanged.
- `python -m compileall -q src` — package remains syntactically valid.
- Cross-repository Markdown link audit — all BKMeeting targets exist.
- `git diff --check` — no whitespace errors.

## Progress Log

- 2026-07-17: Read-only discovery completed and this plan was created before any other tracked change in this repository.

## Completion

Fill this section only after every task and required gate passes.

**Completion Status:** Active

**Verification Evidence:** Pending.

**Canonical Docs Updated:** Pending.

**Repository Update Notes:** Pending.
