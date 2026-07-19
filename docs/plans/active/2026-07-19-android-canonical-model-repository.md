# Android Canonical Model Repository

## Status

Active

## Goal

Materialize one manifest-v2 model repository for BKMeeting containing the canonical Zipformer and VPCD FP32 and Qualcomm HTP artifacts, support components, and fixtures. Remove the separate benchmark model-export path and make Android production and benchmark packaging consume the same checksummed source.

## Scope

- Add the `android-model-repository` CLI and deterministic atomic materialization.
- Validate canonical artifact identities, graph contracts, component roles, and retained compiled checksums.
- Remove the separate benchmark model export and payload identities from current executable code and canonical documentation.
- Update Android handoff documentation and write retained integration evidence.

BKMeeting owns Android runtime, Gradle packaging, UI, and physical-device validation.

## Options Considered

1. Keep separate production bundle and benchmark payload materializers. Rejected because model bytes and metadata can diverge.
2. Synchronize files by legacy Android namespace. Rejected because paths become a second identity and legacy manifests remain authoritative.
3. Materialize one manifest-v2 repository with an index and checksummed artifact descriptors. Selected because production and benchmark can resolve the same bytes by technical identity.

## Canonical Docs

- `docs/architecture.md`
- `docs/source-code-guide.md`
- `docs/aihub-android-operations.md`
- `docs/zipformer-recipe.md`
- `docs/vpcd-recipe.md`
- `docs/evidence/2026-07-19-android-model-repository-handoff.md`

## Tasks

- [x] Add failing contracts for model-index parsing, deterministic atomic materialization, canonical checksums, path safety, and dry-run behavior.
- [x] Implement the canonical Android model repository and CLI.
- [x] Remove the separate benchmark model-export path, CLI surface, tests, and current canonical references.
- [x] Materialize and verify the repository in BKMeeting modelassets.
- [ ] Update canonical documentation and evidence.
- [ ] Run the full Python, graph, documentation, naming, path, and Git gates.
- [ ] Record closure evidence and move this plan to `completed/`.

## Verification Gates

```bash
pytest
python -m compileall -q src
python -m model_pipeline android-model-repository \
  --build-root build/android-integration \
  --destination <BKMEETING_ROOT>/modelassets/src/main/assets/model-repository \
  --dry-run
git diff --check
```

Additional gates cover Zipformer 278 encoder MatMul, VPCD `96/168/1`, retained compiled checksums, manifest determinism, Markdown links, docstrings, secrets, paths, and current naming.

## Progress Log

- 2026-07-19: Created the active plan before other tracked changes.
- 2026-07-19: Added repository identity, manifest-v2, atomic promotion, checksum, unsafe-path, retained-artifact, and CLI dry-run tests; 14 focused repository/CLI tests passed.
- 2026-07-19: Implemented `android-model-repository` with exactly four canonical artifacts, shared support components, runtime metadata, and portable five-fixture manifests.
- 2026-07-19: Removed the separate benchmark model exporter and AIMET-service endpoint; benchmark aggregation now accepts only FP32 CPU and post-compile NPU representations.
- 2026-07-19: Materialized the repository into BKMeeting and validated exact compiled checksums. BKMeeting CPU/QNN unit, packaging, instrumentation compilation, and Appium package tests passed in focused runs.
- 2026-07-19: Updated architecture, source guide, model recipes, root README, and Android operations to describe the shared canonical repository.

## Completion Evidence

Pending.

## Repository Update Notes

No push is authorized. Historical evidence and completed plans are not rewritten.
