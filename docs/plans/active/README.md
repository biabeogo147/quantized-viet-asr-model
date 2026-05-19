# Active Plans

This folder is for plans that are still being actively executed.

Current active planning documents:

- `2026-05-19-option1-vpcd-default-lane-and-workflow-refresh-plan.md`
  - promote VPCD AIMET parity to the default pilot lane, remove retired VPCD AI Hub-quantize code/artifacts, and rewrite the workflow docs around the kept NPU lanes
- `2026-05-19-vpcd-aimet-quantize-cli-plan.md`
  - move VPCD AIMET quantization into the reusable `python -m quantize` workflow, backed by a Docker service on a mapped host port, so the notebook starts from a prebuilt local AIMET artifact instead of exporting AIMET itself
- `2026-05-19-bkmeeting-android-option1-export-plan.md`
  - umbrella plan for freezing Option 1 evidence and handing the selected artifacts into BKMeeting Android
- `2026-05-19-option1-phase6-contract-sync-plan.md`
  - concrete implementation plan for the next pass: extend `sync_android_bundle.py` so it can consume Phase 5 contract packages and stage an Android dry run into `BKMeeting`

Historical or superseded plans should be moved to `docs/plans/archive/`.
