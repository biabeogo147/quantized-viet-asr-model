# Active Plans

This folder is for plans that are still being actively executed.

Current active VPCD planning documents:

- `2026-05-13-vpcd-option1-debug-results.md`
  - current evidence log for the VPCD failure, attribution, and rerun outcomes
- `2026-05-14-vpcd-quantize-vs-compile-isolation-plan.md`
  - keeps the AI Hub quantize attribution flow active, including the `B/C/D` fallback matrix
- `2026-05-18-vpcd-local-qdq-aihub-compile-plan.md`
  - current investigation plan for replacing AI Hub quantize with a local-QDQ-to-AI-Hub-compile lane
- `2026-05-18-vpcd-aimet-local-quantize-aihub-compile-plan.md`
  - implementation plan for the official local quantize path based on AIMET `.aimet` packages
  - current result: compile-compatible on AI Hub, but the default `w8a8 + min_max` variant still fails local teacher-forced step `2`

Historical or superseded plans should be moved to `docs/plans/archive/`.
