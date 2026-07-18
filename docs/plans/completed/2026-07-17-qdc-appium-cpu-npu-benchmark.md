# Qualcomm Device Cloud Appium CPU-NPU Benchmark Implementation Plan

**Status:** Completed

**Goal:** Produce reproducible Android model-level CPU and Qualcomm HTP NPU benchmarks for Zipformer and VPCD on the same physical device per model.

**Scope:** Add benchmark payload generation, QDQ export, result aggregation, tests, and evidence to `quantized-viet-asr-model`; coordinate with the BKMeeting Android/Appium harness. Cloud execution waits for device access. No new quantization recipe or AI Hub compile job is introduced.

**Counterpart plan:** BKMeeting `docs/plans/completed/2026-07-17-qdc-appium-cpu-npu-benchmark.md`.

## Options Considered

| Approach | Advantages | Trade-offs | Decision |
|---|---|---|---|
| Keep all benchmark code in this repository | Model generation remains centralized | Cannot establish Android runtime or physical-device truth | Rejected |
| Keep all benchmark code in BKMeeting | Device workflow is colocated | Duplicates quantization and model-evidence responsibilities | Rejected |
| Split ownership between the two repositories | Preserves model and Android boundaries while sharing a checksum contract | Requires coordinated manifests and plans | Selected |

## Selected Approach

This repository owns fixed-shape FP32 and AIMET QDQ preparation, fixture materialization, checksum manifests, and final evidence. BKMeeting owns Android execution, the minimal benchmark activity, Appium automation, and device result capture. Two model-specific Appium packages avoid a combined multi-gigabyte upload while keeping all three configurations on one allocated device for each comparison.

## Expected Repository Changes

- `src/model_pipeline/benchmarks/`: benchmark-only QDQ export, Android payload materialization, and result aggregation.
- `src/model_pipeline/cli.py`: `android-benchmark-payload` and `android-benchmark-report` commands.
- `test/`: payload, graph-contract, report, and CLI tests.
- `build/android-benchmark/`: ignored generated model payloads and imported QDC results.

## Canonical Docs To Update

- `README.md` and `docs/README.md`: link the benchmark workflow and evidence.
- `docs/architecture.md`: define the model-to-Android benchmark boundary.
- `docs/aihub-android-operations.md`: document QDQ, EPContext, and device execution roles.
- `docs/evidence/2026-07-17-qdc-appium-cpu-npu-performance.md`: record measured results after QDC access becomes available.

## Tasks

- [x] Add failing tests and implement benchmark-only QDQ export and graph validation.
- [x] Add failing tests and implement deterministic Android payload/fixture materialization.
- [x] Add failing tests and implement QDC result validation, statistics, and report generation.
- [x] Add CLI tests and expose payload/report commands.
- [x] Rebuild both model payloads from an empty build root and pass local quality contracts.
- [x] Synchronize canonical documentation and the BKMeeting checksum contract.
- [x] Run all local verification gates and commit the source checkpoint.
- [x] Run both QDC Automated Jobs; use the complete Zipformer job as smoke and measurement before submitting VPCD.
- [x] Publish final evidence and close the plan only when both comparisons are valid.

## Verification Gates

- `pytest` — expected: all tests pass independently of order.
- `python -m compileall -q src` — expected: exit code zero.
- `python -m model_pipeline android-benchmark-payload --model zipformer --output build/android-benchmark/zipformer --dry-run` — expected: no generated writes or cloud calls.
- `python -m model_pipeline android-benchmark-payload --model vpcd --output build/android-benchmark/vpcd --dry-run` — expected: no generated writes or cloud calls.
- `git diff --check` — expected: no whitespace errors.
- AST docstring, Markdown-link, secret, absolute-path, checksum, and graph audits — expected: no violations.

## Progress Log

- 2026-07-17: Verified `HEAD == origin/main`, removed the explicitly authorized dirty worktree, ignored assets, caches, and full `build/`, then created this plan before any other tracked change.
- 2026-07-17: Added strict AIMET encoding restoration and benchmark-only QDQ export, canonical Zipformer/VPCD graph gates, portable raw-tensor payloads, deterministic statistics, invalid-comparison reasons, and both public CLI commands. Focused tests pass: 12 benchmark/AIMET/CLI cases.
- 2026-07-17: Locked comparison provenance to one device, one artifact ID, one payload-manifest checksum, three distinct repetition indexes, positive finite timings, strict QNN HTP, and the model-specific quality contract. Runtime-only finite-output smoke is intentionally insufficient for a valid speedup.
- 2026-07-17: Updated architecture, recipes, operations, source tour, README indexes, and the active evidence record. Full local source gates pass with `90 passed, 2 skipped`; the skips are asset-contract checks because the authorized clean removed ignored model assets.
- 2026-07-17: Blocked before real payload generation: no team-provided VLSP parquet/materialized 24/100 split or exact retained AIMET encodings remain after the authorized clean. Recreating QDQ from different calibration inputs would break pre/post provenance, so no placeholder payload or speedup is being produced.
- 2026-07-17: Committed the source checkpoint as `1d0aaaa` after a fresh `91 passed, 2 skipped`, `compileall`, both payload dry-runs, and `git diff --check`. The plan stays active because asset-dependent graph/checksum gates and real payload generation cannot run yet.
- 2026-07-17: Retained AI Hub access was revalidated. Compile sources and targets for `jp1vnn07p` and `jgn71e3rp` were downloaded into ignored `build/qdc-benchmark/retained/`; their inner compiled ONNX checksums match `8568fdc...9415d` and `c2886b67...4cb4`. All ten hosted inference input/output datasets remain downloadable, so exact five-fixture recovery can proceed without recalibration or stale BKMeeting model bytes.
- 2026-07-17: Pre-device review found that the Android result currently labels only finite tensors as quality evidence while the Python aggregator requires transcript/top-1 contracts. Device placement is likewise asserted from the requested provider without a validation profile. Both contracts must be corrected and test-driven before any QDC upload or device allocation.
- 2026-07-17: Recovered exact retained AIMET sources, compiled `EPContext` targets, and ten hosted fixtures by AI Hub record ID. Strict QDQ restoration exposed and fixed an allowlist bug that incorrectly forced symmetric weight quantizers; activation symmetry remains enforced while 72 retained VPCD weight encodings remain asymmetric. Graph gates pass at Zipformer `278/278` and VPCD `96/168/1`.
- 2026-07-17: Materialized both checksum-locked payloads. Zipformer FP32 and QDQ each achieved transcript parity `5/5`; VPCD FP32 and QDQ each achieved teacher-forced first-five top-1 parity `25/25`. Full repository gates pass with `91 passed, 2 skipped`, `compileall`, and `git diff --check`.
- 2026-07-18: QDC Zipformer job `704393` passed the complete nine-process Appium schedule on Snapdragon 8 Gen 2 HDK8550. Imported nine result files and three ONNX Runtime placement profiles into ignored build evidence. Aggregation is valid with 300 observations per configuration: FP32 CPU median `545.577 ms`, QDQ CPU median `669.529 ms`, and EPContext QNN HTP median `431.728 ms`; speedups are `1.264x` FP32 CPU over NPU and `1.551x` QDQ CPU over NPU. At this checkpoint VPCD job `704409` was still running, so only Zipformer values were available.
- 2026-07-18: QDC VPCD job `704409` passed the complete nine-process schedule. Aggregation is valid with 300 observations per configuration: FP32 CPU median `2482.567 ms`, QDQ CPU median `2591.874 ms`, and EPContext QNN HTP median `625.446 ms`; speedups are `3.969x` FP32 CPU over NPU and `4.144x` QDQ CPU over NPU.
- 2026-07-18: Verified all six NPU profile files byte-for-byte against result SHA-256, one device fingerprint within each model, Android 14/API 34, strict CPU-fallback disable, `QNNExecutionProvider`, `libQnnHtp.so`, and HTP v73 loading through CDSP. Canonical report now records complete statistics, per-run variation, artifact/job provenance, limitations, and the four failed-job diagnoses.
- 2026-07-18: Final verification passed: `91 passed, 2 skipped`; compileall; both payload dry-runs; aggregate regenerated both comparisons with `valid=true`; AST audit passed 73 files and 375 functions; 14 canonical Markdown files resolved every local target; naming/path/secret and whitespace gates passed. The two ignored raw evidence sets contain 18 result JSON files and six checksum-matched placement profiles.

## Completion

Fill this section only after every task and required gate passes.

**Completion Status:** Completed on 2026-07-18.

**Verification Evidence:** Zipformer QDC job `704393` and VPCD job `704409` passed. Each model has nine result files, 900 measured inferences, three checksum-verified QNN profiles, one consistent device fingerprint, model-specific quality evidence, strict CPU-fallback disable and HTP v73 execution. Fresh local gates passed with `91 passed, 2 skipped`, compileall, two dry-runs, deterministic aggregate, AST/Markdown/naming/path/secret audits and `git diff --check`.

**Canonical Docs Updated:** `docs/evidence/2026-07-17-qdc-appium-cpu-npu-performance.md` now owns complete device results, provenance, interpretation, failure analysis and limits. `docs/README.md` links the completed evidence. BKMeeting's QDC benchmark and Qualcomm NPU operations guides contain the Android execution summary and link back through `<QUANTIZED_MODEL_ROOT>`.

**Repository Update Notes:** Source and tests were committed at earlier checkpoints; this closure commits only canonical documentation and plan lifecycle. Generated payloads, APKs, Appium ZIPs, raw QDC logs, profiles and aggregate JSON remain ignored under `build/`. No model bytes, recipe, compile target, production bundle or deployment default changed.
