# VLSP Benchmark Reproducibility Implementation Plan

**Status:** Completed

**Goal:** Provide a tested command-line workflow and canonical documentation that reproduce the VLSP 24/100 quantization, local evaluation, Qualcomm AI Hub compile, and bounded hosted-validation protocol.

**Scope:** Change only `quantized-viet-asr-model` source, tests, and canonical documentation. Keep QDQ benchmark-only, preserve AIMET packages as compile inputs, do not change Android artifacts or BKMeeting, and do not submit real cloud jobs during implementation.

## Options Considered

| Approach | Advantages | Trade-offs | Decision |
|---|---|---|---|
| Patch the historical evidence with manual commands | Small documentation-only change | Commands would duplicate internal APIs and drift quickly; quantized-runtime semantics would remain ambiguous | Rejected |
| Document long Python snippets around existing APIs | Avoids a new public command | Difficult to test, resume, and keep synchronized with graph and quota contracts | Rejected |
| Add one benchmark CLI backed by existing dataset, evaluation, AIMET, and AI Hub boundaries | Reproducible, testable, checksum-keyed, and concise enough for onboarding documentation | Requires a small orchestration layer and benchmark-only QDQ export | Selected |

## Selected Approach

Add `benchmark-vlsp` with cumulative `local`, `compile`, and `hosted` stop points. Export explicit QDQ from the exact AIMET encodings only for local ONNX Runtime evaluation; continue compiling the canonical AIMET model-plus-encodings package. This makes the historical protocol executable without mislabeling the plain AIMET compile-source ONNX as a quantized runtime model.

## Expected Repository Changes

- `src/model_pipeline/benchmarks/`: VLSP benchmark request, orchestration, quality gates, evidence layout, and resume contract.
- `src/model_pipeline/models/aimet_service.py`: strict benchmark-only AIMET-to-QDQ export operation.
- `src/model_pipeline/cli.py`: public `benchmark-vlsp` parser, dry-run, and explicit cloud opt-in.
- `test/`: red-green coverage for CLI, QDQ restoration, graph/quality gates, provider truth, resume, and hosted quota.

## Canonical Docs To Update

- `docs/benchmarking.md`: new canonical benchmark guide because no current document owns end-to-end benchmark reproduction.
- `README.md` and `docs/README.md`: add the benchmark guide to the learning path.
- `docs/getting-started.md`: link local benchmark onboarding without duplicating the guide.
- `docs/zipformer-recipe.md` and `docs/vpcd-recipe.md`: synchronize model-specific graph and quality contracts.
- `docs/aihub-android-operations.md`: define the compile and hosted stop points and cloud safety.
- `docs/evidence/2026-07-15-vlsp100-quantization-compile.md`: add reproduction guidance and distinguish historical latency from protocol truth.

## Tasks

- [x] Add failing CLI and benchmark-contract tests, then implement request parsing, dry-run, stage semantics, and cloud opt-in.
- [x] Add failing strict-QDQ tests, then implement benchmark-only AIMET QDQ export and graph-scope validation.
- [x] Add failing workflow tests, then implement deterministic VLSP local evaluation, quality gates, evidence, resume, compile, and five-input hosted orchestration.
- [x] Write and synchronize canonical benchmark documentation; verify every documented command and link.
- [x] Run plan-level verification and close the plan.

## Verification Gates

- `pytest` — expected: the complete suite passes without order-dependent failures.
- `python -m compileall -q src` — expected: exit code zero.
- `python -m model_pipeline benchmark-vlsp --model all --dataset-root "$VLSP_PARQUET_ROOT" --build-root build/vlsp-benchmark --providers cpu,cuda --through local --dry-run` — expected: portable JSON plan and no external calls.
- `python -m model_pipeline benchmark-vlsp --model all --dataset-root "$VLSP_PARQUET_ROOT" --build-root build/vlsp-benchmark --providers cpu,cuda --through compile --submit-cloud --device "Samsung Galaxy S23 (Family)" --qairt-version 2.45 --dry-run` — expected: two compile requests and no external calls.
- `git diff --check` — expected: no whitespace errors.
- AST docstring, Markdown link/command, secret, and absolute-path audits — expected: no violations in current source or canonical docs.

## Progress Log

- 2026-07-21: Plan created before other tracked changes; worktree was clean at `7a7a009`.
- 2026-07-21: Added `benchmark-vlsp` request and dry-run contracts after observing the expected missing-module failure. Focused CLI/contract verification passed `5/5` with a workspace `--basetemp`; compile and hosted requests now require explicit cloud opt-in, device, and QAIRT version.
- 2026-07-21: Restored benchmark-only explicit-QDQ export after the expected missing-symbol failures. Encoding restoration is strict with missing quantizers disabled, conversion uses signed activations and prequantized constants, and graph inspection rejects missing or out-of-policy MatMul coverage. Focused AIMET/QDQ tests passed `13/13`.
- 2026-07-21: Added production and fake VLSP benchmark orchestration after red tests for missing workflow, environment evidence, hosted resume, and model-source invalidation. The production backend reuses the canonical pipeline, emits explicit-QDQ only under benchmark output, evaluates CPU plus profiler-proven CUDA, validates downloaded EPContext packages, and enforces exactly five hosted inputs. Hosted tensor outputs are now persisted for quota-safe per-input resume. Focused dataset/evaluation/AIMET/AI Hub/workflow verification passed `45/45`.
- 2026-07-21: Added the canonical Vietnamese benchmark guide and synchronized README, onboarding, both recipes, AI Hub operations, and the historical VLSP evidence. Local and compile dry-runs emitted the expected side-effect-free plans, every local Markdown link target resolved, `git diff --check` passed, and current docs contain only portable placeholders.
- 2026-07-21: Completed plan-level verification: full pytest passed `102` tests with `2` intentional skips; compileall, local/compile/hosted dry-runs, AST docstring audit, Markdown link audit, naming/path/secret scans, and `git diff --check` all exited zero. No VLSP100 rerun or real AI Hub job was submitted because this implementation closes the reproducibility tooling and documentation contract without external data or quota.

## Completion

Fill this section only after every task and required gate passes.

**Completion Status:** Completed

**Verification Evidence:**

- `pytest -q --basetemp build/pytest-full` — `102 passed, 2 skipped`.
- `python -m compileall -q src` — exit code zero.
- `benchmark-vlsp` dry-runs through `local`, `compile`, and `hosted` — correct cumulative stages, `writes=false`, and `cloud_calls=false`.
- AST docstring audit — all handwritten source/test functions satisfy the repository contract.
- Markdown link, canonical naming, absolute-path, secret, and `git diff --check` audits — passed.

**Canonical Docs Updated:**

- `docs/benchmarking.md` — canonical executable reproduction guide and artifact semantics.
- `README.md`, `docs/README.md`, and `docs/getting-started.md` — benchmark discovery and learning path.
- `docs/zipformer-recipe.md` and `docs/vpcd-recipe.md` — benchmark-only QDQ and model-specific gates.
- `docs/aihub-android-operations.md` — cumulative compile/hosted semantics and explicit cloud opt-in.
- `docs/evidence/2026-07-15-vlsp100-quantization-compile.md` — protocol reproduction, historical-latency boundary, and AIMET/QDQ clarification.

**Repository Update Notes:** `benchmark-vlsp` is the maintained public reproduction surface. Explicit QDQ remains ignored benchmark output, canonical AI Hub compile input remains the AIMET package, successful hosted outputs are checksum-persisted for quota-safe resume, and BKMeeting/Android artifacts were not changed.
