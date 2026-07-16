# VLSP Quantization, Compilation, Evaluation, And Naming Implementation Plan

**Status:** Completed

**Goal:** Replace opaque model terminology, rebuild Zipformer and VPCD quantization from an empty Python build directory, evaluate 100 held-out VLSP records locally, compile eligible quantized artifacts, validate at most five hosted inputs per model, and publish evidence-backed Vietnamese results.

**Scope:** Only `python-model-test`; delete only its `build/` directory, preserve BKMeeting and Docker caches, do not sync Android assets, and do not commit or publish changes.

## Options Considered

| Approach | Advantages | Trade-offs | Decision |
|---|---|---|---|
| Descriptive configuration contract plus ORT-QNN-first Zipformer fallback | Makes every artifact self-describing, preserves the requested quantizer priority, and limits cloud work | Requires a breaking CLI migration and new evaluation infrastructure | Selected |
| Restore the removed Zipformer implementation unchanged | Smaller code change | Retains misleading component claims, opaque identifiers, and an FP32 compile input disconnected from local quantization | Rejected |
| Use AIMET for both models immediately | Uses the most direct AI Hub package format | Ignores the requested ORT-QNN priority and loses a useful historical baseline | Rejected |

## Selected Approach

Replace profile and rollout naming with explicit configuration identifiers. Materialize non-overlapping VLSP calibration and evaluation sets, evaluate the ORT-QNN Zipformer encoder first, and use AIMET only if local quality or compile compatibility fails. Keep Zipformer decoder/joiner FP32. Rebuild VPCD with AIMET using source length 384, decoder length 64, signed 8-bit weights, signed 16-bit activations, and encoder-MatMul-only coverage. Run CPU always, CUDA only with verified provider placement, and use hosted Qualcomm inference only for the bounded post-compile checks that cannot run on the local x86/NVIDIA machine.

## Expected Repository Changes

- `src/model_pipeline/`: descriptive configuration contract, Zipformer ORT-QNN/AIMET support, reusable VLSP evaluation, provider evidence, and bounded hosted validation.
- `test/`: red-green contract, graph, fallback, metrics, provider, quota, and report coverage.
- `pyproject.toml`: mutually exclusive documented CPU/GPU ONNX Runtime installation surfaces.
- `docs/evidence/`: one Vietnamese 100-sample quantization and compile report backed by generated evidence.
- `src/model_pipeline/integrations/android/compatibility.py`: removed without replacement.

## Canonical Docs To Update

- `README.md`: replace the old profile CLI and link the new evidence report.
- `AGENTS.md`: add the descriptive naming contract.
- `docs/architecture.md`: configuration identity, evaluation flow, and dependency boundaries.
- `docs/zipformer-recipe.md`: ORT-QNN-first encoder quantization and AIMET fallback truth.
- `docs/vpcd-recipe.md`: explicit 384/64 shape and encoder-MatMul quantization wording.
- `docs/aihub-android-operations.md`: clean rebuild, local/provider validation, compile, and hosted input cap.

## Tasks

- [x] Record the clean-start inventory, delete only `python-model-test/build/`, and verify the resolved target is absent.
- [x] Replace opaque naming and the profile API with descriptive configurations; remove the Android compatibility alias and pass focused contract tests.
- [x] Add deterministic disjoint VLSP calibration/evaluation materialization and pass dataset tests.
- [x] Add provider auditing, metrics, model runtimes, and deterministic evaluation reports; pass focused tests.
- [x] Add Zipformer ORT-QNN encoder quantization, graph validation, quality gate, and AIMET fallback; pass focused tests.
- [x] Generalize AIMET service integration and run the VPCD 384/64 encoder-MatMul flow; pass focused tests.
- [x] Add checksum-keyed hosted inference evidence, five-input enforcement, downloaded-model validation, and pass integration tests.
- [x] Configure the requested environment, materialize VLSP data, and run the real 100-sample local evaluations.
- [x] Compile eligible quantized artifacts, download them, run at most five hosted inputs per model, and record evidence.
- [x] Update canonical docs and publish the Vietnamese evidence report.
- [x] Run plan-level verification and close the plan.

## Verification Gates

- `<speech2text-python> -m pytest -q` — all tests pass independently of order.
- `<speech2text-python> -m compileall -q src` — exits zero.
- CLI dry-runs for every descriptive configuration — emit canonical artifact IDs and no `--profile` support.
- AST docstring audit — every handwritten function documents purpose, arguments, and returns or yields.
- Naming scan — no prohibited opaque identifier remains in current source, tests, or canonical docs.
- Zipformer graph evidence — 278 encoder MatMul operations are quantized while decoder and joiner remain FP32.
- VPCD graph evidence — MatMul inventory remains 96 encoder, 168 decoder, and one language-model head, with encodings only for the encoder set.
- Local evidence — 100 held-out VLSP records run on CPU; CUDA is reported only when provider placement proves execution.
- Hosted evidence — no more than five inputs per model and downloaded packages pass structural/checksum validation.
- `git diff --check` and tracked path/secret scans — no formatting errors, machine-local paths, or secret values.

## Progress Log

- 2026-07-15: Plan created on a clean worktree before any other tracked change; implementation is running directly on the user-authorized current branch without commits.
- 2026-07-15: Resolved the deletion target to `python-model-test/build/`, inventoried 219 readable files totaling 2,247,119,826 bytes, and deleted no BKMeeting, Gradle, or Docker cache paths. Two pytest temporary directories had restrictive Windows ACLs; after confirming the literal target, they were removed through the existing local AIMET Docker image bind mount. Host verification confirms `python-model-test/build/` is absent.
- 2026-07-15: Replaced recipe/CLI/runtime naming with explicit configurations, added ORT-QNN and AIMET Zipformer identities, removed the Android namespace compatibility module, renamed model data-flow arguments by state, and synchronized `AGENTS.md`, README, architecture, and recipe/operations docs. The prohibited-name scan returned no matches in current source, tests, or canonical docs; focused naming tests passed 18/18 and the full suite passed 41/41 using `speech2text`.
- 2026-07-15: Added deterministic VLSP calibration/evaluation selection and materialization. Calibration comes only from the first shard; evaluation comes from later shards, applies the 2–12 second and 4–40 word filters, and enforces shard/row/transcription disjointness. The portable manifest records relative audio paths plus audio/text SHA-256 checksums. Focused dataset tests passed 4/4.
- 2026-07-15: Added normalized transcript and VPCD parity metrics, invalid-output detection, deterministic latency/JSON/JSONL evidence, profiler-backed CPU/CUDA node attribution, a fixed-shape Zipformer RNN-T local runtime with FP32 CPU decoder/joiner, and a fixed-shape VPCD autoregressive runtime with CPU tokenizer/host loop. Focused evaluation/runtime tests passed 7/7 and the full suite passed 50/50.
- 2026-07-15: Added Zipformer ORT-QNN static PTQ with MinMax, unsigned 8-bit weights, unsigned 16-bit activations, `per_channel=False`, and MatMul-only scope; added Q/DQ coverage inventory, 278-MatMul AIMET fallback policy using the same calibration inputs, explicit CER/WER/empty/collapse gates, and package-aware AI Hub compile inputs. Decoder, joiner, and tokens are copied byte-for-byte. Focused tests passed 6/6 and the full suite passed 56/56.
- 2026-07-15: Replaced the VPCD-specific AIMET service with one model-independent service and shared calibration package/config code; removed duplicate VPCD export implementation. The VPCD fake integration exercised source 384, decoder 64, and 96/168/1 policy packaging. Initial Docker resolution attempted an unpinned CUDA Torch stack, so the build was stopped and Dockerfile was corrected to pin Torch `2.13.0+cpu` and Torchvision `0.28.0+cpu`. The rebuilt AIMET `2.31.0` image succeeded, container health returned `ok`, focused tests passed 13/13, and the full suite passed 58/58.
- 2026-07-15: Added a hard pre-submission cap of five hosted inputs per model, deterministic input/output checksums over tensor names, dtypes, shapes, and bytes, checksum-keyed inference records, and downloaded ONNX validation for `EPContext`, I/O dtypes, `qnn-htp`, and artifact quantization scope. VPCD validation explicitly rejects retained int64 target input instead of accepting a missing int64-to-int32 transform. Focused AI Hub integration tests passed 9/9 and the full suite passed 62/62.
- 2026-07-15: Reworked VLSP selection to stream parquet rows and stop after 24 calibration plus 100 held-out evaluation samples instead of retaining the complete corpus in memory. Materialization used `train-00000-of-00035.parquet` only for calibration and `train-00001-of-00035.parquet` only for evaluation; manifest paths and checksums remain portable. The environment originally contained overlapping ONNX Runtime CPU 1.26.0 and GPU 1.22.2 distributions. A single GPU 1.26.0 install registered CUDA and executed a CUDA MatMul, but reproducibly failed Zipformer's fully fixed batch/time graph during Extended optimization. GPU 1.22.0 ran that exact optimizer path and executed CUDA nodes, so CPU/GPU extras and the local environment were pinned to 1.22.0 without changing transform order.
- 2026-07-15: Clean Zipformer ORT-QNN validation initially exposed that `MatMulAddFusion` converted eight target MatMul operations to Gemm. The optimizer is now explicitly disabled while all other Extended optimization remains enabled; rebuilt evidence passed 278/278 MatMul Q/DQ coverage. Local runtime comparison against the pre-refactor reference also found two evaluation bugs: Kaldi features were used instead of centered log-Mel, and decode allowed only one symbol per encoder frame. Regression tests now lock normalized log-Mel input and repeated recurrent neural network transducer emissions until blank.
- 2026-07-15: Completed all 100-sample local evaluations. Zipformer CPU FP32 CER/WER were 7.183%/12.490%; ORT-QNN were 7.208%/12.490%, with 84/100 exact FP32 transcript parity, no empty/collapsed output, and a passing quality gate. CPU mean latency was 377.1 ms FP32 and 470.8 ms ORT-QNN. CUDA/mixed mean latency was 142.8 ms FP32 and 530.5 ms ORT-QNN; profiler proved CUDA execution for both and showed substantial CPU fallback/memcpy overhead for the Q/DQ graph. VPCD AIMET achieved 100/100 full-output parity, 500/500 first-five top-1 agreement, zero character edits, early EOS, or collapse. CPU mean latency was 8.924 s FP32 and 9.236 s AIMET; CUDA mean was 0.788 s FP32 and 0.791 s AIMET, with CUDA node execution proven by profiler. Environment and per-sample JSON/JSONL evidence are under `build/`.
- 2026-07-15: The ORT-QNN Zipformer compile path was rejected twice: job `jp2vrw445` required 64-bit I/O truncation, then job `jp1vnjk8p` rejected `com.microsoft::DequantizeLinear`. Per the planned fallback, the AIMET encoder-MatMul configuration was rebuilt with name-allowlisted quantizers. Its local CPU and CUDA/mixed evaluations matched the FP32 transcript on 100/100 samples with unchanged 7.183% CER and 12.490% WER. Representative provider profiles attributed 2,513 nodes to CPU in the CPU run and 2,259 nodes to CUDA plus 245 to CPU in the mixed run.
- 2026-07-15: Zipformer AIMET compile job `jp1vnn07p` succeeded on Samsung Galaxy S23 Family with QAIRT 2.45. The downloaded package checksum is `ff1572ca3be7758e552dab4dd0315ecfb4fe8cb954e14dddbaadd64bd450453b`; its ONNX model contains one `EPContext` node and fixed float32/int32 encoder I/O. Exactly five hosted inference jobs (`jpy7oynrp`, `jgo4l9m45`, `jgdzdm0l5`, `jgdzdm8l5`, and `jgk92k4o5`) succeeded, and local FP32 decoder/joiner decoding produced transcript parity on 5/5 inputs.
- 2026-07-15: VPCD compile job `jgk92knn5` failed before target-model creation, so no VPCD hosted inference quota was used. QAIRT context validation identified `/model/encoder/Cast_3` as an unsupported floating-point-to-boolean conversion. Graph tracing proved the value is exactly `1.0 - attention_mask`, hence binary. A red-green graph contract now changes only the first of the two consecutive boolean casts to signed 32-bit integer, preserving the final boolean tensor while using QAIRT-supported floating-point-to-int32 and int32-to-boolean conversions. VPCD rebuild and parity reruns remain in progress.
- 2026-07-16: The first VPCD cast bridge rebuilt and passed local CPU/CUDA parity again, but pinned QAIRT 2.45 job `jp49y6wq5` proved the converter folds consecutive casts back into unsupported `FLOAT16 -> BOOL`. The unpinned diagnostic job `j56d891np` is not accepted as final evidence. The selected second transform derives the condition directly as `Cast(attention_mask, INT32) -> Equal(0)`, which is exactly equivalent for the binary mask and leaves the additive mask plus all 96 encoder MatMul quantizers unchanged. Its contract test passes; a clean AIMET rebuild produced SHA-256 `edc771657c346a02573d0be351dfa6bbff0b48d42820be4cbe13e338574e7e5d` and validation remains 96/168/1. Final local reruns are in progress before a new pinned compile submission.
- 2026-07-16: Pinned QAIRT 2.45 job `jgl1y9oe5` confirmed the integer comparison removed the unsupported boolean Cast, then exposed asymmetric signed 16-bit offsets at the first encoder attention MatMul. The final VPCD policy now disables every quantizer first, enables tensors associated only with the 96 named encoder MatMul operations, and forces all selected activation quantizers to symmetric signed 16-bit. The exported package contains 168 activation encodings, all with offset `-32768`, and 72 signed 8-bit initializer-weight encodings; decoder and language-head tensors are absent. CPU and CUDA/mixed reruns again achieved 100/100 full-output parity, 500/500 first-five agreement, zero edit distance, early EOS, or collapse. Mean latency was 9.134 s on CPU and 0.799 s on CUDA/mixed, with CUDA execution proven by profiler.
- 2026-07-16: Final VPCD compile job `jgn71e3rp` succeeded on Samsung Galaxy S23 Family with QAIRT 2.45. The downloaded package checksum is `6a6b8f0995812373c795dc35e17f88bf888744fc695d5586b5c3949d95c7863d`; the primary ONNX checksum is `c2886b67e06461ddb9d8ee311afa7ef7bf4c48dc17fc9b27b5f26102a2384cb4`. It contains one `EPContext`, four int32 inputs, two float32 outputs, target `qnn-htp`, and encoder-MatMul scope. Exactly five hosted prefix jobs (`j5w1y9ozg`, `jp1vo8xlp`, `jp49exll5`, `jp3wo4jz5`, and `jgo4d1zd5`) succeeded with FP32/local/HTP top-1 parity 5/5. Compile and hosted evidence is complete for both models.
- 2026-07-16: Published the Vietnamese evidence report and synchronized README, architecture, both model recipes, AI Hub operations, and retained artifact evidence. Final verification passed: 77 pytest tests, compileall, a 66-file AST docstring audit, six CLI dry-runs, Markdown link validation across 12 files, naming/absolute-path scans, and `git diff --check`. The VPCD compile stage was rerun after the stronger encoding validator and reused checksum-matched job `jgn71e3rp` evidence without another cloud submission.

## Completion

Fill this section only after every task and required gate passes.

**Completion Status:** Completed

**Verification Evidence:** `77 passed`; compileall exited zero; AST docstring audit reported `PASS: 66 files`; all six supported model/configuration CLI dry-runs exited zero; local-link, naming, path, secret, and diff checks passed. Zipformer and VPCD each passed 100-sample local gates, compiled with QAIRT 2.45, and passed exactly five hosted inputs.

**Canonical Docs Updated:** `README.md`, `AGENTS.md`, `docs/architecture.md`, `docs/zipformer-recipe.md`, `docs/vpcd-recipe.md`, `docs/aihub-android-operations.md`, `docs/evidence/retained-artifacts.json`, and `docs/evidence/2026-07-15-vlsp100-quantization-compile.md`.

**Repository Update Notes:** No BKMeeting file was changed. Generated model, dataset, profiling, compile, and hosted evidence remains under ignored `build/`; no commit, reset, push, app-default change, or local `EPContext` execution was performed.
