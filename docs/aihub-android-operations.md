# Qualcomm AI Hub → Android Operations

Tài liệu này mô tả boundary từ validated model package đến repository mà BKMeeting consume.

## Artifact truth

Post-compile deployment không phải AIMET `model.onnx`. Nó là cặp adjacent:

```text
EPContext ONNX
model.bin
```

Zipformer còn cần FP32 decoder, joiner và tokens. VPCD còn cần tokenizer encode/decode, ID maps và CPU autoregressive-loop contract.

Retained compiled ONNX checksums:

- Zipformer: `8568fdc6902679c5eda866c7ea5ce82a203a2d79a628c8d89d838e353539415d`.
- VPCD: `c2886b67e06461ddb9d8ee311afa7ef7bf4c48dc17fc9b27b5f26102a2384cb4`.

Mismatch phải dừng; không ghép support/compiled files khác provenance và không tự compile lại trong Android integration workflow.

## Cloud boundary

AI Hub compile request resolve bằng source checksum, không bằng rollout name. Download validation kiểm tra `EPContext`, external data, I/O dtype, QAIRT target và component scope.

Hosted smoke giới hạn năm inputs/model:

- Zipformer: compiled encoder, sau đó decode bằng cùng FP32 decoder/joiner.
- VPCD: teacher-forced prefixes và top-1 parity.

Hosted pass không thay Android app/device proof.

## Materialize canonical repository

Dry-run:

```bash
python -m model_pipeline android-model-repository \
  --build-root build/android-integration \
  --destination <BKMEETING_ROOT>/modelassets/src/main/assets/model-repository \
  --dry-run
```

Promote:

```bash
python -m model_pipeline android-model-repository \
  --build-root build/android-integration \
  --destination <BKMEETING_ROOT>/modelassets/src/main/assets/model-repository
```

Materializer:

1. resolve exact prepared FP32 and retained compiled targets;
2. validate graph, I/O, checksums và provenance;
3. write index, manifests, components và five-fixture sets vào staging;
4. validate staging;
5. atomically replace destination.

Repository chỉ chứa bốn artifact IDs canonical: FP32 CPU và compiled QNN HTP cho Zipformer/VPCD.

## BKMeeting consumption

BKMeeting không nhận model path từ config hoặc UI. `model-index.json` và manifest v2 là public contract.

- `cpuCompat`: package FP32 artifacts.
- `qnnOfficialArm64`: package compiled primary components + CPU support.
- `benchmark`: APK model-free; Appium ZIP lọc FP32 và NPU của một model từ cùng repository.

Canonical Android docs:

- [BKMeeting documentation index](../../BKMeeting/docs/README.md)
- [Android architecture](../../BKMeeting/docs/architecture.md)
- [Model repository contract](../../BKMeeting/modelassets/README.md)
- [Qualcomm NPU operations](../../BKMeeting/docs/qualcomm-npu-operations.md)
- [QDC benchmark](../../BKMeeting/docs/qdc-appium-benchmark.md)

## Acceptance

Host gates:

```bash
pytest
python -m compileall -q src
python -m model_pipeline android-model-repository \
  --build-root build/android-integration \
  --destination <BKMEETING_ROOT>/modelassets/src/main/assets/model-repository \
  --dry-run
```

Android gates validate CPU/QNN package separation and model-free benchmark APK. Physical Snapdragon gates require strict HTP without fallback, five-fixture parity for each model, ten main-app sample goldens and controlled fail-fast behavior.

Benchmark current workflow compares only:

- FP32 fixed-shape on ONNX Runtime CPU;
- post-compile `EPContext + model.bin` on QNN HTP.

Each representation runs three fresh processes, 10 warm-up and 100 timed inferences/process. Zipformer timing covers encoder only; VPCD timing covers one model invocation only.

CPU và NPU là hai artifact canonical khác nhau. Benchmark provenance yêu cầu một
`artifact_id` ổn định trong ba repetitions của từng configuration, cùng một payload
manifest checksum cho toàn comparison và cùng device fingerprint; không yêu cầu hai
representations dùng chung artifact ID.

Kết quả canonical gần nhất nằm trong [Android repository handoff evidence](evidence/2026-07-19-android-model-repository-handoff.md):

- Zipformer encoder: FP32 CPU / QNN HTP median speedup `1.255×`.
- VPCD một model invocation: FP32 CPU / QNN HTP median speedup `3.780×`.

## Ownership

`quantized-viet-asr-model` owns artifact bytes, graph/shape/scope, provenance, index/manifests and fixtures. BKMeeting owns Android resolver, Gradle filtering, providers, strict device validation, UI and release delivery. Neither side may infer backend from folder name.
