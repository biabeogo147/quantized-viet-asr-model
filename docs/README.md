# Tài liệu Model Pipeline

Đây là điểm bắt đầu canonical cho dev mới của `quantized-viet-asr-model`. Repo sở hữu việc chuẩn bị model, quantization, validation, Qualcomm AI Hub compile, evidence và đóng gói artifact cho BKMeeting; repo không sở hữu Android runtime.

## Tuyến đọc chung

Đọc theo thứ tự sau trước khi thay đổi source:

1. [Root README](../README.md) — mục tiêu repo và CLI công khai.
2. [Working agreement](../AGENTS.md) — lifecycle plan, quy tắc docs, naming và verification.
3. [Getting started](getting-started.md) — dựng môi trường và chạy cả Zipformer/VPCD đến local validation.
4. [Architecture](architecture.md) — boundaries, stage flow, artifact lifecycle và execution target.
5. [Source-code guide](source-code-guide.md) — lần theo call chain từ CLI tới model/integration.
6. [Zipformer recipe](zipformer-recipe.md) hoặc [VPCD recipe](vpcd-recipe.md) — graph và quantization contract riêng từng model.
7. [AI Hub → Android operations](aihub-android-operations.md) — compile, hosted validation và handoff boundary.
8. [Báo cáo VLSP](evidence/2026-07-15-vlsp100-quantization-compile.md) cùng [retained artifact record](evidence/retained-artifacts.json) — trạng thái thực nghiệm hiện tại.
9. [QDC Appium CPU–NPU benchmark](evidence/2026-07-17-qdc-appium-cpu-npu-performance.md) — protocol Android model-level và trạng thái chờ thiết bị.

Khi công việc đi vào Android runtime, tiếp tục tại `../../BKMeeting/README.md`, `../../BKMeeting/AGENTS.md`, `../../BKMeeting/docs/architecture/overview.md` và `../../BKMeeting/docs/qnn/playbook.md`. Các path sibling này mô tả layout workspace mong đợi; chúng không biến BKMeeting thành một phần của package Python.

## Nhánh đọc theo vai trò

### Model và quantization engineer

Sau phần chung, đọc:

1. recipe của model;
2. `models/<model>/recipe.py` và `adapter.py`;
3. `graph.py`, `quantization.py`, rồi `runtime.py`;
4. `models/aimet.py` và `models/aimet_service.py`;
5. graph-contract test tương ứng.

Kết thúc khi có thể giải thích operator nào được quantize, operator nào giữ nguyên, calibration đi vào AIMET như thế nào và validation nào ngăn artifact sai được compile.

### Core và pipeline engineer

Sau phần chung, đọc:

1. `core/specs.py`, `core/manifest.py`, `core/runner.py`;
2. `models/base.py`;
3. `pipeline.py`, rồi `runtime.py` và `cli.py`;
4. core, boundary và full-flow tests.

Kết thúc khi có thể giải thích artifact identity, digest/resume rule, stage ownership và cách thêm một configuration mà không phá dependency boundary.

### AI Hub và Android integration engineer

Sau phần chung, đọc:

1. `integrations/aihub/compile.py`, `evidence.py`, `validation.py`, `inference.py`;
2. `integrations/android/bundle.py` và `sync.py`;
3. [AI Hub → Android operations](aihub-android-operations.md);
4. BKMeeting runtime/config/testing/QNN docs được link ở trên.

Kết thúc khi phân biệt được AIMET package trước compile, `EPContext + model.bin` sau compile, Android bundle metadata và physical-device validation.

## Muốn thay đổi gì thì bắt đầu ở đâu?

| Mục tiêu | Source bắt đầu | Test bắt đầu | Tài liệu phải đồng bộ |
|---|---|---|---|
| Thêm/sửa configuration | `models/<model>/recipe.py` | `test_model_pipeline_recipes.py` | architecture và model recipe |
| Đổi fixed shape hoặc graph rewrite | `models/<model>/graph.py`, `adapter.py` | graph-contract test của model | architecture và model recipe |
| Đổi calibration/quantization | `models/<model>/quantization.py`, `models/aimet*.py` | AIMET service và graph-contract tests | getting started và model recipe |
| Đổi stage/cache/manifest | `core/`, `pipeline.py` | core, boundary và flow tests | architecture và source-code guide |
| Đổi dataset/evaluation | `datasets/`, `evaluation/` | dataset/evaluation tests | architecture và evidence liên quan |
| Đổi AI Hub compile/live-run | `integrations/aihub/` | integration tests | AI Hub → Android operations |
| Đổi package/sync Android | `integrations/android/` | integration/flow tests | AI Hub → Android operations và BKMeeting docs tương ứng |
| Đổi Android CPU–NPU benchmark | `benchmarks/` | `test_android_benchmark.py` | architecture, AI Hub → Android operations và QDC evidence |
| Điều tra output sai | model runtime → adapter → graph/quantization | focused contract test, rồi local parity | model recipe và evidence nếu kết luận thay đổi |

## Tiêu chí hoàn thành onboarding

Một dev đã hoàn thành onboarding khi có thể:

- chạy full pytest và hai CLI dry-run;
- materialize VLSP calibration/evaluation theo manifest portable;
- chạy AIMET local pipeline cho cả hai model đến `validate`;
- lần theo `__main__ → CLI → recipe → runtime → pipeline → adapter → integration`;
- chỉ ra quantization scope, CPU/GPU/HTP boundary và artifact bàn giao cho Android;
- tìm đúng test và canonical doc trước khi thay đổi một module;
- giải thích vì sao filename hoặc thư mục không đủ để chứng minh NPU execution.

## Kế hoạch công việc

- [Active plans](plans/active/README.md) chứa công việc đang thực hiện hoặc bị block.
- [Plan template](plans/TEMPLATE.md) là contract bắt buộc cho tracked change.
- [Completed plans](plans/completed/README.md) là operational history, không thay thế canonical docs ở trên.
