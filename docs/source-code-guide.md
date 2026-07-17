# Source-code guide

Guide này giải thích thứ tự đọc và cách lần theo một request, không thay thế docstring hay liệt kê toàn bộ symbol. Bắt đầu sau khi đã chạy [getting started](getting-started.md) và hiểu [architecture](architecture.md).

## Call chain chuẩn

```text
src/model_pipeline/__main__.py
→ cli.main
→ models.get_recipe
→ runtime.run_pipeline
→ ModelPipeline.run
→ ModelAdapter
→ StageRunner
→ AI Hub/Android integrations
```

### 1. `__main__.py` và `cli.py`

`python -m model_pipeline` đi vào `cli.main`. Parser sở hữu duy nhất public command `run`, model/configuration allowlist, `--through`, paths, device và `--dry-run`.

Đọc `cli.py` để trả lời:

- user có thể chọn contract nào;
- dry-run có side effect hay không;
- stage cuối được biểu diễn thế nào;
- lúc nào import runtime nặng mới xảy ra.

Test đầu tiên: `test/test_model_pipeline_cli.py`.

### 2. `models.get_recipe` và model `recipe.py`

`models.get_recipe(model, configuration)` dispatch tới `zipformer_recipe` hoặc `vpcd_recipe`. Recipe tạo `ArtifactSpec`, component roles và parameters mà adapter/pipeline cùng đọc.

Đọc recipe trước adapter. Nếu chưa biết fixed shape, quantize action, compile scope hoặc graph contract mong đợi, phần còn lại của code dễ bị hiểu sai.

Tests: `test_model_pipeline_recipes.py` và `test_model_pipeline_core.py`.

### 3. `runtime.run_pipeline`

Runtime là composition root:

- resolve repo/build path;
- tạo AIMET client nếu recipe chạy tới quantize;
- chọn model adapter;
- chỉ authenticate AI Hub khi requested stage chạm compile;
- tạo evidence store và `ModelPipeline`;
- in kết quả JSON.

Đây là nơi kiểm tra dependency/environment wiring, không phải nơi thêm model semantics.

Tests: CLI, flow và integration tests.

### 4. `ModelPipeline.run`

`pipeline.py` là orchestration duy nhất của bảy stage. Đọc một lần từ đầu tới cuối, chú ý ba mapping có trạng thái khác nhau:

```text
sources
→ prepared_components
→ quantized_components / validated_components
→ compiled_components
→ bundle
```

Mỗi stage action gọi adapter/integration, sau đó giao output cho `StageRunner`. Pipeline không được đếm MatMul, build tokenizer hay biết AIMET encoding semantics.

Tests: `test_model_pipeline_flow.py` cho end-to-end fake client và `test_model_pipeline_boundaries.py` cho import direction.

### 5. `ModelAdapter` protocol

`models/base.py` định nghĩa extension contract:

| Method | Trách nhiệm |
|---|---|
| `source_files` | resolve đầy đủ source/calibration inventory |
| `prepare` | fixed shape và model-specific graph preparation |
| `quantize` | explicit skip hoặc tạo quantized package |
| `validate` | trả `ValidationResult`; không compile artifact sai |
| `compile_inputs` | chọn component/package và target I/O contract |
| `bundle_components` | khai component format, precision và execution target |

Adapter không gọi AI Hub hay sync Android. Integration không import model adapter.

## Đọc model adapter theo thứ tự

### Zipformer

1. `models/zipformer/recipe.py` — identity và contract.
2. `adapter.py` — lifecycle của encoder/decoder/joiner/tokens.
3. `graph.py` — fixed shape, optimizer và boolean mask.
4. `quantization.py` — calibration, ORT-QNN/AIMET policy và quality gate.
5. `runtime.py` — feature extraction và recurrent neural network transducer decode.
6. `test_zipformer_quantization_contract.py` — executable graph truth.

Điểm cần giữ: encoder có 278 MatMul; decoder/joiner Gemm giữ FP32 CPU. ORT-QNN và AIMET không dùng cùng package format.

### VPCD

1. `models/vpcd/recipe.py` — identity, shapes và execution roles.
2. `adapter.py` — model/tokenizer/calibration lifecycle.
3. `graph.py` — MatMul classification và attention-mask rewrite.
4. `quantization.py` — 96-node allowlist và encoding inspection.
5. `calibration.py`/`tokenizer.py` — fixed inputs và CPU support artifacts.
6. `runtime.py` — full autoregressive loop.
7. `test_vpcd_quantization_contract.py` — executable graph/encoding truth.

Điểm cần giữ: graph `96/168/1`; chỉ encoder được quantize; tokenizer và loop luôn CPU.

## Shared AIMET code

`models/aimet.py` sở hữu config và portable `.npz` calibration format. `models/aimet_service.py` có hai phía:

- `AimetServiceClient`: host path → container path và HTTP calls;
- service/export: tạo `QuantizationSimModel`, apply policy, compute encodings và export.

Nếu export lỗi, debug theo boundary:

1. host files có nằm dưới repo mount không;
2. `/healthz` có pass không;
3. request paths có map sang `/workspace` không;
4. config/policy có node đúng không;
5. exported ONNX/encodings có tồn tại và qua model validator không.

Tests: `test_aimet_service_contract.py` trước, rồi graph-contract test của model.

## Core: identity, truth và resume

Đọc theo thứ tự:

1. `core/specs.py`: `Stage`, quantization/compile allowlists, artifact grammar, recipe digest.
2. `core/files.py`: file/directory SHA-256 và stable JSON digest.
3. `core/runner.py`: stage directory, cache match, output confinement.
4. `core/manifest.py`: component/provenance/validation serialization.

Một thay đổi core thường ảnh hưởng mọi model. Viết focused test trong `test_model_pipeline_core.py`, chạy boundary/flow tests, rồi cả suite.

## Dataset và evaluation

`datasets/vlsp.py` stream và split data; `datasets/audio.py` decode/probe; `datasets/records.py` định nghĩa fixture records; `datasets/golden.py` refresh expected output.

`evaluation/metrics.py` không chạy model. `providers.py` tạo/profile ONNX Runtime session. `vlsp100.py` gọi model runtime trên sample và tổng hợp record. `reports.py` ghi deterministic JSON/JSONL.

Khi metric sai, tách ba câu hỏi:

1. sample/reference có đúng không;
2. runtime output có đúng không;
3. metric aggregation/normalization có đúng không.

Không sửa model graph để chữa lỗi report.

## AI Hub integration

Đọc theo data flow:

1. `client.py`: protocol, fake client và Qualcomm SDK adapter.
2. `compile.py`: request, checksum reuse, download/materialization.
3. `evidence.py`: checksum-keyed compile record.
4. `validation.py`: `EPContext`, I/O dtype, target/scope checks.
5. `inference.py`: input checksum, quota tối đa năm và hosted evidence.

Model adapter chỉ khai `CompileInput`; integration không quyết định operator nào được quantize. Khi cloud job lỗi, lưu job ID và error trước khi thay graph. Không resubmit lớn chỉ vì local process timeout nếu cloud job còn chạy.

Tests: `test_model_pipeline_integrations.py`, dùng fake client trước khi gọi cloud.

## Android integration boundary

`integrations/android/bundle.py` copy component và tạo manifest v2. `sync.py` reconcile đúng danh sách file trong manifest đó. Hiện sync không sở hữu toàn bộ BKMeeting live-bundle contract; xem cảnh báo trong [operations guide](aihub-android-operations.md).

Phía BKMeeting chịu trách nhiệm:

- asset pack và resolver;
- `bundle_manifest.json`/`io_contract.json`;
- ONNX Runtime Android QNN provider;
- CPU fallback hoặc strict HTP;
- app end-to-end tests và physical-device evidence.

Không thêm Java/runtime logic vào repo Python chỉ để vượt qua integration test.

## Debug theo thứ tự bằng chứng

```text
request/configuration
→ source inventory + checksums
→ stage-state input/output digests
→ prepared graph
→ quantization policy + encodings
→ validation.json
→ compile record + downloaded contract
→ hosted output
→ Android bundle
→ Android provider diagnostics
```

Quy tắc:

- tái hiện bằng focused test hoặc build root mới;
- tìm boundary đầu tiên khác expected;
- sửa owner của boundary đó;
- không dùng filename để suy execution target;
- không gọi local CPU result là post-compile HTP result;
- nếu Android output sai nhưng hosted đúng, điều tra packaging/I/O/runtime trước quantization.

## Extension checklist

Khi thêm model configuration hoặc thay adapter:

1. tạo active plan;
2. khóa artifact/recipe contract bằng test;
3. giữ dependency boundaries;
4. triển khai model-specific behavior trong adapter/graph/quantization;
5. cập nhật focused tests và fake full flow;
6. cập nhật architecture, recipe và operations doc liên quan;
7. chạy local parity trước compile;
8. nếu bytes/shape/scope đổi, tạo evidence AI Hub và Android mới.

Nếu muốn biết lệnh chạy, quay lại [getting started](getting-started.md). Nếu muốn biết trạng thái model hiện tại, đọc [evidence report](evidence/2026-07-15-vlsp100-quantization-compile.md).
Model-level device benchmark đi theo call chain riêng:

```text
cli.main
→ benchmarks.runtime
→ AIMET service QDQ export
→ benchmarks.graph/payload
→ BKMeeting BenchmarkActivity + Appium
→ benchmarks.report
```

`benchmarks/` không sở hữu Android session và không được dùng để thay production manifest. Bắt đầu debug ở `test_android_benchmark.py`; chỉ sang BKMeeting sau khi payload checksum và graph contract đã xanh.
