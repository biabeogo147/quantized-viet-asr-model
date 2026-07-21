# Tái tạo benchmark VLSP 100 mẫu

Tài liệu này là nguồn hướng dẫn canonical để tái tạo protocol của [báo cáo quantization và compile ngày 2026-07-15](evidence/2026-07-15-vlsp100-quantization-compile.md). Mục tiêu là tái tạo cùng model, dataset split, recipe, graph contract và quality gate. Latency được đo lại trên host hiện tại và không phải khớp tuyệt đối với số lịch sử.

## Bức tranh tổng thể

```text
VLSP parquet
→ 24 calibration + 100 held-out evaluation
→ fixed-shape FP32 control
→ AIMET model + encodings
→ explicit QDQ dành riêng cho local benchmark
→ CPU và CUDA/mixed evaluation
→ tùy chọn AI Hub compile
→ tùy chọn 5 hosted inputs/model
```

CLI công khai:

```bash
python -m model_pipeline benchmark-vlsp \
  --model <zipformer|vpcd|all> \
  --dataset-root "$VLSP_PARQUET_ROOT" \
  --build-root build/vlsp-benchmark \
  --providers cpu,cuda \
  --through <local|compile|hosted>
```

`--through` có nghĩa là “chạy đến hết bước này”, không phải chọn một bước độc lập:

| Giá trị | Những gì được thực hiện |
|---|---|
| `local` | Dataset, prepare, AIMET quantization, QDQ export và FP32/QDQ local evaluation |
| `compile` | Toàn bộ `local`, sau đó compile AIMET package và validate downloaded `EPContext + model.bin` |
| `hosted` | Toàn bộ `compile`, sau đó chạy đúng 5 hosted inputs cho mỗi model |

`compile` ở đây là Qualcomm AI Hub compile, không phải biên dịch Python. `hosted` kiểm tra output trên Qualcomm HTP nhưng không phải benchmark latency NPU và không thay physical Android validation.

## Prerequisite thuộc project

Benchmark cần:

- Zipformer FP32 encoder, decoder, joiner và token table theo layout trong [getting started](getting-started.md);
- VPCD FP32 model và tokenizer/config gốc;
- ít nhất hai VLSP parquet shards theo thứ tự tên ổn định;
- AIMET service đang healthy và mount repository vào `/workspace`;
- đúng một ONNX Runtime distribution, CPU hoặc GPU.

`--build-root` phải nằm trong repository vì AIMET container chỉ truy cập các path dưới repository mount. Model binary, VLSP và toàn bộ output benchmark ở `build/` không được track.

## Dry-run trước khi chạy

Local dry-run không đọc dataset, không gọi AIMET và không tạo file:

```bash
python -m model_pipeline benchmark-vlsp \
  --model all \
  --dataset-root "$VLSP_PARQUET_ROOT" \
  --build-root build/vlsp-benchmark \
  --providers cpu,cuda \
  --through local \
  --dry-run
```

JSON phải liệt kê hai model, hai artifact ID/model, dataset `24/100`, providers và stages `dataset, local`.

Cloud dry-run phải khai báo intent rõ ràng dù không tạo job:

```bash
python -m model_pipeline benchmark-vlsp \
  --model all \
  --dataset-root "$VLSP_PARQUET_ROOT" \
  --build-root build/vlsp-benchmark \
  --providers cpu,cuda \
  --through compile \
  --submit-cloud \
  --device "Samsung Galaxy S23 (Family)" \
  --qairt-version 2.45 \
  --dry-run
```

Trong dry-run, `writes=false` và `cloud_calls=false`. Không có job ID nào được tạo.

## Chạy local benchmark

```bash
python -m model_pipeline benchmark-vlsp \
  --model all \
  --dataset-root "$VLSP_PARQUET_ROOT" \
  --build-root build/vlsp-benchmark \
  --providers cpu,cuda \
  --through local
```

Dataset contract:

- 24 calibration records từ shard đầu tiên;
- 100 evaluation records từ shard tiếp theo;
- evaluation audio dài 2–12 giây và transcription có 4–40 từ;
- không trùng shard, row hoặc normalized transcription;
- manifest chỉ ghi path tương đối và SHA-256.

CPU luôn chạy. Khi yêu cầu `cuda`, kết quả chỉ được gọi là CUDA nếu ONNX Runtime profiler quan sát `CUDAExecutionProvider`. Nếu CUDA không đăng ký, evidence ghi `unavailable`. Nếu CUDA và CPU cùng thực thi node, evidence ghi `cuda-mixed`.

### FP32, AIMET package và QDQ khác nhau thế nào

```text
fixed-shape FP32 ONNX
├── local FP32 control
└── AIMET QuantizationSimModel
    ├── model.onnx + model.encodings  → AI Hub compile source
    └── explicit QDQ ONNX             → local quantized benchmark only
```

AIMET compile-source `model.onnx` không tự chứa `QuantizeLinear/DequantizeLinear`. Encodings nằm trong sidecar `model.encodings` và được AI Hub/AIMET hiểu cùng package. Vì vậy chạy riêng `model.onnx` bằng stock ONNX Runtime chỉ là chạy graph ONNX, không chứng minh quantized inference.

Local quantized benchmark dùng explicit QDQ được dựng lại từ đúng FP32 graph, AIMET config, operator allowlist và encodings. Export bắt buộc strict matching, tắt quantizer thiếu encoding, dùng signed 8-bit weight, signed 16-bit activation và không được đưa QDQ vào compile package hoặc Android repository.

### Quality gate

Zipformer:

- 278/278 encoder `MatMul` thuộc QDQ scope;
- decoder và joiner tiếp tục FP32 CPU;
- CER tăng không quá 1 điểm phần trăm;
- WER tăng không quá 2 điểm phần trăm;
- không output rỗng hoặc repetition collapse;
- evidence ghi exact transcript parity trên 100 mẫu.

VPCD:

- graph giữ inventory `96 encoder / 168 decoder / 1 language-model head`;
- chỉ 96 encoder `MatMul` có QDQ;
- first-five top-1 đạt `500/500`;
- restored-output parity ít nhất `95/100`;
- không early EOS hoặc punctuation collapse;
- tokenizer và autoregressive loop vẫn chạy CPU.

VLSP không có punctuation/capitalization ground truth phù hợp, nên VPCD được đánh giá bằng parity với FP32, không gọi là punctuation accuracy.

## Compile và hosted validation

Compile thật:

```bash
python -m model_pipeline benchmark-vlsp \
  --model all \
  --dataset-root "$VLSP_PARQUET_ROOT" \
  --build-root build/vlsp-benchmark \
  --providers cpu,cuda \
  --through compile \
  --submit-cloud \
  --device "Samsung Galaxy S23 (Family)" \
  --qairt-version 2.45
```

Cloud stage bị từ chối trước khi đọc dataset nếu thiếu `--submit-cloud`, `--device` hoặc `--qairt-version`. Compile chỉ bắt đầu sau khi local quality gate đạt. Input là canonical AIMET package; QDQ không được upload làm compile source.

Downloaded package phải có:

- một ONNX graph chứa `EPContext`;
- adjacent `model.bin` hoặc external data tương ứng;
- QNN HTP execution target;
- I/O đúng fixed shape và transform int64-to-int32;
- source/package checksum khớp evidence.

Hosted validation thật:

```bash
python -m model_pipeline benchmark-vlsp \
  --model all \
  --dataset-root "$VLSP_PARQUET_ROOT" \
  --build-root build/vlsp-benchmark \
  --providers cpu,cuda \
  --through hosted \
  --submit-cloud \
  --device "Samsung Galaxy S23 (Family)" \
  --qairt-version 2.45
```

Mỗi model dùng đúng 5 inputs:

- Zipformer gửi 5 fixed-shape encoder inputs, sau đó decode hosted output bằng FP32 decoder/joiner local và yêu cầu transcript parity `5/5`;
- VPCD gửi 5 teacher-forced decoder prefixes và yêu cầu top-1 parity `5/5`.

Mỗi successful hosted input lưu cả input checksum, output checksum và output tensor để lần chạy sau resume mà không submit lại. Không tăng quota để bù một request thất bại.

## Đọc evidence và resume

```text
build/vlsp-benchmark/
├── dataset/
├── environment.json
├── model-pipeline/
├── zipformer/
│   ├── local/
│   ├── compile/
│   └── hosted/
├── vpcd/
│   ├── local/
│   ├── compile/
│   └── hosted/
└── comparison.json
```

`local/` chứa per-sample JSONL, provider summaries, QDQ graph inventory, quality gate và năm hosted fixtures. `compile/` chứa compile/download validation. `hosted/` chứa quota-safe job/output records. `comparison.json` là entrypoint machine-readable của lần chạy.

Một step chỉ resume khi input digest và checksum của mọi evidence file còn khớp. Local digest bao gồm dataset, recipe, environment, provider request và model source bytes. Compile digest thêm compile-source checksum, device và QAIRT. Hosted digest thêm compiled target và năm input fixtures.

Muốn clean rerun, dùng build root mới:

```bash
python -m model_pipeline benchmark-vlsp \
  --model all \
  --dataset-root "$VLSP_PARQUET_ROOT" \
  --build-root build/vlsp-benchmark-clean \
  --providers cpu,cuda \
  --through local
```

## So sánh với báo cáo lịch sử

Các giá trị cần khớp để gọi là tái tạo cùng protocol:

- source model checksum và artifact identity;
- VLSP shard/row/text/audio checksums;
- fixed shapes và graph inventory;
- AIMET config, encodings và quantization policy;
- quality gates và hosted parity contract.

Latency CPU/CUDA phụ thuộc CPU/GPU, driver, ONNX Runtime, thermal state và background load. Hãy báo cáo số mới cùng `environment.json`; không thay số lịch sử chỉ vì một lần chạy mới khác latency.

Các số CPU/CUDA local không được dùng để suy diễn NPU speedup. `EPContext` không chạy trên local CPU/NVIDIA GPU. Benchmark Android CPU–NPU và app end-to-end thuộc boundary khác, được mô tả trong [AI Hub → Android operations](aihub-android-operations.md).

## Lỗi thường gặp

- Không đủ hai VLSP shards hoặc không chọn đủ 24/100 records: kiểm tra quyền dữ liệu, schema và filter.
- AIMET không reachable hoặc path nằm ngoài repository mount: kiểm tra service health và dùng build root dưới repository.
- Encoding/policy mismatch: không nới `strict=True`; kiểm tra model revision và quantization output.
- CUDA có trong provider list nhưng không có node CUDA: giữ nhãn `cuda-not-observed`, không công bố GPU latency.
- Local quality gate fail: dừng trước cloud; xem per-sample JSONL và graph evidence.
- Compile download thiếu `EPContext` hoặc `model.bin`: giữ evidence failed, không chạy hosted.
- Hosted chạy dở: chạy lại cùng build root để resume successful inputs theo checksum.
