# AI Hub → Android operations

Tài liệu này bắt đầu sau local validation. Dev mới phải hoàn thành [getting started](getting-started.md) và hiểu [artifact taxonomy](architecture.md#phân-biệt-các-artifact) trước khi dùng cloud hoặc chạm Android assets.

## Ranh giới local và cloud

| Công việc | Cần AI Hub token? | Output |
|---|---:|---|
| `prepare`, `quantize`, `validate` | Không | local ONNX/AIMET package và validation evidence |
| `compile` | Có | downloaded `EPContext` ONNX + external context |
| `package` | Có nếu recipe compile | manifest-v2 bundle |
| hosted inference | Có | bounded post-compile output evidence |
| Android strict validation | Không dùng AI Hub token, cần Snapdragon | provider/app/device evidence |

Không dùng cloud compile để thay thế local graph/quality validation. Không gọi package AIMET trước compile là Android NPU model.

## 1. AIMET service và local gate

Lệnh Docker, dependency pins, asset/VLSP prerequisites và walkthrough của cả hai model nằm trong [getting started](getting-started.md#aimet-service). Local gate phải `passed` trước compile.

Image pin AIMET ONNX `2.31.0`, ONNX `1.20.0`, ONNX Runtime `1.22.0`, NumPy `1.26.4`, Torch CPU `2.13.0` và Torchvision CPU `0.28.0`. Service không sở hữu model semantics; adapter cung cấp calibration và operator policy.

## 2. Compile trên Qualcomm AI Hub

Đặt secret trong environment của process, không commit `.env`:

```bash
export QAI_HUB_API_TOKEN="<QAI_HUB_API_TOKEN>"
export QAIRT_VERSION="2.45"
device="Samsung Galaxy S23 (Family)"
```

Chạy đúng artifact vừa validate:

```bash
python -m model_pipeline run \
  --model zipformer \
  --configuration aimet-int8-int16-encoder-matmul \
  --through package \
  --device "$device"

python -m model_pipeline run \
  --model vpcd \
  --configuration aimet-int8-int16-encoder-matmul \
  --through package \
  --device "$device"
```

Compile record được index bằng checksum package đầu vào. Chỉ reuse khi artifact ID, component và input checksum trùng, đồng thời downloaded blob vẫn khớp output checksum. Process timeout sau submit không đồng nghĩa cloud job thất bại; reconnect bằng job ID trước khi resubmit model lớn.

### Downloaded contract

Validator bắt buộc kiểm tra:

- checksum của package;
- đúng một ONNX `EPContext` node;
- input/output dtype và shape;
- execution target `qnn-htp`;
- quantization/compiled scope từ artifact ID;
- external context file nằm cạnh ONNX wrapper.

I/O 64-bit được truncate ở compile boundary:

- Zipformer `x_lens`: source int64 → target int32;
- VPCD: cả bốn input source int64 → target int32.

Nếu target còn int64, package không đạt Android contract.

## 3. Post-compile package format

### Zipformer

Deployment bundle cần:

```text
encoder.onnx       # ONNX wrapper chứa EPContext
model.bin          # external Qualcomm context của encoder
decoder.onnx       # FP32 CPU
joiner.onnx        # FP32 CPU
tokens.txt         # CPU text table
artifact-manifest.json
```

`encoder.onnx` và `model.bin` là một cặp; đổi tên hoặc tách directory có thể phá external-data reference. Decoder/joiner không được gửi lên HTP trong recipe hiện tại.

### VPCD

Deployment bundle cần:

```text
role=model                         # ONNX wrapper chứa EPContext
role=model_external_data           # model.bin, external Qualcomm context
role=tokenizer_encode              # CPU ONNX
role=tokenizer_decode              # CPU ONNX
role=tokenizer_to_model_id_map     # CPU mapping
role=model_to_tokenizer_id_map     # CPU mapping
role=autoregressive_loop           # host-runtime contract
artifact-manifest.json             # ánh xạ role → filename/checksum
```

Tên support file trong package do manifest v2 quyết định từ role và source suffix; consumer phải đọc manifest thay vì hard-code tên suy đoán. Tokenizer và autoregressive loop không nằm trong QNN context. Android phải lặp model session theo decode contract; một hosted teacher-forced step không chứng minh full loop.

## 4. Hosted validation

Hosted runner nhận một đến tối đa năm input độc lập mỗi model. Input identity hash tensor name, dtype, shape và bytes; không dùng job name.

- Zipformer: chạy compiled encoder, sau đó decode output bằng cùng FP32 decoder/joiner local; so transcript.
- VPCD: so full logits/top-1 cho teacher-forced prefixes; đây là bounded smoke, không phải full autoregressive quality run.

Evidence hiện tại đạt 5/5 cho mỗi model, nhưng chưa thay thế Android end-to-end, latency, memory hoặc 100-sample post-compile validation. Xem [báo cáo VLSP](evidence/2026-07-15-vlsp100-quantization-compile.md).

## 5. Android bundle materialization và sync limitation

`integrations/android/bundle.py` hiện tạo:

- `artifact-manifest.json` schema v2;
- component files do adapter khai báo.

`integrations/android/sync.py` copy đúng danh sách này và xóa file khác trong destination. BKMeeting live namespace còn yêu cầu ít nhất:

- `bundle_manifest.json`;
- `io_contract.json`;
- fixture/golden files;
- naming/layout mà asset resolver hiện hành hiểu.

Vì các file đó chưa nằm trong Python manifest-v2 output, **không chạy `--android-destination` trực tiếp vào live BKMeeting namespace**. Sync hiện tại có thể xóa Android metadata/fixtures cần thiết. Materialize vào staging directory để inspect; chỉ promote sau khi Android bundle adapter được mở rộng và kiểm thử bảo toàn toàn bộ BKMeeting contract.

## 6. Handoff sang BKMeeting

Workspace boundary mong đợi:

```text
<WORKSPACE_ROOT>/quantized-viet-asr-model   # model artifact producer
<WORKSPACE_ROOT>/BKMeeting           # Android artifact consumer
```

Khi bundle adapter đã đáp ứng Android contract:

1. materialize vào staging;
2. verify checksums, `.onnx + model.bin`, manifests và fixtures;
3. sync atomically vào đúng namespace;
4. chạy BKMeeting unit/package/build gates;
5. chạy strict tests trên Snapdragon trước promotion.

Canonical Android entrypoints nằm ở:

- `../../BKMeeting/docs/architecture/overview.md`;
- `../../BKMeeting/docs/architecture/runtime-config.md`;
- `../../BKMeeting/docs/architecture/testing.md`;
- `../../BKMeeting/docs/qnn/playbook.md`.

Python repo không sở hữu Java asset resolver, QNN provider options, CPU fallback, app performance hay release packaging.

## 7. Android acceptance

Local preflight tối thiểu trong BKMeeting:

```bash
./gradlew :app:verifyQnnRuntimePackageInputs --no-daemon
./gradlew :app:testCpuCompatDebugUnitTest :app:testQnnOfficialArm64DebugUnitTest --no-daemon
./gradlew :app:assembleCpuCompatDebug --no-daemon
./gradlew :app:assembleQnnOfficialArm64Debug :app:assembleQnnOfficialArm64DebugAndroidTest --no-daemon
```

Physical-device gate phải chứng minh:

- session dùng `QNN_HTP` và strict mode không CPU fallback;
- Zipformer full transcript qua HTP encoder + CPU decoder/joiner;
- VPCD full autoregressive restored output;
- startup, latency, memory và app-level audio/final-output timing;
- parity so với pre-compile/local baseline trên cùng fixture.

Nếu model bytes, input shape, quantization scope hoặc QAIRT target đổi, phải tạo compile/hosted/device evidence mới. Đổi manifest/path nhưng giữ bytes không tạo claim hiệu năng mới.

## 8. QDC Appium model benchmark

QDC Automated Job nhận một APK benchmark mỏng và một Appium ZIP theo model. Không nhúng model vào APK và không dùng Monkey Test: Appium đẩy payload vào external files, force-stop app giữa từng run, mở `BenchmarkActivity`, click một nút duy nhất, chờ trạng thái terminal rồi pull JSON vào `/qdc/logs/`.

```bash
python -m model_pipeline android-benchmark-payload \
  --model zipformer \
  --output build/android-benchmark/zipformer

python -m model_pipeline android-benchmark-report \
  --results-root <QDC_RESULTS_ROOT> \
  --output build/android-benchmark/comparison
```

Pre-compile CPU có hai control riêng: FP32 fixed-shape và AIMET QDQ. Post-compile NPU luôn là cặp adjacent `EPContext ONNX + model.bin`, chạy strict QNN HTP; không thử load cặp này bằng CPU. Mỗi model phải giữ cả ba configuration trong cùng một Automated Job/device allocation. Nếu VPCD ZIP bị từ chối vì kích thước, không tách CPU và NPU sang hai handset rồi công bố speedup.

Xem [QDC benchmark evidence](evidence/2026-07-17-qdc-appium-cpu-npu-performance.md) và BKMeeting `docs/qnn/qdc-appium-model-benchmark.md`.

## Retained clean-rebuild evidence

Canonical checksum-keyed records hiện tại:

- Zipformer compile job `jp1vnn07p`;
- VPCD compile job `jgn71e3rp`;
- target Samsung Galaxy S23 (Family), Qualcomm AI Runtime 2.45;
- đúng năm hosted inputs mỗi model.

Checksums, target IDs và inference job IDs nằm trong [retained artifact record](evidence/retained-artifacts.json). Đây là cloud evidence; chưa phải Android-production acceptance.
