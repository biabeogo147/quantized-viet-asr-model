# Bàn giao Model Repository và Benchmark CPU–NPU trên Android

## Trạng thái

Model repository canonical đã được materialize và cả hai benchmark model-level trên Qualcomm Device Cloud (QDC) đều hợp lệ. Main-app physical-device validation đã chạy nhưng gate exact CPU–NPU chỉ đạt `8/10`; artifact NPU chưa được promote thành deployment mặc định từ evidence này.

## Kết luận

Trên Snapdragon 8 Gen 2 HDK8550, Android 14/API 34, post-compile `EPContext ONNX + model.bin` chạy strict Qualcomm Neural Network (QNN) Hexagon Tensor Processor (HTP) nhanh hơn FP32 ONNX Runtime CPU:

| Model boundary | FP32 CPU median | QNN HTP median | Speedup CPU/NPU | PSS giảm |
|---|---:|---:|---:|---:|
| Zipformer encoder | 536.321 ms | 427.413 ms | **1.255×** | **49.20%** |
| VPCD một model invocation | 2,335.588 ms | 617.943 ms | **3.780×** | **93.66%** |

Đây là so sánh trên cùng device fingerprint trong từng model job. Zipformer chỉ đo encoder; VPCD chỉ đo một fixed-shape invocation, không đo toàn bộ autoregressive loop.

## Artifact provenance

| Thuộc tính | Zipformer | VPCD |
|---|---|---|
| FP32 artifact | `zipformer__q-none-fp32-fp32-none__s-enc1x2009x80-dec1x2-join1x512__c-none-cpu-none` | `vpcd__q-none-fp32-fp32-none__s-src1x384-dec1x64__c-none-cpu-none` |
| NPU artifact | `zipformer__q-aimet-int8-int16-encoder-matmul__s-enc1x2009x80-dec1x2-join1x512__c-aihub-qnn-htp-encoder` | `vpcd__q-aimet-int8-int16-encoder-matmul__s-src1x384-dec1x64__c-aihub-qnn-htp-model` |
| Compiled ONNX SHA-256 | `8568fdc6902679c5eda866c7ea5ce82a203a2d79a628c8d89d838e353539415d` | `c2886b67e06461ddb9d8ee311afa7ef7bf4c48dc17fc9b27b5f26102a2384cb4` |
| Benchmark payload SHA-256 | `1d92ca2cf2db32b8a0c45598155177e1b6c5b5dda46cd23d7117145e1ddad0ee` | `ea66d997b41f1a3ea83e7d20e08dbef40de8c378004922213fe27c7f5ea80026` |
| QDC job | [704740](https://qdc.qualcomm.com/reports/job/automated/704740) | [704742](https://qdc.qualcomm.com/reports/job/automated/704742) |

Benchmark APK không chứa model, SHA-256 `0bd419a17a8f7162d06e8e131879de9fc9a538a0de58b60b52f10abf359db4e7`. Appium package SHA-256 là `aba11ed75065bff1bc79924563425daad7eea17eaa4eb0c509c689d68ad2c71d` cho Zipformer và `f72f080aee832586765de44e65dfb3d496e4466131a9dee60f3159c89da1c112` cho VPCD.

## Phương pháp đo

Mỗi model chạy một Automated Job riêng. Mỗi configuration chạy ba fresh processes theo lịch cân bằng; mỗi process có 10 warm-up và 100 timed inference. Mỗi hàng tổng hợp vì vậy có 300 observations.

Timer `SystemClock.elapsedRealtimeNanos()` chỉ bao quanh `OrtSession.run()`. Việc đọc tensor, tạo session, quality validation và aggregation nằm ngoài timing.

Hai configurations:

- `fp32-fixed-shape-onnxruntime-cpu`: FP32 fixed-shape ONNX, ONNX Runtime CPU.
- `aimet-int8-int16-encoder-matmul-aihub-qnn-htp`: post-compile `EPContext ONNX + model.bin`, strict QNN HTP.

## Kết quả Zipformer encoder

| Configuration | Median | p95 | Mean | Stddev | Min–max | Session create median | PSS median |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP32 CPU | 536.321 ms | 566.940 ms | 536.597 ms | 21.674 ms | 483.328–572.605 ms | 325.146 ms | 235,116 KiB |
| QNN HTP | **427.413 ms** | **437.070 ms** | **430.102 ms** | **4.391 ms** | 425.057–438.945 ms | 612.405 ms | **119,446 KiB** |

Median theo repetition:

| Configuration | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| FP32 CPU | 519.271 ms | 536.219 ms | 561.033 ms |
| QNN HTP | 436.201 ms | 427.018 ms | 426.940 ms |

NPU giảm median latency 20.31% và giảm p95 22.91%. Session creation của NPU chậm hơn CPU 88.35%, nên speedup trên chỉ đại diện cho repeated encoder inference sau khi session đã sẵn sàng.

## Kết quả VPCD một model invocation

| Configuration | Median | p95 | Mean | Stddev | Min–max | Session create median | PSS median |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP32 CPU | 2,335.588 ms | 2,502.086 ms | 2,329.533 ms | 134.864 ms | 1,985.282–2,534.850 ms | 1,994.540 ms | 1,840,516 KiB |
| QNN HTP | **617.943 ms** | **620.136 ms** | **618.055 ms** | **1.127 ms** | 615.456–622.352 ms | **954.723 ms** | **116,695 KiB** |

Median theo repetition:

| Configuration | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| FP32 CPU | 2,171.625 ms | 2,335.588 ms | 2,491.966 ms |
| QNN HTP | 617.941 ms | 617.175 ms | 618.979 ms |

NPU giảm median latency 73.54%, giảm p95 75.22%, giảm session creation 52.13% và giảm PSS 93.66%.

## Quality và HTP placement

- Zipformer: cả sáu results đạt `zipformer-transcript-parity-5-of-5`; encoder output được decode bằng cùng FP32 decoder/joiner ngoài timing.
- VPCD: cả sáu results đạt `vpcd-teacher-forced-top1-25-of-25`.
- Cả sáu NPU runs có `requested_provider=qnn-htp`, `execution_provider=qnn-htp`, `qnn_backend=libQnnHtp.so`, `cpu_fallback_disabled=true`, `strict_npu=true` và `htp_execution_observed=true`.
- Mỗi NPU run ghi SHA-256 riêng cho ONNX Runtime profile. Ba profile checksum/model khác nhau và đều được Appium result validator xác nhận.
- Cả hai job dùng device fingerprint `qti/kalama/kalama:14/UKQ1.240819.001/eng.lnxbui.20240910.040454:userdebug/test-keys`.

## Sự cố và cách xử lý

Job Zipformer `704739` thất bại trước inference vì fixture paths trong manifest là relative với canonical model repository, nhưng Android benchmark runtime lại resolve từ outer payload root. Regression test mới khóa repository-root contract; runtime sau đó verify và đọc tensor từ đúng canonical root. Không có model byte, quantization policy hoặc provider policy nào bị thay đổi.

Khi tổng hợp job mới, aggregator ban đầu yêu cầu CPU và NPU có cùng `artifact_id`. Điều này mâu thuẫn với canonical repository, nơi mỗi representation có identity riêng. Contract được sửa theo hướng đúng: một artifact ID ổn định trong ba repetitions của từng configuration, còn payload checksum phải chung cho cả comparison.

## Giới hạn

- Năm retained fixed-shape fixtures được dùng cho quality gate; đây không phải 100 VLSP samples chạy trực tiếp trên device.
- Zipformer decoder/joiner và VPCD autoregressive loop nằm ngoài timing.
- Không so latency tuyệt đối giữa hai model như một workload tương đương.
- Đây là model-level Android benchmark, chưa phải streaming ASR, punctuation end-to-end, thermal soak, battery hoặc UI benchmark.
- Kết quả chưa đóng main-app gate `10/10`; main app cần physical-device validation riêng từ cùng model repository.

Machine-readable aggregate nằm tại ignored path `build/qdc-benchmark/comparison/comparison.json`; raw six-run JSON nằm tại `build/qdc-device-results/<model>/raw/`.

## Main-app physical-device validation

Interactive Session [704755](https://qdc.qualcomm.com/reports/job/interactive/704755)
chạy trên cùng HDK8550 fingerprint từ 22:38:45 đến 23:26:47 ngày 2026-07-19.

Strict-load instrumentation đạt trong `2.976 s`:

- Zipformer encoder và VPCD model resolve đúng retained compiled checksums;
- `QNNExecutionProvider` khởi tạo `libQnnHtp.so`;
- `libQnnHtpV73Skel.so` được mở trên CDSP domain 3;
- Zipformer decoder/joiner và VPCD tokenizer giữ CPU.

Full main-app instrumentation chạy cùng 10 bundled audio samples:

| Surface | Thời gian 10 samples | Raw exact so với CPU | Final exact so với CPU |
|---|---:|---:|---:|
| FP32 CPU | 1,126.091 s | control | control |
| QNN HTP | 385.502 s | `8/10` | `8/10` |
| QNN HTP repeat | 393.335 s | `8/10` | `8/10` |

Tỷ lệ tổng thời gian CPU/NPU là `2.921×`. Đây là instrumentation elapsed time
bao gồm model load và full orchestration, không thay thế latency benchmark quanh
`OrtSession.run()`.

Hai mismatches ổn định:

- `sample-1`: raw CPU có `... HAI MƯƠI NĂM ...`, NPU có
  `... HAI MƯƠI LĂM ...`; final đổi từ chuỗi chứa `21/12/2020, năm` thành
  `21/12/2025`.
- `sample-8`: NPU bỏ từ đầu `MỘT`, nên final đổi từ `Một số tiền...` thành
  `Số tiền...`.

Hai NPU runs tự khớp raw/final `10/10`, chứng minh mismatch deterministic và
không phải runtime variance. Theo exact-promotion contract, comparison main app
`valid=false`; active plans phải giữ nguyên cho tới khi model/acceptance decision
được xử lý có chủ đích.

Machine-readable records nằm trong ignored
`build/qdc-device-results/main-app/`.

Evidence SHA-256:

- `cpu.json`: `ac28e9788b6b92e8587ed6ad56f44171967c3608d68c1428818c4397e12c0d68`.
- `npu.json` và `npu-repeat.json`: `4b2ee7f7fe3524e106a90cdf552dca2edd9f889899977a7c3e660f08e015fe94`.
- `comparison.json`: `b16dc7896df297c5be1860dfa8c1f3267ece557ee429d7176748306b9223a265`.
