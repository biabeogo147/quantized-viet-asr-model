# So sánh hiệu năng CPU–NPU bằng QDC Appium

**Trạng thái:** Hoàn thành ngày 2026-07-18. Cả Zipformer và VPCD đều có comparison hợp lệ.

## Kết luận chính

Post-compile `EPContext ONNX + model.bin` chạy bằng Qualcomm Neural Network (QNN) Hexagon Tensor Processor (HTP) nhanh hơn cả hai CPU controls trên cùng thiết bị của từng model:

| Model boundary | FP32 CPU / NPU | QDQ CPU / NPU | Ảnh hưởng của QDQ trên CPU |
|---|---:|---:|---:|
| Zipformer encoder | **1.264×** | **1.551×** | chậm hơn FP32 **22.72%** |
| VPCD một model invocation | **3.969×** | **4.144×** | chậm hơn FP32 **4.40%** |

Quantization-aware artifact không tự tạo lợi ích khi vẫn chạy bằng ONNX Runtime CPU trong phép đo này. Lợi ích rõ ràng xuất hiện khi artifact post-compile được thực thi trên HTP. VPCD hưởng lợi lớn hơn Zipformer tại model boundary; không được dùng bảng này để suy ra toàn bộ luồng ASR hoặc punctuation nhanh tương ứng vì decoder/joiner và autoregressive control nằm ngoài timing.

## Phương pháp đo

Mỗi model chạy trong một Qualcomm Device Cloud (QDC) Automated Job riêng trên Snapdragon 8 Gen 2 HDK8550. Trong một job, ba configurations dùng cùng payload, device fingerprint và lịch cân bằng:

```text
Round 1: FP32 CPU → QDQ CPU → EPContext NPU
Round 2: QDQ CPU → EPContext NPU → FP32 CPU
Round 3: EPContext NPU → FP32 CPU → QDQ CPU
```

Mỗi launch là một process mới, gồm 10 warm-up và 100 lần đo. Mỗi configuration vì thế có 300 observations; mỗi model có 900 observations. Timer `SystemClock.elapsedRealtimeNanos()` chỉ bao quanh `OrtSession.run()`. Việc đọc tensor, tạo session, validation output và aggregation nằm ngoài timing.

Ba representations:

| Configuration | Representation | Runtime |
|---|---|---|
| `fp32-fixed-shape-onnxruntime-cpu` | FP32 fixed-shape ONNX | ONNX Runtime CPU |
| `aimet-int8-int16-encoder-matmul-onnxruntime-cpu` | QDQ phục hồi từ exact AIMET encodings | ONNX Runtime CPU |
| `aimet-int8-int16-encoder-matmul-aihub-qnn-htp` | `EPContext ONNX + model.bin` | strict QNN HTP |

Measured QNN session tắt profiling để tránh overhead. Một validation session riêng bật ONNX Runtime/QNN profiling; comparison chỉ hợp lệ khi profile có node event thuộc `QNNExecutionProvider`, backend là `libQnnHtp.so`, CPU fallback bị tắt và Android log xác nhận `libQnnHtpV73Skel.so` được mở qua CDSP.

## Artifact provenance

| Thuộc tính | Zipformer | VPCD |
|---|---|---|
| Artifact ID | `zipformer__q-aimet-int8-int16-encoder-matmul__s-enc1x2009x80-dec1x2-join1x512__c-aihub-qnn-htp-encoder` | `vpcd__q-aimet-int8-int16-encoder-matmul__s-src1x384-dec1x64__c-aihub-qnn-htp-model` |
| Compile job | `jp1vnn07p` | `jgn71e3rp` |
| AI Hub target | `mqep43d7m` | `mqpyjggoq` |
| Compile input SHA-256 | `af88f55d3a287d3059ca7813b515d20173387a6762aee9bbb1b668a73b041429` | `b44f5db3d1ad9054425c6965096e734778d4653a71f2cac4a0a946b91a5a2e19` |
| Compiled ONNX SHA-256 | `8568fdc6902679c5eda866c7ea5ce82a203a2d79a628c8d89d838e353539415d` | `c2886b67e06461ddb9d8ee311afa7ef7bf4c48dc17fc9b27b5f26102a2384cb4` |
| `model.bin` SHA-256 | `8deda7ba477c849626adac2338826daccaf1fce6f9a0abdd2dc2cf2fcaa9747f` | `5e63da601b1162162b2cdb6844a240b7a146c36fef0f90e378d7f842e5cfbda4` |
| Payload manifest SHA-256 | `a2d57b8b47145e5e3846d60b12fb8dae230a325b33eab24ecccf0be06bcce12e` | `3b107ae6bd24cd6fca3b93d36d8c62a0c53726e3276dba737ce8c7de167c9fce` |

QDQ chỉ là benchmark control dưới ignored `build/`; nó không được đưa vào canonical compile package hoặc Android production bundle. Zipformer giữ 278 encoder `MatMul` trong quantization scope; decoder/joiner FP32. VPCD giữ graph contract `96/168/1`: chỉ 96 encoder `MatMul` có encodings, 168 decoder và một language-model-head `MatMul` giữ FP32.

APK benchmark đã upload cho hai successful jobs có SHA-256 `10bd98aeddb34dca6b1c94633c53147a8e15483f06d1d47a3a20025f4d3231c1`; APK không chứa production modelassets. Appium ZIP SHA-256 là `bef61d50cfc8c90b86c81560bec28734c6444182786ef4aba42a966bb5ac13e9` cho Zipformer và `9c7d69e87a17b0c895182b3b4c6b3ef326aef5cf34f2ed401ed16c83c72710c3` cho VPCD.

## QDC jobs và device truth

| Model | QDC job | Trạng thái | Test duration | Device |
|---|---|---:|---:|---|
| Zipformer | [704393](https://qdc.qualcomm.com/reports/job/automated/704393) | Pass | 10 phút 33 giây | Snapdragon 8 Gen 2 HDK8550 |
| VPCD | [704409](https://qdc.qualcomm.com/reports/job/automated/704409) | Pass | 37 phút 08 giây | Snapdragon 8 Gen 2 HDK8550 |

QDC cấp hai device serial khác nhau (`47874641` và `3384ca9e`). Cả hai có fingerprint `qti/kalama/kalama:14/UKQ1.240819.001/eng.lnxbui.20240910.040454:userdebug/test-keys`. Appium capability xác nhận Android 14/API 34; nhãn portal `Android U` không được diễn giải thành Android 16.

## Kết quả Zipformer encoder

| Configuration | Median (ms) | p95 (ms) | Mean (ms) | Stddev (ms) | Min–max (ms) | Session create median (ms) | PSS median (KiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP32 CPU | 545.577 | 590.764 | 535.913 | 35.533 | 452.112–599.953 | 329.740 | 230,322 |
| QDQ CPU | 669.529 | 731.768 | 683.679 | 32.593 | 622.108–735.925 | 549.441 | 173,656 |
| EPContext NPU | **431.728** | **433.394** | **431.797** | **0.857** | 429.073–434.551 | 633.351 | **114,417** |

Median theo repetition:

| Configuration | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| FP32 CPU | 497.232 | 550.002 | 569.551 |
| QDQ CPU | 657.801 | 668.467 | 724.165 |
| EPContext NPU | 431.677 | 431.803 | 431.695 |

NPU giảm median latency 20.87% so với FP32 CPU và 35.52% so với QDQ CPU. Median PSS giảm 50.32% so với FP32 CPU và 34.11% so với QDQ CPU. Ba NPU runs gần như trùng nhau, trong khi CPU runs tăng dần theo lịch chạy; đây là dấu hiệu thermal/order sensitivity cần được theo dõi nếu sau này đặt product budget.

NPU session creation chậm hơn FP32 CPU khoảng 304 ms. Vì vậy lợi ích 1.264× áp dụng cho repeated encoder inference; một luồng chỉ tạo session rồi chạy rất ít lần cần tính cả startup riêng.

## Kết quả VPCD một model invocation

| Configuration | Median (ms) | p95 (ms) | Mean (ms) | Stddev (ms) | Min–max (ms) | Session create median (ms) | PSS median (KiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP32 CPU | 2,482.567 | 2,518.578 | 2,397.163 | 137.848 | 2,071.072–2,539.843 | 1,995.293 | 1,842,806 |
| QDQ CPU | 2,591.874 | 2,759.118 | 2,599.617 | 86.363 | 2,408.475–2,786.338 | 1,507.984 | 1,410,860 |
| EPContext NPU | **625.446** | **631.605** | **624.805** | **5.211** | 616.413–632.728 | **1,007.629** | **112,922** |

Median theo repetition:

| Configuration | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| FP32 CPU | 2,220.163 | 2,491.099 | 2,492.588 |
| QDQ CPU | 2,569.015 | 2,560.747 | 2,666.016 |
| EPContext NPU | 618.087 | 630.911 | 625.446 |

NPU giảm median latency 74.81% so với FP32 CPU và 75.87% so với QDQ CPU. Median PSS giảm 93.87% so với FP32 CPU và 92.00% so với QDQ CPU. Cả session creation và repeated inference đều tốt hơn hai CPU controls tại model boundary.

## Quality và placement evidence

- Zipformer: cả chín results có `quality_passed=true`; contract trên device là transcript parity `5/5` sau FP32 decoder/joiner chung.
- VPCD: cả chín results có `quality_passed=true`; contract trên device là teacher-forced first-five-step top-1 `25/25`.
- Sáu NPU results có `requested_provider=qnn-htp`, `qnn_backend=libQnnHtp.so`, `cpu_fallback_disabled=true`, `htp_execution_observed=true`.
- Ba ONNX Runtime profile mỗi model đều tồn tại và khớp SHA-256 ghi trong result JSON.
- Android host logs có `QNNExecutionProvider` và `Successfully opened file libQnnHtpV73Skel.so`; FastRPC mở HTP skeleton trên CDSP domain 3.
- Host-log SHA-256: Zipformer `f66ecea6ad5e2869f989449905f5a1a63ab893b743918397090da3073c1b5895`; VPCD `ced2f7ac81616cdd65f478380d3f0a194a4216293be274d3fd9079347d84efc0`.

VPCD device job không chạy full autoregressive loop, do đó không tuyên bố restored-output parity `5/5`, early end-of-sequence hay punctuation-collapse trên device từ benchmark này. Retained AI Hub hosted inference trước đó đạt `5/5`, nhưng đó là evidence khác và không thay thế full Android autoregressive validation.

## Sự cố đã gặp và cách xử lý

| QDC job | Triệu chứng | Nguyên nhân | Cách xử lý và regression gate |
|---|---|---|---|
| `704301` | pytest dừng khi collect | Appium ZIP không bootstrap sibling `qdc_benchmark` khi QDC chỉ thêm `tests/` vào `sys.path` | Entrypoint tự thêm extracted package root; thêm isolated-package collection test |
| `704360` | WebDriver trả HTTP 404 | Harness gọi Appium root thay vì QDC base path `/wd/hub` | Đổi mặc định thành `http://127.0.0.1:4723/wd/hub`; thêm endpoint test |
| `704366` | ONNX Runtime fail trước FP32 inference | `ALL_OPT` tối ưu lại Zipformer control-flow graph đã prepare | Benchmark sessions dùng `NO_OPT`; production runtime không đổi; thêm session-policy test |
| `704377` | CPU runs xong nhưng placement session NPU fail | QNN `profiling_level=basic` thiếu `profiling_file_path` | Cấp CSV path riêng trong payload và vẫn dùng ONNX Runtime JSON làm placement proof; thêm configurer test |

Các retry chỉ được submit sau khi local Android/Appium gates và archive integrity chạy lại. Hai successful jobs sau cùng không cần Interactive Session; Zipformer full job đóng vai trò smoke và measurement trước khi submit VPCD.

## Giới hạn và cách dùng kết quả

- Zipformer timing chỉ đo encoder; decoder, joiner và tokenizer không nằm trong timing.
- VPCD timing chỉ đo một fixed-shape model invocation; tokenizer và autoregressive loop không nằm trong timing.
- Fixtures là năm retained fixed-shape inputs, không phải 100 bản ghi VLSP chạy trực tiếp trên device.
- Hai model chạy trên hai physical HDK serial khác nhau; không so latency tuyệt đối chéo model như cùng một máy.
- Đây là Android model-level benchmark, chưa phải production app startup, streaming, final-output, thermal, battery hoặc user-flow benchmark.
- System profile của QDC chỉ là context; các latency ở trên đến từ in-app timer quanh `OrtSession.run()`.
- Không có performance pass/fail threshold tùy ý. Comparison được đánh dấu hợp lệ theo provenance, completeness, quality và strict HTP placement.

Kết quả đủ để kết luận post-compile artifact thực thi được trên Qualcomm HTP và đo được lợi ích CPU–NPU ở model boundary. Nó chưa đủ để quyết định trải nghiệm BKMeeting end-to-end; bước tiếp theo phù hợp là tích hợp canonical bundle vào production namespace rồi chạy strict app smoke và end-to-end latency trên Snapdragon.

## Evidence tái lập

Machine-readable aggregate nằm tại ignored path:

```text
build/qdc-benchmark/comparison/comparison.json
```

Raw result/profile/system logs nằm tại:

```text
build/qdc-benchmark/qdc-results/zipformer/
build/qdc-benchmark/qdc-results/vpcd/
```

Các file lớn và raw device logs không được track trong Git. Canonical provenance nằm trong [retained artifact record](retained-artifacts.json); Android harness và execution contract nằm trong [BKMeeting QDC benchmark guide](../../../BKMeeting/docs/qdc-appium-benchmark.md).
