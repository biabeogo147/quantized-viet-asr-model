# So sánh CPU–NPU bằng QDC Appium

**Trạng thái:** Chờ Qualcomm Device Cloud access và model payload được dựng lại từ VLSP assets ngoài Git.

## Mục tiêu và representations

Mỗi model được đo trên cùng một điện thoại trong một Automated Job:

| Configuration | Representation | Runtime |
|---|---|---|
| `fp32-fixed-shape-onnxruntime-cpu` | FP32 fixed-shape ONNX | ONNX Runtime CPU |
| `aimet-int8-int16-encoder-matmul-onnxruntime-cpu` | QDQ từ exact AIMET encodings | ONNX Runtime CPU |
| `aimet-int8-int16-encoder-matmul-aihub-qnn-htp` | `EPContext ONNX + model.bin` | strict QNN HTP NPU |

Mỗi configuration chạy ba fresh process. Mỗi process có 10 warm-up và 100 measured inference. Lịch Latin-square đổi vị trí configuration giữa ba round. Timer Android chỉ bao quanh `OrtSession.run()`.

## Validity contract

Comparison cần đủ 900 observations mỗi model, checksum khớp retained artifact, latency hữu hạn, output contract đạt và không có CPU/GPU fallback trong NPU runs. Báo cáo sẽ tách:

- ảnh hưởng quantization: FP32 CPU so với QDQ CPU;
- deployment speedup: QDQ CPU so với EPContext NPU;
- app reference: FP32 CPU so với EPContext NPU.

Zipformer chỉ đo encoder. VPCD chỉ đo một model invocation; tokenizer và autoregressive loop không nằm trong timing. Kết quả này không phải Android app end-to-end.

## Source checkpoint

- BKMeeting benchmark APK: `qnnOfficialArm64Benchmark`, không chứa modelassets.
- Appium: hai model-specific ZIP, Zipformer chạy trước VPCD.
- Generated payload và raw QDC output: ignored `build/`, không track trong Git.
- Retained compiled checksums/job IDs: [retained artifact record](retained-artifacts.json).

## Kết quả

Chưa có số liệu device. Không suy diễn CPU–NPU speedup từ local x86/NVIDIA hoặc AI Hub profiling cũ. Phần này chỉ được cập nhật sau khi Interactive Session smoke và cả hai Automated Job trả về evidence hợp lệ.
