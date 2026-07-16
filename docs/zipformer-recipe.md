# Zipformer recipe

FP32 AI Hub encoder artifact:

```text
zipformer__q-none-fp32-fp32-none__s-enc1x2009x80-dec1x2-join1x512__c-aihub-qnn-htp-encoder
```

## Điều thực sự xảy ra

Configuration `fp32-fixed-shape-aihub-encoder` **không local-quantize**. Stage `quantize` là explicit skip và manifest ghi `q-none`.

Prepare cố định shape cho ba ONNX component. Riêng encoder được ORT extended-optimize với `MatMulAddFusion` bị tắt để tám MatMul không bị đổi thành Gemm trước quantization, sau đó symbolic shape inference rồi sửa đúng sáu boundary bool-mask:

- ba `Slice`: `/encoder/Slice_1`, `/encoder/Slice_3`, `/encoder/Slice_5` đọc mask qua một Cast `BOOL → UINT8` dùng chung;
- ba `Unsqueeze_15` đầu mỗi encoder stack xuất `UINT8`, sau đó Cast về `BOOL`.

AI Hub compile **encoder FP32 đã prepare** sang ONNX `EPContext`. Source `x_lens` là int64 nên compile bật `truncate_64bit_io`; target `x_lens` trở thành int32. Decoder và joiner không gửi lên compile.

## Layer/component truth

| Component | MatMul | Quantization | Execution |
|---|---:|---|---|
| encoder | 278 | giữ FP32; không Q/DQ local | QNN HTP sau compile |
| decoder | 0 | giữ FP32 (`Gemm` không quantize) | CPU ONNX |
| joiner | 0 | giữ FP32 (`Gemm` không quantize) | CPU ONNX |
| tokens | n/a | không áp dụng | CPU |

Configuration `ortqnn-uint8-uint16-encoder-matmul` dùng ONNX Runtime Qualcomm Neural Network static post-training quantization, MinMax, unsigned 8-bit weight, unsigned 16-bit activation, `per_channel=False`, và chỉ quantize 278 encoder MatMul. Decoder và joiner tiếp tục là FP32 CPU. Configuration `aimet-int8-int16-encoder-matmul` là fallback khi local quality gate không đạt hoặc AI Hub không chấp nhận artifact ONNX Runtime QNN.

ORT-QNN xuất Q/DQ ONNX và graph validation yêu cầu đủ 278/278 MatMul encoder có Q/DQ. AIMET fallback dùng cùng calibration inputs, signed 8-bit weight, signed 16-bit activation, MinMax, `per_channel=False`, và xuất package `model.onnx` + `model.encodings`. Compile input của ORT-QNN là file ONNX; compile input của AIMET là cả package directory. Local gate cho ORT-QNN yêu cầu CER không tăng quá 1.0 điểm phần trăm, WER không tăng quá 2.0 điểm phần trăm và không có output rỗng hay repetition collapse.

Local transcript evaluation dùng centered log-Mel spectrogram trên waveform chuẩn hóa. Greedy transducer decoder có thể emit nhiều token trên một encoder frame và tiếp tục frame kế tiếp khi joiner trả blank; decoder và joiner luôn chạy CPU kể cả khi encoder chạy CUDA/mixed.

## Evidence 2026-07-15/16

Clean rebuild xác nhận ORT-QNN đạt local quality gate nhưng AI Hub từ chối `com.microsoft::DequantizeLinear`. AIMET fallback đạt 100/100 transcript parity với FP32 trên VLSP, compile job `jp1vnn07p` thành công và hosted transcript parity đạt 5/5. Chi tiết nằm trong [báo cáo VLSP](evidence/2026-07-15-vlsp100-quantization-compile.md).
