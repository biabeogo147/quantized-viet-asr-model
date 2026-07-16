# VPCD recipe

AIMET encoder-MatMul artifact:

```text
vpcd__q-aimet-int8-int16-encoder-matmul__s-src1x384-dec1x64__c-aihub-qnn-htp-model
```

## Quantize như thế nào

1. Freeze bốn input thành batch 1, source length 384, decoder length 64.
2. Sinh calibration prefix từ greedy autoregressive decode của FP32 control, pad bằng token/mask value đúng contract source 384 và decoder 64.
3. AIMET ONNX PTQ dùng MinMax, signed 8-bit weight, signed 16-bit activation, `per_channel=False`.
4. Config chỉ bật `MatMul`; policy allow đúng encoder MatMul và disable toàn bộ decoder/lm-head MatMul.
5. Xuất package `model.onnx` + `model.encodings`; AI Hub compile cả package sang ONNX `EPContext` và dùng `truncate_64bit_io` để đổi I/O int64 thành int32 trên target.

## Layer nào thay đổi, layer nào giữ nguyên

Graph FP32 có 265 MatMul:

| Scope | Số MatMul | Trạng thái |
|---|---:|---|
| 12 encoder layers | 96 | AIMET signed 8-bit weight, signed 16-bit activation |
| 12 decoder layers | 168 | giữ FP32 |
| `lm_head` | 1 | giữ FP32 |

Mỗi encoder layer có tám MatMul được quantize: q/k/v projection, attention score, attention value, out projection, `fc1`, `fc2`.

Không quantize Add/Mul/Div/LayerNorm/Softmax hay operator khác. Tokenizer encode/decode, ID bridge và greedy autoregressive loop luôn chạy CPU. Việc model session là EPContext không biến các host operation này thành NPU.

VPCD và Zipformer dùng chung model-independent AIMET service và calibration package format. Riêng VPCD sở hữu policy 96 encoder MatMul được bật và 169 non-encoder MatMul bị disable; service không tự suy ra scope từ tên model.

Configuration `fp32-fixed-shape` dùng cùng source length 384 và decoder length 64 làm control, đồng thời explicit-skip quantize/compile. Refactor không mở rộng decoder coverage, không thêm policy thử nghiệm và không đưa ra claim latency mới.

## Qualcomm HTP compatibility đã kiểm chứng

Canonical package dùng MinMax, signed 8-bit weight, symmetric signed 16-bit activation và `per_channel=False`. Service tắt toàn bộ quantizer rồi name-allowlist đúng 96 encoder MatMul. Package có 168 activation encodings, tất cả offset `-32768`, và 72 initializer-weight encodings; decoder và language-model head không có encoding.

Encoder attention-mask condition dùng `Cast(attention_mask, INT32) → Equal(0)` để giữ ngữ nghĩa mask nhị phân mà không tạo floating-point-to-boolean Cast bị HTP từ chối. Clean rebuild đạt local full-output parity 100/100, first-five top-1 500/500, compile job `jgn71e3rp` thành công và hosted top-1 parity 5/5. Chi tiết nằm trong [báo cáo VLSP](evidence/2026-07-15-vlsp100-quantization-compile.md).
