# VPCD recipe

Production artifact:

```text
vpcd__q-aimet-int8-int16-encoder-matmul__s-src1x384-dec1x64__c-aihub-qnn-htp-model
```

## Quantize như thế nào

1. Freeze bốn input thành batch 1, source length 384, decoder length 64.
2. Sinh calibration prefix từ greedy autoregressive decode của FP32 control, pad bằng token/mask value đúng contract A4.
3. AIMET ONNX PTQ dùng MinMax, weight INT8, activation INT16, `per_channel=False`.
4. Config chỉ bật `MatMul`; policy allow đúng encoder MatMul và disable toàn bộ decoder/lm-head MatMul.
5. Xuất package `model.onnx` + `model.encodings`; AI Hub compile cả package sang ONNX `EPContext` và dùng `truncate_64bit_io` để đổi I/O int64 thành int32 trên target.

## Layer nào thay đổi, layer nào giữ nguyên

Graph FP32 có 265 MatMul:

| Scope | Số MatMul | Trạng thái |
|---|---:|---|
| 12 encoder layers | 96 | AIMET W8A16 |
| 12 decoder layers | 168 | giữ FP32 |
| `lm_head` | 1 | giữ FP32 |

Mỗi encoder layer có tám MatMul được quantize: q/k/v projection, attention score, attention value, out projection, `fc1`, `fc2`.

Không quantize Add/Mul/Div/LayerNorm/Softmax hay operator khác. Tokenizer encode/decode, ID bridge và greedy autoregressive loop luôn chạy CPU. Việc model session là EPContext không biến các host operation này thành NPU.

Profile `fp32` dùng cùng shape A4 làm control và explicit-skip quantize/compile. Refactor không mở rộng decoder coverage, không thêm policy thử nghiệm và không đưa ra claim latency mới.
