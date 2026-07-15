# Zipformer recipe

Production artifact:

```text
zipformer__q-none-fp32-fp32-none__s-enc1x2009x80-dec1x2-join1x512__c-aihub-qnn-htp-encoder
```

## Điều thực sự xảy ra

Zipformer production **không local-quantize**. Stage `quantize` là explicit skip và manifest ghi `q-none`.

Prepare cố định shape cho ba ONNX component. Riêng encoder được ORT extended-optimize, symbolic shape inference, rồi sửa đúng sáu boundary bool-mask:

- ba `Slice`: `/encoder/Slice_1`, `/encoder/Slice_3`, `/encoder/Slice_5` đọc mask qua một Cast `BOOL → UINT8` dùng chung;
- ba `Unsqueeze_15` đầu mỗi encoder stack xuất `UINT8`, sau đó Cast về `BOOL`.

AI Hub compile **encoder FP32 đã prepare** sang ONNX `EPContext`. Decoder và joiner không gửi lên compile.

## Layer/component truth

| Component | MatMul | Quantization | Execution |
|---|---:|---|---|
| encoder | 278 | giữ FP32; không Q/DQ local | QNN HTP sau compile |
| decoder | 0 | giữ FP32 (`Gemm` không quantize) | CPU ONNX |
| joiner | 0 | giữ FP32 (`Gemm` không quantize) | CPU ONNX |
| tokens | n/a | không áp dụng | CPU |

ORT-QNN static PTQ cũ nhắm `MatMul` từng bị hiểu nhầm là quantize cả bundle: thực tế chỉ encoder có MatMul. Lane đó đã bị xóa vì không phải đầu vào production AI Hub và report sai decoder/joiner.
