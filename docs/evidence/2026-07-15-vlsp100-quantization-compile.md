# Báo cáo quantization và compile trên 100 mẫu VLSP

**Thời gian thực hiện:** 2026-07-15 đến 2026-07-16
**Thiết bị AI Hub:** Samsung Galaxy S23 (Family)
**Qualcomm AI Runtime:** 2.45
**Môi trường local:** `speech2text`, ONNX Runtime GPU 1.22.0, NVIDIA GeForce RTX 4060 Ti

## Kết luận

Cả Zipformer và VPCD đều đã được dựng lại từ `build/` trống, quantize lại, đánh giá local trên 100 mẫu VLSP, compile thành ONNX `EPContext` cho Qualcomm HTP và kiểm tra đúng 5 input trên AI Hub.

- Zipformer chọn AIMET signed 8-bit weight, signed 16-bit activation, encoder-MatMul-only. Phương án ONNX Runtime QNN đạt quality gate local nhưng AI Hub không nhận graph Q/DQ do operator `com.microsoft::DequantizeLinear`, nên pipeline chuyển sang AIMET theo đúng fallback contract.
- VPCD chọn AIMET signed 8-bit weight, symmetric signed 16-bit activation, encoder-MatMul-only. Decoder, language-model head, tokenizer và vòng lặp autoregressive không bị quantize.
- Zipformer đạt transcript parity local 100/100 với FP32 và hosted parity 5/5.
- VPCD đạt restored-output parity local 100/100, first-five top-1 500/500 và hosted top-1 parity 5/5.
- Không có NPU local. `EPContext` chỉ được chạy qua hosted inference trên Qualcomm HTP; không chạy package này bằng CPU hoặc NVIDIA GPU local.

## Dataset và provenance

Manifest machine-readable nằm tại `build/datasets/vlsp/vlsp-calibration-evaluation-manifest.json`.

| Partition | Số mẫu | Shard | Điều kiện |
|---|---:|---|---|
| calibration | 24 | `train-00000-of-00035.parquet` | dùng chung transcription cho hai model |
| evaluation | 100 | `train-00001-of-00035.parquet` | audio 2–12 giây, transcription 4–40 từ |

Hai partition không trùng shard, row hoặc transcription. Manifest chỉ lưu đường dẫn tương đối, shard, row, tên audio nguồn, SHA-256 audio và SHA-256 text; không lưu đường dẫn máy.

## Zipformer

Artifact:

```text
zipformer__q-aimet-int8-int16-encoder-matmul__s-enc1x2009x80-dec1x2-join1x512__c-aihub-qnn-htp-encoder
```

### Quantization và graph coverage

- Encoder fixed shape `1×2009×80`, ORT optimization, symbolic shape inference và boolean-mask rewrite được thực hiện trước quantization.
- 278/278 encoder MatMul thuộc scope quantization.
- Decoder và joiner không có MatMul; các Gemm giữ FP32 và chạy CPU.
- AIMET dùng MinMax, signed 8-bit weight, signed 16-bit activation, `per_channel=False`.

ORT-QNN static PTQ ban đầu dùng unsigned 8-bit weight, unsigned 16-bit activation và đạt local quality gate. Compile job `jp2vrw445` yêu cầu 64-bit I/O truncation; job `jp1vnjk8p` sau đó vẫn bị từ chối vì graph chứa `com.microsoft::DequantizeLinear`. Vì vậy artifact được chọn dùng AIMET, không dùng ORT-QNN.

### Kết quả local trên 100 mẫu

| Runtime | CER | WER | Exact parity với FP32 | Empty/collapse | Mean latency |
|---|---:|---:|---:|---:|---:|
| FP32 CPU | 7.183% | 12.490% | control | 0/0 | 377.057 ms |
| AIMET CPU | 7.183% | 12.490% | 100/100 | 0/0 | 372.806 ms |
| AIMET CUDA/mixed | 7.183% | 12.490% | 100/100 | 0/0 | 153.395 ms |

Provider profiler ghi 2,513 CPU node events cho CPU run. CUDA/mixed ghi 2,259 CUDA và 245 CPU node events; vì vậy kết quả này được mô tả là mixed, không phải CUDA-only.

### Compile và hosted validation

- Compile job: `jp1vnn07p` — success.
- Package checksum: `ff1572ca3be7758e552dab4dd0315ecfb4fe8cb954e14dddbaadd64bd450453b`.
- Primary ONNX checksum: `8568fdc6902679c5eda866c7ea5ce82a203a2d79a628c8d89d838e353539415d`.
- Downloaded graph: một `EPContext`; input `x` float32, `x_lens` int32; output encoder float32 và length int32.
- Hosted jobs: `jpy7oynrp`, `jgo4l9m45`, `jgdzdm0l5`, `jgdzdm8l5`, `jgk92k4o5`.
- Hosted encoder output được decode bằng decoder/joiner FP32 local; transcript parity đạt 5/5.

## VPCD

Artifact:

```text
vpcd__q-aimet-int8-int16-encoder-matmul__s-src1x384-dec1x64__c-aihub-qnn-htp-model
```

### Quantization và graph coverage

- Bốn input fixed shape: source IDs/mask `1×384`, decoder IDs/mask `1×64`.
- Graph có 265 MatMul: 96 encoder, 168 decoder, một language-model head.
- Policy tắt toàn bộ quantizer rồi chỉ bật tensor gắn với đúng 96 encoder MatMul.
- Package có 168 symmetric signed 16-bit activation encodings, tất cả offset `-32768`, và 72 signed 8-bit initializer-weight encodings. Attention score/value MatMul không có initializer weight nên không tạo weight encoding.
- Decoder MatMul, language-model head, non-MatMul operator, tokenizer và autoregressive loop giữ nguyên.
- Encoder attention mask dùng `Cast(attention_mask, INT32) → Equal(0)` thay cho floating-point-to-boolean Cast không được HTP hỗ trợ. Với attention mask nhị phân, phép đổi này giữ nguyên ngữ nghĩa.

Hai compile diagnostic trước final artifact giúp khóa compatibility:

- `jp49y6wq5`: QAIRT gộp cầu nối Cast và tạo lại `FLOAT16 → BOOL`, bị HTP từ chối.
- `jgl1y9oe5`: integer comparison đã qua, nhưng asymmetric signed 16-bit offset `-34126` không đạt MatMul contract yêu cầu `-32768`.

Final policy name-allowlist cộng symmetric activation giải quyết cả hai lỗi mà không tăng quantization coverage.

### Kết quả local trên 100 mẫu

| Runtime | Full-output parity | First-five top-1 | Edit distance | Early EOS/collapse | Mean latency |
|---|---:|---:|---:|---:|---:|
| FP32 CPU | control | control | 0 | 0/0 | 8.924 s |
| AIMET CPU | 100/100 | 500/500 | 0 | 0/0 | 9.134 s |
| FP32 CUDA/mixed | control | control | 0 | 0/0 | 0.788 s |
| AIMET CUDA/mixed | 100/100 | 500/500 | 0 | 0/0 | 0.799 s |

CUDA profiler ghi 998,006 CUDA node events trong phần profile được lưu. Tokenizer và autoregressive host loop vẫn chạy CPU. Các số liệu trên là parity với FP32, không phải punctuation accuracy vì VLSP không có ground truth dấu câu/viết hoa phù hợp.

### Compile và hosted validation

- Compile job: `jgn71e3rp` — success.
- Package checksum: `6a6b8f0995812373c795dc35e17f88bf888744fc695d5586b5c3949d95c7863d`.
- Primary ONNX checksum: `c2886b67e06461ddb9d8ee311afa7ef7bf4c48dc17fc9b27b5f26102a2384cb4`.
- Downloaded graph: một `EPContext`; bốn input đã đổi int64 thành int32; hai output float32.
- Hosted jobs: `j5w1y9ozg`, `jp1vo8xlp`, `jp49exll5`, `jp3wo4jz5`, `jgo4d1zd5`.
- Năm teacher-forced prefixes đều có FP32 top-1 = local AIMET top-1 = hosted HTP top-1; parity đạt 5/5.

## Giới hạn

- Không có Qualcomm NPU local để chạy full 100-sample post-compile evaluation.
- Hosted validation bị giới hạn đúng 5 input mỗi model; không suy diễn latency hoặc accuracy 100 mẫu trên thiết bị từ 5 input này.
- CPU/GPU latency đo trên máy phát triển và gồm host preprocessing/autoregressive work theo runtime tương ứng; không đại diện cho latency Android.
- VPCD được đánh giá bằng parity với FP32 vì dataset không có punctuation/capitalization ground truth phù hợp.

Machine-readable evidence nằm dưới `build/evaluation/`, `build/model-pipeline/aihub-evidence/records/` và stage directories của hai artifact.
