# Kiến trúc Model Pipeline

`src/model_pipeline/` là package công khai duy nhất.

```text
core                 specs, manifest v2, checksum, cache/resume
models/zipformer     fixed shapes, HTP bool-mask rewrite, graph validation
models/vpcd          calibration 384x64, AIMET INT8/INT16 policy, tokenizer/runtime
models/aimet*        shared calibration format and model-independent Docker service
evaluation           normalized metrics, ORT profiler attribution, JSON/JSONL evidence
datasets             VLSP extraction, fixture selection/audio/golden refresh
integrations/aihub   compile client, wait/download/live-run, checksum evidence
integrations/android deterministic bundle và BKMeeting sync
pipeline.py          điều phối bảy stage; được phép phụ thuộc mọi adapter
```

Dependency boundary được khóa bằng test: `core` không import model/integration; model không import integration; integration không import model.

AIMET service nhận duy nhất fixed-shape FP32 model, calibration directory, MatMul-only config, operator policy và output directory. Service không biết Zipformer hay VPCD; model adapter sở hữu calibration semantics và graph-scope policy. Docker image pin CPU-only Torch dependencies để không kéo CUDA runtime vào quantization service.

## Artifact identity

Grammar:

```text
<model>__q-<engine>-<weight>-<activation>-<scope>__s-<shape>__c-<compiler>-<target>-<compiled-scope>
```

Allowlist hiện có ba quantization contract:

- không quantize: `none-fp32-fp32-none`;
- AIMET signed 8-bit weight, signed 16-bit activation, encoder MatMul: `aimet-int8-int16-encoder-matmul`;
- ONNX Runtime QNN unsigned 8-bit weight, unsigned 16-bit activation, encoder MatMul: `ortqnn-uint8-uint16-encoder-matmul`.

Không có alias theo tên thử nghiệm hay công cụ tương tác cũ. Recipe được chọn bằng `configuration`; mỗi giá trị phải tự mô tả engine, precision, shape, operator scope hoặc execution target.

Các configuration công khai:

- VPCD: `fp32-fixed-shape`, `aimet-int8-int16-encoder-matmul`;
- Zipformer: `fp32-fixed-shape`, `fp32-fixed-shape-aihub-encoder`, `ortqnn-uint8-uint16-encoder-matmul`, `aimet-int8-int16-encoder-matmul`.

## Manifest v2

Mỗi component ghi: `role`, `file`, `format`, `precision`, `input_shapes`, quantization engine/scope, `execution_target`, checksum. Manifest còn ghi source checksum, recipe digest, validation và runtime metadata. Android chọn provider từ `execution_target`, không từ tên thư mục.

## Cache và provenance

Một stage chỉ resume khi artifact ID, recipe digest, toàn bộ input digest và checksum output đều khớp. Package AIMET được hash theo danh sách `{relative path, file checksum}`, không chứa đường dẫn máy. AI Hub evidence resolve bằng checksum package đầu vào và kiểm tra lại blob đầu ra trước khi reuse.

## VLSP calibration và evaluation

Dataset adapter chọn calibration từ shard VLSP đầu tiên và evaluation từ các shard còn lại. Evaluation chỉ nhận audio 2–12 giây với transcription 4–40 từ; shard, row và transcription không được trùng giữa hai partition. Manifest chỉ lưu audio path tương đối, shard, row, audio checksum và text checksum. Hai model dùng chung transcription calibration để giữ input provenance có thể so sánh.

## Local evaluation runtime

Zipformer local runtime tạo centered log-Mel spectrogram trực tiếp từ waveform chuẩn hóa, chạy fixed-shape encoder trên provider ONNX Runtime được cấu hình, rồi dùng decoder/joiner FP32 trên CPU. Recurrent neural network transducer greedy loop được phép emit nhiều token trên cùng encoder frame và chỉ chuyển frame sau khi gặp blank. VPCD local runtime chạy model session fixed shape, còn tokenizer và autoregressive loop luôn chạy CPU. CPU/GPU latency được tách theo run; CUDA chỉ được ghi nhận khi ONNX Runtime profiler cho thấy ít nhất một node thực sự chạy trên `CUDAExecutionProvider`. Per-sample JSONL và summary JSON được serialize deterministic dưới `build/`.

VLSP selector stream parquet rows và dừng ngay khi đủ 24 calibration cùng 100 evaluation records; không giữ toàn bộ audio corpus trong RAM. ONNX Runtime CPU và GPU là hai installation surface loại trừ nhau. Version `1.22.0` được pin cho local runtime vì fixed-shape Zipformer Extended optimizer chạy đúng ở version này; `1.26.0` đã được tái hiện lỗi khi cả batch và time dimensions cùng cố định.

## Development contract

`AGENTS.md` là nguồn quy tắc làm việc duy nhất cho repository. Mọi thay đổi tracked phải có plan đang active trước khi implementation bắt đầu; plan ghi các phương án đã cân nhắc, quyết định, task, verification và canonical docs cần đồng bộ. Plan chỉ được chuyển sang `docs/plans/completed/` sau khi mọi gate đạt.

Source change luôn đi cùng canonical-doc update ngoài plan. Function Python viết tay trong `src/model_pipeline/` và `test/` dùng Google-style English docstring để mô tả purpose, input, output/yield và exception contract.

## AIMET operator boundary và Qualcomm HTP

Operator-name allowlist là boundary thực thi quantization: service tắt tất cả tensor quantizer, sau đó chỉ bật quantizer gắn với node MatMul được model policy chọn. VPCD yêu cầu selected activation quantizer dùng symmetric signed 16-bit range với offset `-32768`, vì Qualcomm HTP MatMul không chấp nhận asymmetric signed 16-bit offset. Decoder và language-model head không được phép xuất hiện trong VPCD encodings.
