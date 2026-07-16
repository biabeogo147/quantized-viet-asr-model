# Getting started: chạy local Model Pipeline

Tài liệu này cung cấp bản đồ để dev mới chạy Zipformer và VPCD đến local validation. Nó chỉ mô tả prerequisite và lệnh đặc thù của `quantized-viet-asr-model`; các thao tác Python, Git, Docker hoặc shell cơ bản được giả định là đã biết.

## Bức tranh local

```text
model assets + VLSP
→ fixed-shape preparation
→ AIMET quantization
→ graph/encoding validation
→ local artifact và evidence dưới build/
```

Local walkthrough không gọi Qualcomm AI Hub, không cần API token và không tạo Android deployment package. Compile cloud chỉ bắt đầu sau khi local validation đạt.

## Workspace và dữ liệu ngoài Git

Layout tham chiếu:

```text
<WORKSPACE_ROOT>/
├── quantized-viet-asr-model/
└── BKMeeting/
```

Git chỉ chứa source, test, docs và fixture nhỏ. Team cung cấp riêng:

- `<MODEL_ASSET_ROOT>/zipformer/`: encoder, decoder, joiner FP32 và token table;
- `<MODEL_ASSET_ROOT>/vpcd/`: model FP32 và tokenizer/config gốc;
- `<VLSP_PARQUET_ROOT>/`: VLSP parquet shards đã được cấp quyền sử dụng;
- checksum baseline để xác nhận đúng model revision.

Materialize model theo contract adapter:

```text
assets/zipformer/
├── encoder-epoch-20-avg-1.onnx
├── decoder-epoch-20-avg-1.onnx
├── joiner-epoch-20-avg-1.onnx
└── tokens.txt

assets/vietnamese-punc-cap-denorm-v1/
├── onnx/model.fp32.onnx
├── sentencepiece.bpe.model
├── tokenizer_config.json
├── special_tokens_map.json
├── dict.txt
├── generation_config.json
└── config.json
```

Khi local assets không đầy đủ, adapter thử FP32 bundle đã materialize dưới sibling BKMeeting. Đây chỉ là filesystem fallback, không phải downloader. Không trộn component từ các model revision khác nhau và không commit binary/VLSP/output `build/`.

## Runtime contract

Project yêu cầu Python 3.10+ và đúng một ONNX Runtime distribution:

- `runtime-cpu` cho CPU-only host;
- `runtime-gpu` cho CUDA host;
- version 1.22.0 được pin vì Zipformer fixed-shape optimizer đã được kiểm chứng ở version này.

GPU chỉ được ghi nhận khi profiler cho thấy node thực thi trên `CUDAExecutionProvider`; provider xuất hiện trong danh sách chưa đủ chứng minh GPU execution.

## VLSP calibration/evaluation

Dataset module stream parquet và materialize:

- 24 calibration samples từ shard đầu;
- 100 evaluation samples từ shard sau;
- evaluation audio 2–12 giây, transcription 4–40 từ;
- không trùng shard/row/transcription giữa hai partition.

Manifest canonical nằm tại:

```text
build/datasets/vlsp/vlsp-calibration-evaluation-manifest.json
```

Nếu cần materialize từ raw shards, gọi `iter_vlsp_rows`, `select_vlsp_calibration_evaluation` và `write_vlsp_calibration_evaluation` trong `model_pipeline.datasets`. Output manifest phải giữ relative audio paths và checksum; VPCD calibration text lấy từ 24 calibration transcriptions.

## AIMET service

AIMET chạy trong Linux container; host truyền những path nằm dưới repository mount.

```bash
docker build -t bkmeeting-aimet -f docker/aimet-onnx-ubuntu2204/Dockerfile .
docker run --rm --name bkmeeting-aimet-onboarding \
  -p 18080:8080 \
  -v "$PWD:/workspace" \
  bkmeeting-aimet
```

Service mặc định ở `http://127.0.0.1:18080` và phải trả `status=ok` tại `/healthz`. Image pin AIMET/ONNX/ONNX Runtime cùng CPU-only Torch dependencies; không thay bằng AIMET host package tùy ý.

## Dry-run

Dry-run là điểm bắt đầu để kiểm tra recipe, artifact ID và stage list mà không đọc model hoặc gọi service/cloud:

```bash
python -m model_pipeline run \
  --model zipformer \
  --configuration aimet-int8-int16-encoder-matmul \
  --through validate \
  --build-root build/onboarding-model-pipeline \
  --dry-run

python -m model_pipeline run \
  --model vpcd \
  --configuration aimet-int8-int16-encoder-matmul \
  --through validate \
  --build-root build/onboarding-model-pipeline \
  --dry-run
```

Hai kết quả phải liệt kê `source → prepare → quantize → validate`, với `quantize=aimet`. Không truyền `--device` ở local walkthrough.

## Chạy Zipformer đến validate

```bash
python -m model_pipeline run \
  --model zipformer \
  --configuration aimet-int8-int16-encoder-matmul \
  --through validate \
  --build-root build/onboarding-model-pipeline
```

Contract cần thấy:

- encoder input `1×2009×80`;
- 278/278 encoder MatMul thuộc quantization scope;
- signed 8-bit weight, signed 16-bit activation, MinMax, không per-channel;
- decoder/joiner FP32 CPU;
- boolean-mask rewrite đã áp dụng;
- validation `passed`.

## Chạy VPCD đến validate

```bash
python -m model_pipeline run \
  --model vpcd \
  --configuration aimet-int8-int16-encoder-matmul \
  --through validate \
  --build-root build/onboarding-model-pipeline
```

Contract cần thấy:

- source input `1×384`, decoder input `1×64`;
- MatMul inventory `96 encoder / 168 decoder / 1 language-model head`;
- chỉ 96 encoder MatMul thuộc quantization policy;
- decoder, language-model head, tokenizer và autoregressive loop giữ nguyên/CPU;
- selected activation encodings là symmetric signed 16-bit với offset `-32768`;
- validation `passed`.

VPCD AIMET export cần nhiều tài nguyên hơn Zipformer. Clean verification gần nhất mất khoảng 13 phút, dùng gần 9 GiB container memory và tạo hơn 3 GiB intermediate/output. Đây là capacity reference, không phải benchmark.

## Cache và điều tra lỗi

Chạy lại cùng recipe/build root phải resume những stage có artifact ID, recipe digest, input digest và output checksum không đổi. Dùng build root mới khi cần clean comparison; không xóa retained evidence chỉ để buộc rerun.

Điều tra theo boundary đầu tiên sai:

```text
source inventory/checksum
→ prepared graph
→ calibration + quantization policy
→ AIMET model/encodings
→ validation.json
→ stage-state.json
```

Các nhóm lỗi thường gặp:

- thiếu hoặc trộn model assets;
- thiếu VLSP manifest/calibration text;
- AIMET service/mount không đúng;
- cài chồng ONNX Runtime CPU/GPU hoặc sai version;
- CUDA provider có mặt nhưng graph thực thi trên CPU;
- thiếu RAM/disk cho VPCD export;
- graph count, encoding scope hoặc mask rewrite không đạt contract.

Đọc [architecture](architecture.md) để hiểu boundaries và [source-code guide](source-code-guide.md) để tìm đúng owner/test. Chỉ chuyển sang [AI Hub → Android operations](aihub-android-operations.md) sau khi local validation đạt.
