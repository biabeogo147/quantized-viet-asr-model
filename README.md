# quantized-viet-asr-model

`quantized-viet-asr-model` là nguồn sự thật duy nhất cho việc chuẩn bị, quantize, validate, compile và đóng gói Zipformer/VPCD cho BKMeeting.

## Bắt đầu từ đây

Dev mới đọc [documentation index](docs/README.md), sau đó hoàn thành [local walkthrough](docs/getting-started.md) trước khi thay đổi model hoặc pipeline. Index cung cấp tuyến đọc chung, nhánh theo vai trò, source-code tour và tiêu chí hoàn thành onboarding.

Clean clone không chứa model binary, VLSP parquet hoặc output `build/`. Team phải cấp các asset ngoài Git cùng checksum baseline; walkthrough dùng placeholder portable và giải thích đúng layout cần materialize. Sibling BKMeeting cũng không tự biến binary bị ignore thành tracked source.

## Public contract

```text
RecipeSpec → source → prepare → quantize/explicit-skip → validate → compile → package → sync
```

Các CLI công khai:

```bash
python -m model_pipeline run --model zipformer --configuration ortqnn-uint8-uint16-encoder-matmul --through sync
python -m model_pipeline run --model vpcd --configuration aimet-int8-int16-encoder-matmul --through sync
python -m model_pipeline android-model-repository --build-root build/android-integration --destination <BKMEETING_ROOT>/modelassets/src/main/assets/model-repository --dry-run
```

Thêm `--dry-run` để xem recipe và stage mà không đọc model hay gọi AI Hub. Configuration `fp32-fixed-shape` là control local; tên configuration mô tả trực tiếp engine, precision, shape, operator scope hoặc compile target. Manifest v2 ghi execution target riêng cho từng component.

`runtime-cpu` và `runtime-gpu` là hai lựa chọn loại trừ nhau; không cài đồng thời hai distribution ONNX Runtime. Pipeline pin `1.22.0` vì fixed-shape Zipformer optimizer đã được kiểm chứng ở version này, trong khi `1.26.0` lỗi khi đồng thời cố định batch và time dimensions. Chỉ ghi kết quả GPU khi profiler cho thấy node chạy trên `CUDAExecutionProvider`.

AIMET ONNX chạy trong Docker Linux đã pin dependency; xem [AI Hub → Android operations](docs/aihub-android-operations.md).

## Working agreement

Mọi thay đổi tracked phải tuân theo [AGENTS.md](AGENTS.md): thực hiện read-only discovery, tạo plan từ [template](docs/plans/TEMPLATE.md), cập nhật task trong [active plans](docs/plans/active/), đồng bộ canonical docs, rồi chuyển plan đã kiểm chứng sang [completed plans](docs/plans/completed/).

## Tài liệu canonical

- [Documentation index và learning paths](docs/README.md)
- [Local getting started](docs/getting-started.md)
- [Kiến trúc pipeline](docs/architecture.md)
- [Source-code guide](docs/source-code-guide.md)
- [Zipformer recipe](docs/zipformer-recipe.md)
- [VPCD recipe](docs/vpcd-recipe.md)
- [AI Hub → Android operations](docs/aihub-android-operations.md)
- [Báo cáo VLSP 100 mẫu và compile Qualcomm HTP](docs/evidence/2026-07-15-vlsp100-quantization-compile.md)
- [Retained artifact evidence](docs/evidence/retained-artifacts.json)
- [QDC Appium CPU–NPU benchmark](docs/evidence/2026-07-17-qdc-appium-cpu-npu-performance.md)

Model adapter ưu tiên model đã materialize trong `assets/`, rồi fallback sang FP32 bundle đã materialize ở repo `BKMeeting` cùng cấp. Fallback là filesystem convention, không phải downloader và không bảo đảm clean clone có binary. Các fixture speech/golden nhỏ được track để test và refresh deterministic. `build/` chỉ là cache/output, không phải nguồn sự thật.
