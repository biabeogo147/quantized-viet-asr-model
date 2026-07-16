# Model Pipeline

Repo này là nguồn sự thật duy nhất cho việc chuẩn bị, quantize, validate, compile và đóng gói Zipformer/VPCD cho BKMeeting.

## Public contract

```text
RecipeSpec → source → prepare → quantize/explicit-skip → validate → compile → package → sync
```

CLI công khai duy nhất:

```powershell
python -m model_pipeline run --model zipformer --configuration ortqnn-uint8-uint16-encoder-matmul --through sync
python -m model_pipeline run --model vpcd --configuration aimet-int8-int16-encoder-matmul --through sync
```

Thêm `--dry-run` để xem recipe và stage mà không đọc model hay gọi AI Hub. Configuration `fp32-fixed-shape` là control local; tên configuration mô tả trực tiếp engine, precision, shape, operator scope hoặc compile target. Manifest v2 ghi execution target riêng cho từng component.

## Cài đặt

```powershell
python -m pip install -e ".[onnx,runtime-gpu,datasets,aihub,test]"
python -m pytest -q
```

`runtime-cpu` và `runtime-gpu` là hai lựa chọn loại trừ nhau; không cài đồng thời hai distribution ONNX Runtime. Pipeline pin `1.22.0` vì fixed-shape Zipformer optimizer đã được kiểm chứng ở version này, trong khi `1.26.0` lỗi khi đồng thời cố định batch và time dimensions. Chỉ ghi kết quả GPU khi profiler cho thấy node chạy trên `CUDAExecutionProvider`.

AIMET ONNX chạy trong Docker Linux đã pin dependency; xem [AI Hub → Android operations](docs/aihub-android-operations.md).

## Working agreement

Mọi thay đổi tracked phải tuân theo [AGENTS.md](AGENTS.md): thực hiện read-only discovery, tạo plan từ [template](docs/plans/TEMPLATE.md), cập nhật task trong [active plans](docs/plans/active/), đồng bộ canonical docs, rồi chuyển plan đã kiểm chứng sang [completed plans](docs/plans/completed/).

## Tài liệu canonical

- [Kiến trúc pipeline](docs/architecture.md)
- [Zipformer recipe](docs/zipformer-recipe.md)
- [VPCD recipe](docs/vpcd-recipe.md)
- [AI Hub → Android operations](docs/aihub-android-operations.md)
- [Báo cáo VLSP 100 mẫu và compile Qualcomm HTP](docs/evidence/2026-07-15-vlsp100-quantization-compile.md)
- [Retained artifact evidence](docs/evidence/retained-artifacts.json)

Model adapter ưu tiên model đã materialize trong `assets/`, rồi fallback sang FP32 bundle được track ở repo `BKMeeting` cùng cấp; vì vậy clean clone của workspace hai repo vẫn resolve được source. Các fixture speech/golden nhỏ được track để test và refresh deterministic. `build/` chỉ là cache/output, không phải nguồn sự thật.
