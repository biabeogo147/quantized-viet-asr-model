# Model Pipeline

Repo này là nguồn sự thật duy nhất cho việc chuẩn bị, quantize, validate, compile và đóng gói Zipformer/VPCD cho BKMeeting.

## Public contract

```text
RecipeSpec → source → prepare → quantize/explicit-skip → validate → compile → package → sync
```

CLI công khai duy nhất:

```powershell
python -m model_pipeline run --model zipformer --profile production --through sync
python -m model_pipeline run --model vpcd --profile production --through sync
```

Thêm `--dry-run` để xem recipe và stage mà không đọc model hay gọi AI Hub. Profile `fp32` là control; profile `production` không có nghĩa mọi component đều chạy NPU. Manifest v2 ghi execution target riêng cho từng component.

## Cài đặt

```powershell
python -m pip install -e ".[onnx,datasets,aihub,test]"
python -m pytest -q
```

AIMET ONNX chạy trong Docker Linux đã pin dependency; xem [AI Hub → Android operations](docs/aihub-android-operations.md).

## Working agreement

Mọi thay đổi tracked phải tuân theo [AGENTS.md](AGENTS.md): thực hiện read-only discovery, tạo plan từ [template](docs/plans/TEMPLATE.md), cập nhật task trong [active plans](docs/plans/active/), đồng bộ canonical docs, rồi chuyển plan đã kiểm chứng sang [completed plans](docs/plans/completed/).

## Tài liệu canonical

- [Kiến trúc pipeline](docs/architecture.md)
- [Zipformer recipe](docs/zipformer-recipe.md)
- [VPCD recipe](docs/vpcd-recipe.md)
- [AI Hub → Android operations](docs/aihub-android-operations.md)
- [Retained artifact evidence](docs/evidence/retained-artifacts.json)

Model adapter ưu tiên model đã materialize trong `assets/`, rồi fallback sang FP32 bundle được track ở repo `BKMeeting` cùng cấp; vì vậy clean clone của workspace hai repo vẫn resolve được source. Các fixture speech/golden nhỏ được track để test và refresh deterministic. `build/` chỉ là cache/output, không phải nguồn sự thật.
