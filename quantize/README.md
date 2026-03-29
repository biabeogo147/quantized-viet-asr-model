# Quantize CLI

Module `quantize` dung de quantize ONNX model `vietnamese-punc-cap-denorm-v1` theo nhieu preset phu hop cho Snapdragon 8 Gen 2.

## Entry Point

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m quantize --help
```

Neu `python` da co trong PATH:

```powershell
python -m quantize --help
```

## Presets

- `sd8g2_quality`: static PTQ, uu tien giu quality.
- `sd8g2_balanced`: static PTQ, can bang quality va size.
- `sd8g2_aggressive`: static PTQ, mo rong pham vi quantize.
- `baseline_dynamic_int8`: dynamic INT8 baseline.

## Calibration Source

Static quantization mac dinh se dung thu muc:

```text
quantize/calibration
```

Thu muc nay co the chua nhieu file `.txt`. Moi dong khong rong duoc xem la mot sample calibration.

Ban cung co the truyen:
- mot file txt duy nhat
- hoac mot thu muc chua nhieu file txt

Thong qua:

```powershell
--calibration-source <path>
```

Hoac alias:

```powershell
--calibration-text <path>
```

## Lenh Thuong Dung

Dry-run de xem preset va so node bi exclude:

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m quantize --dry-run --preset sd8g2_quality
```

Static quantization voi calibration mac dinh trong `quantize/calibration`:

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m quantize --preset sd8g2_quality
```

Static quantization voi mot file calibration cu the:

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m quantize --preset sd8g2_quality --calibration-source quantize\calibration\vpcd_calibration_ngan.txt
```

Static quantization voi output rieng:

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m quantize --preset sd8g2_balanced --output test\_tmp\sd8g2_balanced.onnx
```

Dynamic baseline:

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m quantize --preset baseline_dynamic_int8 --output test\_tmp\baseline_dynamic_int8.onnx
```

## Chay Nhieu Preset

Chay lan luot tung preset:

```powershell
$py = "D:\Anaconda\envs\speech2text\python.exe"

& $py -m quantize --preset baseline_dynamic_int8 --output test\_tmp\baseline_dynamic_int8.onnx
& $py -m quantize --preset sd8g2_quality --output test\_tmp\sd8g2_quality.onnx
& $py -m quantize --preset sd8g2_balanced --output test\_tmp\sd8g2_balanced.onnx
& $py -m quantize --preset sd8g2_aggressive --output test\_tmp\sd8g2_aggressive.onnx
```

Hoac loop cho 3 preset static:

```powershell
$py = "D:\Anaconda\envs\speech2text\python.exe"

foreach ($preset in @("sd8g2_quality", "sd8g2_balanced", "sd8g2_aggressive")) {
    & $py -m quantize --preset $preset --output "test\_tmp\$preset.onnx"
}
```

Nen chay tuan tu, khong nen chay song song nhieu job quantize cung luc.

## Mot Vai Option Quan Trong

- `--max-calibration-samples`: gioi han so sample calibration duoc doc.
- `--max-generation-length`: gioi han do dai decoder khi tao calibration records.
- `--percentile`: threshold cho calibration method `percentile`.
- `--calibration-method`: `minmax`, `entropy`, `percentile`, `distribution`.
- `--per-channel` hoac `--no-per-channel`: override setting cua preset.
- `--extra-exclude-pattern`: bo sung pattern node can giu FP32.
- `--size-budget-mb`: muc tieu size de in PASS/FAIL sau quantize.

## Vi Du Smoke Run

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m quantize --preset sd8g2_quality --max-calibration-samples 4 --max-generation-length 8 --output test\_tmp\model.static.smoke.onnx
```

Sau khi chay xong, CLI se in:
- preset dang dung
- calibration stats
- duong dan output
- kich thuoc file
- size budget PASS/FAIL
- goi y tinh chinh tiep theo
