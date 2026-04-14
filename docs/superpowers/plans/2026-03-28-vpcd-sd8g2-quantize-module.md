# VPCD Snapdragon 8 Gen 2 Quantize Module

Trang thai hien tai cua module `quantize` da o dang package chinh thuc, khong con giu wrapper `quantize_vpcd.py` hay unit test rieng cho pha migrate.

## Entry point

Chay module bang:

```powershell
python -m quantize --help
```

Hoac chi dinh Python environment cu the:

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m quantize --help
```

## Cau truc package

- `quantize/config.py`: duong dan mac dinh, gioi han size, calibration defaults, temp root.
- `quantize/types.py`: cac dataclass co ban cho calibration va preset metadata.
- `quantize/presets.py`: preset Snapdragon 8 Gen 2 va logic build `QuantizationPlan`.
- `quantize/model_introspection.py`: doc graph ONNX va in summary plan.
- `quantize/calibration.py`: tao calibration records cho static PTQ.
- `quantize/runtime.py`: temp-workspace workaround cho ONNX Runtime tren Windows.
- `quantize/runner.py`: thuc thi static/dynamic quantization, report size va goi y buoc tiep.
- `quantize/cli.py`: parse args va dieu phoi toan bo flow.
- `quantize/__main__.py`: entrypoint package.

## Preset dang ho tro

- `sd8g2_quality`: static PTQ, giu nhieu node nhay cam o FP32.
- `sd8g2_balanced`: static PTQ, giam bot exclusion de benchmark.
- `sd8g2_aggressive`: static PTQ, chi giu lai `lm_head`.
- `baseline_dynamic_int8`: dynamic INT8 baseline.

## Luong chay

1. `python -m quantize ...`
2. `quantize.cli.parse_args()`
3. `quantize.model_introspection.load_model_node_names()`
4. `quantize.presets.build_quantization_plan()`
5. Neu `--dry-run`: in summary va dung
6. Neu preset la `static`: dung `quantize.calibration.build_calibration_records()`
7. `quantize.runner.run_static_quantization()` hoac `run_dynamic_quantization()`
8. In output size, size budget va goi y tuning

## Nhung gi da bo di

- Wrapper `quantize_vpcd.py`
- Unit test file `test/test_quantize_vpcd.py`
- Plan migrate cu trong `docs/superpowers/plans/2026-03-26-quantize-vpcd.md`

## Ghi chu van hanh

- Module da duoc smoke-verify bang dry-run va quantization that su.
- Dynamic baseline can temp-workspace workaround giong static de tranh loi quyen tren Windows.
- Neu muc tieu van la < 500 MB thi can them mot chien luoc nen khac ngoai selective INT8 QDQ hien tai, vi ba preset static dang cho kich thuoc gan nhu nhau.
