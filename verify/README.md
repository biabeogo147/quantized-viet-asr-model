# Verify Module

`verify/` chua cac CLI dung de kiem tra artifact da export co con theo dung contract va quality gate hay khong.

## Muc tieu

- cung cap mot diem vao duy nhat de verify bundle theo `project`
- giu cho `quantize` va smoke test khong phai tu viet logic so sanh
- in mismatch report ro rang khi candidate bundle lech reference

## File map

```text
python-model-test/verify/
  __init__.py
  model_bundle.py
  README.md
```

## Tung script giai quyet van de gi

### `model_bundle.py`

Day la CLI canonical de verify bundle.

Van de no giai quyet:
- voi `vpcd`, no verify parity encode/decode giua bundle va Hugging Face tokenizer
- voi `zipformer`, no verify transcript giua model-dir runtime va bundle runtime, hoac giua reference bundle va candidate bundle

Ham chinh:
- `build_argument_parser()`
  - parse `--project`
  - parse 3 kieu input:
    - `--model-dir` + `--bundle-dir`
    - `--reference-bundle` + `--candidate-bundle`
    - hoac mac dinh adapter defaults
- `main(argv=None)`
  - resolve project adapter
  - build kwargs hop le cho project do
  - goi `verify_model_bundle(...)`
  - in summary:
    - encode/decode sample count cho `vpcd`
    - checked samples, passed, mismatches cho `zipformer`

## Cach doc output

- Neu output la:
  - `Encode samples : ...`
  - `Decode samples : ...`
  Thi day la verify tokenizer bundle cua `vpcd`.

- Neu output la:
  - `Checked samples: ...`
  - `Passed : True/False`
  - danh sach `mismatches`
  Thi day la verify transcript cua `zipformer`.

## Lenh hay dung

### Verify punctuation bundle

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m verify.model_bundle `
  --project vpcd `
  --model-dir D:\DS-AI\BKMeeting-Research\python-model-test\assets\vietnamese-punc-cap-denorm-v1 `
  --bundle-dir D:\DS-AI\BKMeeting-Research\python-model-test\build\model_bundle\vpcd\fp32
```

### Verify zipformer FP32 bundle voi model-dir

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m verify.model_bundle `
  --project zipformer `
  --model-dir D:\DS-AI\BKMeeting-Research\python-model-test\assets\zipformer `
  --bundle-dir D:\DS-AI\BKMeeting-Research\python-model-test\build\model_bundle\zipformer\fp32
```

### Verify zipformer quantized candidate voi FP32 reference

```powershell
& D:\Anaconda\envs\speech2text\python.exe -m verify.model_bundle `
  --project zipformer `
  --reference-bundle D:\DS-AI\BKMeeting-Research\python-model-test\build\model_bundle\zipformer\fp32 `
  --candidate-bundle D:\DS-AI\BKMeeting-Research\python-model-test\build\model_bundle\zipformer\qnn_u16u8
```

## Quan he voi module khac

- CLI nay chi la wrapper
- logic verify that su nam trong `model_bundle/verifier.py`
- project-specific comparison nam trong:
  - `model_bundle/projects/vpcd.py`
  - `model_bundle/projects/zipformer.py`
