# sd8g2_balanced QNN QDQ Plan

## Goal

Chuyen preset `sd8g2_balanced` sang `PTQ + QDQ` cho Qualcomm QNN/HTP, chi dung:

- `activation_type = QUInt16`
- `weight_type = QUInt8`

Preset nay uu tien output doc duoc va deploy-NPU phu hop hon FP16, trong khi van co co hoi dua size ve gan muc `~800 MB`.

## Constraints

- Khong dung `QInt8`, `QInt16`, `FP16` cho `sd8g2_balanced`
- Phai dung `QDQ`
- Phai co calibration text
- Phai giu lai nhung node nhay cam nhat o FP32 de tranh vo decoder

## Model Findings

Nhung nhom `MatMul` chinh trong model:

- encoder self-attn: 72
- encoder fc1: 12
- encoder fc2: 12
- decoder self-attn: 72
- decoder encoder-attn: 72
- decoder fc1: 12
- decoder fc2: 12
- lm_head: 1

Nhung vung nhay cam nhat:

- `/lm_head/MatMul`
- `decoder/*/self_attn/*`
- `decoder/*/encoder_attn/*`

Nhung vung co the noi quantize dan:

- encoder self-attn q/k/v/out va fc1/fc2
- decoder fc1/fc2
- layer-thap cua decoder feed-forward

## Balanced Ladder

### Balanced-A

Exclude:

- chi giu decoder attention score o FP32:
  - `decoder/*/self_attn/{MatMul,MatMul_1}`
  - `decoder/*/encoder_attn/{MatMul,MatMul_1}`
- `/lm_head/MatMul`

Quantize:

- toan bo encoder
- q/k/v/out_proj cua decoder attention
- toan bo decoder `fc1/fc2`

### Balanced-B

Giong `Balanced-A`, nhung giu them `fc1/fc2` cua 4 layer cuoi decoder o FP32:

- `/model/decoder/layers.8/fc1/MatMul`
- `/model/decoder/layers.8/fc2/MatMul`
- ...
- `/model/decoder/layers.11/fc1/MatMul`
- `/model/decoder/layers.11/fc2/MatMul`

### Balanced-C

Giong `Balanced-B`, va day la muc mac dinh cho preset `sd8g2_balanced` sau khi thu nghiem:

- quantize encoder
- quantize q/k/v/out_proj cua decoder attention
- quantize decoder `fc1/fc2` cua `layers.0-7`
- giu decoder attention score (`MatMul`, `MatMul_1`), `lm_head`, va decoder `fc1/fc2` cua `layers.8-11` o FP32

## Implementation

1. Tao runner `qnn_static`
2. Dung `qnn_preprocess_model()` truoc quantization
3. Dung `get_qnn_qdq_config()` de sinh `StaticQuantConfig`
4. Dung `quantize()` thay vi tu xay config bang tay
5. Giu `calibration_providers = ["CPUExecutionProvider"]`
6. Dung calibration stride an toan de giam peak RAM
7. Chon `Balanced-C` lam exclude mac dinh cho `sd8g2_balanced`
8. Verify them mot lan voi output thuc te de dam bao khong bi collapse decoder thanh dau cau hoac ky tu rac

## Verification

Can verify:

- `python -m quantize --dry-run --preset sd8g2_balanced`
- quantize that su tren calibration nho
- infer CPU tren it nhat 2 cau mau
- output khong duoc co:
  - `<unk>` lap
  - chuoi dau cau vo nghia
  - ky tu rac khong doc duoc
- output duoc phep lech phrasing nhe, nhung phai con doc duoc va khong collapse
- do size output va so sanh voi target `~800 MB`

## Expected Outcome

- `sd8g2_balanced` tro thanh preset `QNN-friendly`
- output on dinh hon cac ban INT8 static truoc day
- size nam trong vung trung gian giua `quality` va `dynamic baseline`
