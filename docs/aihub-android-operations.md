# AI Hub → Android operations

## 1. Chuẩn bị AIMET service

```powershell
docker build -t bkmeeting-aimet -f docker/aimet-onnx-ubuntu2204/Dockerfile .
docker run --rm -p 18080:8080 -v "${PWD}:/workspace" bkmeeting-aimet
```

Image pin AIMET ONNX `2.31.0`, ONNX `1.20.0`, ONNX Runtime `1.22.0`, NumPy `1.26.4`, Torch CPU `2.13.0` và Torchvision CPU `0.28.0`. CPU wheels được cài từ PyTorch CPU index trước AIMET để service không tự resolve CUDA runtime không cần thiết. Đây là pin của pipeline mới; retained artifact cũ không ghi lại AIMET version chính xác.

## 2. Dry-run và local stages

```powershell
python -m model_pipeline run --model vpcd --configuration aimet-int8-int16-encoder-matmul --through validate --dry-run
python -m model_pipeline run --model vpcd --configuration aimet-int8-int16-encoder-matmul --through validate
```

VLSP calibration text mặc định được materialize tại `build/calibration/vlsp2020/transcriptions.txt`. Source model/tokenizer nằm dưới `assets/` và không được nhúng đường dẫn tuyệt đối vào manifest.

## 3. Compile

Đặt `QAI_HUB_API_TOKEN` trong environment của process, không commit `.env`. Khi chạy qua `compile`, truyền `--device` bằng tên device AI Hub.

```powershell
python -m model_pipeline run --model vpcd --configuration aimet-int8-int16-encoder-matmul --through package --device "Samsung Galaxy S23 (Family)"
```

Compile record được index bằng checksum package đầu vào. Chỉ reuse khi artifact ID/component trùng và blob đầu ra còn khớp checksum. Retained VPCD input AIMET package lịch sử đã mất, nên record đó chỉ là evidence; không đủ điều kiện auto-reuse.

Sau download, validator bắt buộc kiểm tra checksum, node `EPContext`, dtype của từng input/output, execution target `qnn-htp` và quantization scope lấy từ artifact ID. VPCD và Zipformer `x_lens` đều phải đổi input target từ int64 sang int32; nếu còn `int64` thì `truncate_64bit_io` chưa đạt contract và package bị từ chối.

Hosted validation nhận từ một đến tối đa năm input độc lập cho mỗi model. Quota được kiểm tra trước lần submit đầu tiên. Mỗi kết quả được lưu bằng checksum tính trên tensor name, dtype, shape và bytes của input; không dùng job name làm identity. Zipformer so sánh transcript sau khi decode bằng decoder/joiner FP32 local, còn VPCD so sánh top-1 trên teacher-forced prefix.

## 4. Sync BKMeeting

`--android-destination` trỏ tới namespace đích. Sync copy đúng file có trong manifest v2 và xóa file dư trong chính namespace đó. Runtime default của app không bị thay đổi: Zipformer encoder precompiled vẫn là QNN default; punctuation default lịch sử vẫn CPU.

Sau sync chạy ít nhất:

```powershell
.\gradlew.bat :app:testDebugUnitTest --no-daemon
.\gradlew.bat :app:assembleCpuCompatDebug --no-daemon
.\gradlew.bat :app:assembleQnnOfficialArm64Debug --no-daemon
```

Nếu model bytes, shape hoặc quantization scope đổi, phải compile/live-run AI Hub và strict Snapdragon lại. Thay đổi manifest/path nhưng giữ bytes không tạo claim hiệu năng mới.

## Retained clean-rebuild evidence

Đặt `QAIRT_VERSION=2.45` trước compile để job ghi rõ toolchain trong options; job thiếu pin này không được dùng làm retained evidence. Checksum-keyed records canonical hiện là Zipformer compile job `jp1vnn07p` và VPCD compile job `jgn71e3rp`. Mỗi model đã chạy đúng 5 hosted inputs, không vượt quota. Xem [báo cáo VLSP](evidence/2026-07-15-vlsp100-quantization-compile.md) để biết checksums và inference job IDs.
