# AI Hub → Android operations

## 1. Chuẩn bị AIMET service

```powershell
docker build -t bkmeeting-aimet -f docker/aimet-onnx-ubuntu2204/Dockerfile .
docker run --rm -p 18080:8080 -v "${PWD}:/workspace" bkmeeting-aimet
```

Image pin AIMET ONNX `2.31.0`. Đây là pin của pipeline mới; retained artifact cũ không ghi lại AIMET version chính xác.

## 2. Dry-run và local stages

```powershell
python -m model_pipeline run --model vpcd --profile production --through validate --dry-run
python -m model_pipeline run --model vpcd --profile production --through validate
```

VLSP calibration text mặc định được materialize tại `build/calibration/vlsp2020/transcriptions.txt`. Source model/tokenizer nằm dưới `assets/` và không được nhúng đường dẫn tuyệt đối vào manifest.

## 3. Compile

Đặt `QAI_HUB_API_TOKEN` trong environment của process, không commit `.env`. Khi chạy qua `compile`, truyền `--device` bằng tên device AI Hub.

```powershell
python -m model_pipeline run --model vpcd --profile production --through package --device "Samsung Galaxy S23 (Family)"
```

Compile record được index bằng checksum package đầu vào. Chỉ reuse khi artifact ID/component trùng và blob đầu ra còn khớp checksum. Retained VPCD input AIMET package lịch sử đã mất, nên record đó chỉ là evidence; không đủ điều kiện auto-reuse.

## 4. Sync BKMeeting

`--android-destination` trỏ tới namespace đích. Sync copy đúng file có trong manifest v2 và xóa file dư trong chính namespace đó. Runtime default của app không bị thay đổi: Zipformer encoder precompiled vẫn là default QNN lane; punctuation default lịch sử vẫn CPU.

Sau sync chạy ít nhất:

```powershell
.\gradlew.bat :app:testDebugUnitTest --no-daemon
.\gradlew.bat :app:assembleCpuCompatDebug --no-daemon
.\gradlew.bat :app:assembleQnnOfficialArm64Debug --no-daemon
```

Nếu model bytes, shape hoặc quantization scope đổi, phải compile/live-run AI Hub và strict Snapdragon lại. Thay đổi manifest/path nhưng giữ bytes không tạo claim hiệu năng mới.
