# Kiến trúc Model Pipeline

`src/model_pipeline/` là package công khai duy nhất.

```text
core                 specs, manifest v2, checksum, cache/resume
models/zipformer     fixed shapes, HTP bool-mask rewrite, graph validation
models/vpcd          A4 calibration, AIMET W8A16 policy, tokenizer/runtime
datasets             VLSP extraction, fixture selection/audio/golden refresh
integrations/aihub   compile client, wait/download/live-run, checksum evidence
integrations/android deterministic bundle và BKMeeting sync
pipeline.py          điều phối bảy stage; được phép phụ thuộc mọi adapter
```

Dependency boundary được khóa bằng test: `core` không import model/integration; model không import integration; integration không import model.

## Artifact identity

Grammar:

```text
<model>__q-<engine>-<weight>-<activation>-<scope>__s-<shape>__c-<compiler>-<target>-<compiled-scope>
```

Allowlist hiện chỉ có hai quantization contract:

- không quantize: `none-fp32-fp32-none`;
- VPCD production: `aimet-int8-int16-encoder-matmul`.

Không có alias theo tên thử nghiệm hay công cụ tương tác cũ. App namespace lịch sử duy nhất được giữ trong `integrations/android/compatibility.py`; nó không parse được thành artifact ID.

## Manifest v2

Mỗi component ghi: `role`, `file`, `format`, `precision`, `input_shapes`, quantization engine/scope, `execution_target`, checksum. Manifest còn ghi source checksum, recipe digest, validation và runtime metadata. Android chọn provider từ `execution_target`, không từ tên thư mục.

## Cache và provenance

Một stage chỉ resume khi artifact ID, recipe digest, toàn bộ input digest và checksum output đều khớp. Package AIMET được hash theo danh sách `{relative path, file checksum}`, không chứa đường dẫn máy. AI Hub evidence resolve bằng checksum package đầu vào và kiểm tra lại blob đầu ra trước khi reuse.

## Development contract

`AGENTS.md` là nguồn quy tắc làm việc duy nhất cho repository. Mọi thay đổi tracked phải có plan đang active trước khi implementation bắt đầu; plan ghi các phương án đã cân nhắc, quyết định, task, verification và canonical docs cần đồng bộ. Plan chỉ được chuyển sang `docs/plans/completed/` sau khi mọi gate đạt.

Source change luôn đi cùng canonical-doc update ngoài plan. Function Python viết tay trong `src/model_pipeline/` và `test/` dùng Google-style English docstring để mô tả purpose, input, output/yield và exception contract.
