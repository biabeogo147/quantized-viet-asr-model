# Python Model Test Repo

`python-model-test/` la workspace Python de:

- export ONNX tu model source
- dong goi bundle theo contract chung
- verify parity giua reference runtime va bundle runtime
- quantize model cho deployment sau nay
- smoke test artifact truoc khi dua sang Android

## Repo nay giai quyet bai toan gi

Repo hien tai co 2 model family chinh:

- `vpcd`
  - punctuation / capitalization seq2seq
  - output la text da phuc hoi dau/cau truc
- `zipformer`
  - acoustic RNNT model
  - output la transcript

Thay vi de moi model co mot he script rieng, repo da duoc dua ve 4 khoi chinh:

- `export/`
  - entrypoint tao artifact
- `verify/`
  - entrypoint kiem tra parity / mismatch
- `model_bundle/`
  - shared contract cho bundle
- `quantize/`
  - shared framework quantization multi-project

## Cau truc repo

```text
python-model-test/
  assets/                # model source, audio/text fixtures
  build/                 # artifact sinh ra trong qua trinh export/quantize
  docs/                  # plan va tai lieu implementation
  export/                # CLI export
  model_bundle/          # shared bundle contract
  quantize/              # quantization framework
  test/                  # smoke runners + pytest suites
  verify/                # CLI verify
  convert_bpe2token.py   # helper tao tokens.txt tu bpe.model
```

## Module nao lam gi

### `export/`

Tao artifact dau vao:
- bundle chung cho `vpcd` va `zipformer`
- ONNX source cho punctuation

Chi tiet: `export/README.md`

### `verify/`

Kiem tra bundle:
- punctuation encode/decode parity
- Zipformer reference-vs-bundle
- Zipformer reference-vs-candidate

Chi tiet: `verify/README.md`

### `model_bundle/`

Shared core:
- manifest
- fixtures
- generic exporter/verifier
- project adapters

Chi tiet:
- `model_bundle/README.md`
- `model_bundle/projects/README.md`

### `quantize/`

Quantization framework:
- calibration
- presets
- QNN PTQ + QDQ
- reports
- project adapters cho `vpcd` va `zipformer`

Chi tiet:
- `quantize/README.md`
- `quantize/projects/README.md`

### `test/`

Noi chua:
- smoke runners canonical
- pytest suite khoa contract
- reference script cu de so sanh behavior

Chi tiet: `test/README.md`

## Pipeline tong the

### 1. Export source artifact

Punctuation:
- neu can tai tao ONNX source, dung `python -m export.punctuation_onnx`

Bundle:
- dung `python -m export.model_bundle --project <project>`

Output:
- `build/model_bundle/vpcd/...`
- `build/model_bundle/zipformer/...`

### 2. Verify reference bundle

Punctuation:
- `python -m verify.model_bundle --project vpcd --model-dir ... --bundle-dir ...`

Zipformer FP32:
- `python -m verify.model_bundle --project zipformer --model-dir ... --bundle-dir ...`

Muc tieu:
- bundle runtime phai theo dung behavior cua reference runtime

### 3. Quantize candidate bundle

Cho Zipformer:
- `python -m quantize --project zipformer ...`

Pipeline ben trong:
- collect calibration audio
- freeze fixed shapes
- QNN PTQ + QDQ tung component
- export candidate bundle `qnn_u16u8`
- verify candidate voi reference bundle
- ghi `quantization_report.json` va `evaluation_report.json`

### 4. Smoke test artifact

Punctuation:
- `python -m test.test_punctuation_model_onnx --bundle-manifest ...`

Zipformer:
- `python -m test.test_acoustic_model_onnx --bundle-manifest ...`

Muc tieu:
- xac nhan artifact that su chay duoc end-to-end

### 5. Hand-off sang Android

Sau khi bundle reference/candidate on:
- sync bundle sang `bkmeeting/modelassets`
- Android dung cung `bundle_manifest.json` va layout do Python export

## Hai pipeline cu the

### VPCD

`model-dir`
-> Hugging Face tokenizer + ONNX model
-> export tokenizer ONNX + bridge map
-> bundle `vpcd/fp32`
-> verify encode/decode parity
-> smoke run bundle-only

### Zipformer

`model-dir`
-> export FP32 reference bundle
-> verify bundle vs model-dir
-> quantize tung component thanh `qnn_u16u8`
-> export candidate bundle
-> verify candidate vs reference
-> smoke run quantized bundle

## Artifact quan trong trong `build/`

### Punctuation

```text
build/model_bundle/vpcd/fp32/
  bundle_manifest.json
  model.mobile.onnx
  tokenizer.encode.onnx
  tokenizer.decode.onnx
  tokenizer.to_model_id_map.json
  tokenizer.from_model_id_map.json
  golden_samples.jsonl
```

### Zipformer reference

```text
build/model_bundle/zipformer/fp32/
  bundle_manifest.json
  encoder.onnx
  decoder.onnx
  joiner.onnx
  tokens.txt
  sample_manifest.jsonl
  expected_outputs.jsonl
```

### Zipformer quantized candidate

```text
build/model_bundle/zipformer/qnn_u16u8/
  bundle_manifest.json
  encoder.onnx
  decoder.onnx
  joiner.onnx
  tokens.txt
  sample_manifest.jsonl
  expected_outputs.jsonl
  quantization_report.json
  evaluation_report.json
```

## Script nho o root repo

### `convert_bpe2token.py`

Vai tro:
- doc `assets/zipformer/bpe.model`
- sinh `assets/zipformer/tokens.txt`

Dung khi:
- token table bi thieu
- can dong bo lai `tokens.txt` tu SentencePiece model

## Trang thai hien tai

- shared bundle contract da dung chung cho `vpcd` va `zipformer`
- quantize da ho tro 2 project
- candidate `zipformer/qnn_u16u8` da runnable
- candidate nay chua exact-match 100% voi FP32 reference tren bo sample hien tai, nen can tiep tuc tune neu muc tieu la parity chat
