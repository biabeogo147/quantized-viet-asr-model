from pathlib import Path


DEFAULT_MODEL_DIR = Path("assets") / "vietnamese-punc-cap-denorm-v1"
DEFAULT_FP32_ONNX = DEFAULT_MODEL_DIR / "onnx" / "model.fp32.onnx"
DEFAULT_OUTPUT_ONNX = DEFAULT_MODEL_DIR / "onnx" / "model.int8.sd8g2.onnx"
DEFAULT_DYNAMIC_OUTPUT_ONNX = DEFAULT_MODEL_DIR / "onnx" / "model.dynamic.int8.sd8g2.onnx"
DEFAULT_CALIBRATION_SOURCE = Path("quantize") / "calibration_400_cau"
DEFAULT_ORT_PROVIDER = "cuda"

DEFAULT_MAX_CALIBRATION_SAMPLES = 128
DEFAULT_MAX_GENERATION_LENGTH = 128
DEFAULT_CALIBRATION_CHUNK_SIZE = 32
DEFAULT_SIZE_BUDGET_MB = 500.0
DEFAULT_PERCENTILE = 99.99
DEFAULT_TEMP_ROOT = Path("test") / "_tmp" / "ort_quant_temp"
