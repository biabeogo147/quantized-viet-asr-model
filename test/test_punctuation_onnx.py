import argparse
import os
import time
from typing import List

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


MODEL_DIR = os.path.join("assets", "vietnamese-punc-cap-denorm-v1")
ONNX_PATH = os.path.join(MODEL_DIR, "onnx", "vpcd_quality.onnx")

DEFAULT_TEXTS = [
    "hôm nay là buổi nhậm chức của tôi phước thành",
    "chào các bạn hôm nay chúng ta cùng nhau đến với bài học deep learning phần số mười ba đáng lý bài này đã học từ ngày hai mốt tháng mười hai năm hai nghìn không trăm hai mươi lăm nhưng vì nghỉ tết chúng ta dời lịch đến ngày hai mươi hai tháng hai năm hai nghìn không trăm hai mươi sáu",
]


class VietnamesePuncCapDenormOnnx:
    def __init__(self, model_dir: str, onnx_path: str, provider: str = "CPUExecutionProvider"):
        self.model_dir = model_dir
        self.onnx_path = onnx_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.session = ort.InferenceSession(onnx_path, providers=[provider])

        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        generation_config_path = os.path.join(model_dir, "generation_config.json")
        self.decoder_start_token_id = self.tokenizer.eos_token_id
        if os.path.exists(generation_config_path):
            import json

            with open(generation_config_path, "r", encoding="utf-8") as f:
                generation_config = json.load(f)
            self.decoder_start_token_id = generation_config.get(
                "decoder_start_token_id",
                self.decoder_start_token_id,
            )

    def restore(self, text: str, max_length: int = 128) -> str:
        encoded = self.tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=512,
        )

        input_ids = encoded["input_ids"].astype(np.int64)
        attention_mask = encoded["attention_mask"].astype(np.int64)
        decoder_input_ids = np.array([[self.decoder_start_token_id]], dtype=np.int64)

        for _ in range(max_length):
            decoder_attention_mask = np.ones_like(decoder_input_ids, dtype=np.int64)
            outputs = self.session.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "decoder_input_ids": decoder_input_ids,
                    "decoder_attention_mask": decoder_attention_mask,
                },
            )

            logits = outputs[0]
            next_token_id = int(np.argmax(logits[:, -1, :], axis=-1)[0])
            decoder_input_ids = np.concatenate(
                [decoder_input_ids, np.array([[next_token_id]], dtype=np.int64)],
                axis=1,
            )

            if next_token_id == self.eos_token_id:
                break

        generated_ids = decoder_input_ids[0, 1:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def load_inputs(args: argparse.Namespace) -> List[str]:
    if args.text:
        return [args.text.strip()]

    if args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    return DEFAULT_TEXTS


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ONNX inference for vietnamese-punc-cap-denorm-v1.")
    parser.add_argument("--text", help="Single input text.")
    parser.add_argument("--text-file", help="Text file with one sample per line.")
    parser.add_argument("--max-length", type=int, default=128, help="Maximum generated token length.")
    parser.add_argument(
        "--provider",
        default="CPUExecutionProvider",
        help="ONNX Runtime provider, e.g. CPUExecutionProvider or CUDAExecutionProvider.",
    )
    args = parser.parse_args()

    if not os.path.exists(ONNX_PATH):
        raise FileNotFoundError(f"Khong tim thay file ONNX: {ONNX_PATH}")

    inputs = load_inputs(args)
    model = VietnamesePuncCapDenormOnnx(
        model_dir=MODEL_DIR,
        onnx_path=ONNX_PATH,
        provider=args.provider,
    )

    for idx, text in enumerate(inputs, start=1):
        started = time.time()
        output = model.restore(text, max_length=args.max_length)
        elapsed = time.time() - started

        print(f"\n========== SAMPLE {idx} ==========")
        print("Input   :", text)
        print("Output  :", output)
        print("Latency :", round(elapsed, 3), "s")


if __name__ == "__main__":
    main()
