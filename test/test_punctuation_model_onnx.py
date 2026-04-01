import argparse
import time
from pathlib import Path
from typing import List, Protocol

from model_bundle.projects.vpcd import BundleOnnxRuntime, DEFAULT_MODEL_VARIANT, resolve_variant_onnx_path


MODEL_DIR = str(Path('assets') / 'vietnamese-punc-cap-denorm-v1')

DEFAULT_TEXTS = [
    'hom nay la buoi nham chuc cua toi phuoc thanh',
    'chao cac ban hom nay chung ta cung nhau den voi bai hoc deep learning phan so muoi ba',
]


class PunctuationRuntime(Protocol):
    def restore(self, text: str, max_length: int = 128) -> str:
        ...


class ModelDirOnnxRuntime:
    def __init__(self, *, model_dir: str, onnx_path: str, provider: str = 'CPUExecutionProvider'):
        import json

        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.model_dir = model_dir
        self.onnx_path = onnx_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.session = ort.InferenceSession(onnx_path, providers=[provider])

        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.decoder_start_token_id = self.tokenizer.eos_token_id

        generation_config_path = Path(model_dir) / 'generation_config.json'
        if generation_config_path.exists():
            generation_config = json.loads(generation_config_path.read_text(encoding='utf-8'))
            self.decoder_start_token_id = generation_config.get('decoder_start_token_id', self.decoder_start_token_id)

    def restore(self, text: str, max_length: int = 128) -> str:
        import numpy as np

        encoded = self.tokenizer(text, return_tensors='np', truncation=True, max_length=512)
        input_ids = encoded['input_ids'].astype(np.int64)
        attention_mask = encoded['attention_mask'].astype(np.int64)
        decoder_input_ids = np.array([[self.decoder_start_token_id]], dtype=np.int64)

        for _ in range(max_length):
            decoder_attention_mask = np.ones_like(decoder_input_ids, dtype=np.int64)
            outputs = self.session.run(
                None,
                {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'decoder_input_ids': decoder_input_ids,
                    'decoder_attention_mask': decoder_attention_mask,
                },
            )
            logits = outputs[0]
            next_token_id = int(np.argmax(logits[:, -1, :], axis=-1)[0])
            decoder_input_ids = np.concatenate([decoder_input_ids, np.array([[next_token_id]], dtype=np.int64)], axis=1)
            if next_token_id == self.eos_token_id:
                break

        generated_ids = decoder_input_ids[0, 1:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


VietnamesePuncCapDenormOnnx = ModelDirOnnxRuntime


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run ONNX inference for vietnamese-punc-cap-denorm-v1.')
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--model-dir', help='Model directory containing Hugging Face tokenizer/config files and the onnx/ folder.')
    input_group.add_argument('--bundle-manifest', help='Bundle manifest path for bundle-only runtime mode.')
    parser.add_argument('--model-variant', default=DEFAULT_MODEL_VARIANT, help='ONNX variant under <model-dir>/onnx. Used for model-dir mode only.')
    parser.add_argument('--text', help='Single input text.')
    parser.add_argument('--text-file', help='Text file with one sample per line.')
    parser.add_argument('--max-length', type=int, default=128, help='Maximum generated token length.')
    parser.add_argument('--provider', default='CPUExecutionProvider', help='ONNX Runtime provider, e.g. CPUExecutionProvider or CUDAExecutionProvider.')
    return parser


def load_inputs(args: argparse.Namespace) -> List[str]:
    if args.text:
        return [args.text.strip()]
    if args.text_file:
        with open(args.text_file, 'r', encoding='utf-8') as handle:
            return [line.strip() for line in handle if line.strip()]
    return DEFAULT_TEXTS


def create_runtime(args: argparse.Namespace) -> PunctuationRuntime:
    if args.bundle_manifest:
        return BundleOnnxRuntime.from_manifest_path(args.bundle_manifest, provider=args.provider)
    model_dir = args.model_dir or MODEL_DIR
    onnx_path = resolve_variant_onnx_path(model_dir, args.model_variant)
    return ModelDirOnnxRuntime(model_dir=model_dir, onnx_path=str(onnx_path), provider=args.provider)


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    inputs = load_inputs(args)
    runtime = create_runtime(args)

    for idx, text in enumerate(inputs, start=1):
        started = time.time()
        output = runtime.restore(text, max_length=args.max_length)
        elapsed = time.time() - started
        print(f'\n========== SAMPLE {idx} ==========')
        print('Input   :', text)
        print('Output  :', output)
        print('Latency :', round(elapsed, 3), 's')


if __name__ == '__main__':
    main()


