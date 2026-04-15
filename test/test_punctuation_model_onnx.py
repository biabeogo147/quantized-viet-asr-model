import argparse
import time
from typing import List, Protocol

from model_bundle.projects.vpcd import (
    BundleOnnxRuntime,
    DEFAULT_MODEL_DIR,
    DEFAULT_MODEL_VARIANT,
    DEFAULT_TEXTS as VPCD_DEFAULT_TEXTS,
    ModelDirOnnxRuntime,
    VietnamesePuncCapDenormOnnx,
    resolve_variant_onnx_path,
)


MODEL_DIR = str(DEFAULT_MODEL_DIR)
DEFAULT_TEXTS = list(VPCD_DEFAULT_TEXTS)


class PunctuationRuntime(Protocol):
    def restore(self, text: str, max_length: int = 128) -> str:
        ...


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
