import argparse
import sys
import time
from pathlib import Path
from typing import List, Protocol

from model_bundle.projects.zipformer import (
    BundleAcousticRuntime,
    DEFAULT_AUDIO_FIXTURES,
    DEFAULT_MODEL_DIR,
    ModelDirAcousticRuntime,
)
from tools.paths import resolve_repo_path


class AcousticRuntime(Protocol):
    def transcribe(self, audio_path: str | Path) -> dict:
        ...


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run ONNX inference for the Zipformer acoustic model.')
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--model-dir', help='Model directory containing encoder/decoder/joiner/tokens.')
    input_group.add_argument('--bundle-manifest', help='Bundle manifest path for bundle-only runtime mode.')
    parser.add_argument('--audio-file', help='Single audio input.')
    parser.add_argument('--audio-manifest', help='Text file with one audio path per line.')
    parser.add_argument(
        '--provider',
        default='CPUExecutionProvider',
        help='ONNX Runtime provider, e.g. CPUExecutionProvider or CUDAExecutionProvider.',
    )
    return parser


def load_inputs(args: argparse.Namespace) -> List[Path]:
    if args.audio_file:
        return [Path(args.audio_file)]
    if args.audio_manifest:
        manifest_path = Path(args.audio_manifest)
        return [Path(line.strip()) for line in manifest_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    return [resolve_repo_path(fixture.audio_path, anchor=__file__) for fixture in DEFAULT_AUDIO_FIXTURES]


def create_runtime(args: argparse.Namespace) -> AcousticRuntime:
    if args.bundle_manifest:
        return BundleAcousticRuntime.from_manifest_path(args.bundle_manifest, provider=args.provider)
    model_dir = args.model_dir or str(DEFAULT_MODEL_DIR)
    return ModelDirAcousticRuntime(model_dir=model_dir, provider=args.provider)


def main() -> None:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    parser = build_argument_parser()
    args = parser.parse_args()
    runtime = create_runtime(args)

    for index, audio_path in enumerate(load_inputs(args), start=1):
        started = time.time()
        result = runtime.transcribe(audio_path)
        elapsed = time.time() - started
        print(f'\n========== SAMPLE {index} ==========')
        print('Audio   :', audio_path)
        print('Output  :', result['text'])
        print('Tokens  :', result['num_tokens'])
        print('Latency :', round(elapsed, 3), 's')


if __name__ == '__main__':
    main()
