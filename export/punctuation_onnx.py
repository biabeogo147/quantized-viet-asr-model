import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from typing import Sequence

from onnxruntime.quantization import QuantType, quantize_dynamic


DEFAULT_MODEL_DIR = os.path.join('assets', 'vietnamese-punc-cap-denorm-v1')
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_MODEL_DIR, 'onnx')
DEFAULT_FP32_NAME = 'model.fp32.onnx'
DEFAULT_INT8_NAME = 'model.int8.onnx'


def has_local_transformers_onnx() -> bool:
    return importlib.util.find_spec('transformers.onnx') is not None


def can_run_transformers_onnx(python_exe: str) -> bool:
    try:
        result = subprocess.run(
            [
                python_exe,
                '-c',
                "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('transformers.onnx') else 1)",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0
    except OSError:
        return False


def resolve_export_python(preferred_python: str | None) -> str:
    candidates: list[str] = []
    if preferred_python:
        candidates.append(preferred_python)

    candidates.extend(
        exe
        for exe in [
            sys.executable,
            shutil.which('python'),
            r'D:\laragon\bin\python\python-3.10\python.exe',
        ]
        if exe
    )

    seen = set()
    for candidate in candidates:
        normalized = os.path.abspath(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        if can_run_transformers_onnx(normalized):
            return normalized

    raise RuntimeError(
        "Khong tim thay Python co module 'transformers.onnx'. Hay cai dat mot phien ban transformers ho tro ONNX, hoac chi dinh --export-python toi interpreter phu hop."
    )


def build_command(export_python: str, model_dir: str, output_dir: str, opset: int, atol: float) -> list[str]:
    return [
        export_python,
        '-m',
        'transformers.onnx',
        '--model',
        model_dir,
        '--feature',
        'seq2seq-lm',
        '--framework',
        'pt',
        '--opset',
        str(opset),
        '--atol',
        str(atol),
        '--export_with_transformers',
        output_dir,
    ]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Export assets/vietnamese-punc-cap-denorm-v1 to ONNX.')
    parser.add_argument('--model-dir', default=DEFAULT_MODEL_DIR, help='Local model directory.')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR, help='Output directory for ONNX files.')
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset version.')
    parser.add_argument('--atol', type=float, default=5e-5, help='Validation tolerance used by transformers.onnx.')
    parser.add_argument('--clean', action='store_true', help='Delete the existing output directory before exporting.')
    parser.add_argument('--skip-int8', action='store_true', help='Only export the FP32 ONNX file and skip INT8 quantization.')
    parser.add_argument('--export-python', help='Python executable used for the ONNX export step. If omitted, the script will auto-detect one.')
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argument_parser().parse_args(argv)

    model_dir = os.path.abspath(args.model_dir)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f'Khong tim thay model dir: {model_dir}')

    if args.clean and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    env = os.environ.copy()
    env.setdefault('TRANSFORMERS_NO_TF', '1')
    env.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

    export_python = resolve_export_python(args.export_python)
    if os.path.abspath(export_python) != os.path.abspath(sys.executable):
        print(f'Using exporter Python: {export_python}')
    elif not has_local_transformers_onnx():
        raise RuntimeError('Interpreter hien tai khong co transformers.onnx va khong tim thay interpreter thay the.')

    command = build_command(export_python, model_dir, output_dir, args.opset, args.atol)
    print('Running:', ' '.join(command))
    subprocess.run(command, check=True, env=env)

    raw_model_path = os.path.join(output_dir, 'model.onnx')
    fp32_model_path = os.path.join(output_dir, DEFAULT_FP32_NAME)
    int8_model_path = os.path.join(output_dir, DEFAULT_INT8_NAME)

    if not os.path.exists(raw_model_path):
        raise FileNotFoundError(f'Export hoan tat nhung khong thay file: {raw_model_path}')

    if os.path.exists(fp32_model_path):
        os.remove(fp32_model_path)
    os.replace(raw_model_path, fp32_model_path)
    print(f'\nFP32 export xong: {fp32_model_path}')

    if not args.skip_int8:
        if os.path.exists(int8_model_path):
            os.remove(int8_model_path)
        quantize_dynamic(model_input=fp32_model_path, model_output=int8_model_path, weight_type=QuantType.QInt8)
        print(f'INT8 quantize xong: {int8_model_path}')


if __name__ == '__main__':
    main()
