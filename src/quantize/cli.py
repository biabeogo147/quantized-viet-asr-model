import argparse
from typing import Sequence

from quantize.projects import list_quantize_projects, resolve_quantize_project

DEFAULT_PROJECT = 'vpcd'


def _build_project_probe_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--project', choices=list_quantize_projects(), default=DEFAULT_PROJECT)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    probe_parser = _build_project_probe_parser()
    probe_args, _ = probe_parser.parse_known_args(argv)
    project_module = resolve_quantize_project(probe_args.project)

    parser = argparse.ArgumentParser(
        description='Quantize ONNX model bundles for Android and Snapdragon deployment.',
    )
    parser.add_argument('--project', choices=list_quantize_projects(), default=probe_args.project)
    project_module.apply_default_arguments(parser)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_module = resolve_quantize_project(args.project)
    project_module.validate_args(args)
    return int(project_module.run(args))
