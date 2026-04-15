import argparse
import sys
from pathlib import Path
from typing import Sequence

from model_bundle.exporter import export_model_bundle
from model_bundle.projects import resolve_bundle_project


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Export a shared model bundle.')
    parser.add_argument('--project', choices=('vpcd', 'zipformer'), required=True)
    parser.add_argument('--model-dir', help='Source model directory.')
    parser.add_argument('--output-dir', help='Output directory for the bundle.')
    parser.add_argument('--asset-namespace', help='Logical asset namespace stored in the bundle manifest.')
    parser.add_argument('--model-variant', help='Optional model variant name for the selected project.')
    parser.add_argument('--provider', default='CPUExecutionProvider')
    parser.add_argument('--max-decode-length', type=int, default=128)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    args = build_argument_parser().parse_args(argv)
    adapter = resolve_bundle_project(args.project)
    manifest = export_model_bundle(
        project=args.project,
        model_dir=args.model_dir or adapter.default_model_dir,
        output_dir=args.output_dir or adapter.default_output_dir,
        asset_namespace=args.asset_namespace or adapter.default_asset_namespace,
        model_variant=args.model_variant or adapter.default_variant,
        provider=args.provider,
        max_decode_length=args.max_decode_length,
    )
    output_dir = Path(args.output_dir or adapter.default_output_dir).resolve()
    print('Model bundle exported.')
    print('Project    :', args.project)
    print('Output dir :', output_dir)
    print('Manifest   :', output_dir / 'bundle_manifest.json')
    print('Model name :', manifest.model_name)


if __name__ == '__main__':
    main()
