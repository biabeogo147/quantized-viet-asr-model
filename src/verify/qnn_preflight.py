from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from verify.qnn_preflight_core import verify_qnn_preflight


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run QNN preflight checks for a shared model bundle.')
    parser.add_argument('--project', choices=('vpcd',), required=True)
    parser.add_argument('--bundle-dir', required=True)
    parser.add_argument('--output')
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    args = build_argument_parser().parse_args(argv)
    report = verify_qnn_preflight(project=args.project, bundle_dir=args.bundle_dir)

    output = Path(args.output) if args.output else None
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    print('QNN preflight complete.')
    print('Project :', args.project)
    print('Bundle  :', args.bundle_dir)
    print('Passed  :', report['passed'])
    if output is not None:
        print('Report  :', output)


if __name__ == '__main__':
    main()
