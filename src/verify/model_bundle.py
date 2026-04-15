import argparse
import json
import sys
from typing import Sequence

from model_bundle.projects import resolve_bundle_project
from model_bundle.verifier import verify_model_bundle


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Verify a shared model bundle.')
    parser.add_argument('--project', choices=('vpcd', 'zipformer'), required=True)
    parser.add_argument('--model-dir')
    parser.add_argument('--bundle-dir')
    parser.add_argument('--reference-bundle')
    parser.add_argument('--candidate-bundle')
    parser.add_argument('--provider', default='CPUExecutionProvider')
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    args = build_argument_parser().parse_args(argv)
    adapter = resolve_bundle_project(args.project)
    kwargs = {'provider': args.provider}
    if args.model_dir:
        kwargs['model_dir'] = args.model_dir
    if args.bundle_dir:
        kwargs['bundle_dir'] = args.bundle_dir
    if args.reference_bundle:
        kwargs['reference_bundle'] = args.reference_bundle
    if args.candidate_bundle:
        kwargs['candidate_bundle'] = args.candidate_bundle
    if not kwargs.keys() - {'provider'}:
        kwargs['model_dir'] = adapter.default_model_dir
        kwargs['bundle_dir'] = adapter.default_output_dir

    report = verify_model_bundle(project=args.project, **kwargs)
    print('Verification complete.')
    print('Project        :', args.project)
    if isinstance(report, tuple):
        print('Encode samples :', report[0])
        print('Decode samples :', report[1])
        return
    print('Checked samples:', report.get('checked_samples'))
    print('Passed         :', report.get('passed'))
    if report.get('mismatches'):
        for mismatch in report['mismatches']:
            print('-', json.dumps(mismatch, ensure_ascii=False))


if __name__ == '__main__':
    main()
