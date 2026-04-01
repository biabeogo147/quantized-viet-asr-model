from types import SimpleNamespace

import pytest


def _make_fake_project(name: str, seen: dict):
    def apply_default_arguments(parser):
        parser.add_argument('--dry-run', action='store_true')
        parser.add_argument('--model-dir', default=f'assets/{name}')

    def validate_args(args):
        seen['validated_project'] = name
        seen['model_dir'] = args.model_dir
        seen['dry_run'] = args.dry_run

    def run(args):
        seen['ran_project'] = name
        seen['run_model_dir'] = args.model_dir
        seen['run_dry_run'] = args.dry_run
        return 0

    return SimpleNamespace(
        NAME=name,
        apply_default_arguments=apply_default_arguments,
        validate_args=validate_args,
        run=run,
    )


def test_quantize_cli_dispatches_dry_run_to_vpcd_project(monkeypatch):
    from quantize.cli import main

    seen = {}
    monkeypatch.setattr('quantize.cli.list_quantize_projects', lambda: ('vpcd', 'zipformer'))
    monkeypatch.setattr('quantize.cli.resolve_quantize_project', lambda project: _make_fake_project(project, seen))

    exit_code = main(['--project', 'vpcd', '--dry-run', '--model-dir', 'assets/vietnamese-punc-cap-denorm-v1'])

    assert exit_code == 0
    assert seen['validated_project'] == 'vpcd'
    assert seen['ran_project'] == 'vpcd'
    assert seen['run_dry_run'] is True


def test_quantize_cli_dispatches_dry_run_to_zipformer_project(monkeypatch):
    from quantize.cli import main

    seen = {}
    monkeypatch.setattr('quantize.cli.list_quantize_projects', lambda: ('vpcd', 'zipformer'))
    monkeypatch.setattr('quantize.cli.resolve_quantize_project', lambda project: _make_fake_project(project, seen))

    exit_code = main(['--project', 'zipformer', '--dry-run', '--model-dir', 'assets/zipformer'])

    assert exit_code == 0
    assert seen['validated_project'] == 'zipformer'
    assert seen['ran_project'] == 'zipformer'
    assert seen['run_model_dir'] == 'assets/zipformer'
