from pathlib import Path


def test_export_model_bundle_module_delegates_to_shared_exporter(monkeypatch, capsys):
    seen = {}

    class FakeAdapter:
        default_model_dir = 'assets/model'
        default_output_dir = 'build/model_bundle/project/fp32'
        default_asset_namespace = 'models/project/fp32'
        default_variant = 'fp32'

    class FakeManifest:
        model_name = 'project/fp32'

    def fake_resolve_bundle_project(name):
        seen['project'] = name
        return FakeAdapter()

    def fake_export_model_bundle(**kwargs):
        seen['kwargs'] = kwargs
        return FakeManifest()

    monkeypatch.setattr('export.model_bundle.resolve_bundle_project', fake_resolve_bundle_project)
    monkeypatch.setattr('export.model_bundle.export_model_bundle', fake_export_model_bundle)

    from export.model_bundle import main

    main(['--project', 'zipformer'])
    output = capsys.readouterr().out

    assert seen['project'] == 'zipformer'
    assert seen['kwargs']['project'] == 'zipformer'
    assert 'Model bundle exported.' in output


def test_verify_model_bundle_module_delegates_to_shared_verifier(monkeypatch, capsys):
    seen = {}

    class FakeAdapter:
        default_model_dir = 'assets/model'
        default_output_dir = 'build/model_bundle/project/fp32'

    def fake_resolve_bundle_project(name):
        seen['project'] = name
        return FakeAdapter()

    def fake_verify_model_bundle(**kwargs):
        seen['kwargs'] = kwargs
        return {'checked_samples': 2, 'passed': True, 'mismatches': []}

    monkeypatch.setattr('verify.model_bundle.resolve_bundle_project', fake_resolve_bundle_project)
    monkeypatch.setattr('verify.model_bundle.verify_model_bundle', fake_verify_model_bundle)

    from verify.model_bundle import main

    main(['--project', 'zipformer'])
    output = capsys.readouterr().out

    assert seen['project'] == 'zipformer'
    assert seen['kwargs']['project'] == 'zipformer'
    assert 'Verification complete.' in output
    assert 'Passed         : True' in output
