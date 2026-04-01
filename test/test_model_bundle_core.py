from model_bundle.manifest import ModelBundleManifest
from model_bundle.projects import list_bundle_projects, resolve_bundle_project


def test_model_bundle_manifest_round_trips_generic_artifacts():
    manifest = ModelBundleManifest(
        bundle_version=1,
        project='vpcd',
        model_family='bartpho-seq2seq',
        model_name='vpcd/fp32',
        model_variant='fp32',
        asset_namespace='models/punctuation/vpcd',
        runtime_kind='text_seq2seq',
        artifacts={'model': 'model.mobile.onnx', 'tokenizer_encode': 'tokenizer.encode.onnx'},
        fixtures={'golden_samples': 'golden_samples.jsonl'},
        metadata={'max_decode_length': 128},
    )

    restored = ModelBundleManifest.from_dict(manifest.to_dict())

    assert restored.project == 'vpcd'
    assert restored.artifacts['model'] == 'model.mobile.onnx'
    assert restored.fixtures['golden_samples'] == 'golden_samples.jsonl'
    assert restored.metadata['max_decode_length'] == 128


def test_model_bundle_project_registry_resolves_vpcd_and_zipformer():
    assert set(list_bundle_projects()) >= {'vpcd', 'zipformer'}
    assert resolve_bundle_project('vpcd').name == 'vpcd'
    assert resolve_bundle_project('zipformer').name == 'zipformer'
