import shutil
import uuid
from pathlib import Path

import pytest

from model_bundle.fixtures import AudioExpectedOutput, AudioSampleFixture
from model_bundle.manifest import ModelBundleManifest

TEST_TMP_ROOT = Path(__file__).resolve().parent / '_tmp' / 'zipformer_bundle'


@pytest.fixture
def tmp_case_dir():
    path = TEST_TMP_ROOT / uuid.uuid4().hex
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_create_runtime_uses_bundle_runtime_when_bundle_manifest_is_provided(monkeypatch):
    from test.test_acoustic_model_onnx import build_argument_parser, create_runtime

    parser = build_argument_parser()
    seen = {}

    class FakeBundleRuntime:
        @classmethod
        def from_manifest_path(cls, manifest_path, provider):
            seen['manifest_path'] = manifest_path
            seen['provider'] = provider
            return cls()

    monkeypatch.setattr('test.test_acoustic_model_onnx.BundleAcousticRuntime', FakeBundleRuntime)

    args = parser.parse_args(['--bundle-manifest', 'bundle_manifest.json', '--provider', 'CPUExecutionProvider'])
    runtime = create_runtime(args)

    assert isinstance(runtime, FakeBundleRuntime)
    assert seen == {'manifest_path': 'bundle_manifest.json', 'provider': 'CPUExecutionProvider'}


def test_create_runtime_uses_model_dir_runtime_by_default(monkeypatch):
    from test.test_acoustic_model_onnx import build_argument_parser, create_runtime

    parser = build_argument_parser()
    seen = {}

    class FakeModelDirRuntime:
        def __init__(self, *, model_dir, provider):
            seen['model_dir'] = model_dir
            seen['provider'] = provider

    monkeypatch.setattr('test.test_acoustic_model_onnx.ModelDirAcousticRuntime', FakeModelDirRuntime)

    args = parser.parse_args([])
    runtime = create_runtime(args)

    assert isinstance(runtime, FakeModelDirRuntime)
    assert seen['provider'] == 'CPUExecutionProvider'
    assert seen['model_dir'].endswith(str(Path('assets') / 'zipformer'))


def test_model_bundle_manifest_round_trips_for_zipformer():
    manifest = ModelBundleManifest(
        bundle_version=1,
        project='zipformer',
        model_family='zipformer-rnnt',
        model_name='zipformer/fp32',
        model_variant='fp32',
        asset_namespace='models/asr/zipformer/fp32',
        runtime_kind='rnnt_greedy',
        artifacts={
            'encoder': 'encoder.onnx',
            'decoder': 'decoder.onnx',
            'joiner': 'joiner.onnx',
            'tokens': 'tokens.txt',
        },
        fixtures={
            'sample_manifest': 'sample_manifest.jsonl',
            'expected_outputs': 'expected_outputs.jsonl',
        },
        metadata={'sample_rate': 16000, 'feature_dim': 80, 'blank_id': 0, 'context_size': 2},
    )

    restored = ModelBundleManifest.from_dict(manifest.to_dict())

    assert restored.project == 'zipformer'
    assert restored.artifacts['encoder'] == 'encoder.onnx'
    assert restored.metadata['context_size'] == 2


def test_audio_fixtures_round_trip():
    sample = AudioSampleFixture(sample_id='sample-1', audio_path='assets/speech/sample-1.mp3')
    expected = AudioExpectedOutput(sample_id='sample-1', audio_path='assets/speech/sample-1.mp3', text='xin chao')

    assert AudioSampleFixture.from_dict(sample.to_dict()) == sample
    assert AudioExpectedOutput.from_dict(expected.to_dict()) == expected


def test_export_zipformer_bundle_writes_manifest_and_model_files(tmp_case_dir):
    from model_bundle.projects.zipformer import export_bundle

    model_dir = tmp_case_dir / 'model'
    model_dir.mkdir()
    for file_name in ('encoder-epoch-20-avg-1.onnx', 'decoder-epoch-20-avg-1.onnx', 'joiner-epoch-20-avg-1.onnx', 'tokens.txt'):
        (model_dir / file_name).write_text(file_name, encoding='utf-8')

    output_dir = tmp_case_dir / 'bundle'
    export_bundle(
        model_dir=model_dir,
        output_dir=output_dir,
        asset_namespace='models/asr/zipformer/fp32',
        sample_fixtures=[AudioSampleFixture(sample_id='sample-1', audio_path='assets/speech/sample-1.mp3')],
        expected_outputs=[AudioExpectedOutput(sample_id='sample-1', audio_path='assets/speech/sample-1.mp3', text='xin chao')],
    )

    assert (output_dir / 'bundle_manifest.json').exists()
    assert (output_dir / 'encoder.onnx').exists()
    assert (output_dir / 'decoder.onnx').exists()
    assert (output_dir / 'joiner.onnx').exists()
    assert (output_dir / 'tokens.txt').exists()
    assert (output_dir / 'sample_manifest.jsonl').exists()
    assert (output_dir / 'expected_outputs.jsonl').exists()


def test_bundle_runtime_uses_only_manifest_paths(tmp_case_dir, monkeypatch):
    from model_bundle.projects.zipformer import BundleAcousticRuntime, export_bundle

    model_dir = tmp_case_dir / 'model'
    model_dir.mkdir()
    for file_name in ('encoder-epoch-20-avg-1.onnx', 'decoder-epoch-20-avg-1.onnx', 'joiner-epoch-20-avg-1.onnx', 'tokens.txt'):
        (model_dir / file_name).write_text(file_name, encoding='utf-8')

    output_dir = tmp_case_dir / 'bundle'
    export_bundle(
        model_dir=model_dir,
        output_dir=output_dir,
        asset_namespace='models/asr/zipformer/fp32',
        sample_fixtures=[],
        expected_outputs=[],
    )

    seen = {}

    def fake_init(self, *, model_dir, provider='CPUExecutionProvider', component_paths=None, sample_rate=16000, feature_dim=80, blank_id=0, context_size=2, fixed_encoder_frames=None):
        seen['model_dir'] = str(model_dir)
        seen['provider'] = provider
        seen['component_paths'] = {k: str(v) for k, v in (component_paths or {}).items()}
        seen['sample_rate'] = sample_rate
        seen['feature_dim'] = feature_dim
        seen['blank_id'] = blank_id
        seen['context_size'] = context_size
        seen['fixed_encoder_frames'] = fixed_encoder_frames

    monkeypatch.setattr(BundleAcousticRuntime, '__init__', fake_init)

    runtime = BundleAcousticRuntime.from_manifest_path(output_dir / 'bundle_manifest.json')

    assert isinstance(runtime, BundleAcousticRuntime)
    assert seen['component_paths']['encoder'].endswith('encoder.onnx')
    assert seen['component_paths']['tokens'].endswith('tokens.txt')
    assert seen['context_size'] == 2


def test_verify_exported_bundle_matches_model_dir_mode(monkeypatch, tmp_case_dir):
    from model_bundle.projects.zipformer import verify_bundle

    bundle_dir = tmp_case_dir / 'bundle'
    bundle_dir.mkdir()
    repo_root = Path(__file__).resolve().parent.parent

    manifest = ModelBundleManifest(
        bundle_version=1,
        project='zipformer',
        model_family='zipformer-rnnt',
        model_name='zipformer/fp32',
        model_variant='fp32',
        asset_namespace='models/asr/zipformer/fp32',
        runtime_kind='rnnt_greedy',
        artifacts={'encoder': 'encoder.onnx', 'decoder': 'decoder.onnx', 'joiner': 'joiner.onnx', 'tokens': 'tokens.txt'},
        fixtures={'sample_manifest': 'sample_manifest.jsonl', 'expected_outputs': 'expected_outputs.jsonl'},
        metadata={'sample_rate': 16000, 'feature_dim': 80, 'blank_id': 0, 'context_size': 2},
    )
    manifest.write_json(bundle_dir / 'bundle_manifest.json')
    (bundle_dir / 'sample_manifest.jsonl').write_text(AudioSampleFixture(sample_id='sample-1', audio_path='assets/speech/sample-1.mp3').to_jsonl_line(), encoding='utf-8')
    (bundle_dir / 'expected_outputs.jsonl').write_text(AudioExpectedOutput(sample_id='sample-1', audio_path='assets/speech/sample-1.mp3', text='xin chao').to_jsonl_line(), encoding='utf-8')

    class FakeRuntime:
        def __init__(self, text):
            self.text = text
            self.audio_paths = []

        def transcribe(self, audio_path):
            self.audio_paths.append(Path(audio_path))
            return {'text': self.text, 'audio_path': str(audio_path)}

    model_runtime = FakeRuntime('xin chao')
    bundle_runtime = FakeRuntime('xin chao')
    monkeypatch.setattr('model_bundle.projects.zipformer.ModelDirAcousticRuntime', lambda **kwargs: model_runtime)
    monkeypatch.setattr(
        'model_bundle.projects.zipformer.BundleAcousticRuntime',
        type('FakeBundleRuntime', (), {'from_manifest_path': classmethod(lambda cls, manifest_path, provider='CPUExecutionProvider': bundle_runtime)}),
    )

    report = verify_bundle(model_dir=tmp_case_dir / 'model', bundle_dir=bundle_dir)

    assert report['passed'] is True
    assert report['checked_samples'] == 1
    assert model_runtime.audio_paths == [repo_root / 'assets' / 'speech' / 'sample-1.mp3']
    assert bundle_runtime.audio_paths == [repo_root / 'assets' / 'speech' / 'sample-1.mp3']

