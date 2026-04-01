import shutil
import uuid
from pathlib import Path

import numpy as np
import pytest

from model_bundle.manifest import ModelBundleManifest

TEST_TMP_ROOT = Path(__file__).resolve().parent / '_tmp' / 'zipformer_quantize'


@pytest.fixture
def tmp_case_dir():
    path = TEST_TMP_ROOT / uuid.uuid4().hex
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_prepare_encoder_inputs_pads_features_to_fixed_frames():
    from model_bundle.projects.zipformer import prepare_encoder_inputs

    features = np.arange(6, dtype=np.float32).reshape(3, 2)

    encoder_inputs = prepare_encoder_inputs(features, fixed_encoder_frames=5)

    assert encoder_inputs['x'].shape == (1, 5, 2)
    assert encoder_inputs['x_lens'].tolist() == [3]
    np.testing.assert_array_equal(encoder_inputs['x'][0, :3, :], features)
    np.testing.assert_array_equal(encoder_inputs['x'][0, 3:, :], np.zeros((2, 2), dtype=np.float32))


def test_bundle_runtime_reads_fixed_encoder_frames_from_manifest(tmp_case_dir, monkeypatch):
    from model_bundle.projects.zipformer import BundleAcousticRuntime

    manifest = ModelBundleManifest(
        bundle_version=1,
        project='zipformer',
        model_family='zipformer-rnnt',
        model_name='zipformer/qnn_u16u8',
        model_variant='qnn_u16u8',
        asset_namespace='models/asr/zipformer/qnn_u16u8',
        runtime_kind='rnnt_greedy',
        artifacts={'encoder': 'encoder.onnx', 'decoder': 'decoder.onnx', 'joiner': 'joiner.onnx', 'tokens': 'tokens.txt'},
        fixtures={'sample_manifest': 'sample_manifest.jsonl', 'expected_outputs': 'expected_outputs.jsonl'},
        metadata={
            'sample_rate': 16000,
            'feature_dim': 80,
            'blank_id': 0,
            'context_size': 2,
            'fixed_input_shapes': {
                'encoder': {'x': [1, 128, 80], 'x_lens': [1]},
                'decoder': {'y': [1, 2]},
                'joiner': {'encoder_out': [1, 512], 'decoder_out': [1, 512]},
            },
        },
    )
    manifest.write_json(tmp_case_dir / 'bundle_manifest.json')

    seen = {}

    def fake_init(self, *, model_dir, provider='CPUExecutionProvider', component_paths=None, sample_rate=16000, feature_dim=80, blank_id=0, context_size=2, fixed_encoder_frames=None):
        seen['model_dir'] = str(model_dir)
        seen['provider'] = provider
        seen['fixed_encoder_frames'] = fixed_encoder_frames

    monkeypatch.setattr(BundleAcousticRuntime, '__init__', fake_init)

    runtime = BundleAcousticRuntime.from_manifest_path(tmp_case_dir / 'bundle_manifest.json')

    assert isinstance(runtime, BundleAcousticRuntime)
    assert seen['fixed_encoder_frames'] == 128
