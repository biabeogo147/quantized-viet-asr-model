import shutil
import uuid
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from model_bundle.manifest import ModelBundleManifest
from model_bundle.fixtures import AudioSampleFixture

TEST_TMP_ROOT = Path(__file__).resolve().parent / '_tmp' / 'zipformer_quantize'


@pytest.fixture
def tmp_case_dir():
    path = TEST_TMP_ROOT / uuid.uuid4().hex
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_zipformer_bool_slice_model() -> onnx.ModelProto:
    bool_mask = helper.make_tensor_value_info("/GreaterOrEqual_output_0", TensorProto.BOOL, [1, 8])
    graph_output = helper.make_tensor_value_info("masked", TensorProto.FLOAT, [1, 1, 4])

    starts = helper.make_tensor("starts", TensorProto.INT64, [1], [0])
    ends = helper.make_tensor("ends", TensorProto.INT64, [1], [8])
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [1])
    steps = helper.make_tensor("steps", TensorProto.INT64, [1], [2])
    unsqueeze_axes = helper.make_tensor("unsqueeze_axes", TensorProto.INT64, [1], [1])
    where_true = helper.make_tensor("where_true", TensorProto.FLOAT, [1], [1.0])
    where_false = helper.make_tensor("where_false", TensorProto.FLOAT, [1], [0.0])

    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                "Slice",
                ["/GreaterOrEqual_output_0", "starts", "ends", "axes", "steps"],
                ["/encoder/Slice_1_output_0"],
                name="/encoder/Slice_1",
            ),
            helper.make_node(
                "Unsqueeze",
                ["/encoder/Slice_1_output_0", "unsqueeze_axes"],
                ["/encoder/1/encoder/0/self_attn_weights/Unsqueeze_15_output_0"],
                name="/encoder/1/encoder/0/self_attn_weights/Unsqueeze_15",
            ),
            helper.make_node(
                "Where",
                [
                    "/encoder/1/encoder/0/self_attn_weights/Unsqueeze_15_output_0",
                    "where_true",
                    "where_false",
                ],
                ["masked"],
                name="/encoder/1/encoder/0/self_attn_weights/Where",
            ),
        ],
        name="zipformer-bool-slice",
        inputs=[bool_mask],
        outputs=[graph_output],
        initializer=[starts, ends, axes, steps, unsqueeze_axes, where_true, where_false],
        value_info=[
            helper.make_tensor_value_info("/encoder/Slice_1_output_0", TensorProto.BOOL, [1, 4]),
            helper.make_tensor_value_info(
                "/encoder/1/encoder/0/self_attn_weights/Unsqueeze_15_output_0",
                TensorProto.BOOL,
                [1, 1, 4],
            ),
            helper.make_tensor_value_info("masked", TensorProto.FLOAT, [1, 1, 4]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    return model


def test_prepare_encoder_inputs_pads_features_to_fixed_frames():
    from model_bundle.zipformer_runtime import prepare_encoder_inputs

    features = np.arange(6, dtype=np.float32).reshape(3, 2)

    encoder_inputs = prepare_encoder_inputs(features, fixed_encoder_frames=5)

    assert encoder_inputs['x'].shape == (1, 5, 2)
    assert encoder_inputs['x_lens'].tolist() == [3]
    np.testing.assert_array_equal(encoder_inputs['x'][0, :3, :], features)
    np.testing.assert_array_equal(encoder_inputs['x'][0, 3:, :], np.zeros((2, 2), dtype=np.float32))


def test_bundle_runtime_reads_fixed_encoder_frames_from_manifest(tmp_case_dir, monkeypatch):
    from model_bundle.zipformer_runtime import BundleAcousticRuntime

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


def test_load_audio_fixtures_supports_utf8_bom_manifest(tmp_case_dir):
    from quantize.projects.zipformer import _load_audio_fixtures

    manifest_path = tmp_case_dir / 'audio_manifest.txt'
    manifest_path.write_text('\ufeffassets/speech/sample-1.mp3\n', encoding='utf-8')

    fixtures = _load_audio_fixtures(str(manifest_path))

    assert fixtures == [AudioSampleFixture(sample_id='audio-1', audio_path='assets/speech/sample-1.mp3')]


def test_collect_component_records_resolves_audio_paths_from_repo_root(monkeypatch):
    from quantize.projects.zipformer import _collect_component_records

    repo_root = Path(__file__).resolve().parent.parent
    seen = {}

    class FakeEncoderSession:
        def run(self, _, inputs):
            return [
                np.zeros((1, 1, 4), dtype=np.float32),
                np.asarray([1], dtype=np.int64),
            ]

    class FakeDecoderSession:
        def run(self, _, inputs):
            return [np.zeros((1, 4), dtype=np.float32)]

    class FakeJoinerSession:
        def run(self, _, inputs):
            return [np.asarray([[1.0, 0.0]], dtype=np.float32)]

    class FakeRuntime:
        sample_rate = 16000
        feature_dim = 80
        blank_id = 0
        context_size = 2
        encoder_sess = FakeEncoderSession()
        decoder_sess = FakeDecoderSession()
        joiner_sess = FakeJoinerSession()

        def _load_features(self, audio_path, sample_rate=16000, feature_dim=80):
            seen['audio_path'] = Path(audio_path)
            return np.zeros((2, feature_dim), dtype=np.float32)

    records, stats = _collect_component_records(
        FakeRuntime(),
        [AudioSampleFixture(sample_id='sample-1', audio_path='assets/speech/sample-1.mp3')],
    )

    assert seen['audio_path'] == repo_root / 'assets' / 'speech' / 'sample-1.mp3'
    assert len(records['encoder']) == 1
    assert stats['sample_count'] == 1


def test_zipformer_validate_args_accepts_balanced_preset():
    from quantize.projects.zipformer import apply_default_arguments, validate_args
    import argparse

    parser = argparse.ArgumentParser()
    apply_default_arguments(parser)

    args = parser.parse_args(['--preset', 'zipformer_sd8g2_balanced'])

    validate_args(args)


def test_zipformer_validate_args_rejects_retired_legacy_alias():
    from quantize.projects.zipformer import apply_default_arguments, validate_args
    import argparse

    parser = argparse.ArgumentParser()
    apply_default_arguments(parser)

    args = parser.parse_args(['--preset', 'zipformer_sd8g2_qnn_u16u8'])

    with pytest.raises(ValueError, match="Unsupported zipformer preset"):
        validate_args(args)


def test_prepare_zipformer_aihub_encoder_rewrites_bool_slice_path(tmp_case_dir):
    from quantize.projects.zipformer import prepare_zipformer_aihub_encoder

    source_model_path = tmp_case_dir / 'encoder.fixed.onnx'
    output_path = tmp_case_dir / 'aihub_compile' / 'encoder.aihub.option1.onnx'
    onnx.save(_build_zipformer_bool_slice_model(), source_model_path.as_posix())

    prepared_path = prepare_zipformer_aihub_encoder(source_model_path, output_path)

    prepared_model = onnx.load(prepared_path.as_posix())
    nodes_by_name = {node.name: node for node in prepared_model.graph.node}
    assert prepared_path == output_path.resolve()
    assert "/GreaterOrEqual_output_0_u8_cast" in nodes_by_name
    assert nodes_by_name["/encoder/Slice_1"].input[0] == "/GreaterOrEqual_output_0_u8"
    assert nodes_by_name["/encoder/1/encoder/0/self_attn_weights/Unsqueeze_15"].output[0].endswith("_u8")
    assert "/encoder/1/encoder/0/self_attn_weights/Unsqueeze_15_cast_bool" in nodes_by_name
    assert sorted(path.name for path in output_path.parent.iterdir()) == ["encoder.aihub.option1.onnx"]


def test_quantization_report_can_record_aihub_prepared_encoder_metadata():
    from quantize.reports import QuantizationReport

    report = QuantizationReport(
        project='zipformer',
        preset='zipformer_sd8g2_balanced',
        output_root='build/quantize/zipformer/qnn_u16u8',
        bundle_output_dir='build/model_bundle/zipformer/qnn_u16u8',
        sample_count=1,
        trace_records=2,
        metadata={
            'aihub_prepared_encoder_path': 'build/quantize/zipformer/qnn_u16u8/aihub_compile/encoder.aihub.option1.onnx',
            'aihub_prepare_applied': True,
            'aihub_prepare_steps': [
                'ort_optimize',
                'symbolic_shape_inference',
                'zipformer_bool_mask_rewrite',
            ],
        },
    )

    payload = report.to_dict()

    assert payload['metadata']['aihub_prepared_encoder_path'].endswith('aihub_compile/encoder.aihub.option1.onnx')
    assert payload['metadata']['aihub_prepare_applied'] is True
    assert payload['metadata']['aihub_prepare_steps'] == [
        'ort_optimize',
        'symbolic_shape_inference',
        'zipformer_bool_mask_rewrite',
    ]
