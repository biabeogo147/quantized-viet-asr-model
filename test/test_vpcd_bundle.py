import json
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from model_bundle.fixtures import TextGoldenSample, serialize_jsonl
from model_bundle.manifest import ModelBundleManifest
from model_bundle.vpcd_runtime import (
    BundleOnnxRuntime,
    DEFAULT_GOLDEN_SAMPLES,
    TokenizerIdBridge,
    TokenizerExportArtifacts,
)
from model_bundle.vpcd_shapes import (
    attention_mask_for_length,
    pad_token_row,
    resolve_vpcd_model_input_shapes,
)
from quantize.vpcd_bundle import build_vpcd_metadata, export_bundle, verify_bundle
from test.test_punctuation_model_onnx import (
    DEFAULT_MODEL_VARIANT,
    MODEL_DIR,
    build_argument_parser,
    create_runtime,
)


TEST_TMP_ROOT = Path(__file__).resolve().parent / '_tmp' / 'vpcd_bundle'


@pytest.fixture
def tmp_case_dir():
    path = TEST_TMP_ROOT / 'case'
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_manifest_schema_is_stable_for_vpcd_bundle_contract():
    manifest = ModelBundleManifest(
        bundle_version=1,
        project='vpcd',
        model_family='bartpho-seq2seq',
        model_name='tourmii/vietnamese-punc-cap-denorm-v1',
        model_variant='vpcd_balanced',
        asset_namespace='models/punctuation/vpcd/vpcd_balanced',
        runtime_kind='text_seq2seq',
        artifacts={
            'model': 'model.mobile.onnx',
            'tokenizer_encode': 'tokenizer.encode.onnx',
            'tokenizer_decode': 'tokenizer.decode.onnx',
            'tokenizer_to_model_id_map': 'tokenizer.to_model_id_map.json',
            'model_to_tokenizer_id_map': 'tokenizer.from_model_id_map.json',
        },
        fixtures={'golden_samples': 'golden_samples.jsonl'},
        metadata={
            'pad_token_id': 1,
            'eos_token_id': 2,
            'decoder_start_token_id': 2,
            'max_source_length': 1024,
            'max_decode_length': 128,
            'input_text_case': 'lower',
        },
    )

    payload = manifest.to_dict()

    assert payload['project'] == 'vpcd'
    assert payload['artifacts']['model'] == 'model.mobile.onnx'
    assert payload['fixtures']['golden_samples'] == 'golden_samples.jsonl'
    assert payload['metadata']['input_text_case'] == 'lower'


def test_vpcd_balanced_metadata_declares_qdq_but_not_fixed_shape_ready():
    metadata = build_vpcd_metadata(model_variant='vpcd_balanced', max_decode_length=128)

    assert metadata['input_text_case'] == 'lower'
    assert metadata['quantization'] == {
        'format': 'QDQ',
        'activation_type': 'quint16',
        'weight_type': 'quint8',
        'preset': 'sd8g2_balanced',
        'fixed_shapes': False,
    }
    assert metadata['qnn_readiness']['target_backend'] == 'qnn_htp'
    assert metadata['qnn_readiness']['model_session_candidate'] is True
    assert metadata['qnn_readiness']['tokenizer_policy'] == 'cpu_only_first_slice'
    assert metadata['qnn_readiness']['requires_fixed_shapes'] is True
    assert metadata['qnn_readiness']['fixed_shapes_ready'] is False


def test_resolve_vpcd_model_input_shapes_reads_manifest_metadata():
    metadata = {
        'fixed_input_shapes': {
            'model': {
                'input_ids': [1, 1024],
                'attention_mask': [1, 1024],
                'decoder_input_ids': [1, 128],
                'decoder_attention_mask': [1, 128],
            }
        }
    }

    shapes = resolve_vpcd_model_input_shapes(metadata)

    assert shapes is not None
    assert shapes.input_ids == (1, 1024)
    assert shapes.attention_mask == (1, 1024)
    assert shapes.decoder_input_ids == (1, 128)
    assert shapes.decoder_attention_mask == (1, 128)
    assert shapes.encoder_sequence == 1024
    assert shapes.decoder_sequence == 128


def test_resolve_vpcd_model_input_shapes_returns_none_without_metadata():
    assert resolve_vpcd_model_input_shapes({}) is None


def test_pad_token_row_pads_to_fixed_length():
    assert pad_token_row([7, 8, 2], target_length=5, pad_value=1).tolist() == [[7, 8, 2, 1, 1]]


def test_attention_mask_for_length_marks_padding_as_zero():
    assert attention_mask_for_length(actual_length=3, target_length=5).tolist() == [[1, 1, 1, 0, 0]]


def test_pad_token_row_rejects_values_longer_than_target():
    with pytest.raises(ValueError, match='exceeds fixed target length'):
        pad_token_row([1, 2, 3], target_length=2, pad_value=0)


def test_text_golden_samples_serialize_to_jsonl():
    sample = TextGoldenSample(
        raw_text='hom nay la buoi nham chuc cua toi phuoc thanh',
        input_ids=[0, 12, 18, 2],
        expected_output='Hôm nay là buổi nhậm chức của tôi Phước Thành.',
    )

    serialized = serialize_jsonl([sample])

    assert serialized == sample.to_jsonl_line()


def test_default_vpcd_golden_samples_are_pinned_to_source_of_truth():
    assert [sample.raw_text for sample in DEFAULT_GOLDEN_SAMPLES] == [
        "h\u00f4m nay l\u00e0 bu\u1ed5i nh\u1eadm ch\u1ee9c c\u1ee7a t\u00f4i ph\u01b0\u1edbc th\u00e0nh",
        "ch\u00e0o c\u00e1c b\u1ea1n h\u00f4m nay ch\u00fang ta c\u00f9ng nhau \u0111\u1ebfn v\u1edbi b\u00e0i h\u1ecdc deep learning ph\u1ea7n s\u1ed1 m\u01b0\u1eddi ba",
    ]
    assert [sample.input_ids for sample in DEFAULT_GOLDEN_SAMPLES] == [
        [0, 799, 177, 9, 847, 559, 2306, 115, 7, 80, 1386, 1338, 58, 2],
        [0, 1740, 10, 144, 799, 177, 248, 336, 120, 383, 30, 15, 635, 71, 19466, 18436, 221, 52, 3125, 712, 2],
    ]
    assert [sample.expected_output for sample in DEFAULT_GOLDEN_SAMPLES] == [
        "H\u00f4m nay l\u00e0 bu\u1ed5i nh\u1eadm ch\u1ee9c c\u1ee7a t\u00f4i - Ph\u01b0\u1edbc Th\u00e0nh.",
        "Ch\u00e0o c\u00e1c b\u1ea1n, h\u00f4m nay ch\u00fang ta c\u00f9ng nhau \u0111\u1ebfn v\u1edbi b\u00e0i h\u1ecdc Deep Learning ph\u1ea7n s\u1ed1 13.",
    ]


def test_create_runtime_defaults_to_model_dir_mode():
    parser = build_argument_parser()
    args = parser.parse_args([])
    captured: dict[str, object] = {}

    class FakeModelDirOnnxRuntime:
        def __init__(self, *, model_dir: str, onnx_path: str, provider: str):
            captured['model_dir'] = model_dir
            captured['onnx_path'] = onnx_path
            captured['provider'] = provider

    with patch('test.test_punctuation_model_onnx.ModelDirOnnxRuntime', FakeModelDirOnnxRuntime):
        runtime = create_runtime(args)

    assert runtime is not None
    assert captured == {
        'model_dir': MODEL_DIR,
        'onnx_path': str(Path(MODEL_DIR) / 'onnx' / f'{DEFAULT_MODEL_VARIANT}.onnx'),
        'provider': 'CPUExecutionProvider',
    }


def test_create_runtime_uses_bundle_manifest_mode_without_model_dir():
    parser = build_argument_parser()
    args = parser.parse_args(['--bundle-manifest', 'build/model_bundle/vpcd/vpcd_balanced/bundle_manifest.json'])
    captured: dict[str, object] = {}

    class FakeBundleOnnxRuntime:
        @classmethod
        def from_manifest_path(cls, manifest_path: str, provider: str):
            captured['manifest_path'] = manifest_path
            captured['provider'] = provider
            return cls()

    with patch('test.test_punctuation_model_onnx.BundleOnnxRuntime', FakeBundleOnnxRuntime):
        runtime = create_runtime(args)

    assert runtime is not None
    assert captured == {
        'manifest_path': 'build/model_bundle/vpcd/vpcd_balanced/bundle_manifest.json',
        'provider': 'CPUExecutionProvider',
    }


def test_argument_parser_rejects_model_dir_and_bundle_manifest_together():
    parser = build_argument_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                '--model-dir',
                'assets/vietnamese-punc-cap-denorm-v1',
                '--bundle-manifest',
                'build/model_bundle/vpcd/vpcd_balanced/bundle_manifest.json',
            ]
        )


def test_tokenizer_id_bridge_writes_dense_mapping_files(tmp_case_dir):
    bridge = TokenizerIdBridge(
        tokenizer_to_model_ids=[0, 1, 2, 99],
        model_to_tokenizer_ids=[0, 1, 2, 3, 42],
    )

    tokenizer_to_model_name, model_to_tokenizer_name = bridge.write_files(
        tokenizer_to_model_path=tmp_case_dir / 'tokenizer.to_model_id_map.json',
        model_to_tokenizer_path=tmp_case_dir / 'tokenizer.from_model_id_map.json',
    )

    assert tokenizer_to_model_name == 'tokenizer.to_model_id_map.json'
    assert model_to_tokenizer_name == 'tokenizer.from_model_id_map.json'
    assert json.loads((tmp_case_dir / tokenizer_to_model_name).read_text(encoding='utf-8')) == [0, 1, 2, 99]
    assert json.loads((tmp_case_dir / model_to_tokenizer_name).read_text(encoding='utf-8')) == [0, 1, 2, 3, 42]


def test_bundle_runtime_restores_text_using_only_bundle_artifacts():
    manifest = ModelBundleManifest(
        bundle_version=1,
        project='vpcd',
        model_family='bartpho-seq2seq',
        model_name='tourmii/vietnamese-punc-cap-denorm-v1',
        model_variant='vpcd_balanced',
        asset_namespace='models/punctuation/vpcd/vpcd_balanced',
        runtime_kind='text_seq2seq',
        artifacts={
            'model': 'model.mobile.onnx',
            'tokenizer_encode': 'tokenizer.encode.onnx',
            'tokenizer_decode': 'tokenizer.decode.onnx',
            'tokenizer_to_model_id_map': 'tokenizer.to_model_id_map.json',
            'model_to_tokenizer_id_map': 'tokenizer.from_model_id_map.json',
        },
        fixtures={'golden_samples': 'golden_samples.jsonl'},
        metadata={
            'pad_token_id': 1,
            'eos_token_id': 2,
            'decoder_start_token_id': 2,
            'max_source_length': 8,
            'max_decode_length': 4,
        },
    )

    class FakeSession:
        def __init__(self, responses: list[object]):
            self.responses = list(responses)
            self.inputs: list[dict[str, object]] = []

        def run(self, _outputs: object, feeds: dict[str, object]) -> list[object]:
            self.inputs.append(feeds)
            return [self.responses.pop(0)]

    encode_session = FakeSession([np.asarray([[0, 4, 5, 2]], dtype=np.int64)])
    model_session = FakeSession(
        [
            np.asarray([[[0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 0.0]]], dtype=np.float32),
            np.asarray([[[0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0]]], dtype=np.float32),
        ]
    )
    decode_session = FakeSession([np.asarray(['xin chao.'], dtype=object)])

    runtime = BundleOnnxRuntime(
        manifest=manifest,
        model_session=model_session,
        encode_session=encode_session,
        decode_session=decode_session,
        tokenizer_to_model_ids=np.asarray([0, 1, 2, 3, 11, 12], dtype=np.int64),
        model_to_tokenizer_ids=np.asarray([0, 1, 2, 3, 4, 5, 5], dtype=np.int64),
    )

    restored = runtime.restore('xin chao', max_length=4)

    assert restored == 'xin chao.'
    assert model_session.inputs[0]['input_ids'].tolist() == [[0, 11, 12, 2]]
    assert decode_session.inputs[0]['ids'].tolist() == [5, 2]


def test_bundle_runtime_lowercases_input_when_bundle_requests_it():
    manifest = ModelBundleManifest(
        bundle_version=1,
        project='vpcd',
        model_family='bartpho-seq2seq',
        model_name='tourmii/vietnamese-punc-cap-denorm-v1',
        model_variant='vpcd_balanced',
        asset_namespace='models/punctuation/vpcd/vpcd_balanced',
        runtime_kind='text_seq2seq',
        artifacts={
            'model': 'model.mobile.onnx',
            'tokenizer_encode': 'tokenizer.encode.onnx',
            'tokenizer_decode': 'tokenizer.decode.onnx',
            'tokenizer_to_model_id_map': 'tokenizer.to_model_id_map.json',
            'model_to_tokenizer_id_map': 'tokenizer.from_model_id_map.json',
        },
        fixtures={'golden_samples': 'golden_samples.jsonl'},
        metadata={
            'pad_token_id': 1,
            'eos_token_id': 2,
            'decoder_start_token_id': 2,
            'max_source_length': 8,
            'max_decode_length': 4,
            'input_text_case': 'lower',
        },
    )

    class FakeSession:
        def __init__(self, responses: list[object]):
            self.responses = list(responses)
            self.inputs: list[dict[str, object]] = []

        def run(self, _outputs: object, feeds: dict[str, object]) -> list[object]:
            self.inputs.append(feeds)
            return [self.responses.pop(0)]

    encode_session = FakeSession([np.asarray([[0, 4, 5, 2]], dtype=np.int64)])
    model_session = FakeSession(
        [
            np.asarray([[[0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 0.0]]], dtype=np.float32),
            np.asarray([[[0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0]]], dtype=np.float32),
        ]
    )
    decode_session = FakeSession([np.asarray(['xin chao.'], dtype=object)])

    runtime = BundleOnnxRuntime(
        manifest=manifest,
        model_session=model_session,
        encode_session=encode_session,
        decode_session=decode_session,
        tokenizer_to_model_ids=np.asarray([0, 1, 2, 3, 11, 12], dtype=np.int64),
        model_to_tokenizer_ids=np.asarray([0, 1, 2, 3, 4, 5, 5], dtype=np.int64),
    )

    restored = runtime.restore('XIN CHAO', max_length=4)

    assert restored == 'xin chao.'
    assert encode_session.inputs[0]['inputs'].tolist() == ['xin chao']


def test_bundle_runtime_pads_fixed_shape_model_inputs():
    manifest = ModelBundleManifest(
        bundle_version=1,
        project='vpcd',
        model_family='bartpho-seq2seq',
        model_name='tourmii/vietnamese-punc-cap-denorm-v1',
        model_variant='vpcd_balanced_fixed_1024x128',
        asset_namespace='models/punctuation/vpcd/vpcd_balanced',
        runtime_kind='text_seq2seq',
        artifacts={
            'model': 'model.mobile.onnx',
            'tokenizer_encode': 'tokenizer.encode.onnx',
            'tokenizer_decode': 'tokenizer.decode.onnx',
            'tokenizer_to_model_id_map': 'tokenizer.to_model_id_map.json',
            'model_to_tokenizer_id_map': 'tokenizer.from_model_id_map.json',
        },
        fixtures={'golden_samples': 'golden_samples.jsonl'},
        metadata={
            'pad_token_id': 1,
            'eos_token_id': 2,
            'decoder_start_token_id': 2,
            'max_source_length': 1024,
            'max_decode_length': 128,
            'fixed_input_shapes': {
                'model': {
                    'input_ids': [1, 1024],
                    'attention_mask': [1, 1024],
                    'decoder_input_ids': [1, 128],
                    'decoder_attention_mask': [1, 128],
                }
            },
        },
    )

    class FakeSession:
        def __init__(self, responses: list[object]):
            self.responses = list(responses)
            self.inputs: list[dict[str, object]] = []

        def run(self, _outputs: object, feeds: dict[str, object]) -> list[object]:
            self.inputs.append(feeds)
            return [self.responses.pop(0)]

    logits = np.zeros((1, 128, 7), dtype=np.float32)
    logits[0, 0, 2] = 9.0
    logits[0, 127, 5] = 99.0
    encode_session = FakeSession([np.asarray([[0, 4, 5, 2]], dtype=np.int64)])
    model_session = FakeSession([logits])
    decode_session = FakeSession([np.asarray([''], dtype=object)])

    runtime = BundleOnnxRuntime(
        manifest=manifest,
        model_session=model_session,
        encode_session=encode_session,
        decode_session=decode_session,
        tokenizer_to_model_ids=np.asarray([0, 1, 2, 3, 11, 12], dtype=np.int64),
        model_to_tokenizer_ids=np.asarray([0, 1, 2, 3, 4, 5, 6], dtype=np.int64),
    )

    runtime.restore('xin chao')

    first_feed = model_session.inputs[0]
    assert first_feed['input_ids'].shape == (1, 1024)
    assert first_feed['attention_mask'].shape == (1, 1024)
    assert first_feed['decoder_input_ids'].shape == (1, 128)
    assert first_feed['decoder_attention_mask'].shape == (1, 128)
    assert first_feed['input_ids'][0, :4].tolist() == [0, 11, 12, 2]
    assert first_feed['input_ids'][0, 4] == 1
    assert first_feed['attention_mask'][0, :4].tolist() == [1, 1, 1, 1]
    assert first_feed['attention_mask'][0, 4] == 0
    assert first_feed['decoder_input_ids'][0, :2].tolist() == [2, 1]
    assert first_feed['decoder_attention_mask'][0, :2].tolist() == [1, 0]


def test_bundle_runtime_reads_fixed_decoder_logits_at_active_position():
    manifest = ModelBundleManifest(
        bundle_version=1,
        project='vpcd',
        model_family='bartpho-seq2seq',
        model_name='tourmii/vietnamese-punc-cap-denorm-v1',
        model_variant='vpcd_balanced_fixed_8x4',
        asset_namespace='models/punctuation/vpcd/vpcd_balanced',
        runtime_kind='text_seq2seq',
        artifacts={
            'model': 'model.mobile.onnx',
            'tokenizer_encode': 'tokenizer.encode.onnx',
            'tokenizer_decode': 'tokenizer.decode.onnx',
            'tokenizer_to_model_id_map': 'tokenizer.to_model_id_map.json',
            'model_to_tokenizer_id_map': 'tokenizer.from_model_id_map.json',
        },
        fixtures={'golden_samples': 'golden_samples.jsonl'},
        metadata={
            'pad_token_id': 1,
            'eos_token_id': 2,
            'decoder_start_token_id': 2,
            'max_source_length': 8,
            'max_decode_length': 4,
            'fixed_input_shapes': {
                'model': {
                    'input_ids': [1, 8],
                    'attention_mask': [1, 8],
                    'decoder_input_ids': [1, 4],
                    'decoder_attention_mask': [1, 4],
                }
            },
        },
    )

    class FakeSession:
        def __init__(self, responses: list[object]):
            self.responses = list(responses)
            self.inputs: list[dict[str, object]] = []

        def run(self, _outputs: object, feeds: dict[str, object]) -> list[object]:
            self.inputs.append(feeds)
            return [self.responses.pop(0)]

    first_logits = np.zeros((1, 4, 7), dtype=np.float32)
    first_logits[0, 0, 5] = 9.0
    first_logits[0, 3, 6] = 99.0
    second_logits = np.zeros((1, 4, 7), dtype=np.float32)
    second_logits[0, 1, 2] = 9.0
    second_logits[0, 3, 6] = 99.0

    encode_session = FakeSession([np.asarray([[0, 4, 2]], dtype=np.int64)])
    model_session = FakeSession([first_logits, second_logits])
    decode_session = FakeSession([np.asarray(['xin chao.'], dtype=object)])

    runtime = BundleOnnxRuntime(
        manifest=manifest,
        model_session=model_session,
        encode_session=encode_session,
        decode_session=decode_session,
        tokenizer_to_model_ids=np.asarray([0, 1, 2, 3, 11], dtype=np.int64),
        model_to_tokenizer_ids=np.asarray([0, 1, 2, 3, 4, 5, 6], dtype=np.int64),
    )

    restored = runtime.restore('xin chao', max_length=4)

    assert restored == 'xin chao.'
    assert model_session.inputs[0]['decoder_input_ids'][0, :4].tolist() == [2, 1, 1, 1]
    assert model_session.inputs[0]['decoder_attention_mask'][0, :4].tolist() == [1, 0, 0, 0]
    assert model_session.inputs[1]['decoder_input_ids'][0, :4].tolist() == [2, 5, 1, 1]
    assert model_session.inputs[1]['decoder_attention_mask'][0, :4].tolist() == [1, 1, 0, 0]
    assert decode_session.inputs[0]['ids'].tolist() == [5, 2]


def test_bundle_runtime_restore_with_model_step_accepts_pluggable_runner():
    manifest = ModelBundleManifest(
        bundle_version=1,
        project='vpcd',
        model_family='bartpho-seq2seq',
        model_name='tourmii/vietnamese-punc-cap-denorm-v1',
        model_variant='vpcd_balanced_fixed_8x4',
        asset_namespace='models/punctuation/vpcd/vpcd_balanced',
        runtime_kind='text_seq2seq',
        artifacts={
            'model': 'model.mobile.onnx',
            'tokenizer_encode': 'tokenizer.encode.onnx',
            'tokenizer_decode': 'tokenizer.decode.onnx',
            'tokenizer_to_model_id_map': 'tokenizer.to_model_id_map.json',
            'model_to_tokenizer_id_map': 'tokenizer.from_model_id_map.json',
        },
        fixtures={'golden_samples': 'golden_samples.jsonl'},
        metadata={
            'pad_token_id': 1,
            'eos_token_id': 2,
            'decoder_start_token_id': 2,
            'max_source_length': 8,
            'max_decode_length': 4,
            'fixed_input_shapes': {
                'model': {
                    'input_ids': [1, 8],
                    'attention_mask': [1, 8],
                    'decoder_input_ids': [1, 4],
                    'decoder_attention_mask': [1, 4],
                }
            },
        },
    )

    class FakeSession:
        def __init__(self, responses: list[object]):
            self.responses = list(responses)
            self.inputs: list[dict[str, object]] = []

        def run(self, _outputs: object, feeds: dict[str, object]) -> list[object]:
            self.inputs.append(feeds)
            return [self.responses.pop(0)]

    first_logits = np.zeros((1, 4, 7), dtype=np.float32)
    first_logits[0, 0, 5] = 9.0
    second_logits = np.zeros((1, 4, 7), dtype=np.float32)
    second_logits[0, 1, 2] = 9.0

    encode_session = FakeSession([np.asarray([[0, 4, 2]], dtype=np.int64)])
    decode_session = FakeSession([np.asarray(['xin chao.'], dtype=object)])
    model_step_inputs: list[dict[str, object]] = []
    model_step_outputs = [first_logits, second_logits]

    def fake_model_step(feeds: dict[str, object]) -> object:
        model_step_inputs.append(feeds)
        return model_step_outputs.pop(0)

    runtime = BundleOnnxRuntime(
        manifest=manifest,
        model_session=object(),
        encode_session=encode_session,
        decode_session=decode_session,
        tokenizer_to_model_ids=np.asarray([0, 1, 2, 3, 11], dtype=np.int64),
        model_to_tokenizer_ids=np.asarray([0, 1, 2, 3, 4, 5, 6], dtype=np.int64),
    )

    restored = runtime.restore_with_model_step('xin chao', fake_model_step, max_length=4)

    assert restored['text'] == 'xin chao.'
    assert restored['decode_steps'] == 2
    assert restored['ended_with_eos'] is True
    assert model_step_inputs[0]['decoder_input_ids'][0, :4].tolist() == [2, 1, 1, 1]
    assert model_step_inputs[0]['decoder_attention_mask'][0, :4].tolist() == [1, 0, 0, 0]
    assert model_step_inputs[1]['decoder_input_ids'][0, :4].tolist() == [2, 5, 1, 1]
    assert model_step_inputs[1]['decoder_attention_mask'][0, :4].tolist() == [1, 1, 0, 0]
    assert decode_session.inputs[0]['ids'].tolist() == [5, 2]


def test_export_vpcd_bundle_writes_standardized_layout(tmp_case_dir):
    model_dir = tmp_case_dir / 'model'
    output_dir = tmp_case_dir / 'output'
    (model_dir / 'onnx').mkdir(parents=True, exist_ok=True)
    (model_dir / 'onnx' / 'vpcd_balanced.onnx').write_bytes(b'dummy-onnx')

    def fake_tokenizer_exporter(_model_dir: str, bundle_dir: str) -> TokenizerExportArtifacts:
        bundle_path = Path(bundle_dir)
        (bundle_path / 'tokenizer.encode.onnx').write_bytes(b'encode')
        (bundle_path / 'tokenizer.decode.onnx').write_bytes(b'decode')
        (bundle_path / 'tokenizer.to_model_id_map.json').write_text('[0,1,2]\n', encoding='utf-8')
        (bundle_path / 'tokenizer.from_model_id_map.json').write_text('[0,1,2]\n', encoding='utf-8')
        return TokenizerExportArtifacts(
            encode_file_name='tokenizer.encode.onnx',
            decode_file_name='tokenizer.decode.onnx',
            tokenizer_to_model_id_map_file_name='tokenizer.to_model_id_map.json',
            model_to_tokenizer_id_map_file_name='tokenizer.from_model_id_map.json',
        )

    def fake_golden_sample_builder(**_: object) -> list[TextGoldenSample]:
        return [
            TextGoldenSample(
                raw_text='hom nay la buoi nham chuc cua toi phuoc thanh',
                input_ids=[0, 12, 18, 2],
                expected_output='Hôm nay là buổi nhậm chức của tôi Phước Thành.',
            )
        ]

    manifest = export_bundle(
        model_dir=model_dir,
        output_dir=output_dir,
        model_variant='vpcd_balanced',
        tokenizer_exporter=fake_tokenizer_exporter,
        golden_sample_builder=fake_golden_sample_builder,
    )

    assert manifest.model_variant == 'vpcd_balanced'
    assert (output_dir / 'bundle_manifest.json').exists()
    assert (output_dir / 'model.mobile.onnx').exists()
    assert (output_dir / 'tokenizer.encode.onnx').exists()
    assert (output_dir / 'tokenizer.decode.onnx').exists()
    assert (output_dir / 'tokenizer.to_model_id_map.json').exists()
    assert (output_dir / 'tokenizer.from_model_id_map.json').exists()
    assert (output_dir / 'golden_samples.jsonl').exists()
    assert manifest.metadata['input_text_case'] == 'lower'
    assert manifest.metadata['quantization']['format'] == 'QDQ'
    assert manifest.metadata['quantization']['activation_type'] == 'quint16'
    assert manifest.metadata['quantization']['weight_type'] == 'quint8'
    assert manifest.metadata['qnn_readiness']['fixed_shapes_ready'] is False


def test_verify_vpcd_candidate_bundle_matches_reference(monkeypatch, tmp_case_dir):
    reference_bundle = tmp_case_dir / 'reference'
    candidate_bundle = tmp_case_dir / 'candidate'
    reference_bundle.mkdir()
    candidate_bundle.mkdir()
    samples = [
        TextGoldenSample(
            raw_text='xin chao',
            input_ids=[0, 12, 2],
            expected_output='Xin chao.',
        )
    ]
    (candidate_bundle / 'golden_samples.jsonl').write_text(serialize_jsonl(samples), encoding='utf-8')

    for bundle_dir, variant in ((reference_bundle, 'vpcd_balanced'), (candidate_bundle, 'vpcd_balanced_fixed_1024x128')):
        manifest = ModelBundleManifest(
            bundle_version=1,
            project='vpcd',
            model_family='bartpho-seq2seq',
            model_name='tourmii/vietnamese-punc-cap-denorm-v1',
            model_variant=variant,
            asset_namespace='models/punctuation/vpcd/vpcd_balanced',
            runtime_kind='text_seq2seq',
            artifacts={
                'model': 'model.mobile.onnx',
                'tokenizer_encode': 'tokenizer.encode.onnx',
                'tokenizer_decode': 'tokenizer.decode.onnx',
                'tokenizer_to_model_id_map': 'tokenizer.to_model_id_map.json',
                'model_to_tokenizer_id_map': 'tokenizer.from_model_id_map.json',
            },
            fixtures={'golden_samples': 'golden_samples.jsonl'},
            metadata={
                'pad_token_id': 1,
                'eos_token_id': 2,
                'decoder_start_token_id': 2,
                'max_source_length': 1024,
                'max_decode_length': 128,
            },
        )
        manifest.write_json(bundle_dir / 'bundle_manifest.json')

    class FakeRuntime:
        @classmethod
        def from_manifest_path(cls, manifest_path: str | Path, provider: str = 'CPUExecutionProvider'):
            return cls()

        def restore(self, text: str, max_length: int = 128) -> str:
            return 'Xin chao.'

    monkeypatch.setattr('quantize.vpcd_bundle.BundleOnnxRuntime', FakeRuntime)

    report = verify_bundle(reference_bundle=reference_bundle, candidate_bundle=candidate_bundle)

    assert report == {
        'checked_samples': 1,
        'passed': True,
        'mismatches': [],
    }


def test_verify_vpcd_candidate_bundle_reports_mismatches(monkeypatch, tmp_case_dir):
    reference_bundle = tmp_case_dir / 'reference'
    candidate_bundle = tmp_case_dir / 'candidate'
    reference_bundle.mkdir()
    candidate_bundle.mkdir()
    samples = [
        TextGoldenSample(
            raw_text='xin chao',
            input_ids=[0, 12, 2],
            expected_output='Xin chao.',
        )
    ]
    (candidate_bundle / 'golden_samples.jsonl').write_text(serialize_jsonl(samples), encoding='utf-8')

    for bundle_dir in (reference_bundle, candidate_bundle):
        manifest = ModelBundleManifest(
            bundle_version=1,
            project='vpcd',
            model_family='bartpho-seq2seq',
            model_name='tourmii/vietnamese-punc-cap-denorm-v1',
            model_variant='vpcd_balanced',
            asset_namespace='models/punctuation/vpcd/vpcd_balanced',
            runtime_kind='text_seq2seq',
            artifacts={'model': 'model.mobile.onnx'},
            fixtures={'golden_samples': 'golden_samples.jsonl'},
            metadata={'max_decode_length': 128},
        )
        manifest.write_json(bundle_dir / 'bundle_manifest.json')

    class FakeRuntime:
        def __init__(self, label: str):
            self.label = label

        @classmethod
        def from_manifest_path(cls, manifest_path: str | Path, provider: str = 'CPUExecutionProvider'):
            label = 'candidate' if 'candidate' in str(manifest_path) else 'reference'
            return cls(label)

        def restore(self, text: str, max_length: int = 128) -> str:
            return 'Xin chao?' if self.label == 'candidate' else 'Xin chao.'

    monkeypatch.setattr('quantize.vpcd_bundle.BundleOnnxRuntime', FakeRuntime)

    report = verify_bundle(reference_bundle=reference_bundle, candidate_bundle=candidate_bundle)

    assert report['checked_samples'] == 1
    assert report['passed'] is False
    assert report['mismatches'] == [
        {
            'raw_text': 'xin chao',
            'reference_output': 'Xin chao.',
            'candidate_output': 'Xin chao?',
        }
    ]
