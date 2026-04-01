import json
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from model_bundle.fixtures import TextGoldenSample, serialize_jsonl
from model_bundle.manifest import ModelBundleManifest
from model_bundle.projects.vpcd import (
    BundleOnnxRuntime,
    TokenizerExportArtifacts,
    TokenizerIdBridge,
    export_bundle,
)
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
        asset_namespace='models/punctuation/vpcd',
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

    payload = manifest.to_dict()

    assert payload['project'] == 'vpcd'
    assert payload['artifacts']['model'] == 'model.mobile.onnx'
    assert payload['fixtures']['golden_samples'] == 'golden_samples.jsonl'


def test_text_golden_samples_serialize_to_jsonl():
    sample = TextGoldenSample(
        raw_text='hom nay la buoi nham chuc cua toi phuoc thanh',
        input_ids=[0, 12, 18, 2],
        expected_output='Hôm nay là buổi nhậm chức của tôi Phước Thành.',
    )

    serialized = serialize_jsonl([sample])

    assert serialized == sample.to_jsonl_line()


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
    args = parser.parse_args(['--bundle-manifest', 'build/model_bundle/vpcd/fp32/bundle_manifest.json'])
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
        'manifest_path': 'build/model_bundle/vpcd/fp32/bundle_manifest.json',
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
                'build/model_bundle/vpcd/fp32/bundle_manifest.json',
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
        asset_namespace='models/punctuation/vpcd',
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
