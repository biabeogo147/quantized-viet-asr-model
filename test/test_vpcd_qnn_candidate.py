import json
from pathlib import Path

import pytest

from model_bundle.manifest import ModelBundleManifest
from tools.prepare_vpcd_qnn_candidate import main


def _write_source_bundle(bundle_dir: Path, *, project: str = 'vpcd', quantization: dict | None = None) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    for name, content in {
        'model.mobile.onnx': b'model',
        'tokenizer.encode.onnx': b'encode',
        'tokenizer.decode.onnx': b'decode',
        'tokenizer.to_model_id_map.json': b'[0,1,2]\n',
        'tokenizer.from_model_id_map.json': b'[0,1,2]\n',
        'golden_samples.jsonl': b'{"raw_text":"xin chao","input_ids":[0,2],"expected_output":"Xin chao."}\n',
    }.items():
        (bundle_dir / name).write_bytes(content)

    metadata = {
        'pad_token_id': 1,
        'eos_token_id': 2,
        'decoder_start_token_id': 2,
        'max_source_length': 1024,
        'max_decode_length': 128,
        'input_text_case': 'lower',
        'quantization': quantization
        or {
            'format': 'QDQ',
            'activation_type': 'quint16',
            'weight_type': 'quint8',
            'preset': 'sd8g2_balanced',
            'fixed_shapes': False,
        },
        'qnn_readiness': {
            'target_backend': 'qnn_htp',
            'model_session_candidate': True,
            'tokenizer_policy': 'cpu_only_first_slice',
            'requires_fixed_shapes': True,
            'fixed_shapes_ready': False,
            'fixed_shape_blocker': 'dynamic shapes',
        },
    }
    manifest = ModelBundleManifest(
        bundle_version=1,
        project=project,
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
        metadata=metadata,
    )
    manifest.write_json(bundle_dir / 'bundle_manifest.json')


def test_prepare_vpcd_qnn_candidate_updates_manifest(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    output = tmp_path / 'candidate'
    _write_source_bundle(source)

    frozen_calls = []

    def fake_freeze_model_inputs(model_path: Path, output_path: Path, input_shapes: dict[str, list[int]]) -> Path:
        frozen_calls.append((Path(model_path), Path(output_path), input_shapes))
        Path(output_path).write_bytes(b'frozen')
        return Path(output_path)

    monkeypatch.setattr('tools.prepare_vpcd_qnn_candidate.freeze_model_inputs', fake_freeze_model_inputs)

    main(
        [
            '--source-bundle',
            str(source),
            '--output-dir',
            str(output),
            '--encoder-sequence',
            '1024',
            '--decoder-sequence',
            '128',
        ]
    )

    manifest = ModelBundleManifest.from_path(output / 'bundle_manifest.json')

    assert frozen_calls == [
        (
            source / 'model.mobile.onnx',
            output / 'model.mobile.onnx',
            {
                'input_ids': [1, 1024],
                'attention_mask': [1, 1024],
                'decoder_input_ids': [1, 128],
                'decoder_attention_mask': [1, 128],
            },
        )
    ]
    assert manifest.model_variant == 'vpcd_balanced_fixed_1024x128'
    assert manifest.metadata['fixed_input_shapes']['model']['input_ids'] == [1, 1024]
    assert manifest.metadata['fixed_input_shapes']['model']['decoder_input_ids'] == [1, 128]
    assert manifest.metadata['quantization']['fixed_shapes'] is True
    assert manifest.metadata['qnn_readiness']['fixed_shapes_ready'] is True
    assert 'fixed_shape_blocker' not in manifest.metadata['qnn_readiness']
    assert (output / 'tokenizer.encode.onnx').read_bytes() == b'encode'
    assert (output / 'model.mobile.onnx').read_bytes() == b'frozen'


def test_prepare_vpcd_qnn_candidate_rejects_non_vpcd_bundle(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    output = tmp_path / 'candidate'
    _write_source_bundle(source, project='zipformer')
    monkeypatch.setattr('tools.prepare_vpcd_qnn_candidate.freeze_model_inputs', lambda *_args, **_kwargs: None)

    with pytest.raises(ValueError, match='project must be vpcd'):
        main(['--source-bundle', str(source), '--output-dir', str(output)])


def test_prepare_vpcd_qnn_candidate_rejects_non_qdq_quantization(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    output = tmp_path / 'candidate'
    _write_source_bundle(
        source,
        quantization={
            'format': 'QOperator',
            'activation_type': 'quint16',
            'weight_type': 'quint8',
            'fixed_shapes': False,
        },
    )
    monkeypatch.setattr('tools.prepare_vpcd_qnn_candidate.freeze_model_inputs', lambda *_args, **_kwargs: None)

    with pytest.raises(ValueError, match='QDQ'):
        main(['--source-bundle', str(source), '--output-dir', str(output)])


def test_prepare_vpcd_qnn_candidate_rejects_same_output_dir(tmp_path):
    source = tmp_path / 'source'
    _write_source_bundle(source)

    with pytest.raises(ValueError, match='same as source'):
        main(['--source-bundle', str(source), '--output-dir', str(source)])
