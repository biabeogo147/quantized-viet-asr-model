import json
from pathlib import Path

import onnx
from onnx import TensorProto, helper, numpy_helper

import numpy as np

from model_bundle.manifest import ModelBundleManifest
from model_bundle.qnn_preflight import inspect_onnx_for_qnn_qdq, verify_qnn_preflight
from verify.qnn_preflight import main


def _write_qdq_model(path: Path, *, symbolic: bool = False, include_qdq: bool = True) -> None:
    encoder_dim = 'encoder_sequence' if symbolic else 1024
    decoder_dim = 'decoder_sequence' if symbolic else 128
    inputs = [
        helper.make_tensor_value_info('input_ids', TensorProto.INT64, [1, encoder_dim]),
        helper.make_tensor_value_info('attention_mask', TensorProto.INT64, [1, encoder_dim]),
        helper.make_tensor_value_info('decoder_input_ids', TensorProto.INT64, [1, decoder_dim]),
        helper.make_tensor_value_info('decoder_attention_mask', TensorProto.INT64, [1, decoder_dim]),
        helper.make_tensor_value_info('x', TensorProto.FLOAT, [1]),
    ]
    outputs = [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1])]
    initializers = [
        numpy_helper.from_array(np.asarray([0.25], dtype=np.float32), name='x_scale'),
        numpy_helper.from_array(np.asarray([0], dtype=np.uint16), name='x_zero_point'),
        numpy_helper.from_array(np.asarray([1], dtype=np.uint8), name='weight_u8'),
    ]
    if include_qdq:
        nodes = [
            helper.make_node('QuantizeLinear', ['x', 'x_scale', 'x_zero_point'], ['x_q']),
            helper.make_node('DequantizeLinear', ['x_q', 'x_scale', 'x_zero_point'], ['y']),
        ]
    else:
        nodes = [helper.make_node('Identity', ['x'], ['y'])]
    graph = helper.make_graph(nodes, 'vpcd_qnn_preflight_test', inputs, outputs, initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 17)])
    onnx.save(model, path)


def _write_manifest(bundle_dir: Path, *, fixed_shapes: bool = True, quantization: dict | None = None) -> None:
    metadata = {
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
        'quantization': quantization
        if quantization is not None
        else {
            'format': 'QDQ',
            'activation_type': 'quint16',
            'weight_type': 'quint8',
            'preset': 'sd8g2_balanced',
            'fixed_shapes': fixed_shapes,
        },
        'qnn_readiness': {
            'target_backend': 'qnn_htp',
            'model_session_candidate': True,
            'tokenizer_policy': 'cpu_only_first_slice',
            'requires_fixed_shapes': True,
            'fixed_shapes_ready': fixed_shapes,
        },
    }
    manifest = ModelBundleManifest(
        bundle_version=1,
        project='vpcd',
        model_family='bartpho-seq2seq',
        model_name='tourmii/vietnamese-punc-cap-denorm-v1',
        model_variant='vpcd_balanced_fixed_1024x128',
        asset_namespace='models/punctuation/vpcd/vpcd_balanced',
        runtime_kind='text_seq2seq',
        artifacts={'model': 'model.mobile.onnx'},
        fixtures={},
        metadata=metadata,
    )
    manifest.write_json(bundle_dir / 'bundle_manifest.json')


def _write_bundle(tmp_path: Path, *, symbolic: bool = False, include_qdq: bool = True, fixed_shapes: bool = True, quantization: dict | None = None) -> Path:
    bundle_dir = tmp_path / 'bundle'
    bundle_dir.mkdir()
    _write_qdq_model(bundle_dir / 'model.mobile.onnx', symbolic=symbolic, include_qdq=include_qdq)
    _write_manifest(bundle_dir, fixed_shapes=fixed_shapes, quantization=quantization)
    return bundle_dir


def test_inspect_onnx_for_qnn_qdq_reports_ops_dtypes_and_inputs(tmp_path):
    model_path = tmp_path / 'model.onnx'
    _write_qdq_model(model_path)

    report = inspect_onnx_for_qnn_qdq(model_path)

    assert report['op_counts']['QuantizeLinear'] == 1
    assert report['op_counts']['DequantizeLinear'] == 1
    assert report['initializer_dtypes']['UINT16'] == 1
    assert report['initializer_dtypes']['UINT8'] == 1
    assert report['inputs']['input_ids'] == [1, 1024]
    assert report['symbolic_inputs'] == []


def test_verify_qnn_preflight_passes_for_fixed_vpcd_qdq_bundle(tmp_path):
    bundle_dir = _write_bundle(tmp_path)

    report = verify_qnn_preflight(project='vpcd', bundle_dir=bundle_dir)

    assert report['passed'] is True
    assert report['checks']['manifest_quantization']['passed'] is True
    assert report['checks']['fixed_input_shapes']['passed'] is True
    assert report['checks']['onnx_qdq_graph']['passed'] is True


def test_verify_qnn_preflight_fails_when_manifest_has_dynamic_shapes(tmp_path):
    bundle_dir = _write_bundle(tmp_path, fixed_shapes=False)

    report = verify_qnn_preflight(project='vpcd', bundle_dir=bundle_dir)

    assert report['passed'] is False
    assert report['checks']['manifest_quantization']['passed'] is False
    assert report['checks']['fixed_input_shapes']['passed'] is False


def test_verify_qnn_preflight_fails_when_graph_inputs_are_symbolic(tmp_path):
    bundle_dir = _write_bundle(tmp_path, symbolic=True)

    report = verify_qnn_preflight(project='vpcd', bundle_dir=bundle_dir)

    assert report['passed'] is False
    assert report['checks']['fixed_input_shapes']['passed'] is False
    assert 'input_ids' in report['checks']['fixed_input_shapes']['symbolic_inputs']


def test_verify_qnn_preflight_fails_when_graph_has_no_qdq_nodes(tmp_path):
    bundle_dir = _write_bundle(tmp_path, include_qdq=False)

    report = verify_qnn_preflight(project='vpcd', bundle_dir=bundle_dir)

    assert report['passed'] is False
    assert report['checks']['onnx_qdq_graph']['passed'] is False


def test_verify_qnn_preflight_cli_writes_report(tmp_path):
    bundle_dir = _write_bundle(tmp_path)
    output = tmp_path / 'report.json'

    main(['--project', 'vpcd', '--bundle-dir', str(bundle_dir), '--output', str(output)])

    payload = json.loads(output.read_text(encoding='utf-8'))
    assert payload['passed'] is True
    assert payload['project'] == 'vpcd'
