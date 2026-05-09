from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from model_bundle.contracts import BundleProjectAdapter
from model_bundle.fixtures import TextGoldenSample, read_jsonl, serialize_jsonl
from model_bundle.layout import resolve_bundle_dir
from model_bundle.manifest import ModelBundleManifest
from model_bundle.projects._vpcd_support import (
    DEFAULT_ASSET_NAMESPACE,
    DEFAULT_TEXTS,
    DEFAULT_MODEL_VARIANT,
    GOLDEN_SAMPLES_FILE_NAME,
    MODEL_FILE_NAME,
    BundleOnnxRuntime,
    ModelDirOnnxRuntime,
    TokenizerExportArtifacts,
    TokenizerIdBridge,
    VietnamesePuncCapDenormOnnx,
    bartpho_tokenizer_ortx_alias,
    build_ort_tokenizer_id_bridge,
    default_golden_sample_builder,
    default_tokenizer_exporter,
    ensure_local_vendor_path,
    resolve_variant_onnx_path,
)

DEFAULT_MODEL_DIR = Path('assets') / 'vietnamese-punc-cap-denorm-v1'
DEFAULT_OUTPUT_DIR = resolve_bundle_dir('vpcd', 'fp32')


def build_vpcd_metadata(*, model_variant: str, max_decode_length: int) -> dict[str, object]:
    metadata: dict[str, object] = {
        'pad_token_id': 1,
        'eos_token_id': 2,
        'decoder_start_token_id': 2,
        'max_source_length': 1024,
        'max_decode_length': max_decode_length,
        'input_text_case': 'lower',
    }

    variant_name = Path(str(model_variant)).stem
    if variant_name == DEFAULT_MODEL_VARIANT or 'qnn.qdq.uint16u8' in variant_name:
        metadata['quantization'] = {
            'format': 'QDQ',
            'activation_type': 'quint16',
            'weight_type': 'quint8',
            'preset': 'sd8g2_balanced',
            'fixed_shapes': False,
        }
        metadata['qnn_readiness'] = {
            'target_backend': 'qnn_htp',
            'model_session_candidate': True,
            'tokenizer_policy': 'cpu_only_first_slice',
            'requires_fixed_shapes': True,
            'fixed_shapes_ready': False,
            'fixed_shape_blocker': 'VPCD model inputs remain symbolic: input_ids[batch,encoder_sequence], attention_mask[batch,encoder_sequence], decoder_input_ids[batch,decoder_sequence], decoder_attention_mask[batch,decoder_sequence].',
        }

    return metadata


def export_bundle(
    *,
    model_dir: Path,
    output_dir: Path,
    model_variant: str = DEFAULT_MODEL_VARIANT,
    asset_namespace: str = DEFAULT_ASSET_NAMESPACE,
    tokenizer_exporter=default_tokenizer_exporter,
    golden_sample_builder=default_golden_sample_builder,
    max_decode_length: int = 128,
) -> ModelBundleManifest:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_onnx_path = resolve_variant_onnx_path(str(model_dir), model_variant)
    if not source_onnx_path.exists():
        raise FileNotFoundError(f'Khong tim thay ONNX variant: {source_onnx_path}')

    model_output_path = output_dir / MODEL_FILE_NAME
    shutil.copy2(source_onnx_path, model_output_path)
    tokenizer_artifacts: TokenizerExportArtifacts = tokenizer_exporter(str(model_dir), str(output_dir))

    samples = [
        TextGoldenSample(
            raw_text=sample.raw_text,
            input_ids=list(sample.input_ids),
            expected_output=sample.expected_output,
        )
        for sample in golden_sample_builder(model_dir=str(model_dir), onnx_path=str(model_output_path), max_decode_length=max_decode_length)
    ]
    golden_path = output_dir / GOLDEN_SAMPLES_FILE_NAME
    golden_path.write_text(serialize_jsonl(samples), encoding='utf-8')

    manifest = ModelBundleManifest(
        bundle_version=1,
        project='vpcd',
        model_family='bartpho-seq2seq',
        model_name='tourmii/vietnamese-punc-cap-denorm-v1',
        model_variant=model_variant,
        asset_namespace=asset_namespace,
        runtime_kind='text_seq2seq',
        artifacts={
            'model': model_output_path.name,
            'tokenizer_encode': tokenizer_artifacts.encode_file_name,
            'tokenizer_decode': tokenizer_artifacts.decode_file_name,
            'tokenizer_to_model_id_map': tokenizer_artifacts.tokenizer_to_model_id_map_file_name,
            'model_to_tokenizer_id_map': tokenizer_artifacts.model_to_tokenizer_id_map_file_name,
        },
        fixtures={'golden_samples': golden_path.name},
        metadata=build_vpcd_metadata(model_variant=model_variant, max_decode_length=max_decode_length),
    )
    manifest.write_json(output_dir / 'bundle_manifest.json')
    return manifest


def iter_golden_samples(bundle_dir: str | Path) -> list[dict]:
    manifest = ModelBundleManifest.from_path(Path(bundle_dir) / 'bundle_manifest.json')
    return read_jsonl(Path(bundle_dir) / manifest.fixtures['golden_samples'])


def _verify_candidate_bundle_parity(*, reference_bundle: Path, candidate_bundle: Path, provider: str) -> dict:
    reference_runtime = BundleOnnxRuntime.from_manifest_path(reference_bundle / 'bundle_manifest.json', provider=provider)
    candidate_runtime = BundleOnnxRuntime.from_manifest_path(candidate_bundle / 'bundle_manifest.json', provider=provider)
    candidate_manifest = ModelBundleManifest.from_path(candidate_bundle / 'bundle_manifest.json')
    samples = read_jsonl(candidate_bundle / candidate_manifest.fixtures['golden_samples'])

    mismatches: list[dict[str, str]] = []
    checked = 0
    max_decode_length = int(candidate_manifest.metadata.get('max_decode_length', 128))
    for sample in samples:
        raw_text = str(sample['raw_text'])
        reference_output = reference_runtime.restore(raw_text, max_length=max_decode_length).strip()
        candidate_output = candidate_runtime.restore(raw_text, max_length=max_decode_length).strip()
        checked += 1
        if reference_output != candidate_output:
            mismatches.append(
                {
                    'raw_text': raw_text,
                    'reference_output': reference_output,
                    'candidate_output': candidate_output,
                }
            )

    return {
        'checked_samples': checked,
        'passed': not mismatches,
        'mismatches': mismatches,
    }


def verify_bundle(
    *,
    model_dir: Path | None = None,
    bundle_dir: Path | None = None,
    reference_bundle: Path | None = None,
    candidate_bundle: Path | None = None,
    provider: str = 'CPUExecutionProvider',
) -> tuple[int, int] | dict:
    if reference_bundle is not None or candidate_bundle is not None:
        if reference_bundle is None or candidate_bundle is None:
            raise ValueError('Both reference_bundle and candidate_bundle are required for VPCD parity verification')
        return _verify_candidate_bundle_parity(
            reference_bundle=Path(reference_bundle),
            candidate_bundle=Path(candidate_bundle),
            provider=provider,
        )
    if model_dir is None or bundle_dir is None:
        raise ValueError('model_dir and bundle_dir are required for VPCD bundle verification')

    ensure_local_vendor_path()
    import onnxruntime as ort
    from onnxruntime_extensions import get_library_path

    model_dir = Path(model_dir)
    bundle_dir = Path(bundle_dir)
    manifest = ModelBundleManifest.from_path(bundle_dir / 'bundle_manifest.json')
    tokenizer_to_model_ids = np.asarray(json.loads((bundle_dir / manifest.artifacts['tokenizer_to_model_id_map']).read_text(encoding='utf-8')), dtype=np.int64)
    model_to_tokenizer_ids = np.asarray(json.loads((bundle_dir / manifest.artifacts['model_to_tokenizer_id_map']).read_text(encoding='utf-8')), dtype=np.int64)

    session_options = ort.SessionOptions()
    session_options.register_custom_ops_library(get_library_path())
    encode_session = ort.InferenceSession(str(bundle_dir / manifest.artifacts['tokenizer_encode']), session_options, providers=['CPUExecutionProvider'])
    decode_session = ort.InferenceSession(str(bundle_dir / manifest.artifacts['tokenizer_decode']), session_options, providers=['CPUExecutionProvider'])

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    encode_verified = 0
    decode_verified = 0
    for sample in iter_golden_samples(bundle_dir):
        raw_text = str(sample['raw_text'])
        expected_input_ids = np.asarray(sample['input_ids'], dtype=np.int64)
        ort_tokenizer_ids = np.asarray(encode_session.run(None, {'inputs': np.asarray([raw_text], dtype=object)})[0], dtype=np.int64)
        bridged_model_ids = tokenizer_to_model_ids[ort_tokenizer_ids]
        if not np.array_equal(bridged_model_ids, expected_input_ids):
            raise AssertionError(f"Tokenizer encode mismatch for '{raw_text}': {bridged_model_ids.tolist()} != {expected_input_ids.tolist()}")
        encode_verified += 1

        bridged_tokenizer_ids = model_to_tokenizer_ids[expected_input_ids]
        decoded_text = decode_session.run(None, {'ids': bridged_tokenizer_ids})[0].tolist()[0]
        with bartpho_tokenizer_ortx_alias(tokenizer):
            expected_decoded_text = tokenizer.decode(expected_input_ids.tolist(), skip_special_tokens=True)
        if decoded_text != expected_decoded_text:
            raise AssertionError(f"Tokenizer decode mismatch for '{raw_text}': {decoded_text!r} != {expected_decoded_text!r}")
        decode_verified += 1
    return encode_verified, decode_verified


ADAPTER = BundleProjectAdapter(
    name='vpcd',
    default_model_dir=str(DEFAULT_MODEL_DIR),
    default_output_dir=str(DEFAULT_OUTPUT_DIR),
    default_asset_namespace=DEFAULT_ASSET_NAMESPACE,
    default_variant=DEFAULT_MODEL_VARIANT,
    export_bundle=export_bundle,
    verify_bundle=verify_bundle,
    bundle_runtime_from_manifest=BundleOnnxRuntime.from_manifest_path,
)
