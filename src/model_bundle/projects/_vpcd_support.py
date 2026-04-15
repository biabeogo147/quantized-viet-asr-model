from __future__ import annotations

import json
import shutil
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from model_bundle.fixtures import TextGoldenSample, serialize_jsonl
from model_bundle.manifest import ModelBundleManifest
from tools.paths import resolve_repo_path

DEFAULT_ASSET_NAMESPACE = 'models/punctuation/vpcd'
DEFAULT_MODEL_VARIANT = 'vpcd_balanced'
MODEL_FILE_NAME = 'model.mobile.onnx'
TOKENIZER_ENCODE_FILE_NAME = 'tokenizer.encode.onnx'
TOKENIZER_DECODE_FILE_NAME = 'tokenizer.decode.onnx'
TOKENIZER_TO_MODEL_ID_MAP_FILE_NAME = 'tokenizer.to_model_id_map.json'
MODEL_TO_TOKENIZER_ID_MAP_FILE_NAME = 'tokenizer.from_model_id_map.json'
GOLDEN_SAMPLES_FILE_NAME = 'golden_samples.jsonl'
UNK_TOKEN_ID = 3
DEFAULT_TEXTS = [
    'hom nay la buoi nham chuc cua toi phuoc thanh',
    'chao cac ban hom nay chung ta cung nhau den voi bai hoc deep learning phan so muoi ba',
]


@dataclass(frozen=True)
class TokenizerExportArtifacts:
    encode_file_name: str
    decode_file_name: str
    tokenizer_to_model_id_map_file_name: str
    model_to_tokenizer_id_map_file_name: str


@dataclass(frozen=True)
class TokenizerIdBridge:
    tokenizer_to_model_ids: list[int]
    model_to_tokenizer_ids: list[int]

    def write_files(
        self,
        *,
        tokenizer_to_model_path: str | Path,
        model_to_tokenizer_path: str | Path,
    ) -> tuple[str, str]:
        tokenizer_to_model_file = Path(tokenizer_to_model_path)
        model_to_tokenizer_file = Path(model_to_tokenizer_path)
        tokenizer_to_model_file.write_text(
            json.dumps(self.tokenizer_to_model_ids, ensure_ascii=False, separators=(',', ':')) + '\n',
            encoding='utf-8',
        )
        model_to_tokenizer_file.write_text(
            json.dumps(self.model_to_tokenizer_ids, ensure_ascii=False, separators=(',', ':')) + '\n',
            encoding='utf-8',
        )
        return tokenizer_to_model_file.name, model_to_tokenizer_file.name


TokenizerExporter = Callable[[str, str], TokenizerExportArtifacts]
GoldenSampleBuilder = Callable[..., list[TextGoldenSample]]


def ensure_local_vendor_path() -> None:
    vendor_dir = resolve_repo_path('_vendor', anchor=__file__)
    if vendor_dir.exists():
        vendor_path = str(vendor_dir)
        if vendor_path not in sys.path:
            sys.path.insert(0, vendor_path)


def resolve_variant_onnx_path(model_dir: str | Path, model_variant: str) -> Path:
    variant_file = model_variant if str(model_variant).endswith('.onnx') else f'{model_variant}.onnx'
    return Path(model_dir) / 'onnx' / variant_file


class ModelDirOnnxRuntime:
    def __init__(self, *, model_dir: str, onnx_path: str, provider: str = 'CPUExecutionProvider'):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.model_dir = model_dir
        self.onnx_path = onnx_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.session = ort.InferenceSession(onnx_path, providers=[provider])

        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.decoder_start_token_id = self.tokenizer.eos_token_id

        generation_config_path = Path(model_dir) / 'generation_config.json'
        if generation_config_path.exists():
            generation_config = json.loads(generation_config_path.read_text(encoding='utf-8'))
            self.decoder_start_token_id = generation_config.get('decoder_start_token_id', self.decoder_start_token_id)

    def restore(self, text: str, max_length: int = 128) -> str:
        encoded = self.tokenizer(text, return_tensors='np', truncation=True, max_length=512)
        input_ids = encoded['input_ids'].astype(np.int64)
        attention_mask = encoded['attention_mask'].astype(np.int64)
        decoder_input_ids = np.array([[self.decoder_start_token_id]], dtype=np.int64)

        for _ in range(max_length):
            decoder_attention_mask = np.ones_like(decoder_input_ids, dtype=np.int64)
            outputs = self.session.run(
                None,
                {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'decoder_input_ids': decoder_input_ids,
                    'decoder_attention_mask': decoder_attention_mask,
                },
            )
            logits = outputs[0]
            next_token_id = int(np.argmax(logits[:, -1, :], axis=-1)[0])
            decoder_input_ids = np.concatenate([decoder_input_ids, np.array([[next_token_id]], dtype=np.int64)], axis=1)
            if next_token_id == self.eos_token_id:
                break

        generated_ids = decoder_input_ids[0, 1:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


VietnamesePuncCapDenormOnnx = ModelDirOnnxRuntime


@contextmanager
def bartpho_tokenizer_ortx_alias(tokenizer: object):
    tokenizer_class = tokenizer.__class__
    original_name = tokenizer_class.__name__
    tokenizer_class.__name__ = 'XLMRobertaTokenizer'
    try:
        yield
    finally:
        tokenizer_class.__name__ = original_name


def build_ort_tokenizer_id_bridge(tokenizer: object) -> TokenizerIdBridge:
    sp_model = tokenizer.sp_model
    tokenizer_to_model_ids = [tokenizer.unk_token_id] * (sp_model.get_piece_size() + 1)
    tokenizer_to_model_ids[0] = tokenizer.cls_token_id
    tokenizer_to_model_ids[1] = tokenizer.pad_token_id
    tokenizer_to_model_ids[2] = tokenizer.sep_token_id
    tokenizer_to_model_ids[3] = tokenizer.unk_token_id

    special_model_ids = {
        tokenizer.cls_token_id,
        tokenizer.pad_token_id,
        tokenizer.sep_token_id,
        tokenizer.unk_token_id,
    }

    for token, model_id in tokenizer.fairseq_tokens_to_ids.items():
        if model_id in special_model_ids or token in tokenizer.all_special_tokens:
            continue
        sp_id = sp_model.piece_to_id(token)
        if sp_id >= 0:
            tokenizer_to_model_ids[sp_id + 1] = model_id

    model_to_tokenizer_ids = [tokenizer.unk_token_id] * len(tokenizer.fairseq_tokens_to_ids)
    model_to_tokenizer_ids[tokenizer.cls_token_id] = 0
    model_to_tokenizer_ids[tokenizer.pad_token_id] = 1
    model_to_tokenizer_ids[tokenizer.sep_token_id] = 2
    model_to_tokenizer_ids[tokenizer.unk_token_id] = 3

    for token, model_id in tokenizer.fairseq_tokens_to_ids.items():
        if model_id in special_model_ids:
            continue
        sp_id = sp_model.piece_to_id(token)
        if sp_id >= 0:
            model_to_tokenizer_ids[model_id] = sp_id + 1

    return TokenizerIdBridge(
        tokenizer_to_model_ids=tokenizer_to_model_ids,
        model_to_tokenizer_ids=model_to_tokenizer_ids,
    )


def default_tokenizer_exporter(model_dir: str, bundle_dir: str) -> TokenizerExportArtifacts:
    ensure_local_vendor_path()
    try:
        import onnx
        from onnxruntime_extensions import gen_processing_models
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError('Tokenizer export requires onnx, transformers, and onnxruntime-extensions.') from exc

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    with bartpho_tokenizer_ortx_alias(tokenizer):
        processing_models = gen_processing_models(
            tokenizer,
            pre_kwargs={'fairseq': True},
            post_kwargs={'fairseq': True},
        )
    if len(processing_models) < 2:
        raise RuntimeError('gen_processing_models did not return both encode and decode graphs.')

    bundle_path = Path(bundle_dir)
    encode_path = bundle_path / TOKENIZER_ENCODE_FILE_NAME
    decode_path = bundle_path / TOKENIZER_DECODE_FILE_NAME
    tokenizer_to_model_id_map_path = bundle_path / TOKENIZER_TO_MODEL_ID_MAP_FILE_NAME
    model_to_tokenizer_id_map_path = bundle_path / MODEL_TO_TOKENIZER_ID_MAP_FILE_NAME
    onnx.save_model(processing_models[0], encode_path)
    onnx.save_model(processing_models[1], decode_path)
    bridge = build_ort_tokenizer_id_bridge(tokenizer)
    bridge.write_files(
        tokenizer_to_model_path=tokenizer_to_model_id_map_path,
        model_to_tokenizer_path=model_to_tokenizer_id_map_path,
    )
    return TokenizerExportArtifacts(
        encode_file_name=encode_path.name,
        decode_file_name=decode_path.name,
        tokenizer_to_model_id_map_file_name=tokenizer_to_model_id_map_path.name,
        model_to_tokenizer_id_map_file_name=model_to_tokenizer_id_map_path.name,
    )


def default_golden_sample_builder(
    *,
    model_dir: str,
    onnx_path: str,
    max_decode_length: int,
) -> list[TextGoldenSample]:
    model = VietnamesePuncCapDenormOnnx(model_dir=model_dir, onnx_path=onnx_path)
    samples: list[TextGoldenSample] = []
    for text in DEFAULT_TEXTS:
        encoded = model.tokenizer(
            text,
            return_tensors='np',
            truncation=True,
            max_length=512,
        )
        samples.append(
            TextGoldenSample(
                raw_text=text,
                input_ids=encoded['input_ids'][0].astype(int).tolist(),
                expected_output=model.restore(text, max_length=max_decode_length),
            )
        )
    return samples


def _load_json_array(path: str | Path) -> np.ndarray:
    return np.asarray(json.loads(Path(path).read_text(encoding='utf-8')), dtype=np.int64)


def _flatten_int64_array(value: object) -> np.ndarray:
    return np.asarray(value, dtype=np.int64).reshape(-1)


def _extract_string(value: object) -> str:
    if isinstance(value, str):
        return value
    flattened = np.asarray(value, dtype=object).reshape(-1)
    return '' if flattened.size == 0 else str(flattened[0])


class BundleOnnxRuntime:
    def __init__(
        self,
        *,
        manifest: ModelBundleManifest,
        model_session: object,
        encode_session: object,
        decode_session: object,
        tokenizer_to_model_ids: np.ndarray,
        model_to_tokenizer_ids: np.ndarray,
    ):
        self.manifest = manifest
        self.model_session = model_session
        self.encode_session = encode_session
        self.decode_session = decode_session
        self.tokenizer_to_model_ids = tokenizer_to_model_ids
        self.model_to_tokenizer_ids = model_to_tokenizer_ids
        self.metadata = manifest.metadata

    @classmethod
    def from_manifest_path(cls, manifest_path: str | Path, provider: str = 'CPUExecutionProvider') -> 'BundleOnnxRuntime':
        ensure_local_vendor_path()
        import onnxruntime as ort
        from onnxruntime_extensions import get_library_path

        manifest = ModelBundleManifest.from_path(manifest_path)
        bundle_dir = Path(manifest_path).resolve().parent
        session_options = ort.SessionOptions()
        session_options.register_custom_ops_library(get_library_path())

        return cls(
            manifest=manifest,
            model_session=ort.InferenceSession(str(bundle_dir / manifest.artifacts['model']), providers=[provider]),
            encode_session=ort.InferenceSession(
                str(bundle_dir / manifest.artifacts['tokenizer_encode']),
                sess_options=session_options,
                providers=[provider],
            ),
            decode_session=ort.InferenceSession(
                str(bundle_dir / manifest.artifacts['tokenizer_decode']),
                sess_options=session_options,
                providers=[provider],
            ),
            tokenizer_to_model_ids=_load_json_array(bundle_dir / manifest.artifacts['tokenizer_to_model_id_map']),
            model_to_tokenizer_ids=_load_json_array(bundle_dir / manifest.artifacts['model_to_tokenizer_id_map']),
        )

    def restore(self, text: str, max_length: int = 128) -> str:
        normalized = '' if text is None else text.strip()
        if not normalized:
            return ''

        input_ids = self._encode_to_model_ids(normalized).reshape(1, -1)
        attention_mask = np.ones_like(input_ids, dtype=np.int64)
        decoder_input_ids = np.asarray([[int(self.metadata['decoder_start_token_id'])]], dtype=np.int64)
        effective_max_length = max(1, min(max_length, int(self.metadata['max_decode_length'])))

        for _ in range(effective_max_length):
            decoder_attention_mask = np.ones_like(decoder_input_ids, dtype=np.int64)
            outputs = self.model_session.run(
                None,
                {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'decoder_input_ids': decoder_input_ids,
                    'decoder_attention_mask': decoder_attention_mask,
                },
            )
            next_token_id = self._argmax_last_token(outputs[0])
            decoder_input_ids = np.concatenate([decoder_input_ids, np.asarray([[next_token_id]], dtype=np.int64)], axis=1)
            if next_token_id == int(self.metadata['eos_token_id']):
                break

        generated_ids = decoder_input_ids[0, 1:]
        return self._decode_model_ids(generated_ids).strip()

    def _encode_to_model_ids(self, text: str) -> np.ndarray:
        outputs = self.encode_session.run(None, {'inputs': np.asarray([text], dtype=object)})
        tokenizer_ids = _flatten_int64_array(outputs[0])
        if tokenizer_ids.size == 0:
            return np.asarray([int(self.metadata['eos_token_id'])], dtype=np.int64)

        effective_max_source_length = max(1, int(self.metadata['max_source_length']))
        output_length = min(tokenizer_ids.size, effective_max_source_length)
        model_ids = np.full(output_length, UNK_TOKEN_ID, dtype=np.int64)
        for index in range(output_length):
            tokenizer_id = int(tokenizer_ids[index])
            if 0 <= tokenizer_id < self.tokenizer_to_model_ids.shape[0]:
                model_ids[index] = int(self.tokenizer_to_model_ids[tokenizer_id])
        if tokenizer_ids.size > output_length:
            model_ids[output_length - 1] = int(self.metadata['eos_token_id'])
        return model_ids

    def _decode_model_ids(self, model_ids: np.ndarray) -> str:
        if model_ids.size == 0:
            return ''
        tokenizer_ids = np.full(model_ids.size, UNK_TOKEN_ID, dtype=np.int64)
        for index, model_id in enumerate(model_ids.tolist()):
            if 0 <= model_id < self.model_to_tokenizer_ids.shape[0]:
                tokenizer_ids[index] = int(self.model_to_tokenizer_ids[model_id])
        outputs = self.decode_session.run(None, {'ids': tokenizer_ids})
        return _extract_string(outputs[0])

    @staticmethod
    def _argmax_last_token(logits: object) -> int:
        array = np.asarray(logits)
        if array.ndim == 3:
            return int(np.argmax(array[:, -1, :], axis=-1)[0])
        if array.ndim == 2:
            return int(np.argmax(array[-1]))
        if array.ndim == 1:
            return int(np.argmax(array))
        raise ValueError(f'Unsupported logits shape: {array.shape}')
