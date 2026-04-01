from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from android_bundle.exporter import ensure_local_vendor_path
from android_bundle.manifest import AndroidBundleManifest


UNK_TOKEN_ID = 3


def load_json_array(path: str | Path) -> np.ndarray:
    return np.asarray(json.loads(Path(path).read_text(encoding="utf-8")), dtype=np.int64)


def _flatten_int64_array(value: object) -> np.ndarray:
    return np.asarray(value, dtype=np.int64).reshape(-1)


def _extract_string(value: object) -> str:
    if isinstance(value, str):
        return value
    flattened = np.asarray(value, dtype=object).reshape(-1)
    return "" if flattened.size == 0 else str(flattened[0])


class BundleOnnxRuntime:
    def __init__(
        self,
        *,
        manifest: AndroidBundleManifest,
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

    @classmethod
    def from_manifest_path(
        cls,
        manifest_path: str | Path,
        provider: str = "CPUExecutionProvider",
    ) -> "BundleOnnxRuntime":
        ensure_local_vendor_path()
        import onnxruntime as ort
        from onnxruntime_extensions import get_library_path

        manifest = AndroidBundleManifest.from_path(manifest_path)
        bundle_dir = Path(manifest_path).resolve().parent
        session_options = ort.SessionOptions()
        session_options.register_custom_ops_library(get_library_path())

        return cls(
            manifest=manifest,
            model_session=ort.InferenceSession(
                str(bundle_dir / manifest.model_file),
                providers=[provider],
            ),
            encode_session=ort.InferenceSession(
                str(bundle_dir / manifest.tokenizer_encode_file),
                sess_options=session_options,
                providers=[provider],
            ),
            decode_session=ort.InferenceSession(
                str(bundle_dir / manifest.tokenizer_decode_file),
                sess_options=session_options,
                providers=[provider],
            ),
            tokenizer_to_model_ids=load_json_array(bundle_dir / manifest.tokenizer_to_model_id_map_file),
            model_to_tokenizer_ids=load_json_array(bundle_dir / manifest.model_to_tokenizer_id_map_file),
        )

    def restore(self, text: str, max_length: int = 128) -> str:
        normalized = "" if text is None else text.strip()
        if not normalized:
            return ""

        input_ids = self._encode_to_model_ids(normalized).reshape(1, -1)
        attention_mask = np.ones_like(input_ids, dtype=np.int64)
        decoder_input_ids = np.asarray([[self.manifest.decoder_start_token_id]], dtype=np.int64)
        effective_max_length = max(1, min(max_length, self.manifest.max_decode_length))

        for _ in range(effective_max_length):
            decoder_attention_mask = np.ones_like(decoder_input_ids, dtype=np.int64)
            outputs = self.model_session.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "decoder_input_ids": decoder_input_ids,
                    "decoder_attention_mask": decoder_attention_mask,
                },
            )
            next_token_id = self._argmax_last_token(outputs[0])
            decoder_input_ids = np.concatenate(
                [decoder_input_ids, np.asarray([[next_token_id]], dtype=np.int64)],
                axis=1,
            )
            if next_token_id == self.manifest.eos_token_id:
                break

        generated_ids = decoder_input_ids[0, 1:]
        return self._decode_model_ids(generated_ids).strip()

    def _encode_to_model_ids(self, text: str) -> np.ndarray:
        outputs = self.encode_session.run(None, {"inputs": np.asarray([text], dtype=object)})
        tokenizer_ids = _flatten_int64_array(outputs[0])
        if tokenizer_ids.size == 0:
            return np.asarray([self.manifest.eos_token_id], dtype=np.int64)

        effective_max_source_length = max(1, self.manifest.max_source_length)
        output_length = min(tokenizer_ids.size, effective_max_source_length)
        model_ids = np.full(output_length, UNK_TOKEN_ID, dtype=np.int64)
        for index in range(output_length):
            tokenizer_id = int(tokenizer_ids[index])
            if 0 <= tokenizer_id < self.tokenizer_to_model_ids.shape[0]:
                model_ids[index] = int(self.tokenizer_to_model_ids[tokenizer_id])
        if tokenizer_ids.size > output_length:
            model_ids[output_length - 1] = self.manifest.eos_token_id
        return model_ids

    def _decode_model_ids(self, model_ids: np.ndarray) -> str:
        if model_ids.size == 0:
            return ""

        tokenizer_ids = np.full(model_ids.size, UNK_TOKEN_ID, dtype=np.int64)
        for index, model_id in enumerate(model_ids.tolist()):
            if 0 <= model_id < self.model_to_tokenizer_ids.shape[0]:
                tokenizer_ids[index] = int(self.model_to_tokenizer_ids[model_id])

        outputs = self.decode_session.run(None, {"ids": tokenizer_ids})
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
        raise ValueError(f"Unsupported logits shape: {array.shape}")
