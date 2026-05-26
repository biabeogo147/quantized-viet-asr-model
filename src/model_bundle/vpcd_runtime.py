from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from model_bundle.fixtures import TextGoldenSample, serialize_jsonl
from model_bundle.manifest import ModelBundleManifest
from model_bundle.vpcd_shapes import attention_mask_for_length, pad_token_row, resolve_vpcd_model_input_shapes
from tools.paths import resolve_repo_path


DEFAULT_MODEL_DIR = Path("assets") / "vietnamese-punc-cap-denorm-v1"
DEFAULT_ASSET_NAMESPACE = "models/punctuation/vpcd/vpcd_balanced"
DEFAULT_MODEL_VARIANT = "vpcd_balanced"
MODEL_FILE_NAME = "model.mobile.onnx"
TOKENIZER_ENCODE_FILE_NAME = "tokenizer.encode.onnx"
TOKENIZER_DECODE_FILE_NAME = "tokenizer.decode.onnx"
TOKENIZER_TO_MODEL_ID_MAP_FILE_NAME = "tokenizer.to_model_id_map.json"
MODEL_TO_TOKENIZER_ID_MAP_FILE_NAME = "tokenizer.from_model_id_map.json"
GOLDEN_SAMPLES_FILE_NAME = "golden_samples.jsonl"
UNK_TOKEN_ID = 3
DEFAULT_GOLDEN_SAMPLES = [
    TextGoldenSample(
        raw_text="h\u00f4m nay l\u00e0 bu\u1ed5i nh\u1eadm ch\u1ee9c c\u1ee7a t\u00f4i ph\u01b0\u1edbc th\u00e0nh",
        input_ids=[0, 799, 177, 9, 847, 559, 2306, 115, 7, 80, 1386, 1338, 58, 2],
        expected_output="H\u00f4m nay l\u00e0 bu\u1ed5i nh\u1eadm ch\u1ee9c c\u1ee7a t\u00f4i - Ph\u01b0\u1edbc Th\u00e0nh.",
    ),
    TextGoldenSample(
        raw_text="ch\u00e0o c\u00e1c b\u1ea1n h\u00f4m nay ch\u00fang ta c\u00f9ng nhau \u0111\u1ebfn v\u1edbi b\u00e0i h\u1ecdc deep learning ph\u1ea7n s\u1ed1 m\u01b0\u1eddi ba",
        input_ids=[0, 1740, 10, 144, 799, 177, 248, 336, 120, 383, 30, 15, 635, 71, 19466, 18436, 221, 52, 3125, 712, 2],
        expected_output="Ch\u00e0o c\u00e1c b\u1ea1n, h\u00f4m nay ch\u00fang ta c\u00f9ng nhau \u0111\u1ebfn v\u1edbi b\u00e0i h\u1ecdc Deep Learning ph\u1ea7n s\u1ed1 13.",
    ),
]
DEFAULT_TEXTS = [sample.raw_text for sample in DEFAULT_GOLDEN_SAMPLES]


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
            json.dumps(self.tokenizer_to_model_ids, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        model_to_tokenizer_file.write_text(
            json.dumps(self.model_to_tokenizer_ids, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        return tokenizer_to_model_file.name, model_to_tokenizer_file.name


TokenizerExporter = Callable[[str, str], TokenizerExportArtifacts]
GoldenSampleBuilder = Callable[..., list[TextGoldenSample]]


def ensure_local_vendor_path() -> None:
    vendor_dir = resolve_repo_path("_vendor", anchor=__file__)
    if vendor_dir.exists():
        vendor_path = str(vendor_dir)
        if vendor_path not in sys.path:
            sys.path.insert(0, vendor_path)


def resolve_variant_onnx_path(model_dir: str | Path, model_variant: str) -> Path:
    variant_file = model_variant if str(model_variant).endswith(".onnx") else f"{model_variant}.onnx"
    return Path(model_dir) / "onnx" / variant_file


class ModelDirOnnxRuntime:
    def __init__(self, *, model_dir: str, onnx_path: str, provider: str = "CPUExecutionProvider"):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.model_dir = model_dir
        self.onnx_path = onnx_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.session = ort.InferenceSession(onnx_path, providers=[provider])

        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.decoder_start_token_id = self.tokenizer.eos_token_id

        generation_config_path = Path(model_dir) / "generation_config.json"
        if generation_config_path.exists():
            generation_config = json.loads(generation_config_path.read_text(encoding="utf-8"))
            self.decoder_start_token_id = generation_config.get("decoder_start_token_id", self.decoder_start_token_id)

    def restore(self, text: str, max_length: int = 128) -> str:
        encoded = self.tokenizer(text, return_tensors="np", truncation=True, max_length=512)
        input_ids = encoded["input_ids"].astype(np.int64)
        attention_mask = encoded["attention_mask"].astype(np.int64)
        decoder_input_ids = np.array([[self.decoder_start_token_id]], dtype=np.int64)

        for _ in range(max_length):
            decoder_attention_mask = np.ones_like(decoder_input_ids, dtype=np.int64)
            outputs = self.session.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "decoder_input_ids": decoder_input_ids,
                    "decoder_attention_mask": decoder_attention_mask,
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
    tokenizer_class.__name__ = "XLMRobertaTokenizer"
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
        raise RuntimeError("Tokenizer export requires onnx, transformers, and onnxruntime-extensions.") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    with bartpho_tokenizer_ortx_alias(tokenizer):
        processing_models = gen_processing_models(
            tokenizer,
            pre_kwargs={"fairseq": True},
            post_kwargs={"fairseq": True},
        )
    if len(processing_models) < 2:
        raise RuntimeError("gen_processing_models did not return both encode and decode graphs.")

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
    for sample in DEFAULT_GOLDEN_SAMPLES:
        encoded = model.tokenizer(
            sample.raw_text,
            return_tensors="np",
            truncation=True,
            max_length=512,
        )
        encoded_ids = encoded["input_ids"][0].astype(int).tolist()
        if encoded_ids != sample.input_ids:
            raise ValueError(
                "Pinned VPCD golden sample tokenizer ids drifted for "
                f"{sample.raw_text!r}: expected {sample.input_ids}, got {encoded_ids}"
            )
        samples.append(
            TextGoldenSample(
                raw_text=sample.raw_text,
                input_ids=encoded_ids,
                expected_output=sample.expected_output,
            )
        )
    return samples


def _load_json_array(path: str | Path) -> np.ndarray:
    return np.asarray(json.loads(Path(path).read_text(encoding="utf-8")), dtype=np.int64)


def _flatten_int64_array(value: object) -> np.ndarray:
    return np.asarray(value, dtype=np.int64).reshape(-1)


def _extract_string(value: object) -> str:
    if isinstance(value, str):
        return value
    flattened = np.asarray(value, dtype=object).reshape(-1)
    return "" if flattened.size == 0 else str(flattened[0])


def _normalize_input_text(text: str | None, metadata: dict[str, object]) -> str:
    normalized = "" if text is None else text.strip()
    if not normalized:
        return ""

    input_text_case = str(metadata.get("input_text_case", "") or "").strip().lower()
    if input_text_case == "lower":
        return normalized.lower()
    return normalized


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
        self.fixed_input_shapes = resolve_vpcd_model_input_shapes(self.metadata)

    @classmethod
    def from_manifest_path(cls, manifest_path: str | Path, provider: str = "CPUExecutionProvider") -> "BundleOnnxRuntime":
        ensure_local_vendor_path()
        import onnxruntime as ort
        from onnxruntime_extensions import get_library_path

        manifest = ModelBundleManifest.from_path(manifest_path)
        bundle_dir = Path(manifest_path).resolve().parent
        session_options = ort.SessionOptions()
        session_options.register_custom_ops_library(get_library_path())

        return cls(
            manifest=manifest,
            model_session=ort.InferenceSession(str(bundle_dir / manifest.artifacts["model"]), providers=[provider]),
            encode_session=ort.InferenceSession(
                str(bundle_dir / manifest.artifacts["tokenizer_encode"]),
                sess_options=session_options,
                providers=[provider],
            ),
            decode_session=ort.InferenceSession(
                str(bundle_dir / manifest.artifacts["tokenizer_decode"]),
                sess_options=session_options,
                providers=[provider],
            ),
            tokenizer_to_model_ids=_load_json_array(bundle_dir / manifest.artifacts["tokenizer_to_model_id_map"]),
            model_to_tokenizer_ids=_load_json_array(bundle_dir / manifest.artifacts["model_to_tokenizer_id_map"]),
        )

    def restore(self, text: str, max_length: int = 128) -> str:
        result = self.restore_with_model_step(
            text,
            lambda feeds: self.model_session.run(None, feeds)[0],
            max_length=max_length,
        )
        return str(result["text"])

    def restore_with_model_step(
        self,
        text: str,
        model_step_runner: Callable[[dict[str, np.ndarray]], object],
        *,
        max_length: int = 128,
    ) -> dict[str, object]:
        normalized = _normalize_input_text(text, self.metadata)
        if not normalized:
            return {
                "text": "",
                "decode_steps": 0,
                "generated_ids": np.asarray([], dtype=np.int64),
                "ended_with_eos": False,
            }

        model_ids = self._encode_to_model_ids(normalized)
        if self.fixed_input_shapes is None:
            input_ids = model_ids.reshape(1, -1)
            attention_mask = np.ones_like(input_ids, dtype=np.int64)
        else:
            input_ids = pad_token_row(
                model_ids,
                target_length=self.fixed_input_shapes.encoder_sequence,
                pad_value=int(self.metadata["pad_token_id"]),
            )
            attention_mask = attention_mask_for_length(
                actual_length=int(model_ids.size),
                target_length=self.fixed_input_shapes.encoder_sequence,
            )

        decoder_token_ids = np.asarray([int(self.metadata["decoder_start_token_id"])], dtype=np.int64)
        effective_max_length = max(1, min(max_length, int(self.metadata["max_decode_length"])))

        for _ in range(effective_max_length):
            if self.fixed_input_shapes is None:
                decoder_input_ids = decoder_token_ids.reshape(1, -1)
                decoder_attention_mask = np.ones_like(decoder_input_ids, dtype=np.int64)
                logits_position = None
            else:
                active_decoder_length = int(decoder_token_ids.size)
                decoder_input_ids = pad_token_row(
                    decoder_token_ids,
                    target_length=self.fixed_input_shapes.decoder_sequence,
                    pad_value=int(self.metadata["pad_token_id"]),
                )
                decoder_attention_mask = attention_mask_for_length(
                    actual_length=active_decoder_length,
                    target_length=self.fixed_input_shapes.decoder_sequence,
                )
                logits_position = active_decoder_length - 1

            feeds = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "decoder_input_ids": decoder_input_ids,
                "decoder_attention_mask": decoder_attention_mask,
            }
            logits = model_step_runner(feeds)
            next_token_id = self._argmax_token_at(logits, logits_position)
            decoder_token_ids = np.concatenate([decoder_token_ids, np.asarray([next_token_id], dtype=np.int64)])
            if next_token_id == int(self.metadata["eos_token_id"]):
                break

        generated_ids = decoder_token_ids[1:]
        ended_with_eos = bool(generated_ids.size and int(generated_ids[-1]) == int(self.metadata["eos_token_id"]))
        return {
            "text": self._decode_model_ids(generated_ids).strip(),
            "decode_steps": int(generated_ids.size),
            "generated_ids": generated_ids,
            "ended_with_eos": ended_with_eos,
        }

    def _encode_to_model_ids(self, text: str) -> np.ndarray:
        outputs = self.encode_session.run(None, {"inputs": np.asarray([text], dtype=object)})
        tokenizer_ids = _flatten_int64_array(outputs[0])
        if tokenizer_ids.size == 0:
            return np.asarray([int(self.metadata["eos_token_id"])], dtype=np.int64)

        effective_max_source_length = max(1, int(self.metadata["max_source_length"]))
        output_length = min(tokenizer_ids.size, effective_max_source_length)
        model_ids = np.full(output_length, UNK_TOKEN_ID, dtype=np.int64)
        for index in range(output_length):
            tokenizer_id = int(tokenizer_ids[index])
            if 0 <= tokenizer_id < self.tokenizer_to_model_ids.shape[0]:
                model_ids[index] = int(self.tokenizer_to_model_ids[tokenizer_id])
        if tokenizer_ids.size > output_length:
            model_ids[output_length - 1] = int(self.metadata["eos_token_id"])
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
    def _argmax_token_at(logits: object, position: int | None = None) -> int:
        array = np.asarray(logits)
        if array.ndim == 3:
            index = -1 if position is None else int(position)
            return int(np.argmax(array[:, index, :], axis=-1)[0])
        if array.ndim == 2:
            index = -1 if position is None else int(position)
            return int(np.argmax(array[index]))
        if array.ndim == 1:
            return int(np.argmax(array))
        raise ValueError(f"Unsupported logits shape: {array.shape}")

    @staticmethod
    def _argmax_last_token(logits: object) -> int:
        return BundleOnnxRuntime._argmax_token_at(logits)
