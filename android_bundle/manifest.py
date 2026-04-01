from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AndroidBundleManifest:
    DEFAULT_MODEL_FILE = "model.mobile.onnx"
    DEFAULT_TOKENIZER_ENCODE_FILE = "tokenizer.encode.onnx"
    DEFAULT_TOKENIZER_DECODE_FILE = "tokenizer.decode.onnx"
    DEFAULT_TOKENIZER_TO_MODEL_ID_MAP_FILE = "tokenizer.to_model_id_map.json"
    DEFAULT_MODEL_TO_TOKENIZER_ID_MAP_FILE = "tokenizer.from_model_id_map.json"
    DEFAULT_GOLDEN_SAMPLES_FILE = "golden_samples.jsonl"

    bundle_version: int
    model_name: str
    model_variant: str
    asset_namespace: str
    model_file: str
    tokenizer_encode_file: str
    tokenizer_decode_file: str
    tokenizer_to_model_id_map_file: str
    model_to_tokenizer_id_map_file: str
    golden_samples_file: str
    pad_token_id: int
    eos_token_id: int
    decoder_start_token_id: int
    max_source_length: int
    max_decode_length: int

    @staticmethod
    def normalize_bundle_file(path: str) -> str:
        return Path(path).name

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_version": self.bundle_version,
            "model_name": self.model_name,
            "model_variant": self.model_variant,
            "asset_namespace": self.asset_namespace,
            "model_file": self.DEFAULT_MODEL_FILE,
            "tokenizer_encode_file": self.DEFAULT_TOKENIZER_ENCODE_FILE,
            "tokenizer_decode_file": self.DEFAULT_TOKENIZER_DECODE_FILE,
            "tokenizer_to_model_id_map_file": self.DEFAULT_TOKENIZER_TO_MODEL_ID_MAP_FILE,
            "model_to_tokenizer_id_map_file": self.DEFAULT_MODEL_TO_TOKENIZER_ID_MAP_FILE,
            "golden_samples_file": self.DEFAULT_GOLDEN_SAMPLES_FILE,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
            "decoder_start_token_id": self.decoder_start_token_id,
            "max_source_length": self.max_source_length,
            "max_decode_length": self.max_decode_length,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AndroidBundleManifest":
        normalized = dict(payload)
        for field_name in (
            "model_file",
            "tokenizer_encode_file",
            "tokenizer_decode_file",
            "tokenizer_to_model_id_map_file",
            "model_to_tokenizer_id_map_file",
            "golden_samples_file",
        ):
            normalized[field_name] = cls.normalize_bundle_file(str(normalized[field_name]))

        normalized["model_file"] = cls.DEFAULT_MODEL_FILE
        normalized["tokenizer_encode_file"] = cls.DEFAULT_TOKENIZER_ENCODE_FILE
        normalized["tokenizer_decode_file"] = cls.DEFAULT_TOKENIZER_DECODE_FILE
        normalized["tokenizer_to_model_id_map_file"] = cls.DEFAULT_TOKENIZER_TO_MODEL_ID_MAP_FILE
        normalized["model_to_tokenizer_id_map_file"] = cls.DEFAULT_MODEL_TO_TOKENIZER_ID_MAP_FILE
        normalized["golden_samples_file"] = cls.DEFAULT_GOLDEN_SAMPLES_FILE
        return cls(**normalized)

    @classmethod
    def from_path(cls, manifest_path: str | Path) -> "AndroidBundleManifest":
        manifest_file = Path(manifest_path)
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    def resolve_model_path(self, manifest_path: str | Path) -> str:
        return str(Path(manifest_path).resolve().parent / self.model_file)
