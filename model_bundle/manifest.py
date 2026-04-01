from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelBundleManifest:
    bundle_version: int
    project: str
    model_family: str
    model_name: str
    model_variant: str
    asset_namespace: str
    runtime_kind: str
    artifacts: dict[str, str] = field(default_factory=dict)
    fixtures: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'bundle_version': self.bundle_version,
            'project': self.project,
            'model_family': self.model_family,
            'model_name': self.model_name,
            'model_variant': self.model_variant,
            'asset_namespace': self.asset_namespace,
            'runtime_kind': self.runtime_kind,
            'artifacts': dict(self.artifacts),
            'fixtures': dict(self.fixtures),
            'metadata': dict(self.metadata),
        }

    @classmethod
    def _from_legacy_punctuation(cls, payload: dict[str, Any]) -> 'ModelBundleManifest':
        return cls(
            bundle_version=int(payload['bundle_version']),
            project='vpcd',
            model_family='bartpho-seq2seq',
            model_name=str(payload['model_name']),
            model_variant=str(payload.get('model_variant', 'fp32')),
            asset_namespace=str(payload['asset_namespace']),
            runtime_kind='text_seq2seq',
            artifacts={
                'model': Path(str(payload['model_file'])).name,
                'tokenizer_encode': Path(str(payload['tokenizer_encode_file'])).name,
                'tokenizer_decode': Path(str(payload['tokenizer_decode_file'])).name,
                'tokenizer_to_model_id_map': Path(str(payload['tokenizer_to_model_id_map_file'])).name,
                'model_to_tokenizer_id_map': Path(str(payload['model_to_tokenizer_id_map_file'])).name,
            },
            fixtures={'golden_samples': Path(str(payload['golden_samples_file'])).name},
            metadata={
                'pad_token_id': int(payload['pad_token_id']),
                'eos_token_id': int(payload['eos_token_id']),
                'decoder_start_token_id': int(payload['decoder_start_token_id']),
                'max_source_length': int(payload['max_source_length']),
                'max_decode_length': int(payload['max_decode_length']),
            },
        )

    @classmethod
    def _from_legacy_zipformer(cls, payload: dict[str, Any]) -> 'ModelBundleManifest':
        return cls(
            bundle_version=int(payload['bundle_version']),
            project='zipformer',
            model_family=str(payload['model_family']),
            model_name=str(payload['model_name']),
            model_variant=str(payload.get('model_variant', 'fp32')),
            asset_namespace=str(payload['asset_namespace']),
            runtime_kind='rnnt_greedy',
            artifacts={
                'encoder': Path(str(payload['encoder_file'])).name,
                'decoder': Path(str(payload['decoder_file'])).name,
                'joiner': Path(str(payload['joiner_file'])).name,
                'tokens': Path(str(payload['tokens_file'])).name,
            },
            fixtures={
                'sample_manifest': Path(str(payload['sample_manifest_file'])).name,
                'expected_outputs': Path(str(payload['expected_outputs_file'])).name,
            },
            metadata={
                'sample_rate': int(payload['sample_rate']),
                'feature_dim': int(payload['feature_dim']),
                'blank_id': int(payload['blank_id']),
                'context_size': int(payload['context_size']),
            },
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> 'ModelBundleManifest':
        if 'project' not in payload:
            if 'tokenizer_encode_file' in payload:
                return cls._from_legacy_punctuation(payload)
            if 'encoder_file' in payload:
                return cls._from_legacy_zipformer(payload)
        return cls(
            bundle_version=int(payload['bundle_version']),
            project=str(payload['project']),
            model_family=str(payload['model_family']),
            model_name=str(payload['model_name']),
            model_variant=str(payload.get('model_variant', '')),
            asset_namespace=str(payload['asset_namespace']),
            runtime_kind=str(payload['runtime_kind']),
            artifacts={str(k): Path(str(v)).name for k, v in dict(payload.get('artifacts', {})).items()},
            fixtures={str(k): Path(str(v)).name for k, v in dict(payload.get('fixtures', {})).items()},
            metadata=dict(payload.get('metadata', {})),
        )

    @classmethod
    def from_path(cls, manifest_path: str | Path) -> 'ModelBundleManifest':
        payload = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
        return cls.from_dict(payload)

    def write_json(self, manifest_path: str | Path) -> Path:
        path = Path(manifest_path)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        return path

    def bundle_dir(self, manifest_path: str | Path) -> Path:
        return Path(manifest_path).resolve().parent

    def resolve_artifact_path(self, manifest_path: str | Path, key: str) -> Path:
        return self.bundle_dir(manifest_path) / self.artifacts[key]

    def resolve_fixture_path(self, manifest_path: str | Path, key: str) -> Path:
        return self.bundle_dir(manifest_path) / self.fixtures[key]
