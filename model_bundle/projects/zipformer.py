from __future__ import annotations

import re
import shutil
import time
from pathlib import Path

import numpy as np

from model_bundle.contracts import BundleProjectAdapter
from model_bundle.fixtures import AudioExpectedOutput, AudioSampleFixture, read_jsonl, serialize_jsonl
from model_bundle.layout import resolve_bundle_dir
from model_bundle.manifest import ModelBundleManifest

DEFAULT_MODEL_DIR = Path('assets') / 'zipformer'
DEFAULT_OUTPUT_DIR = resolve_bundle_dir('zipformer', 'fp32')
DEFAULT_ASSET_NAMESPACE = 'models/asr/zipformer/fp32'
DEFAULT_VARIANT = 'fp32'
DEFAULT_COMPONENT_FILES = {
    'encoder': 'encoder-epoch-20-avg-1.onnx',
    'decoder': 'decoder-epoch-20-avg-1.onnx',
    'joiner': 'joiner-epoch-20-avg-1.onnx',
    'tokens': 'tokens.txt',
}
DEFAULT_AUDIO_FIXTURES = [
    AudioSampleFixture(sample_id='sample-1', audio_path='assets/speech/sample-1.mp3'),
    AudioSampleFixture(sample_id='sample-2', audio_path='assets/speech/sample-2.wav'),
]


def prepare_encoder_inputs(features: np.ndarray, fixed_encoder_frames: int | None = None) -> dict[str, np.ndarray]:
    if features.ndim != 2:
        raise ValueError(f'Expected 2D feature matrix, got shape {features.shape}')

    frame_count, feature_dim = features.shape
    if fixed_encoder_frames is None:
        x = features[None, ...].astype(np.float32, copy=False)
    else:
        if frame_count > fixed_encoder_frames:
            raise ValueError(
                f'Audio features contain {frame_count} frames but bundle expects at most {fixed_encoder_frames}.'
            )
        x = np.zeros((1, fixed_encoder_frames, feature_dim), dtype=np.float32)
        x[:, :frame_count, :] = features.astype(np.float32, copy=False)
    x_lens = np.array([frame_count], dtype=np.int64)
    return {'x': x, 'x_lens': x_lens}


def trim_encoder_frames(encoder_frames: np.ndarray, encoder_out_lens: np.ndarray | None) -> np.ndarray:
    if encoder_out_lens is None:
        return encoder_frames
    valid_frames = int(encoder_out_lens.reshape(-1)[0])
    return encoder_frames[:valid_frames]


def resolve_fixed_encoder_frames(metadata: dict) -> int | None:
    fixed_input_shapes = metadata.get('fixed_input_shapes', {}) if isinstance(metadata, dict) else {}
    encoder_shapes = fixed_input_shapes.get('encoder', {}) if isinstance(fixed_input_shapes, dict) else {}
    encoder_x_shape = encoder_shapes.get('x') if isinstance(encoder_shapes, dict) else None
    if isinstance(encoder_x_shape, (list, tuple)) and len(encoder_x_shape) >= 2:
        return int(encoder_x_shape[1])
    fixed_encoder_frames = metadata.get('fixed_encoder_frames') if isinstance(metadata, dict) else None
    return None if fixed_encoder_frames is None else int(fixed_encoder_frames)


class ZipformerRuntimeBase:
    def _load_tokens(self, tokens_path: Path) -> list[str]:
        tokens: list[str] = []
        with tokens_path.open('r', encoding='utf-8') as handle:
            for line in handle:
                parts = line.strip().split()
                if not parts:
                    continue
                tokens.append(parts[0] if len(parts) == 2 else parts[-1])
        return tokens

    @staticmethod
    def _decode_tokens(tokens_table: list[str], result: list[int]) -> str:
        text = ''.join(tokens_table[i] for i in result if i < len(tokens_table))
        text = text.replace('?', ' ').strip()
        text = re.sub(r'\s{2,}', ' ', text)
        return text

    @staticmethod
    def _resolve_providers(provider: str) -> list[str]:
        if provider == 'CPUExecutionProvider':
            return ['CPUExecutionProvider']
        return [provider, 'CPUExecutionProvider'] if provider != 'CPUExecutionProvider' else ['CPUExecutionProvider']

    @staticmethod
    def _load_features(audio_path: str | Path, sample_rate: int = 16000, feature_dim: int = 80) -> np.ndarray:
        import torch
        import torchaudio

        waveform, sr = torchaudio.load(str(audio_path))
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            hop_length=160,
            win_length=400,
            n_mels=feature_dim,
            f_min=20,
            f_max=8000,
            power=2.0,
        )(waveform)
        log_mel = torch.clamp(mel, min=1e-10).log()
        return log_mel.squeeze(0).transpose(0, 1).numpy().astype(np.float32)


class ModelDirAcousticRuntime(ZipformerRuntimeBase):
    def __init__(
        self,
        *,
        model_dir: str | Path,
        provider: str = 'CPUExecutionProvider',
        component_paths: dict[str, str | Path] | None = None,
        sample_rate: int = 16000,
        feature_dim: int = 80,
        blank_id: int = 0,
        context_size: int = 2,
        fixed_encoder_frames: int | None = None,
    ):
        import onnxruntime as ort

        self.model_dir = Path(model_dir)
        self.provider = provider
        self.sample_rate = sample_rate
        self.feature_dim = feature_dim
        self.blank_id = blank_id
        self.context_size = context_size
        self.fixed_encoder_frames = fixed_encoder_frames
        resolved = {key: Path(value) for key, value in (component_paths or {}).items()}
        self.encoder_path = resolved.get('encoder', self.model_dir / DEFAULT_COMPONENT_FILES['encoder'])
        self.decoder_path = resolved.get('decoder', self.model_dir / DEFAULT_COMPONENT_FILES['decoder'])
        self.joiner_path = resolved.get('joiner', self.model_dir / DEFAULT_COMPONENT_FILES['joiner'])
        self.tokens_path = resolved.get('tokens', self.model_dir / DEFAULT_COMPONENT_FILES['tokens'])

        providers = self._resolve_providers(provider)
        self.encoder_sess = ort.InferenceSession(str(self.encoder_path), providers=providers)
        self.decoder_sess = ort.InferenceSession(str(self.decoder_path), providers=providers)
        self.joiner_sess = ort.InferenceSession(str(self.joiner_path), providers=providers)
        self.tokens_table = self._load_tokens(self.tokens_path)

    def transcribe(self, audio_path: str | Path) -> dict:
        features = self._load_features(audio_path, sample_rate=self.sample_rate, feature_dim=self.feature_dim)
        encoder_inputs = prepare_encoder_inputs(features, fixed_encoder_frames=self.fixed_encoder_frames)

        encoder_started = time.time()
        encoder_outputs = self.encoder_sess.run(None, encoder_inputs)
        encoder_elapsed = time.time() - encoder_started

        encoder_out = encoder_outputs[0]
        encoder_out_lens = encoder_outputs[1] if len(encoder_outputs) > 1 else None
        frames = trim_encoder_frames(encoder_out[0].astype(np.float32), encoder_out_lens)
        result: list[int] = []
        history = [self.blank_id] * self.context_size

        decoder_started = time.time()
        for frame in frames:
            enc_frame = frame.reshape(1, -1).astype(np.float32)
            while True:
                dec_in = np.asarray([history[-self.context_size :]], dtype=np.int64)
                dec_out = self.decoder_sess.run(None, {'y': dec_in})[0].astype(np.float32)
                join_out = self.joiner_sess.run(None, {'encoder_out': enc_frame, 'decoder_out': dec_out})[0]
                token = int(np.argmax(join_out, axis=-1)[0])
                if token == self.blank_id:
                    break
                result.append(token)
                history.append(token)
        decoder_elapsed = time.time() - decoder_started

        return {
            'text': self._decode_tokens(self.tokens_table, result),
            'num_tokens': len(result),
            'encoder_time': round(encoder_elapsed, 3),
            'decoder_time': round(decoder_elapsed, 3),
        }


class BundleAcousticRuntime(ModelDirAcousticRuntime):
    @classmethod
    def from_manifest_path(cls, manifest_path: str | Path, provider: str = 'CPUExecutionProvider') -> 'BundleAcousticRuntime':
        manifest = ModelBundleManifest.from_path(manifest_path)
        bundle_dir = Path(manifest_path).resolve().parent
        return cls(
            model_dir=bundle_dir,
            provider=provider,
            component_paths={key: bundle_dir / value for key, value in manifest.artifacts.items() if key in {'encoder', 'decoder', 'joiner', 'tokens'}},
            sample_rate=int(manifest.metadata.get('sample_rate', 16000)),
            feature_dim=int(manifest.metadata.get('feature_dim', 80)),
            blank_id=int(manifest.metadata.get('blank_id', 0)),
            context_size=int(manifest.metadata.get('context_size', 2)),
            fixed_encoder_frames=resolve_fixed_encoder_frames(manifest.metadata),
        )


def _build_expected_outputs(runtime: ModelDirAcousticRuntime, fixtures: list[AudioSampleFixture], workspace_root: Path) -> list[AudioExpectedOutput]:
    outputs: list[AudioExpectedOutput] = []
    for fixture in fixtures:
        text = runtime.transcribe(workspace_root / fixture.audio_path)['text']
        outputs.append(AudioExpectedOutput(sample_id=fixture.sample_id, audio_path=fixture.audio_path, text=text))
    return outputs


def export_bundle(
    *,
    model_dir: Path,
    output_dir: Path,
    asset_namespace: str = DEFAULT_ASSET_NAMESPACE,
    provider: str = 'CPUExecutionProvider',
    sample_fixtures: list[AudioSampleFixture] | None = None,
    expected_outputs: list[AudioExpectedOutput] | None = None,
    component_paths: dict[str, str | Path] | None = None,
    model_variant: str = DEFAULT_VARIANT,
    extra_metadata: dict | None = None,
) -> ModelBundleManifest:
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = {key: Path(value) for key, value in (component_paths or {}).items()}
    source_encoder = resolved.get('encoder', model_dir / DEFAULT_COMPONENT_FILES['encoder'])
    source_decoder = resolved.get('decoder', model_dir / DEFAULT_COMPONENT_FILES['decoder'])
    source_joiner = resolved.get('joiner', model_dir / DEFAULT_COMPONENT_FILES['joiner'])
    source_tokens = resolved.get('tokens', model_dir / DEFAULT_COMPONENT_FILES['tokens'])

    encoder_out = output_dir / 'encoder.onnx'
    decoder_out = output_dir / 'decoder.onnx'
    joiner_out = output_dir / 'joiner.onnx'
    tokens_out = output_dir / 'tokens.txt'
    shutil.copy2(source_encoder, encoder_out)
    shutil.copy2(source_decoder, decoder_out)
    shutil.copy2(source_joiner, joiner_out)
    shutil.copy2(source_tokens, tokens_out)

    fixtures = sample_fixtures or list(DEFAULT_AUDIO_FIXTURES)
    expected = expected_outputs
    if expected is None:
        runtime = ModelDirAcousticRuntime(model_dir=model_dir, provider=provider, component_paths={
            'encoder': source_encoder,
            'decoder': source_decoder,
            'joiner': source_joiner,
            'tokens': source_tokens,
        })
        expected = _build_expected_outputs(runtime, fixtures, Path(__file__).resolve().parents[2])

    sample_manifest_path = output_dir / 'sample_manifest.jsonl'
    expected_outputs_path = output_dir / 'expected_outputs.jsonl'
    sample_manifest_path.write_text(serialize_jsonl(fixtures), encoding='utf-8')
    expected_outputs_path.write_text(serialize_jsonl(expected), encoding='utf-8')

    metadata = {
        'sample_rate': 16000,
        'feature_dim': 80,
        'blank_id': 0,
        'context_size': 2,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    manifest = ModelBundleManifest(
        bundle_version=1,
        project='zipformer',
        model_family='zipformer-rnnt',
        model_name=f'zipformer/{model_variant}',
        model_variant=model_variant,
        asset_namespace=asset_namespace,
        runtime_kind='rnnt_greedy',
        artifacts={
            'encoder': encoder_out.name,
            'decoder': decoder_out.name,
            'joiner': joiner_out.name,
            'tokens': tokens_out.name,
        },
        fixtures={
            'sample_manifest': sample_manifest_path.name,
            'expected_outputs': expected_outputs_path.name,
        },
        metadata=metadata,
    )
    manifest.write_json(output_dir / 'bundle_manifest.json')
    return manifest


def verify_bundle(*, model_dir: Path | None = None, bundle_dir: Path | None = None, reference_bundle: Path | None = None, candidate_bundle: Path | None = None, provider: str = 'CPUExecutionProvider') -> dict:
    mismatches: list[dict] = []

    if reference_bundle is not None and candidate_bundle is not None:
        reference_manifest = ModelBundleManifest.from_path(reference_bundle / 'bundle_manifest.json')
        sample_rows = [AudioSampleFixture.from_dict(row) for row in read_jsonl(reference_bundle / reference_manifest.fixtures['sample_manifest'])]
        expected_rows = {row['sample_id']: AudioExpectedOutput.from_dict(row) for row in read_jsonl(reference_bundle / reference_manifest.fixtures['expected_outputs'])}
        reference_runtime = BundleAcousticRuntime.from_manifest_path(reference_bundle / 'bundle_manifest.json', provider=provider)
        candidate_runtime = BundleAcousticRuntime.from_manifest_path(candidate_bundle / 'bundle_manifest.json', provider=provider)
        workspace_root = Path(__file__).resolve().parents[2]
        for fixture in sample_rows:
            audio_path = workspace_root / fixture.audio_path
            reference_text = reference_runtime.transcribe(audio_path)['text']
            candidate_text = candidate_runtime.transcribe(audio_path)['text']
            expected_text = expected_rows[fixture.sample_id].text if fixture.sample_id in expected_rows else reference_text
            if candidate_text != reference_text:
                mismatches.append({
                    'sample_id': fixture.sample_id,
                    'expected_text': expected_text,
                    'reference_text': reference_text,
                    'candidate_text': candidate_text,
                })
        return {
            'project': 'zipformer',
            'passed': not mismatches,
            'checked_samples': len(sample_rows),
            'mismatches': mismatches,
            'bundle_dir': str(candidate_bundle),
        }

    if model_dir is None or bundle_dir is None:
        raise ValueError('zipformer verification requires either model_dir+bundle_dir or reference_bundle+candidate_bundle')

    manifest = ModelBundleManifest.from_path(bundle_dir / 'bundle_manifest.json')
    sample_rows = [AudioSampleFixture.from_dict(row) for row in read_jsonl(bundle_dir / manifest.fixtures['sample_manifest'])]
    expected_rows = {row['sample_id']: AudioExpectedOutput.from_dict(row) for row in read_jsonl(bundle_dir / manifest.fixtures['expected_outputs'])}
    model_runtime = ModelDirAcousticRuntime(model_dir=model_dir, provider=provider)
    bundle_runtime = BundleAcousticRuntime.from_manifest_path(bundle_dir / 'bundle_manifest.json', provider=provider)
    workspace_root = Path(__file__).resolve().parents[2]
    for fixture in sample_rows:
        audio_path = workspace_root / fixture.audio_path
        model_text = model_runtime.transcribe(audio_path)['text']
        bundle_text = bundle_runtime.transcribe(audio_path)['text']
        expected_text = expected_rows[fixture.sample_id].text if fixture.sample_id in expected_rows else model_text
        if bundle_text != model_text:
            mismatches.append({
                'sample_id': fixture.sample_id,
                'expected_text': expected_text,
                'model_dir_text': model_text,
                'bundle_text': bundle_text,
            })
    return {
        'project': 'zipformer',
        'passed': not mismatches,
        'checked_samples': len(sample_rows),
        'mismatches': mismatches,
        'bundle_dir': str(bundle_dir),
    }


ADAPTER = BundleProjectAdapter(
    name='zipformer',
    default_model_dir=str(DEFAULT_MODEL_DIR),
    default_output_dir=str(DEFAULT_OUTPUT_DIR),
    default_asset_namespace=DEFAULT_ASSET_NAMESPACE,
    default_variant=DEFAULT_VARIANT,
    export_bundle=export_bundle,
    verify_bundle=verify_bundle,
    bundle_runtime_from_manifest=BundleAcousticRuntime.from_manifest_path,
)
