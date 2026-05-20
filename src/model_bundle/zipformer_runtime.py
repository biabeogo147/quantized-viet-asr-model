from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np

from model_bundle.manifest import ModelBundleManifest


DEFAULT_MODEL_DIR = Path("assets") / "zipformer"
DEFAULT_COMPONENT_FILES = {
    "encoder": "encoder-epoch-20-avg-1.onnx",
    "decoder": "decoder-epoch-20-avg-1.onnx",
    "joiner": "joiner-epoch-20-avg-1.onnx",
    "tokens": "tokens.txt",
}


def prepare_encoder_inputs(features: np.ndarray, fixed_encoder_frames: int | None = None) -> dict[str, np.ndarray]:
    if features.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape {features.shape}")

    frame_count, feature_dim = features.shape
    if fixed_encoder_frames is None:
        x = features[None, ...].astype(np.float32, copy=False)
    else:
        if frame_count > fixed_encoder_frames:
            raise ValueError(
                f"Audio features contain {frame_count} frames but bundle expects at most {fixed_encoder_frames}."
            )
        x = np.zeros((1, fixed_encoder_frames, feature_dim), dtype=np.float32)
        x[:, :frame_count, :] = features.astype(np.float32, copy=False)
    x_lens = np.array([frame_count], dtype=np.int64)
    return {"x": x, "x_lens": x_lens}


def trim_encoder_frames(encoder_frames: np.ndarray, encoder_out_lens: np.ndarray | None) -> np.ndarray:
    if encoder_out_lens is None:
        return encoder_frames
    valid_frames = int(encoder_out_lens.reshape(-1)[0])
    return encoder_frames[:valid_frames]


def resolve_fixed_encoder_frames(metadata: dict) -> int | None:
    fixed_input_shapes = metadata.get("fixed_input_shapes", {}) if isinstance(metadata, dict) else {}
    encoder_shapes = fixed_input_shapes.get("encoder", {}) if isinstance(fixed_input_shapes, dict) else {}
    encoder_x_shape = encoder_shapes.get("x") if isinstance(encoder_shapes, dict) else None
    if isinstance(encoder_x_shape, (list, tuple)) and len(encoder_x_shape) >= 2:
        return int(encoder_x_shape[1])
    fixed_encoder_frames = metadata.get("fixed_encoder_frames") if isinstance(metadata, dict) else None
    return None if fixed_encoder_frames is None else int(fixed_encoder_frames)


def decode_encoder_frames_greedy(
    *,
    frames: np.ndarray,
    decoder_session: object,
    joiner_session: object,
    tokens_table: list[str],
    blank_id: int,
    context_size: int,
) -> dict[str, object]:
    frame_rows = np.asarray(frames, dtype=np.float32)
    if frame_rows.ndim != 2:
        raise ValueError(f"Expected 2D encoder frames, got shape {frame_rows.shape}")

    token_ids: list[int] = []
    history = [int(blank_id)] * int(context_size)

    for frame in frame_rows:
        enc_frame = frame.reshape(1, -1).astype(np.float32, copy=False)
        while True:
            dec_in = np.asarray([history[-int(context_size) :]], dtype=np.int64)
            dec_out = decoder_session.run(None, {"y": dec_in})[0]
            join_out = joiner_session.run(
                None,
                {
                    "encoder_out": enc_frame,
                    "decoder_out": np.asarray(dec_out, dtype=np.float32),
                },
            )[0]
            token = int(np.argmax(join_out, axis=-1)[0])
            if token == int(blank_id):
                break
            token_ids.append(token)
            history.append(token)

    return {
        "text": ZipformerRuntimeBase.decode_tokens(tokens_table, token_ids),
        "num_tokens": len(token_ids),
        "token_ids": token_ids,
    }


class ZipformerRuntimeBase:
    def _load_tokens(self, tokens_path: Path) -> list[str]:
        tokens: list[str] = []
        with tokens_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if not parts:
                    continue
                tokens.append(parts[0] if len(parts) == 2 else parts[-1])
        return tokens

    @staticmethod
    def decode_tokens(tokens_table: list[str], result: list[int]) -> str:
        text = "".join(tokens_table[i] for i in result if i < len(tokens_table))
        text = text.replace("?", " ").strip()
        text = re.sub(r"\s{2,}", " ", text)
        return text

    @staticmethod
    def _resolve_providers(provider: str) -> list[str]:
        if provider == "CPUExecutionProvider":
            return ["CPUExecutionProvider"]
        return [provider, "CPUExecutionProvider"] if provider != "CPUExecutionProvider" else ["CPUExecutionProvider"]

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
        provider: str = "CPUExecutionProvider",
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
        self.encoder_path = resolved.get("encoder", self.model_dir / DEFAULT_COMPONENT_FILES["encoder"])
        self.decoder_path = resolved.get("decoder", self.model_dir / DEFAULT_COMPONENT_FILES["decoder"])
        self.joiner_path = resolved.get("joiner", self.model_dir / DEFAULT_COMPONENT_FILES["joiner"])
        self.tokens_path = resolved.get("tokens", self.model_dir / DEFAULT_COMPONENT_FILES["tokens"])

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

        decoder_started = time.time()
        decode_result = decode_encoder_frames_greedy(
            frames=frames,
            decoder_session=self.decoder_sess,
            joiner_session=self.joiner_sess,
            tokens_table=self.tokens_table,
            blank_id=self.blank_id,
            context_size=self.context_size,
        )
        decoder_elapsed = time.time() - decoder_started

        return {
            "text": str(decode_result["text"]),
            "num_tokens": int(decode_result["num_tokens"]),
            "encoder_time": round(encoder_elapsed, 3),
            "decoder_time": round(decoder_elapsed, 3),
        }


class BundleAcousticRuntime(ModelDirAcousticRuntime):
    @classmethod
    def from_manifest_path(cls, manifest_path: str | Path, provider: str = "CPUExecutionProvider") -> "BundleAcousticRuntime":
        manifest = ModelBundleManifest.from_path(manifest_path)
        bundle_dir = Path(manifest_path).resolve().parent
        return cls(
            model_dir=bundle_dir,
            provider=provider,
            component_paths={
                key: bundle_dir / value
                for key, value in manifest.artifacts.items()
                if key in {"encoder", "decoder", "joiner", "tokens"}
            },
            sample_rate=int(manifest.metadata.get("sample_rate", 16000)),
            feature_dim=int(manifest.metadata.get("feature_dim", 80)),
            blank_id=int(manifest.metadata.get("blank_id", 0)),
            context_size=int(manifest.metadata.get("context_size", 2)),
            fixed_encoder_frames=resolve_fixed_encoder_frames(manifest.metadata),
        )
