from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from model_pipeline.evaluation.providers import (
    OrtProfileSummary,
    OrtProviderSelection,
    create_profiled_ort_session,
    summarize_ort_profile,
)


@dataclass(frozen=True)
class ZipformerInferenceResult:
    transcript: str
    token_ids: tuple[int, ...]
    latency_ms: float
    encoder_execution_target: str = "configured-onnx-runtime-provider"
    decoder_execution_target: str = "cpu"
    joiner_execution_target: str = "cpu"


class ZipformerLocalRuntime:
    def __init__(
        self,
        *,
        encoder_session,
        decoder_session,
        joiner_session,
        token_table: Mapping[int, str],
        feature_extractor: Callable[[np.ndarray, int], np.ndarray],
        fixed_encoder_frames: int = 2009,
        blank_id: int = 0,
        context_size: int = 2,
        max_symbols_per_frame: int = 100,
        provider_selection: OrtProviderSelection | None = None,
    ):
        """Initialize fixed-shape RNN-T sessions and CPU host decoding state.

        Args:
            encoder_session: ONNX Runtime-compatible Zipformer encoder session.
            decoder_session: CPU ONNX Runtime-compatible decoder session.
            joiner_session: CPU ONNX Runtime-compatible joiner session.
            token_table: Integer token IDs mapped to token pieces.
            feature_extractor: Callback producing frame-by-feature arrays.
            fixed_encoder_frames: Required padded encoder time dimension.
            blank_id: Recurrent neural network transducer blank token ID.
            context_size: Number of recent tokens supplied to the decoder.
            max_symbols_per_frame: Safety bound for non-blank emissions on one frame.
            provider_selection: Optional encoder provider-selection evidence.

        Returns:
            None.
        """
        self.encoder_session = encoder_session
        self.decoder_session = decoder_session
        self.joiner_session = joiner_session
        self.token_table = dict(token_table)
        self.feature_extractor = feature_extractor
        self.fixed_encoder_frames = int(fixed_encoder_frames)
        self.blank_id = int(blank_id)
        self.context_size = int(context_size)
        self.max_symbols_per_frame = int(max_symbols_per_frame)
        self.provider_selection = provider_selection

    @classmethod
    def from_paths(
        cls,
        *,
        encoder_path: str | Path,
        decoder_path: str | Path,
        joiner_path: str | Path,
        tokens_path: str | Path,
        prefer_cuda: bool,
        fixed_encoder_frames: int = 2009,
    ) -> "ZipformerLocalRuntime":
        """Create a profiled encoder runtime with CPU decoder and joiner sessions.

        Args:
            encoder_path: Fixed-shape FP32 or quantized encoder ONNX file.
            decoder_path: FP32 decoder ONNX file.
            joiner_path: FP32 joiner ONNX file.
            tokens_path: Token table text file.
            prefer_cuda: Whether to request CUDA for encoder execution.
            fixed_encoder_frames: Required padded encoder frame count.

        Returns:
            Configured local Zipformer runtime.
        """
        import onnxruntime as ort

        encoder_session, selection = create_profiled_ort_session(
            encoder_path,
            prefer_cuda=prefer_cuda,
        )
        decoder_session = ort.InferenceSession(
            Path(decoder_path).resolve().as_posix(),
            providers=["CPUExecutionProvider"],
        )
        joiner_session = ort.InferenceSession(
            Path(joiner_path).resolve().as_posix(),
            providers=["CPUExecutionProvider"],
        )
        return cls(
            encoder_session=encoder_session,
            decoder_session=decoder_session,
            joiner_session=joiner_session,
            token_table=load_token_table(tokens_path),
            feature_extractor=extract_zipformer_features,
            fixed_encoder_frames=fixed_encoder_frames,
            provider_selection=selection,
        )

    def transcribe(self, waveform: np.ndarray, *, sample_rate: int) -> ZipformerInferenceResult:
        """Transcribe one waveform through fixed-shape encoder and CPU greedy decode.

        Args:
            waveform: Mono floating-point waveform.
            sample_rate: Waveform sample rate in hertz.

        Returns:
            Transcript, emitted tokens, latency, and component execution targets.

        Raises:
            ValueError: If extracted features exceed fixed shape or have invalid rank.
            RuntimeError: If one encoder frame never emits blank within the safety bound.
        """
        started = time.perf_counter()
        features = np.asarray(self.feature_extractor(np.asarray(waveform), int(sample_rate)))
        if features.ndim != 2 or features.shape[1] != 80:
            raise ValueError(f"Expected feature shape [frames, 80], got {features.shape}")
        if features.shape[0] > self.fixed_encoder_frames:
            raise ValueError(
                f"Feature length {features.shape[0]} exceeds fixed encoder length "
                f"{self.fixed_encoder_frames}"
            )
        encoder_input = np.zeros(
            (1, self.fixed_encoder_frames, 80),
            dtype=np.float32,
        )
        encoder_input[0, : features.shape[0], :] = features.astype(np.float32)
        encoder_outputs = self.encoder_session.run(
            None,
            {
                "x": encoder_input,
                "x_lens": np.asarray([features.shape[0]], dtype=np.int64),
            },
        )
        encoded = np.asarray(encoder_outputs[0])
        encoded_length = int(np.asarray(encoder_outputs[1]).reshape(-1)[0])
        emitted = self._decode_token_ids(encoded, encoded_length)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return ZipformerInferenceResult(
            transcript=render_token_ids(emitted, self.token_table),
            token_ids=tuple(emitted),
            latency_ms=elapsed_ms,
        )

    def decode_encoder_outputs(
        self,
        encoded: np.ndarray,
        *,
        encoded_length: int,
    ) -> ZipformerInferenceResult:
        """Decode hosted or local encoder outputs with FP32 CPU components.

        Args:
            encoded: Batched encoder frames with final embedding dimension.
            encoded_length: Number of valid encoder frames to decode.

        Returns:
            Transcript, emitted token IDs, and local decoder/joiner latency.
        """
        started = time.perf_counter()
        emitted = self._decode_token_ids(np.asarray(encoded), int(encoded_length))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return ZipformerInferenceResult(
            transcript=render_token_ids(emitted, self.token_table),
            token_ids=tuple(emitted),
            latency_ms=elapsed_ms,
        )

    def _decode_token_ids(
        self,
        encoded: np.ndarray,
        encoded_length: int,
    ) -> list[int]:
        """Greedily decode valid encoder frames until each emits blank.

        Args:
            encoded: Batched encoder frames with final embedding dimension.
            encoded_length: Number of valid frames to decode.

        Returns:
            Emitted non-blank token IDs in decoder order.

        Raises:
            RuntimeError: If one frame reaches the non-blank safety bound.
        """
        context = [self.blank_id] * self.context_size
        emitted: list[int] = []
        for frame_index in range(encoded_length):
            encoder_frame = encoded[:, frame_index, :]
            for _ in range(self.max_symbols_per_frame):
                decoder_output = np.asarray(
                    self.decoder_session.run(
                        None,
                        {"y": np.asarray([context[-self.context_size :]], dtype=np.int64)},
                    )[0]
                )
                if decoder_output.ndim == 3 and decoder_output.shape[1] == 1:
                    decoder_output = decoder_output[:, 0, :]
                logits = np.asarray(
                    self.joiner_session.run(
                        None,
                        {
                            "encoder_out": encoder_frame,
                            "decoder_out": decoder_output,
                        },
                    )[0]
                )
                token_id = int(np.argmax(logits.reshape(-1)))
                if token_id == self.blank_id:
                    break
                emitted.append(token_id)
                context.append(token_id)
            else:
                raise RuntimeError(
                    f"Zipformer emitted {self.max_symbols_per_frame} non-blank symbols "
                    f"without blank at encoder frame {frame_index}"
                )
        return emitted

    def finish_provider_profile(self) -> OrtProfileSummary:
        """Finish encoder profiling and summarize actual node placement.

        Returns:
            Executed node counts by ONNX Runtime provider.
        """
        profile_path = self.encoder_session.end_profiling()
        return summarize_ort_profile(profile_path)


def extract_zipformer_features(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """Extract 80-bin Kaldi-compatible filterbank features from mono audio.

    Args:
        waveform: Mono floating-point waveform normalized near minus-one to one.
        sample_rate: Waveform sample rate in hertz.

    Returns:
        Float32 filterbank frames with 80 feature bins.

    Raises:
        ValueError: If the waveform is not mono or sample rate is unsupported.
    """
    if sample_rate != 16_000:
        raise ValueError(f"Zipformer requires 16000 Hz audio, got {sample_rate}")
    values = np.asarray(waveform, dtype=np.float32).squeeze()
    if values.ndim != 1:
        raise ValueError(f"Zipformer requires mono audio, got shape {values.shape}")
    import torch
    import torchaudio

    tensor = torch.from_numpy(values).unsqueeze(0)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=16_000,
        n_fft=400,
        hop_length=160,
        win_length=400,
        n_mels=80,
        f_min=20,
        f_max=8000,
        power=2.0,
    )(tensor)
    features = torch.clamp(mel, min=1.0e-10).log().squeeze(0).transpose(0, 1)
    return features.detach().cpu().numpy().astype(np.float32)


def load_token_table(path: str | Path) -> dict[int, str]:
    """Load an Icefall-style token table from text.

    Args:
        path: Token table whose final field is the integer token ID.

    Returns:
        Token IDs mapped to their serialized pieces.
    """
    table: dict[int, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        piece, token_id = line.rsplit(maxsplit=1)
        table[int(token_id)] = piece
    return table


def render_token_ids(token_ids: list[int], token_table: Mapping[int, str]) -> str:
    """Render emitted token IDs into whitespace-normalized text.

    Args:
        token_ids: Emitted non-blank token IDs.
        token_table: Token IDs mapped to SentencePiece-style pieces.

    Returns:
        Human-readable transcript.
    """
    pieces = [token_table.get(token_id, "") for token_id in token_ids]
    text = "".join(piece for piece in pieces if not (piece.startswith("<") and piece.endswith(">")))
    return re.sub(r"\s+", " ", text.replace("▁", " ")).strip()
