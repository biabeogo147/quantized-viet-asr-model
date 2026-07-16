from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from model_pipeline.evaluation.providers import (
    OrtProfileSummary,
    OrtProviderSelection,
    create_profiled_ort_session,
    summarize_ort_profile,
)


@dataclass(frozen=True)
class VpcdInferenceResult:
    output_text: str
    top1_token_ids: tuple[int, ...]
    latency_ms: float
    model_execution_target: str = "configured-onnx-runtime-provider"
    tokenizer_execution_target: str = "cpu"
    autoregressive_execution_target: str = "cpu"


class VpcdLocalRuntime:
    def __init__(
        self,
        *,
        model_session,
        encode_text: Callable[[str], tuple[np.ndarray, np.ndarray]],
        decode_tokens: Callable[[Sequence[int]], str],
        source_length: int = 384,
        decoder_length: int = 64,
        pad_token_id: int = 1,
        decoder_start_token_id: int = 2,
        eos_token_id: int = 2,
        provider_selection: OrtProviderSelection | None = None,
    ):
        """Initialize fixed-shape VPCD model and CPU host operations.

        Args:
            model_session: ONNX Runtime-compatible fixed-shape VPCD session.
            encode_text: CPU tokenizer callback returning IDs and attention mask.
            decode_tokens: CPU tokenizer callback restoring text from token IDs.
            source_length: Fixed source sequence length.
            decoder_length: Fixed decoder sequence length.
            pad_token_id: Token ID used for padding.
            decoder_start_token_id: First decoder token ID.
            eos_token_id: End-of-sequence token ID.
            provider_selection: Optional model provider-selection evidence.

        Returns:
            None.
        """
        self.model_session = model_session
        self.encode_text = encode_text
        self.decode_tokens = decode_tokens
        self.source_length = int(source_length)
        self.decoder_length = int(decoder_length)
        self.pad_token_id = int(pad_token_id)
        self.decoder_start_token_id = int(decoder_start_token_id)
        self.eos_token_id = int(eos_token_id)
        self.provider_selection = provider_selection

    @classmethod
    def from_paths(
        cls,
        *,
        model_path: str | Path,
        tokenizer_directory: str | Path,
        prefer_cuda: bool,
    ) -> "VpcdLocalRuntime":
        """Create a profiled VPCD model session with a local CPU tokenizer.

        Args:
            model_path: Fixed-shape FP32 or AIMET ONNX model.
            tokenizer_directory: Local Hugging Face tokenizer directory.
            prefer_cuda: Whether to request CUDA for model execution.

        Returns:
            Configured VPCD local runtime.
        """
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            Path(tokenizer_directory).resolve().as_posix(),
            local_files_only=True,
        )

        def encode_text(text: str) -> tuple[np.ndarray, np.ndarray]:
            """Encode one source string with the CPU tokenizer.

            Args:
                text: Source text to encode.

            Returns:
                Unpadded source token IDs and attention mask.
            """
            encoded = tokenizer(text, return_tensors="np", truncation=True, max_length=384)
            return (
                np.asarray(encoded["input_ids"]).reshape(-1),
                np.asarray(encoded["attention_mask"]).reshape(-1),
            )

        def decode_tokens(token_ids: Sequence[int]) -> str:
            """Decode generated token IDs with the CPU tokenizer.

            Args:
                token_ids: Generated non-special model token IDs.

            Returns:
                Restored output text.
            """
            return str(tokenizer.decode(list(token_ids), skip_special_tokens=True)).strip()

        model_session, selection = create_profiled_ort_session(
            model_path,
            prefer_cuda=prefer_cuda,
        )
        return cls(
            model_session=model_session,
            encode_text=encode_text,
            decode_tokens=decode_tokens,
            pad_token_id=int(tokenizer.pad_token_id),
            decoder_start_token_id=2,
            eos_token_id=int(tokenizer.eos_token_id),
            provider_selection=selection,
        )

    def restore(self, text: str) -> VpcdInferenceResult:
        """Restore punctuation and capitalization with fixed-shape greedy decoding.

        Args:
            text: Lowercase unpunctuated source text.

        Returns:
            Restored output, top-1 tokens, latency, and host execution targets.

        Raises:
            ValueError: If tokenizer output exceeds the fixed source shape.
        """
        started = time.perf_counter()
        source_ids, source_mask = self.encode_text(text)
        source_ids = np.asarray(source_ids, dtype=np.int64).reshape(-1)
        source_mask = np.asarray(source_mask, dtype=np.int64).reshape(-1)
        if len(source_ids) > self.source_length or len(source_ids) != len(source_mask):
            raise ValueError(
                f"Tokenizer output length {len(source_ids)} violates source shape "
                f"{self.source_length}"
            )
        fixed_source_ids = np.full(
            (1, self.source_length),
            self.pad_token_id,
            dtype=np.int64,
        )
        fixed_source_mask = np.zeros((1, self.source_length), dtype=np.int64)
        fixed_source_ids[0, : len(source_ids)] = source_ids
        fixed_source_mask[0, : len(source_mask)] = source_mask
        decoder_tokens = [self.decoder_start_token_id]
        top1_tokens: list[int] = []
        while len(decoder_tokens) < self.decoder_length:
            decoder_ids = np.full(
                (1, self.decoder_length),
                self.pad_token_id,
                dtype=np.int64,
            )
            decoder_mask = np.zeros((1, self.decoder_length), dtype=np.int64)
            decoder_ids[0, : len(decoder_tokens)] = decoder_tokens
            decoder_mask[0, : len(decoder_tokens)] = 1
            logits = np.asarray(
                self.model_session.run(
                    None,
                    {
                        "input_ids": fixed_source_ids,
                        "attention_mask": fixed_source_mask,
                        "decoder_input_ids": decoder_ids,
                        "decoder_attention_mask": decoder_mask,
                    },
                )[0]
            )
            next_token = int(np.argmax(logits[0, len(decoder_tokens) - 1, :]))
            top1_tokens.append(next_token)
            if next_token == self.eos_token_id:
                break
            decoder_tokens.append(next_token)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return VpcdInferenceResult(
            output_text=self.decode_tokens(top1_tokens[:-1] if top1_tokens and top1_tokens[-1] == self.eos_token_id else top1_tokens),
            top1_token_ids=tuple(top1_tokens),
            latency_ms=elapsed_ms,
        )

    def finish_provider_profile(self) -> OrtProfileSummary:
        """Finish model profiling and summarize actual node placement.

        Returns:
            Executed node counts by ONNX Runtime provider.
        """
        profile_path = self.model_session.end_profiling()
        return summarize_ort_profile(profile_path)
