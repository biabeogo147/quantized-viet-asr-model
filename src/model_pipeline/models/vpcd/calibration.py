from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np

from model_pipeline.models.vpcd.quantization import CalibrationBatch, pad_calibration_batch


def build_calibration_batches(
    *,
    model_dir: str | Path,
    fp32_model_path: str | Path,
    text_source: str | Path,
    max_samples: int = 24,
    max_decode_length: int = 32,
    tokenizer_encode_path: str | Path | None = None,
    tokenizer_to_model_ids_path: str | Path | None = None,
) -> tuple[list[CalibrationBatch], dict[str, int]]:
    """Generate fixed-shape calibration prefixes from FP32 greedy decoding.

    Args:
        model_dir: Native tokenizer/model metadata directory when available.
        fp32_model_path: FP32 VPCD model used to generate decoder prefixes.
        text_source: Text file, JSONL file, or directory of text files.
        max_samples: Maximum number of source texts to consume.
        max_decode_length: Maximum greedy decoder steps per text.
        tokenizer_encode_path: Optional pre-exported tokenizer encoder graph.
        tokenizer_to_model_ids_path: Optional tokenizer-to-model ID map.

    Returns:
        Padded calibration batches and source/batch count statistics.

    Raises:
        FileNotFoundError: If neither native nor exported tokenizer inputs are available.
        ValueError: If the text source yields no calibration records.
    """
    import onnxruntime as ort

    model_root = Path(model_dir)
    use_native_tokenizer = (model_root / "sentencepiece.bpe.model").is_file()
    if use_native_tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_root.as_posix(), local_files_only=True)
        start_id = _decoder_start_id(model_root, tokenizer)
        pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        eos_id = int(tokenizer.eos_token_id)

        def encode(text: str) -> dict[str, np.ndarray]:
            """Encode text with the locally materialized Hugging Face tokenizer.

            Args:
                text: Normalized calibration text.

            Returns:
                Batch-one model input IDs and attention mask arrays.
            """
            raw = tokenizer(text, return_tensors="np", truncation=True, max_length=384)
            return {
                "input_ids": raw["input_ids"].astype(np.int64),
                "attention_mask": raw["attention_mask"].astype(np.int64),
            }
    else:
        if tokenizer_encode_path is None or tokenizer_to_model_ids_path is None:
            raise FileNotFoundError("Pre-exported tokenizer graph and ID map are required")
        from onnxruntime_extensions import get_library_path

        options = ort.SessionOptions()
        options.register_custom_ops_library(get_library_path())
        encode_session = ort.InferenceSession(
            Path(tokenizer_encode_path).as_posix(),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        tokenizer_to_model = np.asarray(
            json.loads(Path(tokenizer_to_model_ids_path).read_text(encoding="utf-8")), dtype=np.int64
        )
        start_id = 2
        pad_id = 1
        eos_id = 2

        def encode(text: str) -> dict[str, np.ndarray]:
            """Encode text with ONNX Runtime Extensions and bridge token IDs.

            Args:
                text: Normalized calibration text.

            Returns:
                Batch-one model input IDs and attention mask arrays.
            """
            tokenizer_ids = np.asarray(
                encode_session.run(None, {"inputs": np.asarray([text], dtype=object)})[0], dtype=np.int64
            ).reshape(-1)
            model_ids = tokenizer_to_model[tokenizer_ids][None, :]
            return {
                "input_ids": model_ids.astype(np.int64),
                "attention_mask": np.ones_like(model_ids, dtype=np.int64),
            }

    session = ort.InferenceSession(Path(fp32_model_path).as_posix(), providers=["CPUExecutionProvider"])
    batches: list[CalibrationBatch] = []
    sample_count = 0
    for text in iter_texts(text_source, max_samples=max_samples):
        sample_count += 1
        encoder = encode(text)
        decoded = [start_id]
        for _ in range(max_decode_length):
            decoder_ids = np.asarray([decoded], dtype=np.int64)
            decoder_mask = np.ones_like(decoder_ids)
            batch = CalibrationBatch(
                {
                    **encoder,
                    "decoder_input_ids": decoder_ids,
                    "decoder_attention_mask": decoder_mask,
                }
            )
            batches.append(pad_calibration_batch(batch, pad_token_id=pad_id))
            logits = session.run(None, batch.inputs)[0]
            next_id = int(np.argmax(logits[:, -1, :], axis=-1)[0])
            decoded.append(next_id)
            if next_id == eos_id:
                break
    if not batches:
        raise ValueError("Calibration text source produced no records")
    return batches, {"text_samples": sample_count, "batches": len(batches)}


def iter_texts(source: str | Path, *, max_samples: int) -> Iterator[str]:
    """Yield non-empty calibration texts from text or JSONL sources.

    Args:
        source: Text/JSONL file or directory containing text files.
        max_samples: Maximum number of records to yield.

    Yields:
        Normalized non-empty calibration text records.

    Raises:
        FileNotFoundError: If the source does not resolve to readable input files.
    """
    root = Path(source)
    files = [root] if root.is_file() else sorted(root.glob("*.txt")) if root.is_dir() else []
    if not files:
        raise FileNotFoundError(f"Calibration text source not found: {root}")
    count = 0
    for path in files:
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            text = line.strip()
            if path.suffix.lower() == ".jsonl" and text:
                payload = json.loads(text)
                text = str(payload.get("raw_text") or payload.get("text") or "").strip()
            if text:
                yield text
                count += 1
                if count == max_samples:
                    return


def _decoder_start_id(model_dir: Path, tokenizer) -> int:
    """Resolve the decoder start token from generation metadata or tokenizer fallback.

    Args:
        model_dir: Directory containing optional generation configuration.
        tokenizer: Loaded tokenizer providing the fallback EOS token.

    Returns:
        Decoder start token ID.
    """
    path = model_dir / "generation_config.json"
    if path.is_file():
        return int(json.loads(path.read_text(encoding="utf-8"))["decoder_start_token_id"])
    return int(tokenizer.eos_token_id)
