import json
import os
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader
from transformers import AutoTokenizer

from quantize.types import CalibrationSample


class ListCalibrationDataReader(CalibrationDataReader):
    def __init__(self, records: Sequence[CalibrationSample]):
        self._all_records = list(records)
        self._records = iter(self._all_records)

    def get_next(self) -> dict[str, np.ndarray] | None:
        sample = next(self._records, None)
        return None if sample is None else sample.inputs

    def rewind(self) -> None:
        self._records = iter(self._all_records)


def iter_calibration_texts(path: str | os.PathLike[str], max_samples: int) -> Iterator[str]:
    source_path = Path(path)
    count = 0
    for file_path in iter_calibration_files(source_path):
        with open(file_path, "r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                yield text
                count += 1
                if count >= max_samples:
                    return


def iter_calibration_files(path: Path) -> Iterator[Path]:
    if path.is_file():
        yield path
        return
    if path.is_dir():
        for file_path in sorted(path.glob("*.txt")):
            if file_path.is_file():
                yield file_path
        return
    raise FileNotFoundError(f"Khong tim thay calibration source: {path}")


def make_calibration_records(
    encoder_inputs: dict[str, np.ndarray],
    decoded_ids: Sequence[int],
    decoder_start_token_id: int,
) -> list[CalibrationSample]:
    if not decoded_ids or decoded_ids[0] != decoder_start_token_id:
        raise ValueError("decoded_ids must start with decoder_start_token_id")

    if len(decoded_ids) == 1:
        prefixes = [decoded_ids]
    else:
        prefixes = [decoded_ids[:prefix_len] for prefix_len in range(1, len(decoded_ids))]

    records: list[CalibrationSample] = []
    for prefix in prefixes:
        decoder_input_ids = np.asarray([prefix], dtype=np.int64)
        decoder_attention_mask = np.ones_like(decoder_input_ids, dtype=np.int64)
        records.append(
            CalibrationSample(
                inputs={
                    "input_ids": encoder_inputs["input_ids"].astype(np.int64, copy=False),
                    "attention_mask": encoder_inputs["attention_mask"].astype(np.int64, copy=False),
                    "decoder_input_ids": decoder_input_ids,
                    "decoder_attention_mask": decoder_attention_mask,
                }
            )
        )
    return records


def _pad_array(values: np.ndarray, target_length: int, pad_value: int) -> np.ndarray:
    current_length = int(values.shape[1])
    if current_length >= target_length:
        return values
    padded = np.full((values.shape[0], target_length), pad_value, dtype=np.int64)
    padded[:, :current_length] = values
    return padded


def pad_calibration_samples(
    samples: Sequence[CalibrationSample],
    pad_token_id: int,
) -> list[CalibrationSample]:
    if not samples:
        return []

    max_encoder_len = max(int(sample.inputs["input_ids"].shape[1]) for sample in samples)
    max_decoder_len = max(int(sample.inputs["decoder_input_ids"].shape[1]) for sample in samples)
    padded_samples: list[CalibrationSample] = []
    for sample in samples:
        padded_samples.append(
            CalibrationSample(
                inputs={
                    "input_ids": _pad_array(sample.inputs["input_ids"], max_encoder_len, pad_token_id),
                    "attention_mask": _pad_array(sample.inputs["attention_mask"], max_encoder_len, 0),
                    "decoder_input_ids": _pad_array(
                        sample.inputs["decoder_input_ids"],
                        max_decoder_len,
                        pad_token_id,
                    ),
                    "decoder_attention_mask": _pad_array(
                        sample.inputs["decoder_attention_mask"],
                        max_decoder_len,
                        0,
                    ),
                }
            )
        )
    return padded_samples


def load_decoder_start_token_id(model_dir: Path, tokenizer: AutoTokenizer) -> int:
    generation_config_path = model_dir / "generation_config.json"
    if generation_config_path.exists():
        with open(generation_config_path, "r", encoding="utf-8") as handle:
            generation_config = json.load(handle)
        return int(generation_config.get("decoder_start_token_id", tokenizer.eos_token_id))
    return int(tokenizer.eos_token_id)


def greedy_decode_ids(
    session: ort.InferenceSession,
    tokenizer: AutoTokenizer,
    text: str,
    decoder_start_token_id: int,
    max_generation_length: int,
) -> tuple[dict[str, np.ndarray], list[int]]:
    encoded = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=512,
    )
    encoder_inputs = {
        "input_ids": encoded["input_ids"].astype(np.int64),
        "attention_mask": encoded["attention_mask"].astype(np.int64),
    }
    decoded_ids = [decoder_start_token_id]

    for _ in range(max_generation_length):
        decoder_input_ids = np.asarray([decoded_ids], dtype=np.int64)
        decoder_attention_mask = np.ones_like(decoder_input_ids, dtype=np.int64)
        outputs = session.run(
            None,
            {
                "input_ids": encoder_inputs["input_ids"],
                "attention_mask": encoder_inputs["attention_mask"],
                "decoder_input_ids": decoder_input_ids,
                "decoder_attention_mask": decoder_attention_mask,
            },
        )
        logits = outputs[0]
        next_token_id = int(np.argmax(logits[:, -1, :], axis=-1)[0])
        decoded_ids.append(next_token_id)
        if next_token_id == tokenizer.eos_token_id:
            break

    return encoder_inputs, decoded_ids


def build_calibration_records(
    model_dir: Path,
    fp32_onnx_path: Path,
    calibration_source_path: Path,
    max_calibration_samples: int,
    max_generation_length: int,
) -> tuple[list[CalibrationSample], dict[str, int]]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    decoder_start_token_id = load_decoder_start_token_id(model_dir, tokenizer)
    session = ort.InferenceSession(os.fspath(fp32_onnx_path), providers=["CPUExecutionProvider"])

    records: list[CalibrationSample] = []
    max_encoder_len = 0
    max_decoder_len = 0
    sample_count = 0
    source_files = list(iter_calibration_files(calibration_source_path))
    for text in iter_calibration_texts(calibration_source_path, max_samples=max_calibration_samples):
        sample_count += 1
        encoder_inputs, decoded_ids = greedy_decode_ids(
            session=session,
            tokenizer=tokenizer,
            text=text,
            decoder_start_token_id=decoder_start_token_id,
            max_generation_length=max_generation_length,
        )
        max_encoder_len = max(max_encoder_len, int(encoder_inputs["input_ids"].shape[1]))
        max_decoder_len = max(max_decoder_len, len(decoded_ids))
        records.extend(
            make_calibration_records(
                encoder_inputs=encoder_inputs,
                decoded_ids=decoded_ids,
                decoder_start_token_id=decoder_start_token_id,
            )
        )

    stats = {
        "source_files": len(source_files),
        "text_samples": sample_count,
        "records": len(records),
        "max_encoder_len": max_encoder_len,
        "max_decoder_len": max_decoder_len,
    }
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    return pad_calibration_samples(records, pad_token_id=int(pad_token_id)), stats
