from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from model_pipeline.evaluation.metrics import (
    classify_text_output,
    compute_transcript_metrics,
    evaluate_vpcd_output,
    summarize_latency,
)


@dataclass(frozen=True)
class VlspEvaluationSample:
    """Describe one portable held-out VLSP evaluation record."""

    sample_id: str
    audio_path: Path
    transcription: str


def load_vlsp_evaluation_samples(manifest_path: str | Path) -> tuple[VlspEvaluationSample, ...]:
    """Load held-out VLSP records and resolve audio beside the portable manifest.

    Args:
        manifest_path: Calibration/evaluation split manifest path.

    Returns:
        Evaluation samples in manifest order.

    Raises:
        ValueError: If the manifest contains no evaluation samples.
        FileNotFoundError: If a referenced evaluation audio file is missing.
    """
    manifest = Path(manifest_path).resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    samples = tuple(
        VlspEvaluationSample(
            sample_id=str(item["sample_id"]),
            audio_path=(manifest.parent / str(item["audio_path"])).resolve(),
            transcription=str(item["transcription"]),
        )
        for item in payload.get("samples", ())
        if item.get("partition") == "evaluation"
    )
    if not samples:
        raise ValueError("VLSP manifest contains no evaluation samples")
    missing = [sample.audio_path for sample in samples if not sample.audio_path.is_file()]
    if missing:
        raise FileNotFoundError(f"VLSP evaluation audio is missing: {missing!r}")
    return samples


def load_mono_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Decode one audio file as a mono floating-point waveform.

    Args:
        path: Audio file path supported by Torchaudio.

    Returns:
        Mono waveform and integer sample rate.
    """
    import torchaudio

    waveform, sample_rate = torchaudio.load(Path(path).as_posix())
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0).detach().cpu().numpy().astype(np.float32), int(sample_rate)


def evaluate_zipformer_samples(
    runtime: Any,
    samples: Sequence[VlspEvaluationSample],
    *,
    audio_loader: Callable[[Path], tuple[np.ndarray, int]] = load_mono_audio,
) -> tuple[dict[str, object], ...]:
    """Transcribe ordered VLSP samples through one Zipformer runtime.

    Args:
        runtime: Object exposing `transcribe(waveform, sample_rate=...)`.
        samples: Held-out VLSP samples to transcribe.
        audio_loader: Audio decoder returning mono waveform and sample rate.

    Returns:
        Deterministic per-sample transcript, token, latency, and collapse records.
    """
    records: list[dict[str, object]] = []
    for sample in samples:
        waveform, sample_rate = audio_loader(sample.audio_path)
        result = runtime.transcribe(waveform, sample_rate=sample_rate)
        classification = classify_text_output(result.transcript)
        records.append(
            {
                "sample_id": sample.sample_id,
                "reference": sample.transcription,
                "transcript": result.transcript,
                "token_ids": list(result.token_ids),
                "latency_ms": float(result.latency_ms),
                "empty": classification.empty,
                "repetition_collapse": classification.repetition_collapse,
            }
        )
    return tuple(records)


def summarize_zipformer_outputs(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Aggregate Zipformer transcript quality, invalid output, and latency evidence.

    Args:
        records: Per-sample Zipformer evaluation records.

    Returns:
        Corpus transcript metrics, invalid-output counts, and latency summary.
    """
    metrics = compute_transcript_metrics(
        [str(record["reference"]) for record in records],
        [str(record["transcript"]) for record in records],
    )
    return {
        "transcript_metrics": metrics.to_dict(),
        "empty_outputs": sum(bool(record["empty"]) for record in records),
        "repetition_collapses": sum(bool(record["repetition_collapse"]) for record in records),
        "latency": summarize_latency([float(record["latency_ms"]) for record in records]).to_dict(),
    }


def summarize_zipformer_regression(
    fp32_records: Sequence[Mapping[str, object]],
    quantized_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Compare aligned quantized Zipformer outputs with the FP32 control.

    Args:
        fp32_records: Ordered FP32 per-sample outputs.
        quantized_records: Ordered quantized per-sample outputs.

    Returns:
        FP32/quantized summaries and transcript regression counts.

    Raises:
        ValueError: If sample IDs are not aligned.
    """
    _require_aligned_ids(fp32_records, quantized_records)
    return {
        "fp32": summarize_zipformer_outputs(fp32_records),
        "quantized": summarize_zipformer_outputs(quantized_records),
        "exact_transcript_parity": sum(
            str(fp32["transcript"]) == str(quantized["transcript"])
            for fp32, quantized in zip(fp32_records, quantized_records)
        ),
        "sample_count": len(fp32_records),
    }


def evaluate_vpcd_samples(
    runtime: Any,
    samples: Sequence[VlspEvaluationSample],
) -> tuple[dict[str, object], ...]:
    """Restore ordered VLSP transcriptions through one VPCD runtime.

    Args:
        runtime: Object exposing `restore(text)`.
        samples: Held-out VLSP transcription records.

    Returns:
        Deterministic per-sample restored output, top-1 tokens, and latency records.
    """
    return tuple(
        {
            "sample_id": sample.sample_id,
            "source_text": sample.transcription,
            "output_text": result.output_text,
            "top1_token_ids": list(result.top1_token_ids),
            "latency_ms": float(result.latency_ms),
        }
        for sample in samples
        for result in (runtime.restore(sample.transcription),)
    )


def summarize_vpcd_parity(
    fp32_records: Sequence[Mapping[str, object]],
    quantized_records: Sequence[Mapping[str, object]],
    *,
    eos_token_id: int,
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    """Aggregate full-output and first-five-step VPCD parity evidence.

    Args:
        fp32_records: Ordered FP32 VPCD output records.
        quantized_records: Ordered quantized VPCD output records.
        eos_token_id: Model end-of-sequence token ID.

    Returns:
        Aggregate parity/latency summary and aligned per-sample comparison records.

    Raises:
        ValueError: If sample IDs are not aligned.
    """
    _require_aligned_ids(fp32_records, quantized_records)
    comparisons: list[dict[str, object]] = []
    for fp32, quantized in zip(fp32_records, quantized_records):
        metrics = evaluate_vpcd_output(
            fp32_output=str(fp32["output_text"]),
            quantized_output=str(quantized["output_text"]),
            fp32_top1=tuple(int(token) for token in fp32["top1_token_ids"]),
            quantized_top1=tuple(int(token) for token in quantized["top1_token_ids"]),
            eos_token_id=eos_token_id,
        )
        comparisons.append(
            {
                "sample_id": fp32["sample_id"],
                "fp32_output": fp32["output_text"],
                "quantized_output": quantized["output_text"],
                **metrics.to_dict(),
            }
        )
    return (
        {
            "sample_count": len(comparisons),
            "exact_output_matches": sum(bool(row["exact_output_match"]) for row in comparisons),
            "character_edit_distance": sum(int(row["character_edit_distance"]) for row in comparisons),
            "first_five_top1_matches": sum(int(row["first_five_top1_matches"]) for row in comparisons),
            "first_five_step_count": sum(int(row["first_five_step_count"]) for row in comparisons),
            "early_eos_count": sum(bool(row["early_eos"]) for row in comparisons),
            "collapse_count": sum(bool(row["punctuation_collapse"]) for row in comparisons),
            "fp32_latency": summarize_latency(
                [float(record["latency_ms"]) for record in fp32_records]
            ).to_dict(),
            "quantized_latency": summarize_latency(
                [float(record["latency_ms"]) for record in quantized_records]
            ).to_dict(),
        },
        tuple(comparisons),
    )


def _require_aligned_ids(
    left: Sequence[Mapping[str, object]],
    right: Sequence[Mapping[str, object]],
) -> None:
    """Require two model-output sequences to contain identical sample IDs.

    Args:
        left: First ordered output sequence.
        right: Second ordered output sequence.

    Returns:
        None.

    Raises:
        ValueError: If sequences are empty or sample IDs differ.
    """
    left_ids = [str(record["sample_id"]) for record in left]
    right_ids = [str(record["sample_id"]) for record in right]
    if not left_ids or left_ids != right_ids:
        raise ValueError("Evaluation records must be non-empty and aligned by sample ID")
