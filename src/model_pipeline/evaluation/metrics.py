from __future__ import annotations

import math
import re
import statistics
import unicodedata
from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass(frozen=True)
class TranscriptMetrics:
    sample_count: int
    exact_matches: int
    character_errors: int
    reference_characters: int
    word_errors: int
    reference_words: int

    @property
    def character_error_rate(self) -> float:
        """Return normalized character error rate across all samples.

        Returns:
            Total character edits divided by reference characters.
        """
        return self.character_errors / max(self.reference_characters, 1)

    @property
    def word_error_rate(self) -> float:
        """Return normalized word error rate across all samples.

        Returns:
            Total word edits divided by reference words.
        """
        return self.word_errors / max(self.reference_words, 1)

    def to_dict(self) -> dict[str, int | float]:
        """Serialize transcript counts and derived error rates.

        Returns:
            JSON-compatible transcript metric fields.
        """
        return {
            **asdict(self),
            "character_error_rate": self.character_error_rate,
            "word_error_rate": self.word_error_rate,
        }


@dataclass(frozen=True)
class TextOutputClassification:
    empty: bool
    repetition_collapse: bool
    punctuation_collapse: bool

    @property
    def valid(self) -> bool:
        """Return whether no invalid-output condition was detected.

        Returns:
            `True` for a non-empty, non-collapsed text output.
        """
        return not (self.empty or self.repetition_collapse or self.punctuation_collapse)


@dataclass(frozen=True)
class VpcdOutputMetrics:
    exact_output_match: bool
    character_edit_distance: int
    first_five_top1_matches: int
    first_five_step_count: int
    early_eos: bool
    punctuation_collapse: bool

    def to_dict(self) -> dict[str, int | bool]:
        """Serialize one VPCD comparison result.

        Returns:
            JSON-compatible parity and invalid-output fields.
        """
        return asdict(self)


@dataclass(frozen=True)
class LatencySummary:
    sample_count: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    minimum_ms: float
    maximum_ms: float

    def to_dict(self) -> dict[str, int | float]:
        """Serialize deterministic latency summary fields.

        Returns:
            JSON-compatible latency statistics in milliseconds.
        """
        return asdict(self)


def normalize_transcript(text: str) -> str:
    """Normalize transcript case, punctuation, and whitespace for error metrics.

    Args:
        text: Raw reference or hypothesis transcript.

    Returns:
        Unicode-normalized lowercase words separated by single spaces.
    """
    normalized = unicodedata.normalize("NFC", text).casefold()
    normalized = "".join(
        character if character.isalnum() or character.isspace() else " "
        for character in normalized
    )
    return re.sub(r"\s+", " ", normalized).strip()


def compute_transcript_metrics(
    references: Sequence[str],
    hypotheses: Sequence[str],
) -> TranscriptMetrics:
    """Compute corpus exact-match, character, and word error counts.

    Args:
        references: Ordered reference transcripts.
        hypotheses: Ordered model transcripts aligned with references.

    Returns:
        Aggregated normalized transcript metrics.

    Raises:
        ValueError: If the sequences are empty or have different lengths.
    """
    if not references or len(references) != len(hypotheses):
        raise ValueError("Reference and hypothesis sequences must be non-empty and aligned")
    exact_matches = 0
    character_errors = 0
    reference_characters = 0
    word_errors = 0
    reference_words = 0
    for reference, hypothesis in zip(references, hypotheses):
        normalized_reference = normalize_transcript(reference)
        normalized_hypothesis = normalize_transcript(hypothesis)
        exact_matches += int(normalized_reference == normalized_hypothesis)
        reference_character_sequence = normalized_reference.replace(" ", "")
        hypothesis_character_sequence = normalized_hypothesis.replace(" ", "")
        character_errors += edit_distance(
            tuple(reference_character_sequence),
            tuple(hypothesis_character_sequence),
        )
        reference_characters += len(reference_character_sequence)
        reference_word_sequence = tuple(normalized_reference.split())
        hypothesis_word_sequence = tuple(normalized_hypothesis.split())
        word_errors += edit_distance(reference_word_sequence, hypothesis_word_sequence)
        reference_words += len(reference_word_sequence)
    return TranscriptMetrics(
        sample_count=len(references),
        exact_matches=exact_matches,
        character_errors=character_errors,
        reference_characters=reference_characters,
        word_errors=word_errors,
        reference_words=reference_words,
    )


def classify_text_output(text: str) -> TextOutputClassification:
    """Classify empty, repetitive, or punctuation-only text output.

    Args:
        text: Model output to inspect.

    Returns:
        Independent invalid-output flags and a derived validity property.
    """
    stripped = text.strip()
    tokens = stripped.casefold().split()
    repetition_collapse = len(tokens) >= 5 and len(set(tokens)) <= max(1, len(tokens) // 4)
    punctuation_collapse = bool(stripped) and not any(character.isalnum() for character in stripped)
    return TextOutputClassification(not stripped, repetition_collapse, punctuation_collapse)


def evaluate_vpcd_output(
    *,
    fp32_output: str,
    quantized_output: str,
    fp32_top1: Sequence[int],
    quantized_top1: Sequence[int],
    eos_token_id: int,
) -> VpcdOutputMetrics:
    """Compare one quantized VPCD decode with its FP32 control.

    Args:
        fp32_output: Restored text emitted by the FP32 control.
        quantized_output: Restored text emitted by the quantized model.
        fp32_top1: FP32 decoder top-1 token IDs in generation order.
        quantized_top1: Quantized decoder top-1 token IDs in generation order.
        eos_token_id: End-of-sequence token used by the model.

    Returns:
        Full-output and first-five-step parity evidence.
    """
    step_count = min(5, len(fp32_top1), len(quantized_top1))
    matches = sum(
        int(fp32_top1[index] == quantized_top1[index])
        for index in range(step_count)
    )
    early_eos = eos_token_id in quantized_top1[: max(0, len(fp32_top1) - 1)]
    output_classification = classify_text_output(quantized_output)
    return VpcdOutputMetrics(
        exact_output_match=quantized_output == fp32_output,
        character_edit_distance=edit_distance(tuple(fp32_output), tuple(quantized_output)),
        first_five_top1_matches=matches,
        first_five_step_count=step_count,
        early_eos=early_eos,
        punctuation_collapse=output_classification.punctuation_collapse
        or output_classification.repetition_collapse,
    )


def summarize_latency(milliseconds: Sequence[float]) -> LatencySummary:
    """Summarize non-negative latency observations deterministically.

    Args:
        milliseconds: One or more elapsed times in milliseconds.

    Returns:
        Count, mean, median, nearest-rank p95, minimum, and maximum.

    Raises:
        ValueError: If no observations exist or a value is negative.
    """
    values = sorted(float(value) for value in milliseconds)
    if not values or any(value < 0 for value in values):
        raise ValueError("Latency observations must be non-empty and non-negative")
    p95_index = max(0, math.ceil(0.95 * len(values)) - 1)
    return LatencySummary(
        sample_count=len(values),
        mean_ms=statistics.fmean(values),
        median_ms=statistics.median(values),
        p95_ms=values[p95_index],
        minimum_ms=values[0],
        maximum_ms=values[-1],
    )


def edit_distance(left: Sequence[object], right: Sequence[object]) -> int:
    """Compute Levenshtein edit distance for two token sequences.

    Args:
        left: Reference token sequence.
        right: Hypothesis token sequence.

    Returns:
        Minimum insertions, deletions, and substitutions.
    """
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + int(left_value != right_value),
                )
            )
        previous = current
    return previous[-1]
