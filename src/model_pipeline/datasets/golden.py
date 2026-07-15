from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol, Sequence

from model_pipeline.datasets.records import AudioExpectedOutput, AudioSampleFixture, TextGoldenSample


class AcousticRuntime(Protocol):
    def transcribe(self, path: Path):
        """Transcribe one audio fixture with the current acoustic runtime.

        Args:
            path: Absolute path to a fixture audio file.

        Returns:
            Recognized text or a mapping containing a `text` field.
        """
        ...


class PunctuationRuntime(Protocol):
    def encode(self, text: str):
        """Encode normalized text to current model token IDs.

        Args:
            text: Normalized acoustic output.

        Returns:
            Ordered model token IDs.
        """
        ...

    def restore(self, text: str):
        """Restore punctuation and casing with the current runtime.

        Args:
            text: Normalized acoustic output.

        Returns:
            Restored display text.
        """
        ...


def generate_audio_golden(
    fixtures: Sequence[AudioSampleFixture],
    *,
    repo_root: str | Path,
    runtime: AcousticRuntime,
) -> list[AudioExpectedOutput]:
    """Generate acoustic golden outputs using an injected runtime.

    Args:
        fixtures: Audio fixture inventory to evaluate.
        repo_root: Root used to resolve repository-relative fixture paths.
        runtime: Acoustic runtime providing current-model transcription.

    Returns:
        Actual text outputs paired with fixture identity and path.
    """
    root = Path(repo_root)
    results: list[AudioExpectedOutput] = []
    for fixture in fixtures:
        raw_result = runtime.transcribe(root / fixture.audio_path)
        text = raw_result["text"] if isinstance(raw_result, dict) else str(raw_result)
        results.append(AudioExpectedOutput(fixture.sample_id, fixture.audio_path, str(text)))
    return results


def generate_text_golden(
    audio_rows: Sequence[AudioExpectedOutput], *, runtime: PunctuationRuntime
) -> list[TextGoldenSample]:
    """Generate punctuation golden samples from acoustic outputs.

    Args:
        audio_rows: Current acoustic outputs to normalize and restore.
        runtime: Punctuation runtime providing tokenizer IDs and restored text.

    Returns:
        Text golden samples aligned to the end-to-end app flow.
    """
    results: list[TextGoldenSample] = []
    for row in audio_rows:
        raw_text = normalize_acoustic_text(row.text)
        results.append(
            TextGoldenSample(
                raw_text=raw_text,
                input_ids=[int(value) for value in runtime.encode(raw_text)],
                expected_output=str(runtime.restore(raw_text)),
                sample_id=row.sample_id,
            )
        )
    return results


def normalize_acoustic_text(text: str) -> str:
    """Normalize acoustic text for lowercase punctuation-model input.

    Args:
        text: Raw acoustic output, possibly containing SentencePiece markers.

    Returns:
        Lowercase text with normalized spaces and no edge whitespace.
    """
    normalized = str(text).replace("▁", " ")
    return re.sub(r"\s+", " ", normalized).strip().lower()
