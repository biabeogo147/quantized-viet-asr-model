from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Iterable

from model_pipeline.datasets.vlsp import VlspRow


MIN_DURATION_SECONDS = 2.0
MAX_DURATION_SECONDS = 12.0
MIN_WORD_COUNT = 4
MAX_WORD_COUNT = 40
MIN_CHAR_COUNT = 12
MAX_CHAR_COUNT = 180


@dataclass(frozen=True)
class AudioInfo:
    duration_seconds: float
    sample_rate: int


@dataclass(frozen=True)
class SelectedFixture:
    sample_id: str
    source: VlspRow
    normalized_text: str
    duration_seconds: float
    original_sample_rate: int
    exported_sample_rate: int = 16_000


def select_fixtures(
    rows: Iterable[VlspRow],
    *,
    selection_count: int,
    probe: Callable[[VlspRow], AudioInfo],
) -> list[SelectedFixture]:
    """Select deterministic Vietnamese speech fixtures under quality bounds.

    Args:
        rows: Eligible VLSP rows in stable source order.
        selection_count: Exact number of fixtures required.
        probe: Callback returning duration and sample-rate metadata.

    Returns:
        Selected fixtures numbered after the retained default sample.

    Raises:
        ValueError: If the requested count is invalid or cannot be satisfied.
    """
    if selection_count < 1:
        raise ValueError("selection_count must be >= 1")
    selected: list[SelectedFixture] = []
    seen: set[str] = set()
    for row in rows:
        text = re.sub(r"\s+", " ", row.transcription.strip())
        words = text.split()
        if not _has_vietnamese_marks(text):
            continue
        if not MIN_CHAR_COUNT <= len(text) <= MAX_CHAR_COUNT:
            continue
        if not MIN_WORD_COUNT <= len(words) <= MAX_WORD_COUNT:
            continue
        dedupe_key = re.sub(r"[^\w\s]", " ", text.casefold())
        dedupe_key = re.sub(r"\s+", " ", dedupe_key).strip()
        if not dedupe_key or dedupe_key in seen:
            continue
        audio = probe(row)
        if not MIN_DURATION_SECONDS <= audio.duration_seconds <= MAX_DURATION_SECONDS:
            continue
        selected.append(
            SelectedFixture(
                sample_id=f"sample-{len(selected) + 2}",
                source=row,
                normalized_text=text,
                duration_seconds=round(float(audio.duration_seconds), 3),
                original_sample_rate=int(audio.sample_rate),
            )
        )
        seen.add(dedupe_key)
        if len(selected) == selection_count:
            break
    if len(selected) != selection_count:
        raise ValueError(f"Selected {len(selected)} fixtures; expected {selection_count}")
    return selected


def _has_vietnamese_marks(text: str) -> bool:
    """Detect precomposed or combining Vietnamese diacritic marks.

    Args:
        text: Transcription text to inspect.

    Returns:
        `True` when the text contains a Vietnamese-specific marked character.
    """
    if any(character in text.casefold() for character in "đăâêôơư"):
        return True
    return any(unicodedata.combining(character) for character in unicodedata.normalize("NFD", text))
