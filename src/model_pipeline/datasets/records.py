from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping


@dataclass(frozen=True)
class TextGoldenSample:
    raw_text: str
    input_ids: list[int]
    expected_output: str
    sample_id: str = ""

    def to_dict(self) -> dict:
        """Serialize a text golden sample while omitting an empty sample ID.

        Returns:
            JSON-compatible golden-sample fields.
        """
        payload = asdict(self)
        if not self.sample_id:
            payload.pop("sample_id")
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping) -> "TextGoldenSample":
        """Restore a typed text golden sample from JSON-compatible fields.

        Args:
            payload: Mapping containing raw text, IDs, expected output, and optional ID.

        Returns:
            The normalized text golden sample.
        """
        return cls(
            raw_text=str(payload["raw_text"]),
            input_ids=[int(value) for value in payload["input_ids"]],
            expected_output=str(payload["expected_output"]),
            sample_id=str(payload.get("sample_id", "")),
        )


@dataclass(frozen=True)
class AudioSampleFixture:
    sample_id: str
    audio_path: str

    def to_dict(self) -> dict:
        """Serialize an audio fixture to JSON-compatible fields.

        Returns:
            Sample ID and repository-relative audio path.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping) -> "AudioSampleFixture":
        """Restore an audio fixture from JSON-compatible fields.

        Args:
            payload: Mapping containing sample ID and audio path.

        Returns:
            The normalized audio fixture.
        """
        return cls(str(payload["sample_id"]), str(payload["audio_path"]))


@dataclass(frozen=True)
class AudioExpectedOutput:
    sample_id: str
    audio_path: str
    text: str

    def to_dict(self) -> dict:
        """Serialize an acoustic golden output to JSON-compatible fields.

        Returns:
            Sample identity, audio path, and recognized text.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping) -> "AudioExpectedOutput":
        """Restore an acoustic golden output from JSON-compatible fields.

        Args:
            payload: Mapping containing sample ID, audio path, and output text.

        Returns:
            The normalized acoustic output record.
        """
        return cls(str(payload["sample_id"]), str(payload["audio_path"]), str(payload["text"]))


def serialize_jsonl(items: Iterable[object]) -> str:
    """Serialize record objects or mappings as UTF-8-friendly JSON Lines.

    Args:
        items: Objects exposing `to_dict` or already JSON-compatible values.

    Returns:
        Newline-terminated JSONL text preserving Unicode characters.
    """
    return "".join(
        json.dumps(item.to_dict() if hasattr(item, "to_dict") else item, ensure_ascii=False) + "\n"
        for item in items
    )


def read_jsonl(path: str | Path) -> list[dict]:
    """Read non-empty JSONL rows with optional UTF-8 BOM handling.

    Args:
        path: JSON Lines file to decode.

    Returns:
        Decoded row mappings in file order.
    """
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
