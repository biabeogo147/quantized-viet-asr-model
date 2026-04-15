from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class TextGoldenSample:
    raw_text: str
    input_ids: list[int]
    expected_output: str
    sample_id: str = ''

    def to_dict(self) -> dict:
        payload = asdict(self)
        if not payload['sample_id']:
            payload.pop('sample_id')
        return payload

    def to_jsonl_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False) + '\n'


@dataclass(frozen=True)
class AudioSampleFixture:
    sample_id: str
    audio_path: str

    def to_dict(self) -> dict:
        return asdict(self)

    def to_jsonl_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False) + '\n'

    @classmethod
    def from_dict(cls, payload: dict) -> 'AudioSampleFixture':
        return cls(sample_id=str(payload['sample_id']), audio_path=str(payload['audio_path']))


@dataclass(frozen=True)
class AudioExpectedOutput:
    sample_id: str
    audio_path: str
    text: str

    def to_dict(self) -> dict:
        return asdict(self)

    def to_jsonl_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False) + '\n'

    @classmethod
    def from_dict(cls, payload: dict) -> 'AudioExpectedOutput':
        return cls(sample_id=str(payload['sample_id']), audio_path=str(payload['audio_path']), text=str(payload['text']))


def serialize_jsonl(items: Iterable[object]) -> str:
    lines: list[str] = []
    for item in items:
        if hasattr(item, 'to_jsonl_line'):
            lines.append(item.to_jsonl_line())
        else:
            lines.append(json.dumps(item, ensure_ascii=False) + '\n')
    return ''.join(lines)


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows
