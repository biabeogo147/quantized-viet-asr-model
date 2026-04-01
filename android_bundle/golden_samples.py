from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class GoldenSample:
    raw_text: str
    input_ids: list[int]
    expected_output: str

    def to_dict(self) -> dict[str, object]:
        return {
            "raw_text": self.raw_text,
            "input_ids": list(self.input_ids),
            "expected_output": self.expected_output,
        }


def serialize_golden_samples(samples: list[GoldenSample]) -> str:
    return "".join(
        json.dumps(sample.to_dict(), ensure_ascii=True, separators=(",", ":")) + "\n"
        for sample in samples
    )
