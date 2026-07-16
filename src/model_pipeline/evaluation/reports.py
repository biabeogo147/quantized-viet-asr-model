from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping


def write_evaluation_json(path: str | Path, payload: Mapping[str, object]) -> Path:
    """Write deterministic machine-readable evaluation evidence.

    Args:
        path: Destination JSON file.
        payload: JSON-compatible evaluation fields.

    Returns:
        Resolved path to the generated JSON file.
    """
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def write_sample_jsonl(path: str | Path, samples: Iterable[Mapping[str, object]]) -> Path:
    """Write deterministic per-sample evaluation evidence as JSON Lines.

    Args:
        path: Destination JSONL file.
        samples: Ordered JSON-compatible sample records.

    Returns:
        Resolved path to the generated JSONL file.
    """
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(
            json.dumps(dict(sample), ensure_ascii=False, sort_keys=True) + "\n"
            for sample in samples
        ),
        encoding="utf-8",
    )
    return destination
