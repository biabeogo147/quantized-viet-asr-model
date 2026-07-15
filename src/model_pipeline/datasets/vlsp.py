from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from model_pipeline.datasets.records import AudioSampleFixture, serialize_jsonl


@dataclass(frozen=True)
class VlspRow:
    source_shard: str
    row_index: int
    audio_file_name: str
    audio_bytes: bytes
    transcription: str


def iter_vlsp_rows(dataset_root: str | Path, *, batch_size: int = 32) -> Iterator[VlspRow]:
    """Stream usable audio/transcription rows from VLSP parquet shards.

    Args:
        dataset_root: Directory containing parquet shards.
        batch_size: Number of parquet rows decoded per batch.

    Yields:
        Traceable VLSP rows with in-memory audio bytes.

    Raises:
        RuntimeError: If the optional dataset dependency is unavailable.
        FileNotFoundError: If no parquet shards exist below the root.
    """
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("VLSP extraction requires the 'datasets' dependency extra") from exc
    root = Path(dataset_root)
    shards = sorted(root.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No parquet shards found under: {root}")
    for shard in shards:
        row_index = 0
        for batch in parquet.ParquetFile(shard).iter_batches(
            columns=["audio", "transcription"], batch_size=batch_size
        ):
            for raw in batch.to_pylist():
                audio = raw.get("audio") or {}
                if audio.get("bytes") and audio.get("path") and raw.get("transcription") is not None:
                    yield VlspRow(
                        source_shard=shard.name,
                        row_index=row_index,
                        audio_file_name=str(audio["path"]),
                        audio_bytes=bytes(audio["bytes"]),
                        transcription=str(raw["transcription"]),
                    )
                row_index += 1


def select_calibration_rows(rows: Iterable[VlspRow], *, max_samples: int) -> list[VlspRow]:
    """Select deterministic unique non-empty calibration transcriptions.

    Args:
        rows: Candidate VLSP rows in deterministic order.
        max_samples: Maximum number of unique rows to select.

    Returns:
        Selected rows with normalized whitespace.

    Raises:
        ValueError: If the requested maximum is less than one.
    """
    if max_samples < 1:
        raise ValueError("max_samples must be >= 1")
    selected: list[VlspRow] = []
    seen: set[str] = set()
    for row in rows:
        transcription = re.sub(r"\s+", " ", row.transcription.strip())
        key = transcription.casefold()
        if not transcription or key in seen:
            continue
        selected.append(replace(row, transcription=transcription))
        seen.add(key)
        if len(selected) == max_samples:
            break
    return selected


def write_calibration_subset(rows: Sequence[VlspRow], output_dir: str | Path) -> dict[str, Path]:
    """Materialize calibration audio, fixtures, transcripts, and provenance.

    Args:
        rows: Selected traceable VLSP rows.
        output_dir: Directory receiving the reproducible subset.

    Returns:
        Paths to subset manifest, fixture JSONL, and transcription text.
    """
    root = Path(output_dir)
    audio_dir = root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    samples: list[dict] = []
    fixtures: list[AudioSampleFixture] = []
    transcriptions: list[str] = []
    for index, row in enumerate(rows, start=1):
        name = f"{index:06d}__{Path(row.audio_file_name).name}"
        relative_audio_path = (Path("audio") / name).as_posix()
        (audio_dir / name).write_bytes(row.audio_bytes)
        sample_id = f"sample-{index}"
        fixtures.append(AudioSampleFixture(sample_id, relative_audio_path))
        transcriptions.append(row.transcription)
        samples.append(
            {
                "sample_id": sample_id,
                "source_shard": row.source_shard,
                "row_index": row.row_index,
                "source_audio_file_name": row.audio_file_name,
                "audio_path": relative_audio_path,
                "transcription": row.transcription,
            }
        )
    manifest = root / "subset-manifest.json"
    manifest.write_text(
        json.dumps({"sample_count": len(samples), "samples": samples}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    fixture_path = root / "fixture-manifest.jsonl"
    fixture_path.write_text(serialize_jsonl(fixtures), encoding="utf-8")
    transcription_path = root / "transcriptions.txt"
    transcription_path.write_text("\n".join(transcriptions) + "\n", encoding="utf-8")
    return {"manifest": manifest, "fixtures": fixture_path, "transcriptions": transcription_path}
