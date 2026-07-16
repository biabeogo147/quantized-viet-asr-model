from __future__ import annotations

import json
import hashlib
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

from model_pipeline.datasets.records import AudioSampleFixture, serialize_jsonl


@dataclass(frozen=True)
class VlspRow:
    source_shard: str
    row_index: int
    audio_file_name: str
    audio_bytes: bytes
    transcription: str


@dataclass(frozen=True)
class VlspCalibrationEvaluationSplit:
    calibration: tuple[VlspRow, ...]
    evaluation: tuple[VlspRow, ...]


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
        rows: Source VLSP rows in deterministic order.
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


def select_vlsp_calibration_evaluation(
    rows: Iterable[VlspRow],
    *,
    calibration_count: int,
    evaluation_count: int,
    probe: Callable[[VlspRow], object],
) -> VlspCalibrationEvaluationSplit:
    """Select disjoint calibration and held-out evaluation records.

    Args:
        rows: VLSP rows ordered by shard and row index.
        calibration_count: Required records from the first shard.
        evaluation_count: Required filtered records from later shards.
        probe: Callback returning an object with `duration_seconds`.

    Returns:
        Deterministic calibration and evaluation records with normalized text.

    Raises:
        ValueError: If counts are invalid or eligible records are insufficient.
    """
    if calibration_count < 1 or evaluation_count < 1:
        raise ValueError("Calibration and evaluation counts must be positive")
    calibration_shard: str | None = None
    calibration: list[VlspRow] = []
    evaluation: list[VlspRow] = []
    seen_texts: set[str] = set()
    seen_rows: set[tuple[str, int]] = set()

    for row in rows:
        if calibration_shard is None:
            calibration_shard = row.source_shard
        if row.source_shard == calibration_shard:
            if len(calibration) == calibration_count:
                continue
            normalized = _normalize_transcription(row.transcription)
            text_key = normalized.casefold()
            row_key = (row.source_shard, row.row_index)
            if not normalized or text_key in seen_texts or row_key in seen_rows:
                continue
            calibration.append(replace(row, transcription=normalized))
            seen_texts.add(text_key)
            seen_rows.add(row_key)
            continue
        if len(calibration) != calibration_count:
            raise ValueError(
                f"Selected {len(calibration)} calibration records; expected {calibration_count}"
            )
        normalized = _normalize_transcription(row.transcription)
        words = normalized.split()
        text_key = normalized.casefold()
        row_key = (row.source_shard, row.row_index)
        if not 4 <= len(words) <= 40:
            continue
        if not normalized or text_key in seen_texts or row_key in seen_rows:
            continue
        duration_seconds = float(getattr(probe(row), "duration_seconds"))
        if not 2.0 <= duration_seconds <= 12.0:
            continue
        evaluation.append(replace(row, transcription=normalized))
        seen_texts.add(text_key)
        seen_rows.add(row_key)
        if len(evaluation) == evaluation_count:
            break
    if calibration_shard is None:
        raise ValueError("VLSP rows must not be empty")
    if len(calibration) != calibration_count:
        raise ValueError(
            f"Selected {len(calibration)} calibration records; expected {calibration_count}"
        )
    if len(evaluation) != evaluation_count:
        raise ValueError(
            f"Selected {len(evaluation)} evaluation records; expected {evaluation_count}"
        )
    return VlspCalibrationEvaluationSplit(tuple(calibration), tuple(evaluation))


def write_vlsp_calibration_evaluation(
    split: VlspCalibrationEvaluationSplit,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Materialize a portable calibration/evaluation dataset and manifest.

    Args:
        split: Disjoint VLSP calibration and evaluation records.
        output_dir: Directory receiving audio, text, and provenance files.

    Returns:
        Paths to the split manifest and partition transcription files.

    Raises:
        ValueError: If the supplied split violates shard, row, or text disjointness.
    """
    _validate_vlsp_split(split)
    root = Path(output_dir)
    samples: list[dict[str, object]] = []
    transcription_paths: dict[str, Path] = {}
    for partition, rows in (
        ("calibration", split.calibration),
        ("evaluation", split.evaluation),
    ):
        audio_dir = root / partition / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        transcriptions: list[str] = []
        for index, row in enumerate(rows, start=1):
            source_name = Path(row.audio_file_name).name
            file_name = f"{index:06d}__{source_name}"
            relative_audio_path = (Path(partition) / "audio" / file_name).as_posix()
            (audio_dir / file_name).write_bytes(row.audio_bytes)
            transcriptions.append(row.transcription)
            samples.append(
                {
                    "partition": partition,
                    "sample_id": f"{partition}-{index:06d}",
                    "source_shard": row.source_shard,
                    "row_index": row.row_index,
                    "source_audio_file_name": source_name,
                    "audio_path": relative_audio_path,
                    "audio_sha256": hashlib.sha256(row.audio_bytes).hexdigest(),
                    "text_sha256": hashlib.sha256(row.transcription.encode("utf-8")).hexdigest(),
                    "transcription": row.transcription,
                }
            )
        transcription_path = root / partition / "transcriptions.txt"
        transcription_path.write_text("\n".join(transcriptions) + "\n", encoding="utf-8")
        transcription_paths[partition] = transcription_path
    manifest = root / "vlsp-calibration-evaluation-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "calibration_count": len(split.calibration),
                "evaluation_count": len(split.evaluation),
                "samples": samples,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "manifest": manifest,
        "calibration_transcriptions": transcription_paths["calibration"],
        "evaluation_transcriptions": transcription_paths["evaluation"],
    }


def _validate_vlsp_split(split: VlspCalibrationEvaluationSplit) -> None:
    """Validate partition counts and shard, row, and transcription disjointness.

    Args:
        split: Calibration and evaluation records to validate.

    Returns:
        None.

    Raises:
        ValueError: If a partition is empty or overlap is detected.
    """
    if not split.calibration or not split.evaluation:
        raise ValueError("Calibration and evaluation partitions must be non-empty")
    calibration_shards = {row.source_shard for row in split.calibration}
    evaluation_shards = {row.source_shard for row in split.evaluation}
    if calibration_shards & evaluation_shards:
        raise ValueError("Calibration and evaluation shards overlap")
    calibration_rows = {(row.source_shard, row.row_index) for row in split.calibration}
    evaluation_rows = {(row.source_shard, row.row_index) for row in split.evaluation}
    if calibration_rows & evaluation_rows:
        raise ValueError("Calibration and evaluation rows overlap")
    calibration_texts = {row.transcription.casefold() for row in split.calibration}
    evaluation_texts = {row.transcription.casefold() for row in split.evaluation}
    if calibration_texts & evaluation_texts:
        raise ValueError("Calibration and evaluation transcriptions overlap")


def _normalize_transcription(text: str) -> str:
    """Collapse transcription whitespace without altering characters.

    Args:
        text: Raw VLSP transcription.

    Returns:
        Trimmed transcription with internal whitespace collapsed.
    """
    return re.sub(r"\s+", " ", text.strip())


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
