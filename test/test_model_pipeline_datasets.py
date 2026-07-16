from __future__ import annotations

import json
from pathlib import Path

from model_pipeline.datasets import (
    AudioInfo,
    AudioExpectedOutput,
    AudioSampleFixture,
    TextGoldenSample,
    VlspRow,
    read_jsonl,
    select_calibration_rows,
    select_vlsp_calibration_evaluation,
    serialize_jsonl,
    write_calibration_subset,
    write_vlsp_calibration_evaluation,
)


def test_fixture_records_round_trip_utf8() -> None:
    """Verify fixture records round-trip readable Vietnamese text through JSONL.

    Returns:
        None.
    """
    rows = [
        TextGoldenSample("xin chào", [1, 2], "Xin chào.", "sample-1"),
        AudioSampleFixture("sample-1", "assets/speech/sample-1.wav"),
        AudioExpectedOutput("sample-1", "assets/speech/sample-1.wav", "xin chào"),
    ]

    content = serialize_jsonl(rows)

    assert "xin chào" in content
    assert TextGoldenSample.from_dict(json.loads(content.splitlines()[0])) == rows[0]


def test_vlsp_subset_is_deterministic_and_manifest_paths_are_relative(tmp_path: Path) -> None:
    """Verify VLSP subset outputs are deterministic and repository portable.

    Args:
        tmp_path: Isolated calibration-subset output directory.

    Returns:
        None.
    """
    rows = [
        VlspRow("b.parquet", 2, "b.wav", b"b", "  câu thứ hai  "),
        VlspRow("a.parquet", 1, "a.wav", b"a", ""),
        VlspRow("a.parquet", 3, "c.wav", b"c", "câu thứ ba"),
    ]
    selected = select_calibration_rows(rows, max_samples=2)

    outputs = write_calibration_subset(selected, tmp_path)
    payload = json.loads(outputs["manifest"].read_text(encoding="utf-8"))

    assert [row.transcription for row in selected] == ["câu thứ hai", "câu thứ ba"]
    assert payload["sample_count"] == 2
    assert payload["samples"][0]["audio_path"] == "audio/000001__b.wav"
    assert not Path(payload["samples"][0]["audio_path"]).is_absolute()
    assert read_jsonl(outputs["fixtures"])[0]["sample_id"] == "sample-1"


def test_vlsp_calibration_and_evaluation_are_deterministic_and_disjoint() -> None:
    """Verify calibration and evaluation use disjoint shards, rows, and texts.

    Returns:
        None.
    """
    rows = [
        VlspRow("shard-00.parquet", 0, "cal-0.wav", b"cal-0", "mot hai ba bon"),
        VlspRow("shard-00.parquet", 1, "cal-1.wav", b"cal-1", "nam sau bay tam"),
        VlspRow("shard-01.parquet", 0, "eval-0.wav", b"eval-0", "chin muoi muoi mot muoi hai"),
        VlspRow("shard-01.parquet", 1, "short.wav", b"short", "muoi ba muoi bon muoi lam"),
        VlspRow("shard-02.parquet", 0, "eval-1.wav", b"eval-1", "muoi sau muoi bay muoi tam"),
        VlspRow("shard-02.parquet", 1, "eval-2.wav", b"eval-2", "muoi chin hai muoi hai mot"),
    ]
    durations = {b"short": 1.5, b"eval-0": 2.0, b"eval-1": 7.0, b"eval-2": 12.0}

    split = select_vlsp_calibration_evaluation(
        rows,
        calibration_count=2,
        evaluation_count=3,
        probe=lambda row: AudioInfo(durations[row.audio_bytes], 16_000),
    )

    assert [row.row_index for row in split.calibration] == [0, 1]
    assert [row.audio_file_name for row in split.evaluation] == [
        "eval-0.wav",
        "eval-1.wav",
        "eval-2.wav",
    ]
    assert {row.source_shard for row in split.calibration}.isdisjoint(
        row.source_shard for row in split.evaluation
    )
    assert {row.transcription.casefold() for row in split.calibration}.isdisjoint(
        row.transcription.casefold() for row in split.evaluation
    )


def test_vlsp_split_manifest_uses_relative_paths_and_checksums(tmp_path: Path) -> None:
    """Verify split materialization records portable paths and content checksums.

    Args:
        tmp_path: Isolated split output directory.

    Returns:
        None.
    """
    rows = [
        VlspRow("shard-00.parquet", 7, "cal.wav", b"calibration-audio", "mot hai ba bon"),
        VlspRow("shard-01.parquet", 9, "eval.wav", b"evaluation-audio", "nam sau bay tam"),
    ]
    split = select_vlsp_calibration_evaluation(
        rows,
        calibration_count=1,
        evaluation_count=1,
        probe=lambda _row: AudioInfo(4.0, 16_000),
    )

    outputs = write_vlsp_calibration_evaluation(split, tmp_path)
    payload = json.loads(outputs["manifest"].read_text(encoding="utf-8"))

    assert payload["calibration_count"] == 1
    assert payload["evaluation_count"] == 1
    assert payload["samples"][0]["audio_path"] == "calibration/audio/000001__cal.wav"
    assert payload["samples"][1]["audio_path"] == "evaluation/audio/000001__eval.wav"
    assert all(not Path(sample["audio_path"]).is_absolute() for sample in payload["samples"])
    assert all(len(sample["audio_sha256"]) == 64 for sample in payload["samples"])
    assert all(len(sample["text_sha256"]) == 64 for sample in payload["samples"])
    assert outputs["calibration_transcriptions"].read_text(encoding="utf-8").splitlines() == [
        "mot hai ba bon"
    ]


def test_vlsp_selection_stops_stream_after_required_evaluation_rows() -> None:
    """Verify selection does not retain or decode the complete VLSP corpus.

    Returns:
        None.
    """
    def rows():
        """Yield enough records and fail if selection over-consumes the stream.

        Yields:
            Ordered calibration and evaluation rows.
        """
        yield VlspRow("shard-00.parquet", 0, "cal.wav", b"cal", "mot hai ba bon")
        yield VlspRow("shard-01.parquet", 0, "eval.wav", b"eval", "nam sau bay tam")
        raise AssertionError("selection consumed rows after satisfying both partitions")

    split = select_vlsp_calibration_evaluation(
        rows(),
        calibration_count=1,
        evaluation_count=1,
        probe=lambda _row: AudioInfo(4.0, 16_000),
    )

    assert len(split.calibration) == 1
    assert len(split.evaluation) == 1
