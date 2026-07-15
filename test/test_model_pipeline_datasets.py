from __future__ import annotations

import json
from pathlib import Path

from model_pipeline.datasets import (
    AudioExpectedOutput,
    AudioSampleFixture,
    TextGoldenSample,
    VlspRow,
    read_jsonl,
    select_calibration_rows,
    serialize_jsonl,
    write_calibration_subset,
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
