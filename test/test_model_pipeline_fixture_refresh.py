from __future__ import annotations

from pathlib import Path

from model_pipeline.datasets import AudioSampleFixture, VlspRow
from model_pipeline.datasets.golden import generate_audio_golden, generate_text_golden
from model_pipeline.datasets.selection import AudioInfo, select_fixtures


def test_fixture_selection_applies_wip_quality_rules_deterministically() -> None:
    """Verify fixture selection applies quality, dedupe, and duration rules.

    Returns:
        None.
    """
    rows = [
        VlspRow("a", 0, "plain.wav", b"plain", "this sentence has no vietnamese marks"),
        VlspRow("a", 1, "short.wav", b"short", "xin chào"),
        VlspRow("a", 2, "good.wav", b"good", "hôm nay chúng ta kiểm thử mô hình"),
        VlspRow("a", 3, "duplicate.wav", b"dup", "  HÔM NAY chúng ta kiểm thử mô hình "),
        VlspRow("b", 0, "good-2.wav", b"good-2", "đây là câu tiếng Việt thứ hai"),
    ]
    info = {
        b"short": AudioInfo(duration_seconds=1.0, sample_rate=16_000),
        b"good": AudioInfo(duration_seconds=4.0, sample_rate=48_000),
        b"dup": AudioInfo(duration_seconds=4.0, sample_rate=16_000),
        b"good-2": AudioInfo(duration_seconds=5.0, sample_rate=16_000),
    }

    selected = select_fixtures(rows, selection_count=2, probe=lambda row: info[row.audio_bytes])

    assert [row.sample_id for row in selected] == ["sample-2", "sample-3"]
    assert [row.source.audio_file_name for row in selected] == ["good.wav", "good-2.wav"]
    assert selected[0].exported_sample_rate == 16_000


def test_golden_generation_is_runtime_injected_and_has_no_rollout_names(tmp_path: Path) -> None:
    """Verify golden generation uses injected runtimes and canonical records.

    Args:
        tmp_path: Isolated repository root for fixture path resolution.

    Returns:
        None.
    """
    fixtures = [AudioSampleFixture("sample-1", "assets/speech/sample-1.wav")]

    class Acoustic:
        def transcribe(self, path: Path):
            """Return deterministic fake acoustic output for one fixture.

            Args:
                path: Resolved fixture audio path.

            Returns:
                Mapping containing fake recognized text.
            """
            assert path == tmp_path / "assets/speech/sample-1.wav"
            return {"text": "xin▁chào"}

    class Punctuation:
        def encode(self, text: str):
            """Return deterministic fake token IDs for normalized text.

            Args:
                text: Normalized acoustic output.

            Returns:
                Fake model token IDs.
            """
            assert text == "xin chào"
            return [0, 1, 2]

        def restore(self, text: str):
            """Return deterministic fake punctuation output.

            Args:
                text: Normalized acoustic output.

            Returns:
                Fake restored display text.
            """
            return "Xin chào."

    audio_rows = generate_audio_golden(fixtures, repo_root=tmp_path, runtime=Acoustic())
    text_rows = generate_text_golden(audio_rows, runtime=Punctuation())

    assert audio_rows[0].text == "xin▁chào"
    assert text_rows[0].raw_text == "xin chào"
    assert text_rows[0].input_ids == [0, 1, 2]
    assert text_rows[0].expected_output == "Xin chào."
