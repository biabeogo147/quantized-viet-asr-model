from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Sequence

from model_pipeline.datasets.records import AudioSampleFixture
from model_pipeline.datasets.selection import AudioInfo, SelectedFixture
from model_pipeline.datasets.vlsp import VlspRow


def probe_audio(row: VlspRow) -> AudioInfo:
    """Decode one dataset row and report duration and sample rate.

    Args:
        row: VLSP row containing encoded audio bytes.

    Returns:
        Audio duration in seconds and source sample rate.
    """
    waveform, sample_rate = _decode(row)
    return AudioInfo(float(waveform.shape[-1]) / float(sample_rate), int(sample_rate))


def materialize_audio(
    selected: Sequence[SelectedFixture],
    *,
    python_speech_dir: str | Path,
    android_speech_dir: str | Path | None = None,
) -> list[AudioSampleFixture]:
    """Export selected rows as synchronized mono WAV fixture assets.

    Args:
        selected: Deterministically selected source fixtures.
        python_speech_dir: Python repository speech-asset destination.
        android_speech_dir: Optional mirrored Android speech-asset destination.

    Returns:
        Fixture inventory including the stable default MP3 sample.
    """
    destinations = [Path(python_speech_dir)]
    if android_speech_dir is not None:
        destinations.append(Path(android_speech_dir))
    fixtures = [AudioSampleFixture("sample-1", "assets/speech/sample-1.mp3")]
    for item in selected:
        waveform, sample_rate = _decode(item.source)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != item.exported_sample_rate:
            import torchaudio

            waveform = torchaudio.functional.resample(waveform, sample_rate, item.exported_sample_rate)
        file_name = f"{item.sample_id}.wav"
        for destination in destinations:
            destination.mkdir(parents=True, exist_ok=True)
            import torchaudio

            torchaudio.save(
                str(destination / file_name), waveform.contiguous(), item.exported_sample_rate, format="wav"
            )
        fixtures.append(AudioSampleFixture(item.sample_id, f"assets/speech/{file_name}"))
    return fixtures


def _decode(row: VlspRow):
    """Decode in-memory dataset audio through a suffix-preserving temporary file.

    Args:
        row: VLSP row containing encoded audio bytes and a source filename.

    Returns:
        Torchaudio waveform tensor and integer sample rate.
    """
    import torchaudio

    suffix = Path(row.audio_file_name).suffix or ".wav"
    with tempfile.TemporaryDirectory(prefix="model-pipeline-audio-") as temporary:
        source = Path(temporary) / f"source{suffix}"
        source.write_bytes(row.audio_bytes)
        return torchaudio.load(str(source))
