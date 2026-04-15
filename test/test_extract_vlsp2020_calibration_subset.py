import json
import shutil
import uuid
from pathlib import Path


TEST_TMP_ROOT = Path(__file__).resolve().parent / '_tmp' / 'vlsp_calibration_subset'


def tmp_case_dir():
    path = TEST_TMP_ROOT / uuid.uuid4().hex
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_select_subset_rows_keeps_source_order():
    from tools.extract_vlsp2020_calibration_subset import VlspCalibrationRow, select_subset_rows

    rows = [
        VlspCalibrationRow(
            source_shard='train-00000-of-00035.parquet',
            row_index=0,
            audio_file_name='a.wav',
            audio_bytes=b'audio-a',
            transcription='alpha',
        ),
        VlspCalibrationRow(
            source_shard='train-00000-of-00035.parquet',
            row_index=1,
            audio_file_name='b.wav',
            audio_bytes=b'audio-b',
            transcription='beta',
        ),
        VlspCalibrationRow(
            source_shard='train-00001-of-00035.parquet',
            row_index=0,
            audio_file_name='c.wav',
            audio_bytes=b'audio-c',
            transcription='gamma',
        ),
    ]

    selected = select_subset_rows(rows, max_samples=2)

    assert [row.audio_file_name for row in selected] == ['a.wav', 'b.wav']


def test_write_subset_outputs_emits_audio_manifest_and_text_file():
    from tools.extract_vlsp2020_calibration_subset import VlspCalibrationRow, write_subset_outputs

    case_dir = tmp_case_dir()

    rows = [
        VlspCalibrationRow(
            source_shard='train-00000-of-00035.parquet',
            row_index=0,
            audio_file_name='a.wav',
            audio_bytes=b'audio-a',
            transcription='  xin chao  ',
        ),
        VlspCalibrationRow(
            source_shard='train-00000-of-00035.parquet',
            row_index=1,
            audio_file_name='b.wav',
            audio_bytes=b'audio-b',
            transcription='hom nay troi dep',
        ),
    ]

    outputs = write_subset_outputs(rows, output_dir=case_dir)

    zipformer_manifest_path = case_dir / 'zipformer_audio_manifest.txt'
    vpcd_transcriptions_path = case_dir / 'vpcd_transcriptions.txt'
    subset_manifest_path = case_dir / 'subset_manifest.json'

    assert outputs['zipformer_audio_manifest'] == zipformer_manifest_path
    assert outputs['vpcd_transcriptions'] == vpcd_transcriptions_path
    assert outputs['subset_manifest'] == subset_manifest_path

    audio_manifest_lines = zipformer_manifest_path.read_text(encoding='utf-8').splitlines()
    assert len(audio_manifest_lines) == 2
    assert audio_manifest_lines[0].endswith('000001__a.wav')
    assert audio_manifest_lines[1].endswith('000002__b.wav')

    assert vpcd_transcriptions_path.read_text(encoding='utf-8').splitlines() == [
        'xin chao',
        'hom nay troi dep',
    ]

    subset_manifest = json.loads(subset_manifest_path.read_text(encoding='utf-8'))
    assert subset_manifest['sample_count'] == 2
    assert subset_manifest['samples'][0]['audio_file_name'] == 'a.wav'
    assert subset_manifest['samples'][1]['audio_file_name'] == 'b.wav'
