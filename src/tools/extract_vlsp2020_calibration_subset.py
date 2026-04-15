from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence


DEFAULT_OUTPUT_DIR = Path('build') / 'calibration' / 'vlsp2020'
DEFAULT_MAX_SAMPLES = 24


@dataclass(frozen=True)
class VlspCalibrationRow:
    source_shard: str
    row_index: int
    audio_file_name: str
    audio_bytes: bytes
    transcription: str


def _load_pyarrow_parquet():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError('VLSP subset extraction requires pyarrow to read parquet shards.') from exc
    return pq


def iter_vlsp_parquet_rows(dataset_root: str | Path, batch_size: int = 32) -> Iterator[VlspCalibrationRow]:
    pq = _load_pyarrow_parquet()
    root = Path(dataset_root)
    shard_paths = sorted(root.glob('*.parquet'))
    if not shard_paths:
        raise FileNotFoundError(f'No parquet shards found under: {root}')

    for shard_path in shard_paths:
        parquet_file = pq.ParquetFile(shard_path)
        row_index = 0
        for batch in parquet_file.iter_batches(columns=['audio', 'transcription'], batch_size=batch_size):
            table = batch.to_pylist()
            for row in table:
                audio = row.get('audio') or {}
                audio_bytes = audio.get('bytes')
                audio_file_name = audio.get('path')
                transcription = row.get('transcription')
                if not audio_bytes or not audio_file_name or transcription is None:
                    row_index += 1
                    continue
                yield VlspCalibrationRow(
                    source_shard=shard_path.name,
                    row_index=row_index,
                    audio_file_name=str(audio_file_name),
                    audio_bytes=bytes(audio_bytes),
                    transcription=str(transcription),
                )
                row_index += 1


def normalize_transcription(text: str) -> str:
    return text.strip()


def select_subset_rows(rows: Iterable[VlspCalibrationRow], max_samples: int) -> list[VlspCalibrationRow]:
    if max_samples < 1:
        raise ValueError('max_samples must be >= 1')
    selected: list[VlspCalibrationRow] = []
    for row in rows:
        normalized = normalize_transcription(row.transcription)
        if not normalized:
            continue
        selected.append(
            VlspCalibrationRow(
                source_shard=row.source_shard,
                row_index=row.row_index,
                audio_file_name=row.audio_file_name,
                audio_bytes=row.audio_bytes,
                transcription=normalized,
            )
        )
        if len(selected) >= max_samples:
            break
    return selected


def _materialized_audio_name(index: int, original_name: str) -> str:
    return f'{index:06d}__{Path(original_name).name}'


def write_subset_outputs(rows: Sequence[VlspCalibrationRow], output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    audio_dir = root / 'audio'
    root.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    audio_manifest_lines: list[str] = []
    transcription_lines: list[str] = []
    sample_entries: list[dict] = []

    for index, row in enumerate(rows, start=1):
        normalized_transcription = normalize_transcription(row.transcription)
        output_audio_path = (audio_dir / _materialized_audio_name(index, row.audio_file_name)).resolve()
        output_audio_path.write_bytes(row.audio_bytes)
        audio_manifest_lines.append(str(output_audio_path))
        transcription_lines.append(normalized_transcription)
        sample_entries.append(
            {
                'sample_index': index,
                'source_shard': row.source_shard,
                'row_index': row.row_index,
                'audio_file_name': row.audio_file_name,
                'materialized_audio_path': str(output_audio_path),
                'transcription': normalized_transcription,
            }
        )

    zipformer_audio_manifest = root / 'zipformer_audio_manifest.txt'
    vpcd_transcriptions = root / 'vpcd_transcriptions.txt'
    subset_manifest = root / 'subset_manifest.json'

    zipformer_audio_manifest.write_text('\n'.join(audio_manifest_lines) + '\n', encoding='utf-8')
    vpcd_transcriptions.write_text('\n'.join(transcription_lines) + '\n', encoding='utf-8')
    subset_manifest.write_text(
        json.dumps(
            {
                'sample_count': len(sample_entries),
                'audio_dir': str(audio_dir.resolve()),
                'samples': sample_entries,
            },
            ensure_ascii=False,
            indent=2,
        )
        + '\n',
        encoding='utf-8',
    )
    return {
        'zipformer_audio_manifest': zipformer_audio_manifest,
        'vpcd_transcriptions': vpcd_transcriptions,
        'subset_manifest': subset_manifest,
    }


def extract_subset(*, dataset_root: str | Path, output_dir: str | Path = DEFAULT_OUTPUT_DIR, max_samples: int = DEFAULT_MAX_SAMPLES) -> dict[str, Path]:
    rows = select_subset_rows(iter_vlsp_parquet_rows(dataset_root), max_samples=max_samples)
    if not rows:
        raise ValueError(f'No usable VLSP rows found under: {dataset_root}')
    outputs = write_subset_outputs(rows, output_dir)
    subset_manifest_path = outputs['subset_manifest']
    subset_manifest_data = json.loads(subset_manifest_path.read_text(encoding='utf-8'))
    subset_manifest_data['dataset_root'] = str(Path(dataset_root).resolve())
    subset_manifest_path.write_text(json.dumps(subset_manifest_data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return outputs


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Extract a deterministic calibration subset from VLSP 2020 parquet shards.')
    parser.add_argument('--dataset-root', required=True, help='Directory containing VLSP parquet shards.')
    parser.add_argument('--output-dir', default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument('--max-samples', type=int, default=DEFAULT_MAX_SAMPLES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    outputs = extract_subset(dataset_root=args.dataset_root, output_dir=args.output_dir, max_samples=args.max_samples)
    print(f'Dataset root: {Path(args.dataset_root).resolve()}')
    print(f'Max samples : {args.max_samples}')
    print(f"Zipformer manifest: {outputs['zipformer_audio_manifest']}")
    print(f"VPCD texts       : {outputs['vpcd_transcriptions']}")
    print(f"Subset manifest  : {outputs['subset_manifest']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
