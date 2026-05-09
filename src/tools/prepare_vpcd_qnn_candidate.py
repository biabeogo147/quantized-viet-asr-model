from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Sequence

from model_bundle.manifest import ModelBundleManifest
from quantize.fixed_shapes import freeze_model_inputs


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Prepare a fixed-shape VPCD QDQ bundle for Android QNN preflight.')
    parser.add_argument('--source-bundle', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--encoder-sequence', type=int, default=1024)
    parser.add_argument('--decoder-sequence', type=int, default=128)
    parser.add_argument('--model-variant')
    parser.add_argument('--overwrite', action='store_true')
    return parser


def _require_qdq_vpcd_source(manifest: ModelBundleManifest) -> None:
    if manifest.project != 'vpcd':
        raise ValueError(f'Source bundle project must be vpcd, got {manifest.project!r}')
    if not manifest.artifacts.get('model'):
        raise ValueError('Source VPCD bundle is missing artifacts.model')

    quantization = manifest.metadata.get('quantization')
    if not isinstance(quantization, dict):
        raise ValueError('Source VPCD bundle is missing metadata.quantization')
    if quantization.get('format') != 'QDQ':
        raise ValueError('Source VPCD quantization format must be QDQ')
    if quantization.get('activation_type') != 'quint16':
        raise ValueError('Source VPCD activation_type must be quint16')
    if quantization.get('weight_type') != 'quint8':
        raise ValueError('Source VPCD weight_type must be quint8')


def _prepare_output_dir(source_dir: Path, output_dir: Path, overwrite: bool) -> None:
    resolved_source = source_dir.resolve()
    resolved_output = output_dir.resolve()
    if resolved_source == resolved_output:
        raise ValueError('Output directory cannot be the same as source bundle directory')
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise ValueError(f'Output directory already exists and is not empty: {output_dir}')
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _copy_bundle_payload(source_dir: Path, output_dir: Path, model_name: str) -> None:
    for source_path in source_dir.iterdir():
        if source_path.name in {'bundle_manifest.json', model_name}:
            continue
        target_path = output_dir / source_path.name
        if source_path.is_dir():
            shutil.copytree(source_path, target_path)
        else:
            shutil.copy2(source_path, target_path)


def _fixed_input_shapes(encoder_sequence: int, decoder_sequence: int) -> dict[str, list[int]]:
    if encoder_sequence <= 0:
        raise ValueError('--encoder-sequence must be positive')
    if decoder_sequence <= 0:
        raise ValueError('--decoder-sequence must be positive')
    return {
        'input_ids': [1, int(encoder_sequence)],
        'attention_mask': [1, int(encoder_sequence)],
        'decoder_input_ids': [1, int(decoder_sequence)],
        'decoder_attention_mask': [1, int(decoder_sequence)],
    }


def prepare_candidate(
    *,
    source_bundle: Path,
    output_dir: Path,
    encoder_sequence: int,
    decoder_sequence: int,
    model_variant: str | None = None,
    overwrite: bool = False,
) -> ModelBundleManifest:
    source_dir = Path(source_bundle)
    output_path = Path(output_dir)
    manifest = ModelBundleManifest.from_path(source_dir / 'bundle_manifest.json')
    _require_qdq_vpcd_source(manifest)
    _prepare_output_dir(source_dir, output_path, overwrite)

    model_name = manifest.artifacts['model']
    fixed_shapes = _fixed_input_shapes(encoder_sequence, decoder_sequence)
    _copy_bundle_payload(source_dir, output_path, model_name)
    freeze_model_inputs(source_dir / model_name, output_path / model_name, fixed_shapes)

    metadata = dict(manifest.metadata)
    quantization = dict(metadata.get('quantization', {}))
    quantization['fixed_shapes'] = True
    metadata['quantization'] = quantization
    qnn_readiness = dict(metadata.get('qnn_readiness', {}))
    qnn_readiness.update(
        {
            'target_backend': 'qnn_htp',
            'model_session_candidate': True,
            'tokenizer_policy': 'cpu_only_first_slice',
            'requires_fixed_shapes': True,
            'fixed_shapes_ready': True,
        }
    )
    qnn_readiness.pop('fixed_shape_blocker', None)
    metadata['qnn_readiness'] = qnn_readiness
    metadata['fixed_input_shapes'] = {'model': fixed_shapes}

    resolved_variant = model_variant or f'{manifest.model_variant}_fixed_{encoder_sequence}x{decoder_sequence}'
    candidate_manifest = ModelBundleManifest(
        bundle_version=manifest.bundle_version,
        project=manifest.project,
        model_family=manifest.model_family,
        model_name=manifest.model_name,
        model_variant=resolved_variant,
        asset_namespace=manifest.asset_namespace,
        runtime_kind=manifest.runtime_kind,
        artifacts=dict(manifest.artifacts),
        fixtures=dict(manifest.fixtures),
        metadata=metadata,
    )
    candidate_manifest.write_json(output_path / 'bundle_manifest.json')
    return candidate_manifest


def main(argv: Sequence[str] | None = None) -> None:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    args = build_argument_parser().parse_args(argv)
    manifest = prepare_candidate(
        source_bundle=Path(args.source_bundle),
        output_dir=Path(args.output_dir),
        encoder_sequence=args.encoder_sequence,
        decoder_sequence=args.decoder_sequence,
        model_variant=args.model_variant,
        overwrite=args.overwrite,
    )
    print('VPCD QNN candidate prepared.')
    print('Output        :', args.output_dir)
    print('Model variant :', manifest.model_variant)
    print('Encoder seq   :', args.encoder_sequence)
    print('Decoder seq   :', args.decoder_sequence)


if __name__ == '__main__':
    main()
