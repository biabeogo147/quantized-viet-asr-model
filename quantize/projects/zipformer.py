from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from onnxruntime.quantization import CalibrationMethod

from model_bundle.fixtures import AudioExpectedOutput, AudioSampleFixture, read_jsonl
from model_bundle.projects.zipformer import DEFAULT_AUDIO_FIXTURES, ModelDirAcousticRuntime, export_bundle
from quantize.evaluate import evaluate_candidate_bundle
from quantize.fixed_shapes import freeze_model_inputs
from quantize.model_introspection import load_model_node_names
from quantize.qnn import run_qnn_static_quantization
from quantize.reports import ComponentQuantizationReport, QuantizationReport
from quantize.runner import file_size_mb
from quantize.types import CalibrationSample, QuantizationPlan

NAME = 'zipformer'
DEFAULT_MODEL_DIR = Path('assets') / 'zipformer'
DEFAULT_OUTPUT_ROOT = Path('build') / 'quantize' / 'zipformer' / 'qnn_u16u8'
DEFAULT_BUNDLE_OUTPUT_DIR = Path('build') / 'model_bundle' / 'zipformer' / 'qnn_u16u8'
DEFAULT_REFERENCE_BUNDLE_DIR = Path('build') / 'model_bundle' / 'zipformer' / 'fp32'
DEFAULT_PRESET = 'zipformer_sd8g2_qnn_u16u8'
SUPPORTED_PRESETS = (DEFAULT_PRESET,)


def apply_default_arguments(parser) -> None:
    parser.add_argument('--model-dir', default=str(DEFAULT_MODEL_DIR))
    parser.add_argument('--preset', default=DEFAULT_PRESET)
    parser.add_argument('--output-root', default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument('--bundle-output-dir', default=str(DEFAULT_BUNDLE_OUTPUT_DIR))
    parser.add_argument('--reference-bundle-dir', default=str(DEFAULT_REFERENCE_BUNDLE_DIR))
    parser.add_argument('--provider', default='CPUExecutionProvider')
    parser.add_argument('--calibration-chunk-size', type=int, default=16)
    parser.add_argument('--audio-manifest', help='Optional text file with one audio path per line for calibration and evaluation.')
    parser.add_argument('--dry-run', action='store_true')


def validate_args(args) -> None:
    if args.preset not in SUPPORTED_PRESETS:
        raise ValueError(f'Unsupported zipformer preset: {args.preset}')


def _load_audio_fixtures(manifest_path: str | None) -> list[AudioSampleFixture]:
    if not manifest_path:
        return list(DEFAULT_AUDIO_FIXTURES)
    path = Path(manifest_path)
    fixtures: list[AudioSampleFixture] = []
    for index, line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
        audio_path = line.strip()
        if audio_path:
            fixtures.append(AudioSampleFixture(sample_id=f'audio-{index}', audio_path=audio_path))
    return fixtures


def _pad_encoder_records(records: Sequence[CalibrationSample], max_t: int, feature_dim: int) -> list[CalibrationSample]:
    padded: list[CalibrationSample] = []
    for record in records:
        x = record.inputs['x']
        x_lens = record.inputs['x_lens']
        tensor = np.zeros((1, max_t, feature_dim), dtype=np.float32)
        tensor[:, : x.shape[1], :] = x
        padded.append(CalibrationSample(inputs={'x': tensor, 'x_lens': x_lens.astype(np.int64, copy=False)}))
    return padded


def _collect_component_records(runtime: ModelDirAcousticRuntime, fixtures: Sequence[AudioSampleFixture]) -> tuple[dict[str, list[CalibrationSample]], dict[str, int]]:
    workspace_root = Path(__file__).resolve().parents[2]
    encoder_records: list[CalibrationSample] = []
    decoder_records: list[CalibrationSample] = []
    joiner_records: list[CalibrationSample] = []
    max_encoder_t = 0
    trace_records = 0

    for fixture in fixtures:
        audio_path = workspace_root / fixture.audio_path
        features = runtime._load_features(audio_path, sample_rate=runtime.sample_rate, feature_dim=runtime.feature_dim)
        x = features[None, ...].astype(np.float32)
        x_lens = np.array([features.shape[0]], dtype=np.int64)
        max_encoder_t = max(max_encoder_t, int(features.shape[0]))
        encoder_records.append(CalibrationSample(inputs={'x': x, 'x_lens': x_lens}))

        encoder_outputs = runtime.encoder_sess.run(None, {'x': x, 'x_lens': x_lens})
        encoder_out = encoder_outputs[0]
        encoder_out_lens = encoder_outputs[1] if len(encoder_outputs) > 1 else None
        frames = encoder_out[0].astype(np.float32)
        if encoder_out_lens is not None:
            frames = frames[: int(encoder_out_lens.reshape(-1)[0])]
        history = [runtime.blank_id] * runtime.context_size

        for frame in frames:
            enc_frame = frame.reshape(1, -1).astype(np.float32)
            while True:
                dec_in = np.asarray([history[-runtime.context_size :]], dtype=np.int64)
                decoder_records.append(CalibrationSample(inputs={'y': dec_in}))
                dec_out = runtime.decoder_sess.run(None, {'y': dec_in})[0].astype(np.float32)
                joiner_records.append(CalibrationSample(inputs={'encoder_out': enc_frame, 'decoder_out': dec_out}))
                trace_records += 1
                join_out = runtime.joiner_sess.run(None, {'encoder_out': enc_frame, 'decoder_out': dec_out})[0]
                token = int(np.argmax(join_out, axis=-1)[0])
                if token == runtime.blank_id:
                    break
                history.append(token)

    encoder_records = _pad_encoder_records(encoder_records, max_encoder_t, runtime.feature_dim)
    stats = {
        'sample_count': len(fixtures),
        'trace_records': trace_records,
        'max_encoder_t': max_encoder_t,
        'feature_dim': runtime.feature_dim,
        'context_size': runtime.context_size,
        'joiner_dim': int(joiner_records[0].inputs['encoder_out'].shape[-1]) if joiner_records else 512,
    }
    return {
        'encoder': encoder_records,
        'decoder': decoder_records,
        'joiner': joiner_records,
    }, stats


def _fixed_shape_paths(model_dir: Path, output_root: Path, stats: dict[str, int]) -> dict[str, Path]:
    fixed_dir = output_root / 'fixed_shapes'
    fixed_dir.mkdir(parents=True, exist_ok=True)
    return {
        'encoder': freeze_model_inputs(model_dir / 'encoder-epoch-20-avg-1.onnx', fixed_dir / 'encoder.fixed.onnx', {'x': (1, stats['max_encoder_t'], stats['feature_dim']), 'x_lens': (1,)}),
        'decoder': freeze_model_inputs(model_dir / 'decoder-epoch-20-avg-1.onnx', fixed_dir / 'decoder.fixed.onnx', {'y': (1, stats['context_size'])}),
        'joiner': freeze_model_inputs(model_dir / 'joiner-epoch-20-avg-1.onnx', fixed_dir / 'joiner.fixed.onnx', {'encoder_out': (1, stats['joiner_dim']), 'decoder_out': (1, stats['joiner_dim'])}),
    }


def _fixed_input_shapes(stats: dict[str, int]) -> dict[str, dict[str, list[int]]]:
    return {
        'encoder': {
            'x': [1, int(stats['max_encoder_t']), int(stats['feature_dim'])],
            'x_lens': [1],
        },
        'decoder': {
            'y': [1, int(stats['context_size'])],
        },
        'joiner': {
            'encoder_out': [1, int(stats['joiner_dim'])],
            'decoder_out': [1, int(stats['joiner_dim'])],
        },
    }


def _build_component_plan(component_model: Path, preset: str) -> QuantizationPlan:
    _ = load_model_node_names(component_model)
    return QuantizationPlan(
        preset=preset,
        runner_kind='qnn_static',
        op_types_to_quantize=('MatMul',),
        exclusion_patterns=(),
        nodes_to_exclude=(),
        calibration_method='minmax',
        percentile=99.99,
        per_channel=False,
        activation_type='quint16',
        weight_type='quint8',
    )


def _load_reference_expected_outputs(reference_bundle_dir: Path) -> list[AudioExpectedOutput]:
    manifest_rows = read_jsonl(reference_bundle_dir / 'expected_outputs.jsonl')
    return [AudioExpectedOutput.from_dict(row) for row in manifest_rows]


def run(args) -> int:
    validate_args(args)
    model_dir = Path(args.model_dir)
    output_root = Path(args.output_root)
    bundle_output_dir = Path(args.bundle_output_dir)
    reference_bundle_dir = Path(args.reference_bundle_dir)
    fixtures = _load_audio_fixtures(args.audio_manifest)

    if args.dry_run:
        print(f'Project: {NAME}')
        print(f'Preset: {args.preset}')
        print(f'Model dir: {model_dir}')
        print(f'Output root: {output_root}')
        print(f'Bundle output dir: {bundle_output_dir}')
        print(f'Reference bundle dir: {reference_bundle_dir}')
        print(f'Calibration samples: {len(fixtures)}')
        return 0

    runtime = ModelDirAcousticRuntime(model_dir=model_dir, provider=args.provider)
    records, stats = _collect_component_records(runtime, fixtures)
    fixed_paths = _fixed_shape_paths(model_dir, output_root, stats)
    fixed_input_shapes = _fixed_input_shapes(stats)
    quantized_dir = output_root / 'quantized'
    quantized_dir.mkdir(parents=True, exist_ok=True)

    component_reports: list[ComponentQuantizationReport] = []
    quantized_paths: dict[str, Path] = {}
    for component in ('encoder', 'decoder', 'joiner'):
        fixed_model_path = fixed_paths[component]
        output_path = quantized_dir / f'{component}.onnx'
        plan = _build_component_plan(fixed_model_path, args.preset)
        run_qnn_static_quantization(
            fp32_onnx_path=fixed_model_path,
            output_path=output_path,
            plan=plan,
            records=records[component],
            calibration_method=CalibrationMethod.MinMax,
            calibration_chunk_size=args.calibration_chunk_size,
        )
        quantized_paths[component] = output_path
        component_reports.append(
            ComponentQuantizationReport(
                component=component,
                input_model=str(fixed_model_path),
                output_model=str(output_path),
                size_mb=file_size_mb(output_path),
                calibration_records=len(records[component]),
            )
        )

    if not (reference_bundle_dir / 'bundle_manifest.json').exists():
        export_bundle(model_dir=model_dir, output_dir=reference_bundle_dir, sample_fixtures=fixtures, provider=args.provider)

    expected_outputs = _load_reference_expected_outputs(reference_bundle_dir)
    export_bundle(
        model_dir=model_dir,
        output_dir=bundle_output_dir,
        asset_namespace='models/asr/zipformer/qnn_u16u8',
        sample_fixtures=list(fixtures),
        expected_outputs=expected_outputs,
        component_paths={
            'encoder': quantized_paths['encoder'],
            'decoder': quantized_paths['decoder'],
            'joiner': quantized_paths['joiner'],
            'tokens': model_dir / 'tokens.txt',
        },
        model_variant='qnn_u16u8',
        extra_metadata={
            'fixed_input_shapes': fixed_input_shapes,
            'fixed_encoder_frames': int(stats['max_encoder_t']),
            'quantization': {
                'format': 'QDQ',
                'activation_type': 'quint16',
                'weight_type': 'quint8',
                'preset': args.preset,
                'fixed_shapes': True,
            },
        },
    )

    evaluation = evaluate_candidate_bundle(project='zipformer', reference_bundle=reference_bundle_dir, candidate_bundle=bundle_output_dir, provider=args.provider)
    report = QuantizationReport(
        project=NAME,
        preset=args.preset,
        output_root=str(output_root),
        bundle_output_dir=str(bundle_output_dir),
        sample_count=stats['sample_count'],
        trace_records=stats['trace_records'],
        components=component_reports,
        evaluation=evaluation,
    )
    report.write_json(bundle_output_dir / 'quantization_report.json')
    (bundle_output_dir / 'evaluation_report.json').write_text(json.dumps(evaluation, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    print(f'Project: {NAME}')
    print(f'Preset: {args.preset}')
    print(f'Output root: {output_root}')
    print(f'Candidate bundle: {bundle_output_dir}')
    print(f"Samples: {stats['sample_count']}")
    print(f"Trace records: {stats['trace_records']}")
    print(f"Candidate exact-match pass: {evaluation['passed']}")
    return 0



