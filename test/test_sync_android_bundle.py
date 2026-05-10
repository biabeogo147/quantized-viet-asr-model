import json
from pathlib import Path

from model_bundle.manifest import ModelBundleManifest
from tools.sync_android_bundle import sync_android_bundle


def _write_manifest_bundle(bundle_dir: Path, *, project: str, variant: str, namespace: str) -> None:
    bundle_dir.mkdir(parents=True)
    manifest = ModelBundleManifest(
        bundle_version=1,
        project=project,
        model_family='zipformer-rnnt' if project == 'zipformer' else 'bartpho-seq2seq',
        model_name=f'{project}/{variant}' if project == 'zipformer' else 'tourmii/vietnamese-punc-cap-denorm-v1',
        model_variant=variant,
        asset_namespace=namespace,
        runtime_kind='rnnt_greedy' if project == 'zipformer' else 'text_seq2seq',
        artifacts={'encoder': 'encoder.onnx'} if project == 'zipformer' else {'model': 'model.mobile.onnx'},
        fixtures={'sample_manifest': 'sample_manifest.jsonl'} if project == 'zipformer' else {'golden_samples': 'golden_samples.jsonl'},
        metadata={'source': 'test'},
    )
    manifest.write_json(bundle_dir / 'bundle_manifest.json')
    for file_name in set(manifest.artifacts.values()) | set(manifest.fixtures.values()) | {'extra_report.json'}:
        (bundle_dir / file_name).write_text(file_name, encoding='utf-8')


def test_sync_zipformer_fp32_bundle_to_bkmeeting_modelassets(tmp_path):
    source_bundle = tmp_path / 'source' / 'zipformer' / 'fp32'
    bkmeeting_root = tmp_path / 'BKMeeting'
    _write_manifest_bundle(
        source_bundle,
        project='zipformer',
        variant='stale-fp32',
        namespace='models/asr/zipformer/stale-fp32',
    )

    result = sync_android_bundle(
        project='zipformer',
        variant='fp32',
        source_bundle=source_bundle,
        bkmeeting_root=bkmeeting_root,
    )

    target_dir = bkmeeting_root / 'modelassets' / 'src' / 'main' / 'assets' / 'models' / 'asr' / 'zipformer' / 'fp32'
    target_manifest = json.loads((target_dir / 'bundle_manifest.json').read_text(encoding='utf-8'))
    assert result.target_dir == target_dir
    assert (target_dir / 'encoder.onnx').exists()
    assert (target_dir / 'extra_report.json').exists()
    assert target_manifest['model_name'] == 'zipformer/fp32'
    assert target_manifest['model_variant'] == 'fp32'
    assert target_manifest['asset_namespace'] == 'models/asr/zipformer/fp32'


def test_sync_vpcd_qnn_fixed_to_bkmeeting_modelassets(tmp_path):
    source_bundle = tmp_path / 'source' / 'vpcd' / 'qnn_fixed_1024x128'
    bkmeeting_root = tmp_path / 'BKMeeting'
    _write_manifest_bundle(
        source_bundle,
        project='vpcd',
        variant='vpcd_balanced_fixed_1024x128',
        namespace='models/punctuation/vpcd',
    )

    result = sync_android_bundle(
        project='vpcd',
        variant='qnn_fixed_1024x128',
        source_bundle=source_bundle,
        bkmeeting_root=bkmeeting_root,
    )

    target_dir = (
        bkmeeting_root
        / 'modelassets'
        / 'src'
        / 'main'
        / 'assets'
        / 'models'
        / 'punctuation'
        / 'vpcd'
        / 'qnn_fixed_1024x128'
    )
    target_manifest = json.loads((target_dir / 'bundle_manifest.json').read_text(encoding='utf-8'))
    assert result.asset_pack == 'modelassets'
    assert result.target_dir == target_dir
    assert target_manifest['asset_namespace'] == 'models/punctuation/vpcd/qnn_fixed_1024x128'
