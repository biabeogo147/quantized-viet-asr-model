from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from model_bundle.manifest import ModelBundleManifest
from tools.bundle_paths import resolve_bundle_dir


@dataclass(frozen=True)
class AndroidBundleTarget:
    project: str
    variant: str
    asset_pack: str
    asset_namespace: str
    manifest_model_name: str | None = None
    manifest_model_variant: str | None = None


@dataclass(frozen=True)
class SyncResult:
    source_bundle: Path
    target_dir: Path
    asset_pack: str
    asset_namespace: str
    copied_files: tuple[Path, ...]


_TARGETS: dict[tuple[str, str], AndroidBundleTarget] = {
    ('zipformer', 'fp32'): AndroidBundleTarget(
        project='zipformer',
        variant='fp32',
        asset_pack='modelassets',
        asset_namespace='models/asr/zipformer/fp32',
        manifest_model_name='zipformer/fp32',
        manifest_model_variant='fp32',
    ),
    ('zipformer', 'qnn_u16u8'): AndroidBundleTarget(
        project='zipformer',
        variant='qnn_u16u8',
        asset_pack='modelassets',
        asset_namespace='models/asr/zipformer/qnn_u16u8',
        manifest_model_name='zipformer/qnn_u16u8',
        manifest_model_variant='qnn_u16u8',
    ),
    ('vpcd', 'vpcd_balanced'): AndroidBundleTarget(
        project='vpcd',
        variant='vpcd_balanced',
        asset_pack='modelassets',
        asset_namespace='models/punctuation/vpcd/vpcd_balanced',
        manifest_model_variant='vpcd_balanced',
    ),
    ('vpcd', 'qnn_fixed_1024x128'): AndroidBundleTarget(
        project='vpcd',
        variant='qnn_fixed_1024x128',
        asset_pack='modelassets',
        asset_namespace='models/punctuation/vpcd/qnn_fixed_1024x128',
    ),
}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Sync a Python model bundle into BKMeeting Android asset packs.')
    parser.add_argument('--project', required=True, choices=sorted({project for project, _ in _TARGETS}))
    parser.add_argument('--variant', required=True)
    parser.add_argument('--source-bundle')
    parser.add_argument('--bkmeeting-root', default='../BKMeeting')
    parser.add_argument('--overwrite', action='store_true')
    return parser


def _target_for(project: str, variant: str) -> AndroidBundleTarget:
    key = (project, variant)
    if key not in _TARGETS:
        supported = ', '.join(f'{item_project}/{item_variant}' for item_project, item_variant in sorted(_TARGETS))
        raise ValueError(f'Unsupported Android bundle target {project}/{variant}. Supported targets: {supported}')
    return _TARGETS[key]


def _default_source_bundle(project: str, variant: str) -> Path:
    return resolve_bundle_dir(project, variant)


def _target_dir(bkmeeting_root: Path, target: AndroidBundleTarget) -> Path:
    namespace_path = Path(*target.asset_namespace.split('/'))
    return bkmeeting_root / target.asset_pack / 'src' / 'main' / 'assets' / namespace_path


def _prepare_target_dir(target_dir: Path, *, overwrite: bool, bkmeeting_root: Path) -> None:
    resolved_root = bkmeeting_root.resolve()
    resolved_target = target_dir.resolve()
    if not resolved_target.is_relative_to(resolved_root):
        raise ValueError(f'Refusing to write outside BKMeeting root: {target_dir}')
    if target_dir.exists():
        if any(target_dir.iterdir()):
            if not overwrite:
                raise ValueError(f'Target directory already exists and is not empty: {target_dir}. Pass --overwrite to replace it.')
            shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)


def _copy_bundle_payload(source_bundle: Path, target_dir: Path) -> list[Path]:
    copied_files: list[Path] = []
    for source_path in sorted(source_bundle.iterdir(), key=lambda item: item.name):
        if source_path.name == 'bundle_manifest.json':
            continue
        destination = target_dir / source_path.name
        if source_path.is_dir():
            shutil.copytree(source_path, destination)
            copied_files.extend(path for path in destination.rglob('*') if path.is_file())
        else:
            shutil.copy2(source_path, destination)
            copied_files.append(destination)
    return copied_files


def _rewrite_manifest(source_manifest: ModelBundleManifest, target: AndroidBundleTarget, target_dir: Path) -> ModelBundleManifest:
    manifest = ModelBundleManifest(
        bundle_version=source_manifest.bundle_version,
        project=source_manifest.project,
        model_family=source_manifest.model_family,
        model_name=target.manifest_model_name or source_manifest.model_name,
        model_variant=target.manifest_model_variant or source_manifest.model_variant,
        asset_namespace=target.asset_namespace,
        runtime_kind=source_manifest.runtime_kind,
        artifacts=dict(source_manifest.artifacts),
        fixtures=dict(source_manifest.fixtures),
        metadata=dict(source_manifest.metadata),
    )
    manifest.write_json(target_dir / 'bundle_manifest.json')
    return manifest


def sync_android_bundle(
    *,
    project: str,
    variant: str,
    bkmeeting_root: str | Path,
    source_bundle: str | Path | None = None,
    overwrite: bool = False,
) -> SyncResult:
    target = _target_for(project, variant)
    source_dir = Path(source_bundle) if source_bundle else _default_source_bundle(project, variant)
    if not (source_dir / 'bundle_manifest.json').is_file():
        raise FileNotFoundError(f'Missing source bundle manifest: {source_dir / "bundle_manifest.json"}')

    manifest = ModelBundleManifest.from_path(source_dir / 'bundle_manifest.json')
    if manifest.project != project:
        raise ValueError(f'Source bundle project mismatch: expected {project!r}, got {manifest.project!r}')

    bkmeeting_path = Path(bkmeeting_root)
    destination = _target_dir(bkmeeting_path, target)
    _prepare_target_dir(destination, overwrite=overwrite, bkmeeting_root=bkmeeting_path)
    copied_files = _copy_bundle_payload(source_dir, destination)
    _rewrite_manifest(manifest, target, destination)
    return SyncResult(
        source_bundle=source_dir,
        target_dir=destination,
        asset_pack=target.asset_pack,
        asset_namespace=target.asset_namespace,
        copied_files=tuple(copied_files),
    )


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    args = build_argument_parser().parse_args(argv)
    result = sync_android_bundle(
        project=args.project,
        variant=args.variant,
        source_bundle=args.source_bundle,
        bkmeeting_root=args.bkmeeting_root,
        overwrite=args.overwrite,
    )
    print('Android bundle synced.')
    print('Source      :', result.source_bundle)
    print('Asset pack  :', result.asset_pack)
    print('Namespace   :', result.asset_namespace)
    print('Target      :', result.target_dir)
    print('Copied files:', len(result.copied_files))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
