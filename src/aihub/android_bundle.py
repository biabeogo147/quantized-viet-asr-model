from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from aihub.session import DEFAULT_TARGET_RUNTIME, AiHubRuntimeConfig, build_runtime_config
from model_bundle.manifest import ModelBundleManifest


@dataclass(frozen=True)
class AndroidBundleLayout:
    project: str
    manifest_model_name: str | None
    manifest_model_variant: str
    asset_namespace: str
    compiled_artifact_key: str
    external_data_artifact_key: str
    component_target_runtimes: Mapping[str, str]


@dataclass(frozen=True)
class ResolvedAndroidBundleInputs:
    layout: AndroidBundleLayout
    deployment_package_dir: Path
    deployment_manifest_path: Path
    deployment_manifest: dict[str, Any]
    source_bundle_manifest_path: Path
    source_bundle_manifest: ModelBundleManifest
    downloaded_artifact_path: Path
    io_contract_path: Path
    run_label: str
    target_model_id: str


@dataclass(frozen=True)
class AndroidBundleResult:
    project: str
    run_label: str
    bundle_dir: Path
    manifest_path: Path
    compiled_model_path: Path
    external_data_path: Path
    io_contract_path: Path


PROJECT_LAYOUTS: dict[str, AndroidBundleLayout] = {
    "zipformer": AndroidBundleLayout(
        project="zipformer",
        manifest_model_name="zipformer/precompiled_qnn_onnx",
        manifest_model_variant="precompiled_qnn_onnx",
        asset_namespace="models/asr/zipformer/precompiled_qnn_onnx",
        compiled_artifact_key="encoder",
        external_data_artifact_key="encoder_external_data",
        component_target_runtimes={
            "encoder": DEFAULT_TARGET_RUNTIME,
            "decoder": "cpu_onnx",
            "joiner": "cpu_onnx",
        },
    ),
    "vpcd": AndroidBundleLayout(
        project="vpcd",
        manifest_model_name=None,
        manifest_model_variant="precompiled_qnn_onnx",
        asset_namespace="models/punctuation/vpcd/precompiled_qnn_onnx",
        compiled_artifact_key="model",
        external_data_artifact_key="model_external_data",
        component_target_runtimes={
            "model": DEFAULT_TARGET_RUNTIME,
            "tokenizer_encode": "cpu_onnx",
            "tokenizer_decode": "cpu_onnx",
        },
    ),
}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize Android-ready bundles from retained AI Hub deployment packages.")
    parser.add_argument("--project", required=True, choices=("zipformer", "vpcd", "all"))
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--deployment-package")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--device-name", default="Samsung Galaxy S24 (Family)")
    parser.add_argument("--qairt-version", default=None)
    parser.add_argument("--output-dir")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def resolve_android_bundle_inputs(
    *,
    deployment_package_dir: str | Path,
) -> ResolvedAndroidBundleInputs:
    package_dir = Path(deployment_package_dir).resolve()
    deployment_manifest_path = package_dir / "deployment_manifest.json"
    if not deployment_manifest_path.is_file():
        raise FileNotFoundError(f"Missing deployment manifest: {deployment_manifest_path}")

    deployment_manifest = _read_json(deployment_manifest_path)
    project = str(deployment_manifest.get("project") or "").strip()
    if project not in PROJECT_LAYOUTS:
        supported = ", ".join(sorted(PROJECT_LAYOUTS))
        raise ValueError(f"Unsupported Android bundle project {project!r}. Supported projects: {supported}")
    layout = PROJECT_LAYOUTS[project]

    source_bundle_manifest_path = _resolve_required_path(
        package_dir=package_dir,
        value=deployment_manifest.get("source_bundle_manifest"),
        label="source_bundle_manifest",
    )
    source_bundle_manifest = ModelBundleManifest.from_path(source_bundle_manifest_path)
    if source_bundle_manifest.project != layout.project:
        raise ValueError(
            f"Source bundle project mismatch: expected {layout.project!r}, got {source_bundle_manifest.project!r}"
        )

    io_contract_path = _resolve_required_path(
        package_dir=package_dir,
        value=deployment_manifest.get("io_contract_path"),
        label="io_contract_path",
    )

    downloaded_artifact_value = deployment_manifest.get("downloaded_artifact")
    downloaded_artifact_path = _resolve_downloaded_artifact_path(
        package_dir=package_dir,
        downloaded_artifact=downloaded_artifact_value,
    )

    run_label = _normalize_optional_string(deployment_manifest.get("run_label")) or package_dir.name
    target_model_id = _normalize_optional_string(deployment_manifest.get("target_model_id")) or "unknown"
    return ResolvedAndroidBundleInputs(
        layout=layout,
        deployment_package_dir=package_dir,
        deployment_manifest_path=deployment_manifest_path,
        deployment_manifest=deployment_manifest,
        source_bundle_manifest_path=source_bundle_manifest_path,
        source_bundle_manifest=source_bundle_manifest,
        downloaded_artifact_path=downloaded_artifact_path,
        io_contract_path=io_contract_path,
        run_label=run_label,
        target_model_id=target_model_id,
    )


def materialize_android_bundle(
    *,
    deployment_package_dir: str | Path,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> AndroidBundleResult:
    resolved = resolve_android_bundle_inputs(deployment_package_dir=deployment_package_dir)
    bundle_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else _default_bundle_dir(resolved.deployment_package_dir, resolved.layout.project, resolved.run_label)
    )
    _prepare_output_dir(bundle_dir, overwrite=overwrite)

    compiled_output_name = resolved.source_bundle_manifest.artifacts[resolved.layout.compiled_artifact_key]
    compiled_model_path = (bundle_dir / compiled_output_name).resolve()
    external_data_path = (bundle_dir / "model.bin").resolve()
    io_contract_output_path = (bundle_dir / "io_contract.json").resolve()

    _extract_compiled_payload(
        zip_path=resolved.downloaded_artifact_path,
        compiled_output_path=compiled_model_path,
        external_data_output_path=external_data_path,
    )
    _copy_source_bundle_companions(
        source_manifest_path=resolved.source_bundle_manifest_path,
        source_manifest=resolved.source_bundle_manifest,
        target_dir=bundle_dir,
        compiled_artifact_key=resolved.layout.compiled_artifact_key,
    )
    shutil.copy2(resolved.io_contract_path, io_contract_output_path)

    manifest = _build_android_bundle_manifest(
        resolved=resolved,
        compiled_output_name=compiled_output_name,
    )
    manifest_path = manifest.write_json(bundle_dir / "bundle_manifest.json").resolve()
    return AndroidBundleResult(
        project=resolved.layout.project,
        run_label=resolved.run_label,
        bundle_dir=bundle_dir,
        manifest_path=manifest_path,
        compiled_model_path=compiled_model_path,
        external_data_path=external_data_path,
        io_contract_path=io_contract_output_path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_argument_parser().parse_args(argv)
    runtime_config = build_runtime_config(
        device_name=args.device_name,
        qairt_version=_normalize_optional_string(args.qairt_version),
        repo_root=args.repo_root,
    )
    projects = list(PROJECT_LAYOUTS) if args.project == "all" else [args.project]

    if args.deployment_package and args.project == "all":
        raise ValueError("--deployment-package cannot be used with --project all")
    if args.output_dir and args.project == "all":
        raise ValueError("--output-dir cannot be used with --project all")

    for project in projects:
        deployment_package_dir = (
            Path(args.deployment_package).resolve()
            if args.deployment_package
            else _default_deployment_package_dir(runtime_config, project, args.run_label)
        )
        if args.dry_run:
            resolved = resolve_android_bundle_inputs(deployment_package_dir=deployment_package_dir)
            bundle_dir = _default_bundle_dir(deployment_package_dir, project, resolved.run_label)
            print(f"[dry-run] project={project}")
            print("  deployment_package:", deployment_package_dir)
            print("  source_bundle     :", resolved.source_bundle_manifest_path)
            print("  downloaded_zip    :", resolved.downloaded_artifact_path)
            print("  output_bundle     :", bundle_dir)
            continue

        result = materialize_android_bundle(
            deployment_package_dir=deployment_package_dir,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
        print(f"Android bundle ready for {project}.")
        print("  deployment_package:", deployment_package_dir)
        print("  output_bundle     :", result.bundle_dir)
        print("  manifest          :", result.manifest_path)
        print("  compiled_model    :", result.compiled_model_path)
        print("  external_data     :", result.external_data_path)
        print("  io_contract       :", result.io_contract_path)
    return 0


def _copy_source_bundle_companions(
    *,
    source_manifest_path: Path,
    source_manifest: ModelBundleManifest,
    target_dir: Path,
    compiled_artifact_key: str,
) -> None:
    source_bundle_dir = source_manifest.bundle_dir(source_manifest_path)
    for key, file_name in source_manifest.artifacts.items():
        if key == compiled_artifact_key:
            continue
        source_path = source_bundle_dir / file_name
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing source bundle artifact for {key}: {source_path}")
        shutil.copy2(source_path, target_dir / Path(file_name).name)

    for _key, file_name in source_manifest.fixtures.items():
        source_path = source_bundle_dir / file_name
        if not source_path.is_file():
            continue
        shutil.copy2(source_path, target_dir / Path(file_name).name)


def _build_android_bundle_manifest(
    *,
    resolved: ResolvedAndroidBundleInputs,
    compiled_output_name: str,
) -> ModelBundleManifest:
    source_manifest = resolved.source_bundle_manifest
    artifacts = dict(source_manifest.artifacts)
    artifacts[resolved.layout.compiled_artifact_key] = compiled_output_name
    artifacts[resolved.layout.external_data_artifact_key] = "model.bin"
    artifacts["io_contract"] = "io_contract.json"

    fixtures = {
        key: Path(value).name
        for key, value in source_manifest.fixtures.items()
        if (source_manifest.bundle_dir(resolved.source_bundle_manifest_path) / value).is_file()
    }

    metadata = dict(source_manifest.metadata)
    metadata["aihub"] = {
        "run_label": resolved.run_label,
        "target_runtime": _normalize_optional_string(resolved.deployment_manifest.get("target_runtime")) or DEFAULT_TARGET_RUNTIME,
        "device_name": resolved.deployment_manifest.get("device_name"),
        "qairt_version": resolved.deployment_manifest.get("qairt_version"),
        "target_model_id": resolved.target_model_id,
        "io_contract_artifact": "io_contract",
        "compiled_artifact_key": resolved.layout.compiled_artifact_key,
        "external_data_artifact_key": resolved.layout.external_data_artifact_key,
        "special_handling": list(_read_json(resolved.io_contract_path).get("special_handling") or []),
        "components": {
            key: {"target_runtime": value}
            for key, value in resolved.layout.component_target_runtimes.items()
        },
    }

    return ModelBundleManifest(
        bundle_version=source_manifest.bundle_version,
        project=source_manifest.project,
        model_family=source_manifest.model_family,
        model_name=resolved.layout.manifest_model_name or source_manifest.model_name,
        model_variant=resolved.layout.manifest_model_variant,
        asset_namespace=resolved.layout.asset_namespace,
        runtime_kind="onnx",
        artifacts=artifacts,
        fixtures=fixtures,
        metadata=metadata,
    )


def _extract_compiled_payload(
    *,
    zip_path: Path,
    compiled_output_path: Path,
    external_data_output_path: Path,
) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        file_entries = [entry for entry in archive.infolist() if not entry.is_dir()]
        onnx_entries = [entry for entry in file_entries if entry.filename.lower().endswith(".onnx")]
        model_bin_entries = [entry for entry in file_entries if Path(entry.filename).name == "model.bin"]
        if len(onnx_entries) != 1:
            raise ValueError(
                f"Expected compiled payload to contain exactly one compiled ONNX file, found {len(onnx_entries)} in {zip_path}"
            )
        if len(model_bin_entries) != 1:
            raise ValueError(
                f"Expected compiled payload to contain exactly one model.bin, found {len(model_bin_entries)} in {zip_path}"
            )
        expected_entry_names = {onnx_entries[0].filename, model_bin_entries[0].filename}
        extra_entries = [entry.filename for entry in file_entries if entry.filename not in expected_entry_names]
        if extra_entries:
            raise ValueError(
                "Expected compiled payload to contain only one compiled ONNX file plus model.bin; "
                f"found extra payload entries: {extra_entries}"
            )

        compiled_output_path.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(onnx_entries[0]) as source, compiled_output_path.open("wb") as destination:
            shutil.copyfileobj(source, destination)
        with archive.open(model_bin_entries[0]) as source, external_data_output_path.open("wb") as destination:
            shutil.copyfileobj(source, destination)


def _resolve_downloaded_artifact_path(
    *,
    package_dir: Path,
    downloaded_artifact: Any,
) -> Path:
    if isinstance(downloaded_artifact, Mapping):
        candidate = _resolve_required_path(
            package_dir=package_dir,
            value=downloaded_artifact.get("path"),
            label="downloaded_artifact.path",
        )
        return candidate
    download_dir = package_dir / "download"
    zip_candidates = sorted(download_dir.glob("*.zip"))
    if len(zip_candidates) == 1:
        return zip_candidates[0].resolve()
    raise FileNotFoundError(f"Cannot resolve downloaded artifact zip from package: {package_dir}")


def _resolve_required_path(
    *,
    package_dir: Path,
    value: Any,
    label: str,
) -> Path:
    normalized = _normalize_optional_string(value)
    if not normalized:
        raise FileNotFoundError(f"Missing required deployment package field: {label}")
    candidate = Path(normalized)
    resolved = candidate.resolve() if candidate.is_absolute() else (package_dir / candidate).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Missing required deployment package file for {label}: {resolved}")
    return resolved


def _default_bundle_dir(package_dir: Path, project: str, run_label: str) -> Path:
    return (package_dir.parents[2] / "android_bundle" / project / _normalize_run_label(run_label)).resolve()


def _default_deployment_package_dir(
    runtime_config: AiHubRuntimeConfig,
    project: str,
    run_label: str,
) -> Path:
    return (runtime_config.artifact_root / "deploy" / project / _normalize_run_label(run_label)).resolve()


def _prepare_output_dir(bundle_dir: Path, *, overwrite: bool) -> None:
    if bundle_dir.exists():
        if any(bundle_dir.iterdir()):
            if not overwrite:
                raise ValueError(f"Output bundle dir already exists and is not empty: {bundle_dir}. Pass --overwrite to replace it.")
            shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _normalize_run_label(value: str) -> str:
    normalized = "".join(char if char.isalnum() or char in ("-", "_") else "-" for char in str(value).strip())
    collapsed = "-".join(part for part in normalized.split("-") if part)
    return collapsed or "run"


if __name__ == "__main__":
    raise SystemExit(main())
