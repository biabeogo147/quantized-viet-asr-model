from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from aihub.session import (
    DEFAULT_TARGET_RUNTIME,
    AiHubRuntimeConfig,
    build_runtime_config,
    download_compiled_target_model,
    resolve_target_model_id,
    resolve_vpcd_source,
    resolve_zipformer_encoder_source,
    write_deployment_download_record,
)


TargetModelResolver = Callable[[str], Any]


@dataclass(frozen=True)
class DeploymentLayout:
    project: str
    compile_record_group: str
    evaluation_record_group: str
    downloaded_artifact_name: str
    deployment_notes: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedDeploymentInputs:
    layout: DeploymentLayout
    run_label: str
    target_model_id: str
    package_dir: Path
    source_bundle_manifest_path: Path
    prepared_record_path: Path
    compile_record_path: Path
    live_record_path: Path
    hybrid_record_path: Path | None
    prepared_record: dict[str, Any]
    compile_record: dict[str, Any]
    live_record: dict[str, Any]
    hybrid_record: dict[str, Any] | None


@dataclass(frozen=True)
class DeploymentPackageResult:
    project: str
    run_label: str
    package_dir: Path
    manifest_path: Path
    io_contract_path: Path
    deploy_notes_path: Path
    downloaded_artifact_path: Path
    download_record_path: Path
    target_model_id: str


PROJECT_LAYOUTS: dict[str, DeploymentLayout] = {
    "zipformer": DeploymentLayout(
        project="zipformer",
        compile_record_group="zipformer_encoder_option1",
        evaluation_record_group="zipformer_hybrid_option1",
        downloaded_artifact_name="encoder.precompiled_qnn_onnx.onnx",
        deployment_notes=(
            "Encoder runs on the compiled target artifact.",
            "Decoder and joiner remain CPU-side ONNX artifacts.",
        ),
    ),
    "vpcd": DeploymentLayout(
        project="vpcd",
        compile_record_group="vpcd_option1_local_aimet",
        evaluation_record_group="vpcd_hybrid_option1",
        downloaded_artifact_name="model.precompiled_qnn_onnx.onnx",
        deployment_notes=(
            "Model session runs on the compiled target artifact.",
            "Tokenizer encode and tokenizer decode remain CPU-side artifacts.",
        ),
    ),
}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and package retained AI Hub deployment artifacts.")
    parser.add_argument("--project", required=True, choices=("zipformer", "vpcd", "all"))
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--device-name", default="Samsung Galaxy S24 (Family)")
    parser.add_argument("--qairt-version", default=None)
    parser.add_argument("--source-bundle-manifest", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def resolve_deployment_inputs(
    *,
    runtime_config: AiHubRuntimeConfig,
    project: str,
    run_label: str,
    explicit_target_model_id: str | None = None,
    source_bundle_manifest_path: str | Path | None = None,
) -> ResolvedDeploymentInputs:
    layout = _resolve_layout(project)
    normalized_run_label = _normalize_optional_string(run_label) or "latest"

    prepared_record_path = _resolve_required_record_path(
        runtime_config=runtime_config,
        record_group=layout.compile_record_group,
        record_kind="prepared-artifact",
        run_label=normalized_run_label,
    )
    compile_record_path = _resolve_required_record_path(
        runtime_config=runtime_config,
        record_group=layout.compile_record_group,
        record_kind="compile-run",
        run_label=normalized_run_label,
    )
    live_record_path = _resolve_required_record_path(
        runtime_config=runtime_config,
        record_group=layout.compile_record_group,
        record_kind="live-run",
        run_label=normalized_run_label,
    )
    hybrid_record_path = _resolve_optional_record_path(
        runtime_config=runtime_config,
        record_group=layout.evaluation_record_group,
        record_kind="hybrid-run",
        run_label=normalized_run_label,
    )

    target_model_id = resolve_target_model_id(
        pilot_name=layout.compile_record_group,
        runtime_config=runtime_config,
        explicit_target_model_id=explicit_target_model_id,
        run_label=normalized_run_label,
    )
    source_bundle_manifest_path = _resolve_source_bundle_manifest_path(
        runtime_config=runtime_config,
        layout=layout,
        source_bundle_manifest_path=source_bundle_manifest_path,
    )

    return ResolvedDeploymentInputs(
        layout=layout,
        run_label=normalized_run_label,
        target_model_id=target_model_id,
        package_dir=_resolve_package_dir(runtime_config=runtime_config, layout=layout, run_label=normalized_run_label),
        source_bundle_manifest_path=source_bundle_manifest_path,
        prepared_record_path=prepared_record_path,
        compile_record_path=compile_record_path,
        live_record_path=live_record_path,
        hybrid_record_path=hybrid_record_path,
        prepared_record=_read_json(prepared_record_path),
        compile_record=_read_json(compile_record_path),
        live_record=_read_json(live_record_path),
        hybrid_record=_read_json(hybrid_record_path) if hybrid_record_path is not None else None,
    )


def materialize_deployment_package(
    *,
    runtime_config: AiHubRuntimeConfig,
    project: str,
    run_label: str,
    explicit_target_model_id: str | None = None,
    source_bundle_manifest_path: str | Path | None = None,
    target_model_resolver: TargetModelResolver | None = None,
) -> DeploymentPackageResult:
    resolved = resolve_deployment_inputs(
        runtime_config=runtime_config,
        project=project,
        run_label=run_label,
        explicit_target_model_id=explicit_target_model_id,
        source_bundle_manifest_path=source_bundle_manifest_path,
    )
    resolver = target_model_resolver or _default_target_model_resolver

    package_dir = resolved.package_dir
    download_dir = package_dir / "download"
    evidence_dir = package_dir / "evidence"
    package_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    target_model = resolver(resolved.target_model_id)
    downloaded_artifact_path = download_compiled_target_model(
        target_model=target_model,
        output_path=download_dir / resolved.layout.downloaded_artifact_name,
    )
    download_record_path = write_deployment_download_record(
        pilot_name=resolved.layout.compile_record_group,
        runtime_config=runtime_config,
        compile_record_path=resolved.compile_record_path,
        target_model=target_model,
        downloaded_artifact_path=downloaded_artifact_path,
        run_label=resolved.run_label,
    )

    copied_evidence = _copy_evidence_records(
        evidence_dir=evidence_dir,
        record_paths=(
            resolved.prepared_record_path,
            resolved.compile_record_path,
            resolved.live_record_path,
            resolved.hybrid_record_path,
            download_record_path,
        ),
    )
    io_contract_payload = _build_io_contract_payload(resolved)
    io_contract_path = _write_json(package_dir / "io_contract.json", io_contract_payload)
    deploy_notes_path = _write_text(
        package_dir / "deploy_notes.md",
        _build_deploy_notes_markdown(resolved),
    )
    manifest_payload = _build_deployment_manifest_payload(
        resolved=resolved,
        downloaded_artifact_path=downloaded_artifact_path,
        io_contract_path=io_contract_path,
        deploy_notes_path=deploy_notes_path,
        copied_evidence=copied_evidence,
    )
    manifest_path = _write_json(package_dir / "deployment_manifest.json", manifest_payload)
    return DeploymentPackageResult(
        project=resolved.layout.project,
        run_label=resolved.run_label,
        package_dir=package_dir,
        manifest_path=manifest_path,
        io_contract_path=io_contract_path,
        deploy_notes_path=deploy_notes_path,
        downloaded_artifact_path=downloaded_artifact_path,
        download_record_path=download_record_path,
        target_model_id=resolved.target_model_id,
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

    if args.dry_run:
        for project in projects:
            resolved = resolve_deployment_inputs(
                runtime_config=runtime_config,
                project=project,
                run_label=args.run_label,
                source_bundle_manifest_path=args.source_bundle_manifest,
            )
            print(f"[dry-run] project={project}")
            print("  target_model_id:", resolved.target_model_id)
            print("  compile_record :", resolved.compile_record_path)
            print("  live_record    :", resolved.live_record_path)
            print("  package_dir    :", resolved.package_dir)
        return 0

    for project in projects:
        result = materialize_deployment_package(
            runtime_config=runtime_config,
            project=project,
            run_label=args.run_label,
            source_bundle_manifest_path=args.source_bundle_manifest,
        )
        print(f"Deployment package ready for {project}.")
        print("  target_model_id:", result.target_model_id)
        print("  package_dir    :", result.package_dir)
        print("  manifest       :", result.manifest_path)
        print("  io_contract    :", result.io_contract_path)
        print("  deploy_notes   :", result.deploy_notes_path)
    return 0


def _resolve_layout(project: str) -> DeploymentLayout:
    if project not in PROJECT_LAYOUTS:
        supported = ", ".join(sorted(PROJECT_LAYOUTS))
        raise ValueError(f"Unsupported Deployment project {project!r}. Supported projects: {supported}")
    return PROJECT_LAYOUTS[project]


def _resolve_required_record_path(
    *,
    runtime_config: AiHubRuntimeConfig,
    record_group: str,
    record_kind: str,
    run_label: str,
) -> Path:
    path = _resolve_record_path(
        runtime_config=runtime_config,
        record_group=record_group,
        record_kind=record_kind,
        run_label=run_label,
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing required Deployment record: {path}")
    return path


def _resolve_optional_record_path(
    *,
    runtime_config: AiHubRuntimeConfig,
    record_group: str,
    record_kind: str,
    run_label: str,
) -> Path | None:
    path = _resolve_record_path(
        runtime_config=runtime_config,
        record_group=record_group,
        record_kind=record_kind,
        run_label=run_label,
    )
    return path if path.exists() else None


def _resolve_record_path(
    *,
    runtime_config: AiHubRuntimeConfig,
    record_group: str,
    record_kind: str,
    run_label: str,
) -> Path:
    normalized_label = _normalize_record_label(run_label or "latest")
    return (runtime_config.pilot_record_dir(record_group) / f"{record_kind}-{normalized_label}.json").resolve()


def _resolve_package_dir(
    *,
    runtime_config: AiHubRuntimeConfig,
    layout: DeploymentLayout,
    run_label: str,
) -> Path:
    return (runtime_config.artifact_root / "deploy" / layout.project / _normalize_record_label(run_label)).resolve()


def _resolve_source_bundle_manifest_path(
    *,
    runtime_config: AiHubRuntimeConfig,
    layout: DeploymentLayout,
    source_bundle_manifest_path: str | Path | None = None,
) -> Path:
    if source_bundle_manifest_path is not None:
        explicit_path = Path(source_bundle_manifest_path).resolve()
        if not explicit_path.is_file():
            raise FileNotFoundError(f"Missing explicit source bundle manifest: {explicit_path}")
        return explicit_path
    if layout.project == "zipformer":
        return resolve_zipformer_encoder_source(runtime_config.repo_root).bundle_manifest_path.resolve()
    if layout.project == "vpcd":
        return resolve_vpcd_source(runtime_config.repo_root).bundle_manifest_path.resolve()
    raise ValueError(f"Unsupported source bundle resolution for project: {layout.project}")


def _copy_evidence_records(
    *,
    evidence_dir: Path,
    record_paths: Sequence[Path | None],
) -> tuple[Path, ...]:
    copied: list[Path] = []
    for record_path in record_paths:
        if record_path is None:
            continue
        destination = evidence_dir / record_path.name
        shutil.copy2(record_path, destination)
        copied.append(destination.resolve())
    return tuple(copied)


def _build_io_contract_payload(resolved: ResolvedDeploymentInputs) -> dict[str, Any]:
    compile_options = str(resolved.compile_record.get("compile_options") or "")
    special_handling = _build_special_handling(compile_options)
    input_specs = dict(resolved.prepared_record.get("input_specs") or {})
    output_tensors = dict(resolved.live_record.get("output_tensors") or {})
    return {
        "target_runtime": DEFAULT_TARGET_RUNTIME,
        "inputs": _build_input_contract_entries(input_specs=input_specs, special_handling=special_handling),
        "outputs": _build_output_contract_entries(output_tensors=output_tensors),
        "special_handling": special_handling,
        "deployment_notes": list(resolved.layout.deployment_notes),
    }


def _build_input_contract_entries(
    *,
    input_specs: Mapping[str, Any],
    special_handling: Sequence[str],
) -> list[dict[str, Any]]:
    truncate_64bit = "truncate_64bit_io required" in set(special_handling)
    entries: list[dict[str, Any]] = []
    for name, spec in input_specs.items():
        if not isinstance(spec, Mapping):
            continue
        source_dtype = str(spec.get("dtype") or "")
        expected_dtype = "int32" if truncate_64bit and source_dtype == "int64" else source_dtype
        entries.append(
            {
                "name": str(name),
                "shape": [int(dim) for dim in list(spec.get("shape") or [])],
                "dtype": expected_dtype,
                "source_dtype": source_dtype,
            }
        )
    return entries


def _build_output_contract_entries(*, output_tensors: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for name, tensor_summaries in output_tensors.items():
        first_summary = tensor_summaries[0] if isinstance(tensor_summaries, list) and tensor_summaries else None
        if not isinstance(first_summary, Mapping):
            continue
        entries.append(
            {
                "name": str(name),
                "shape": [int(dim) for dim in list(first_summary.get("shape") or [])],
                "dtype": str(first_summary.get("dtype") or ""),
            }
        )
    return entries


def _build_special_handling(compile_options: str) -> list[str]:
    notes: list[str] = []
    if "--truncate_64bit_io" in str(compile_options or ""):
        notes.append("truncate_64bit_io required")
    return notes


def _build_deployment_manifest_payload(
    *,
    resolved: ResolvedDeploymentInputs,
    downloaded_artifact_path: Path,
    io_contract_path: Path,
    deploy_notes_path: Path,
    copied_evidence: Sequence[Path],
) -> dict[str, Any]:
    compile_options = str(resolved.compile_record.get("compile_options") or "")
    special_handling = _build_special_handling(compile_options)
    return {
        "project": resolved.layout.project,
        "compile_record_group": resolved.layout.compile_record_group,
        "evaluation_record_group": resolved.layout.evaluation_record_group,
        "run_label": resolved.run_label,
        "target_model_id": resolved.target_model_id,
        "target_runtime": DEFAULT_TARGET_RUNTIME,
        "device_name": resolved.compile_record.get("device_name"),
        "qairt_version": resolved.compile_record.get("qairt_version"),
        "compile_options": compile_options,
        "downloaded_artifact": _build_file_metadata(downloaded_artifact_path),
        "source_bundle_manifest": resolved.source_bundle_manifest_path.as_posix(),
        "io_contract_path": io_contract_path.as_posix(),
        "deploy_notes_path": deploy_notes_path.as_posix(),
        "record_evidence": [path.as_posix() for path in copied_evidence],
        "special_handling": special_handling,
    }


def _build_deploy_notes_markdown(resolved: ResolvedDeploymentInputs) -> str:
    lines = [
        f"# {resolved.layout.project} Deployment Notes",
        "",
        f"- Run label: `{resolved.run_label}`",
        f"- Target model id: `{resolved.target_model_id}`",
        f"- Source bundle manifest: `{resolved.source_bundle_manifest_path.as_posix()}`",
        "",
        "## Runtime split",
    ]
    lines.extend(f"- {note}" for note in resolved.layout.deployment_notes)
    return "\n".join(lines) + "\n"


def _default_target_model_resolver(target_model_id: str) -> Any:
    import qai_hub as hub

    return hub.get_model(target_model_id)


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path.resolve()


def _write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path.resolve()


def _build_file_metadata(path: Path) -> dict[str, Any]:
    resolved_path = path.resolve()
    return {
        "path": resolved_path.as_posix(),
        "size_bytes": int(resolved_path.stat().st_size),
    }


def _normalize_optional_string(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _normalize_record_label(value: str) -> str:
    normalized = "".join(char if char.isalnum() or char in ("-", "_") else "-" for char in str(value).strip())
    collapsed = "-".join(part for part in normalized.split("-") if part)
    return collapsed or "run"


if __name__ == "__main__":
    raise SystemExit(main())

