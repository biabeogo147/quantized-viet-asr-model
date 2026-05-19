from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from shutil import copy2
from typing import Any, Mapping

import numpy as np

from tools.aihub_option1_phase4_gate import Option1PilotLayout, resolve_option1_pilot_layout
from tools.aihub_option1_pilots import Option1RuntimeConfig

DEPLOYMENT_CANDIDATE = "deployment_candidate"
RESEARCH_ONLY = "research_only"


def map_phase4_recommendation_to_promotion_status(recommendation: Mapping[str, Any]) -> str:
    value = str(recommendation.get("value", "") or "").strip().upper()
    if value == "NO_GO":
        return RESEARCH_ONLY
    return DEPLOYMENT_CANDIDATE


def resolve_phase5_evidence_inputs(
    *,
    pilot_name: str,
    runtime_config: Option1RuntimeConfig,
    run_label: str | None,
    phase2_compile_pilot_name_override: str | None = None,
) -> dict[str, Any]:
    layout = resolve_option1_pilot_layout(pilot_name)
    normalized_label = _normalize_record_label(run_label or "latest")
    phase2_compile_pilot_name = (
        _normalize_optional_string(phase2_compile_pilot_name_override) or layout.phase2_compile_pilot_name
    )
    required_paths = {
        "prepared_artifact_record": (
            runtime_config.pilot_record_dir(phase2_compile_pilot_name) / f"prepared-artifact-{normalized_label}.json"
        ).resolve(),
        "compile_run_record": (
            runtime_config.pilot_record_dir(phase2_compile_pilot_name) / f"compile-run-{normalized_label}.json"
        ).resolve(),
        "live_run_record": (
            runtime_config.pilot_record_dir(phase2_compile_pilot_name) / f"live-run-{normalized_label}.json"
        ).resolve(),
        "hybrid_run_record": (
            runtime_config.pilot_record_dir(layout.phase3_hybrid_pilot_name) / f"hybrid-run-{normalized_label}.json"
        ).resolve(),
        "phase4_gate_record": (
            runtime_config.pilot_record_dir(layout.phase4_gate_pilot_name) / f"phase4-gate-{normalized_label}.json"
        ).resolve(),
    }
    payloads = {name: _read_required_json(path) for name, path in required_paths.items()}
    warnings: list[str] = []
    live_run_record = payloads["live_run_record"]
    profile_artifact = live_run_record.get("profile_artifact")
    if not profile_artifact:
        warnings.append("profile_artifact missing from live run record")
    return {
        "layout": layout,
        "run_label": normalized_label,
        "phase2_compile_pilot_name": phase2_compile_pilot_name,
        "required_record_paths": required_paths,
        **payloads,
        "warnings": warnings,
    }


def build_phase5_io_contract(
    *,
    pilot_name: str,
    prepared_artifact_record: Mapping[str, Any],
    compile_run_record: Mapping[str, Any],
    live_run_record: Mapping[str, Any],
) -> dict[str, Any]:
    layout = resolve_option1_pilot_layout(pilot_name)
    compile_options = str(
        compile_run_record.get("compile_options")
        or prepared_artifact_record.get("compile_options")
        or ""
    )
    truncate_64bit_io = "--truncate_64bit_io" in compile_options

    inputs: dict[str, dict[str, Any]] = {}
    for name, spec in (prepared_artifact_record.get("input_specs") or {}).items():
        dtype = str(spec.get("dtype", "") or "")
        runtime_dtype = "int32" if truncate_64bit_io and dtype == "int64" else dtype
        inputs[str(name)] = {
            "shape": [int(dim) for dim in spec.get("shape", [])],
            "dtype": dtype,
            "runtime_dtype": runtime_dtype,
        }

    outputs: dict[str, dict[str, Any]] = {}
    for name, items in (live_run_record.get("output_tensors") or {}).items():
        if not items:
            continue
        first_item = items[0]
        outputs[str(name)] = {
            "shape": [int(dim) for dim in first_item.get("shape", [])],
            "dtype": str(first_item.get("dtype", "") or ""),
        }

    notes = list(layout.deployment_notes)
    if truncate_64bit_io:
        notes.append(
            "truncate_64bit_io requires int64 host inputs to be cast to int32 for compiled execution."
        )

    return {
        "pilot_name": layout.contract_pilot_name,
        "inputs": inputs,
        "outputs": outputs,
        "notes": notes,
    }


def build_phase5_contract_manifest(
    *,
    pilot_name: str,
    run_label: str,
    package_label: str | None,
    target_model: Mapping[str, Any],
    runtime: Mapping[str, Any],
    promotion_status: str,
    phase4_recommendation: Mapping[str, Any],
    source_artifacts: Mapping[str, Any],
    evidence: Mapping[str, Any],
    io_contract: Mapping[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    layout = resolve_option1_pilot_layout(pilot_name)
    return {
        "record_kind": "phase5_contract_manifest",
        "pilot_name": layout.contract_pilot_name,
        "run_label": _normalize_record_label(run_label),
        "package_label": package_label,
        "promotion_status": promotion_status,
        "phase4_recommendation": _json_safe(phase4_recommendation),
        "target_model": _json_safe(target_model),
        "runtime": _json_safe(runtime),
        "source_artifacts": _json_safe(source_artifacts),
        "evidence": _json_safe(evidence),
        "io_contract_summary": _json_safe(io_contract),
        "warnings": list(warnings),
        "created_at_utc": _utc_now_isoformat(),
    }


def materialize_phase5_contract_package(
    *,
    pilot_name: str,
    runtime_config: Option1RuntimeConfig,
    run_label: str | None,
    phase2_compile_pilot_name_override: str | None = None,
    package_label: str | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    evidence_inputs = resolve_phase5_evidence_inputs(
        pilot_name=pilot_name,
        runtime_config=runtime_config,
        run_label=run_label,
        phase2_compile_pilot_name_override=phase2_compile_pilot_name_override,
    )
    layout: Option1PilotLayout = evidence_inputs["layout"]
    normalized_label = evidence_inputs["run_label"]

    promotion_status = map_phase4_recommendation_to_promotion_status(evidence_inputs["phase4_gate_record"]["recommendation"])
    io_contract = build_phase5_io_contract(
        pilot_name=layout.canonical_name,
        prepared_artifact_record=evidence_inputs["prepared_artifact_record"],
        compile_run_record=evidence_inputs["compile_run_record"],
        live_run_record=evidence_inputs["live_run_record"],
    )
    package_path = _resolve_package_path(
        layout=layout,
        runtime_config=runtime_config,
        run_label=normalized_label,
        package_label=package_label,
        output_root=output_root,
    )
    evidence_dir = package_path / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    copied_evidence = {
        "prepared_artifact_record": _copy_evidence_record(
            source_path=evidence_inputs["required_record_paths"]["prepared_artifact_record"],
            destination_path=evidence_dir / "prepared-artifact-record.json",
        ),
        "compile_run_record": _copy_evidence_record(
            source_path=evidence_inputs["required_record_paths"]["compile_run_record"],
            destination_path=evidence_dir / "compile-run-record.json",
        ),
        "live_run_record": _copy_evidence_record(
            source_path=evidence_inputs["required_record_paths"]["live_run_record"],
            destination_path=evidence_dir / "live-run-record.json",
        ),
        "hybrid_run_record": _copy_evidence_record(
            source_path=evidence_inputs["required_record_paths"]["hybrid_run_record"],
            destination_path=evidence_dir / "hybrid-run-record.json",
        ),
        "phase4_gate_record": _copy_evidence_record(
            source_path=evidence_inputs["required_record_paths"]["phase4_gate_record"],
            destination_path=evidence_dir / "phase4-gate-record.json",
        ),
    }

    manifest = build_phase5_contract_manifest(
        pilot_name=layout.canonical_name,
        run_label=normalized_label,
        package_label=package_label,
        target_model=evidence_inputs["compile_run_record"].get("target_model", {}),
        runtime={
            "device_name": evidence_inputs["compile_run_record"].get("device_name"),
            "qairt_version": evidence_inputs["compile_run_record"].get("qairt_version"),
            "compute_unit": evidence_inputs["compile_run_record"].get("compute_unit"),
        },
        promotion_status=promotion_status,
        phase4_recommendation=evidence_inputs["phase4_gate_record"].get("recommendation", {}),
        source_artifacts={
            "source_model": evidence_inputs["prepared_artifact_record"].get("source_model"),
            "prepared_model": evidence_inputs["prepared_artifact_record"].get("prepared_model"),
        },
        evidence=copied_evidence,
        io_contract=io_contract,
        warnings=list(evidence_inputs["warnings"]),
    )

    package_path.mkdir(parents=True, exist_ok=True)
    manifest_path = package_path / "contract_manifest.json"
    io_contract_path = package_path / "io_contract.json"
    summary_path = package_path / "contract_summary.md"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    io_contract_path.write_text(json.dumps(io_contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        _build_contract_summary_markdown(
            layout=layout,
            run_label=normalized_label,
            promotion_status=promotion_status,
            phase4_recommendation=evidence_inputs["phase4_gate_record"].get("recommendation", {}),
            warnings=evidence_inputs["warnings"],
        ),
        encoding="utf-8",
    )

    return {
        "pilot_name": layout.contract_pilot_name,
        "package_path": package_path,
        "manifest_path": manifest_path,
        "io_contract_path": io_contract_path,
        "summary_path": summary_path,
        "promotion_status": promotion_status,
        "warnings": list(evidence_inputs["warnings"]),
    }


def _resolve_package_path(
    *,
    layout: Option1PilotLayout,
    runtime_config: Option1RuntimeConfig,
    run_label: str,
    package_label: str | None,
    output_root: str | Path | None,
) -> Path:
    base_root = (
        Path(output_root).resolve()
        if output_root is not None
        else (runtime_config.artifact_root / "contracts" / "option1").resolve()
    )
    package_path = base_root / layout.contract_pilot_name / _normalize_record_label(run_label)
    normalized_package_label = _normalize_optional_string(package_label)
    if normalized_package_label:
        package_path = package_path / _normalize_record_label(normalized_package_label)
    return package_path.resolve()


def _copy_evidence_record(*, source_path: str | Path, destination_path: str | Path) -> dict[str, Any]:
    resolved_source = Path(source_path).resolve()
    resolved_destination = Path(destination_path).resolve()
    resolved_destination.parent.mkdir(parents=True, exist_ok=True)
    copy2(resolved_source, resolved_destination)
    return {
        "source_path": resolved_source.as_posix(),
        "packaged_path": resolved_destination.as_posix(),
        "size_bytes": int(resolved_destination.stat().st_size),
        "sha256": _hash_file_sha256(resolved_destination),
    }


def _build_contract_summary_markdown(
    *,
    layout: Option1PilotLayout,
    run_label: str,
    promotion_status: str,
    phase4_recommendation: Mapping[str, Any],
    warnings: list[str],
) -> str:
    lines = [
        f"# Option 1 Contract Summary: {layout.contract_pilot_name}",
        "",
        f"- Run label: `{run_label}`",
        f"- Promotion status: `{promotion_status}`",
        f"- Phase 4 recommendation: `{phase4_recommendation.get('value', 'unknown')}`",
    ]
    reasons = phase4_recommendation.get("reasons") or []
    if reasons:
        lines.append(f"- Recommendation reasons: `{', '.join(str(reason) for reason in reasons)}`")
    if warnings:
        lines.append(f"- Warnings: `{'; '.join(warnings)}`")
    lines.append("")
    lines.append("## Deployment Notes")
    for note in layout.deployment_notes:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def _hash_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_required_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Required Phase 5 input record is missing: {resolved}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def _normalize_optional_string(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _normalize_record_label(value: str) -> str:
    normalized = "".join(character if character.isalnum() or character in {"-", "_"} else "-" for character in str(value).strip())
    collapsed = "-".join(part for part in normalized.split("-") if part)
    return collapsed or "run"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _utc_now_isoformat() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
