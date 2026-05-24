from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from model_bundle.fixtures import read_jsonl
from model_bundle.manifest import ModelBundleManifest
from model_bundle.vpcd_runtime import BundleOnnxRuntime
from model_bundle.zipformer_runtime import BundleAcousticRuntime


ZIPFORMER_SEPARATOR_VARIANTS = ("â–", "Ã¢â€“Â", "▁")
NUMBER_PATTERN = re.compile(r"\d+(?:[/:.\-]\d+)*")
WORD_PATTERN = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]+", re.UNICODE)
TERMINAL_PUNCTUATION = ".!?…"


def normalize_whitespace_text(raw_text: str | None) -> str:
    return re.sub(r"\s+", " ", (raw_text or "").strip())


def normalize_zipformer_text(raw_text: str | None) -> str:
    normalized = raw_text or ""
    for marker in ZIPFORMER_SEPARATOR_VARIANTS:
        normalized = normalized.replace(marker, " ")
    return normalize_whitespace_text(normalized)


def evaluate_vpcd_golden(
    model_root: str | Path,
    *,
    candidate_label: str,
    provider: str = "CPUExecutionProvider",
) -> dict[str, Any]:
    bundle_root = Path(model_root).resolve()
    manifest = ModelBundleManifest.from_path(bundle_root / "bundle_manifest.json")
    golden_path = bundle_root / str(manifest.fixtures["golden_samples"])
    samples = read_jsonl(golden_path)
    try:
        runtime = BundleOnnxRuntime.from_manifest_path(bundle_root / "bundle_manifest.json", provider=provider)
    except Exception as exc:  # pragma: no cover - exercised via unit test with monkeypatch
        raise _rewrite_runtime_error(exc, project="vpcd", bundle_root=bundle_root) from exc
    max_decode_length = int(manifest.metadata.get("max_decode_length", 128))

    exact_match_count = 0
    mismatches: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    total_char_distance = 0
    total_expected_chars = 0

    canonical_index = _resolve_vpcd_canonical_index(samples)
    canonical_exact = False

    for index, sample in enumerate(samples):
        raw_text = str(sample["raw_text"])
        expected_text = str(sample.get("expected_output", ""))
        actual_text = str(runtime.restore(raw_text, max_length=max_decode_length))
        normalized_expected = normalize_whitespace_text(expected_text)
        normalized_actual = normalize_whitespace_text(actual_text)
        exact_match = normalized_expected == normalized_actual
        char_distance = _levenshtein_distance(normalized_expected, normalized_actual)
        expected_chars = max(1, len(normalized_expected))
        cer = round(char_distance / expected_chars, 6)
        critical_regressions = _detect_vpcd_critical_regressions(
            expected_text=normalized_expected,
            actual_text=normalized_actual,
        )
        if exact_match:
            exact_match_count += 1
        else:
            mismatches.append(
                {
                    "sample_index": index,
                    "raw_text": raw_text,
                    "expected_text": expected_text,
                    "actual_text": actual_text,
                    "normalized_expected_text": normalized_expected,
                    "normalized_actual_text": normalized_actual,
                    "cer": cer,
                    "critical_regressions": critical_regressions,
                }
            )
        if index == canonical_index:
            canonical_exact = exact_match
        total_char_distance += char_distance
        total_expected_chars += expected_chars
        reports.append(
            {
                "sample_index": index,
                "raw_text": raw_text,
                "expected_text": expected_text,
                "actual_text": actual_text,
                "normalized_expected_text": normalized_expected,
                "normalized_actual_text": normalized_actual,
                "exact_match": exact_match,
                "cer": cer,
                "critical_regressions": critical_regressions,
            }
        )

    exact_match_rate = exact_match_count / len(samples) if samples else 0.0
    normalized_cer = total_char_distance / max(1, total_expected_chars)
    critical_regression_count = sum(len(row["critical_regressions"]) for row in reports)

    return {
        "project": "vpcd",
        "candidate_label": candidate_label,
        "bundle_manifest_path": (bundle_root / "bundle_manifest.json").as_posix(),
        "golden_path": golden_path.as_posix(),
        "sample_count": len(samples),
        "exact_match_count": exact_match_count,
        "exact_match_rate": round(exact_match_rate, 6),
        "normalized_cer": round(normalized_cer, 6),
        "canonical_sample_index": canonical_index,
        "canonical_exact_match": canonical_exact,
        "critical_regression_count": critical_regression_count,
        "passed": bool(
            samples
            and canonical_exact
            and exact_match_rate >= 0.95
            and normalized_cer <= 0.01
            and critical_regression_count == 0
        ),
        "reports": reports,
        "mismatches": mismatches,
    }


def evaluate_zipformer_golden(
    model_root: str | Path,
    *,
    candidate_label: str,
    repo_root: str | Path | None = None,
    provider: str = "CPUExecutionProvider",
) -> dict[str, Any]:
    bundle_root = Path(model_root).resolve()
    manifest = ModelBundleManifest.from_path(bundle_root / "bundle_manifest.json")
    expected_path = bundle_root / str(manifest.fixtures["expected_outputs"])
    expected_rows = read_jsonl(expected_path)
    try:
        runtime = BundleAcousticRuntime.from_manifest_path(bundle_root / "bundle_manifest.json", provider=provider)
    except Exception as exc:  # pragma: no cover - covered by runtime integration, not unit path
        raise _rewrite_runtime_error(exc, project="zipformer", bundle_root=bundle_root) from exc

    exact_match_count = 0
    mismatches: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    total_char_distance = 0
    total_expected_chars = 0

    for index, row in enumerate(expected_rows):
        sample_id = str(row["sample_id"])
        expected_text = str(row.get("text", ""))
        audio_path = _resolve_audio_path(str(row["audio_path"]), bundle_root=bundle_root, repo_root=repo_root)
        actual_text = str(runtime.transcribe(audio_path)["text"])
        normalized_expected = normalize_zipformer_text(expected_text)
        normalized_actual = normalize_zipformer_text(actual_text)
        exact_match = normalized_expected == normalized_actual
        char_distance = _levenshtein_distance(normalized_expected, normalized_actual)
        expected_chars = max(1, len(normalized_expected))
        cer = round(char_distance / expected_chars, 6)
        critical_regressions = _detect_zipformer_critical_regressions(
            expected_text=normalized_expected,
            actual_text=normalized_actual,
        )
        if exact_match:
            exact_match_count += 1
        else:
            mismatches.append(
                {
                    "sample_index": index,
                    "sample_id": sample_id,
                    "audio_path": str(row["audio_path"]),
                    "expected_text": expected_text,
                    "actual_text": actual_text,
                    "normalized_expected_text": normalized_expected,
                    "normalized_actual_text": normalized_actual,
                    "cer": cer,
                    "critical_regressions": critical_regressions,
                }
            )
        total_char_distance += char_distance
        total_expected_chars += expected_chars
        reports.append(
            {
                "sample_index": index,
                "sample_id": sample_id,
                "audio_path": str(row["audio_path"]),
                "expected_text": expected_text,
                "actual_text": actual_text,
                "normalized_expected_text": normalized_expected,
                "normalized_actual_text": normalized_actual,
                "exact_match": exact_match,
                "cer": cer,
                "critical_regressions": critical_regressions,
            }
        )

    exact_match_rate = exact_match_count / len(expected_rows) if expected_rows else 0.0
    normalized_cer = total_char_distance / max(1, total_expected_chars)
    critical_regression_count = sum(len(row["critical_regressions"]) for row in reports)

    return {
        "project": "zipformer",
        "candidate_label": candidate_label,
        "bundle_manifest_path": (bundle_root / "bundle_manifest.json").as_posix(),
        "expected_outputs_path": expected_path.as_posix(),
        "sample_count": len(expected_rows),
        "exact_match_count": exact_match_count,
        "exact_match_rate": round(exact_match_rate, 6),
        "normalized_cer": round(normalized_cer, 6),
        "critical_regression_count": critical_regression_count,
        "passed": bool(expected_rows and exact_match_count == len(expected_rows) and critical_regression_count == 0),
        "reports": reports,
        "mismatches": mismatches,
    }


def collect_phase7_candidate_metadata(
    *,
    project: str,
    candidate_label: str,
    model_root: str | Path,
    compile_record_path: str | Path | None = None,
    prepared_record_path: str | Path | None = None,
    live_record_path: str | Path | None = None,
    hybrid_record_path: str | Path | None = None,
) -> dict[str, Any]:
    bundle_root = Path(model_root).resolve()
    manifest_path = bundle_root / "bundle_manifest.json"
    manifest = ModelBundleManifest.from_path(manifest_path)
    io_contract_path = bundle_root / "io_contract.json"
    files = [path for path in bundle_root.rglob("*") if path.is_file()]
    total_bytes = sum(path.stat().st_size for path in files)

    payload = {
        "project": project,
        "candidate_label": candidate_label,
        "model_root": bundle_root.as_posix(),
        "bundle": {
            "manifest_path": manifest_path.as_posix(),
            "model_name": manifest.model_name,
            "model_variant": manifest.model_variant,
            "asset_namespace": manifest.asset_namespace,
            "artifact_count": len(files),
            "total_bytes": total_bytes,
            "artifacts": {path.relative_to(bundle_root).as_posix(): path.stat().st_size for path in sorted(files)},
            "quantization": dict(manifest.metadata.get("quantization") or {}),
            "aihub": dict(manifest.metadata.get("aihub") or {}),
            "io_contract_path": io_contract_path.as_posix() if io_contract_path.exists() else None,
            "io_contract": _read_optional_json(io_contract_path),
        },
        "compile": _summarize_record_payload(compile_record_path),
        "prepared": _summarize_record_payload(prepared_record_path),
        "live": _summarize_record_payload(live_record_path),
        "hybrid": _summarize_record_payload(hybrid_record_path),
    }
    return payload


def materialize_vpcd_local_aimet_candidate_bundle(
    *,
    candidate_label: str,
    control_bundle_root: str | Path,
    quantize_report_path: str | Path,
    output_root: str | Path,
) -> Path:
    control_bundle = Path(control_bundle_root).resolve()
    control_manifest = ModelBundleManifest.from_path(control_bundle / "bundle_manifest.json")

    quantize_report = json.loads(Path(quantize_report_path).resolve().read_text(encoding="utf-8"))
    qdq_reference_model_path = Path(str(quantize_report["qdq_reference_model_path"])).resolve()
    if not qdq_reference_model_path.exists():
        raise FileNotFoundError(f"Missing qdq_reference_model_path for Phase 7 candidate: {qdq_reference_model_path}")

    output_dir = Path(output_root).resolve() / _slugify_candidate_label(candidate_label)
    output_dir.mkdir(parents=True, exist_ok=True)

    for artifact_name in control_manifest.artifacts.values():
        if artifact_name == control_manifest.artifacts.get("model"):
            continue
        source_path = control_bundle / artifact_name
        target_path = output_dir / artifact_name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)

    for fixture_name in control_manifest.fixtures.values():
        source_path = control_bundle / fixture_name
        target_path = output_dir / fixture_name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)

    _copy_model_with_sidecars(qdq_reference_model_path, output_dir)

    metadata = dict(control_manifest.metadata)
    quantization = dict(metadata.get("quantization") or {})
    quantization["phase7_lane"] = candidate_label
    metadata["quantization"] = quantization
    metadata["phase7_candidate"] = {
        "candidate_label": candidate_label,
        "source_strategy": str(quantize_report.get("source_strategy") or ""),
        "variant_name": str(quantize_report.get("variant_name") or ""),
        "qdq_reference_model_path": qdq_reference_model_path.as_posix(),
        "packaging_path": str(quantize_report.get("packaging_path") or quantize_report.get("package_dir") or ""),
        "packaging_kind": str(quantize_report.get("packaging_kind") or "aimet_dir"),
        "source_kind": str(quantize_report.get("source_kind") or "local_aimet"),
        "transformation_kind": str(quantize_report.get("transformation_kind") or "aimet_service_export"),
        "aimet": dict(quantize_report.get("aimet") or {}),
    }

    candidate_manifest = ModelBundleManifest(
        bundle_version=control_manifest.bundle_version,
        project=control_manifest.project,
        model_family=control_manifest.model_family,
        model_name=control_manifest.model_name,
        model_variant=_slugify_candidate_label(candidate_label),
        asset_namespace=f"{control_manifest.asset_namespace}/phase7/{_slugify_candidate_label(candidate_label)}",
        runtime_kind=control_manifest.runtime_kind,
        artifacts={
            **dict(control_manifest.artifacts),
            "model": qdq_reference_model_path.name,
        },
        fixtures=dict(control_manifest.fixtures),
        metadata=metadata,
    )
    candidate_manifest.write_json(output_dir / "bundle_manifest.json")
    return output_dir


def materialize_zipformer_component_candidate_bundle(
    *,
    candidate_label: str,
    control_bundle_root: str | Path,
    quantized_bundle_root: str | Path,
    output_root: str | Path,
    component_sources: Mapping[str, str],
) -> Path:
    control_bundle = Path(control_bundle_root).resolve()
    quantized_bundle = Path(quantized_bundle_root).resolve()
    control_manifest = ModelBundleManifest.from_path(control_bundle / "bundle_manifest.json")
    quantized_manifest = ModelBundleManifest.from_path(quantized_bundle / "bundle_manifest.json")
    if control_manifest.project != "zipformer":
        raise ValueError(f"Expected a zipformer control bundle, got: {control_manifest.project!r}")
    if quantized_manifest.project != "zipformer":
        raise ValueError(f"Expected a zipformer quantized bundle, got: {quantized_manifest.project!r}")

    normalized_component_sources = {
        key: str(value).strip().lower()
        for key, value in dict(component_sources).items()
        if str(value).strip()
    }
    for component_name in ("encoder", "decoder", "joiner"):
        source_kind = normalized_component_sources.get(component_name, "control")
        if source_kind not in {"control", "quantized"}:
            raise ValueError(
                f"Unsupported source kind for zipformer component {component_name!r}: {source_kind!r}"
            )
        normalized_component_sources[component_name] = source_kind

    output_dir = Path(output_root).resolve() / _slugify_candidate_label(candidate_label)
    output_dir.mkdir(parents=True, exist_ok=True)

    for artifact_key, artifact_name in control_manifest.artifacts.items():
        if artifact_key in {"encoder", "decoder", "joiner"}:
            source_kind = normalized_component_sources.get(artifact_key, "control")
            source_bundle = quantized_bundle if source_kind == "quantized" else control_bundle
            _copy_model_with_sidecars(source_bundle / artifact_name, output_dir)
            continue
        source_path = control_bundle / artifact_name
        target_path = output_dir / artifact_name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)

    for fixture_name in control_manifest.fixtures.values():
        source_path = control_bundle / fixture_name
        target_path = output_dir / fixture_name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)

    metadata = dict(control_manifest.metadata)
    quantization = dict(metadata.get("quantization") or {})
    quantization["phase7_lane"] = candidate_label
    metadata["quantization"] = quantization
    metadata["phase7_candidate"] = {
        "candidate_label": candidate_label,
        "component_sources": dict(normalized_component_sources),
        "control_bundle_root": control_bundle.as_posix(),
        "quantized_bundle_root": quantized_bundle.as_posix(),
    }

    candidate_manifest = ModelBundleManifest(
        bundle_version=control_manifest.bundle_version,
        project=control_manifest.project,
        model_family=control_manifest.model_family,
        model_name=control_manifest.model_name,
        model_variant=_slugify_candidate_label(candidate_label),
        asset_namespace=f"{control_manifest.asset_namespace}/phase7/{_slugify_candidate_label(candidate_label)}",
        runtime_kind=control_manifest.runtime_kind,
        artifacts=dict(control_manifest.artifacts),
        fixtures=dict(control_manifest.fixtures),
        metadata=metadata,
    )
    candidate_manifest.write_json(output_dir / "bundle_manifest.json")
    return output_dir


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 7 golden evaluation and metadata helpers.")
    parser.add_argument("--project", required=True, choices=("vpcd", "zipformer"))
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--output", default=None)
    parser.add_argument("--compile-record", default=None)
    parser.add_argument("--prepared-record", default=None)
    parser.add_argument("--live-record", default=None)
    parser.add_argument("--hybrid-record", default=None)
    parser.add_argument("--mode", required=True, choices=("golden", "metadata"))
    return parser


def cli(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.mode == "golden":
        if args.project == "vpcd":
            payload = evaluate_vpcd_golden(
                args.model_root,
                candidate_label=args.candidate,
                provider=args.provider,
            )
        else:
            payload = evaluate_zipformer_golden(
                args.model_root,
                candidate_label=args.candidate,
                repo_root=args.repo_root,
                provider=args.provider,
            )
    else:
        payload = collect_phase7_candidate_metadata(
            project=args.project,
            candidate_label=args.candidate,
            model_root=args.model_root,
            compile_record_path=args.compile_record,
            prepared_record_path=args.prepared_record,
            live_record_path=args.live_record,
            hybrid_record_path=args.hybrid_record,
        )

    output_path = Path(args.output).resolve() if args.output else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(output_path.as_posix())
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_model_with_sidecars(model_path: Path, bundle_root: Path) -> None:
    target_path = bundle_root / model_path.name
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_path, target_path)
    for sibling in sorted(model_path.parent.glob(f"{model_path.name}.*")):
        if sibling.is_file():
            shutil.copy2(sibling, bundle_root / sibling.name)


def _rewrite_runtime_error(exc: Exception, *, project: str, bundle_root: Path) -> RuntimeError:
    message = str(exc)
    if "EPContext node generated by 'QNN'" in message:
        return RuntimeError(
            "This bundle appears to be a shipping precompiled_qnn_onnx artifact and cannot be evaluated host-side "
            "with CPUExecutionProvider. Use the source bundle under python-model-test/build/model_bundle for the "
            f"{project} golden gate, then use AI Hub compile/live/hybrid records as evidence for the compiled bundle. "
            f"Bundle root: {bundle_root.as_posix()}"
        )
    if "HasExternalDataInMemory" in message:
        return RuntimeError(
            "This bundle could not be loaded host-side because the ONNX external-data layout is not compatible with "
            "the current ORT path. Use a CPU-loadable source control bundle for the host-side golden gate, or "
            "rebuild the candidate bundle before rerunning the Phase 7 harness."
        )
    return RuntimeError(message)


def _summarize_record_payload(path_like: str | Path | None) -> dict[str, Any] | None:
    if path_like is None:
        return None
    path = Path(path_like).resolve()
    if not path.exists():
        return {"path": path.as_posix(), "exists": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    target_model = payload.get("target_model") or {}
    return {
        "path": path.as_posix(),
        "exists": True,
        "device_name": payload.get("device_name"),
        "qairt_version": payload.get("qairt_version"),
        "compile_options": payload.get("compile_options"),
        "target_model_id": payload.get("target_model_id") or target_model.get("model_id"),
        "target_model_url": target_model.get("url"),
    }


def _resolve_vpcd_canonical_index(samples: Sequence[Mapping[str, Any]]) -> int:
    for index, sample in enumerate(samples):
        normalized = normalize_whitespace_text(str(sample.get("raw_text", ""))).lower()
        if normalized.startswith("chào các bạn hôm nay"):
            return index
    return 0


def _resolve_audio_path(raw_path: str, *, bundle_root: Path, repo_root: str | Path | None) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    if repo_root is not None:
        resolved = Path(repo_root).resolve() / raw_path
        if resolved.exists():
            return resolved
    for parent in (bundle_root, *bundle_root.parents):
        resolved = parent / raw_path
        if resolved.exists():
            return resolved
    raise FileNotFoundError(f"Could not resolve audio fixture path: {raw_path}")


def _detect_vpcd_critical_regressions(*, expected_text: str, actual_text: str) -> list[str]:
    regressions: list[str] = []
    if _terminal_punctuation(expected_text) != _terminal_punctuation(actual_text):
        regressions.append("sentence_final_punctuation")
    if _extract_number_like_tokens(expected_text) != _extract_number_like_tokens(actual_text):
        regressions.append("date_number_formatting")
    if _extract_titlecase_tokens(expected_text) != _extract_titlecase_tokens(actual_text):
        regressions.append("proper_name_capitalization")
    return regressions


def _detect_zipformer_critical_regressions(*, expected_text: str, actual_text: str) -> list[str]:
    regressions: list[str] = []
    if _extract_number_like_tokens(expected_text) != _extract_number_like_tokens(actual_text):
        regressions.append("dates_numbers")
    expected_tokens = _word_tokens(expected_text)
    actual_tokens = _word_tokens(actual_text)
    if expected_tokens != actual_tokens:
        if len(expected_tokens) != len(actual_tokens) or len(set(expected_tokens)) != len(set(actual_tokens)):
            regressions.append("repeated_or_dropped_words")
        elif _extract_titlecase_tokens(expected_text) != _extract_titlecase_tokens(actual_text):
            regressions.append("names")
    return regressions


def _extract_number_like_tokens(text: str) -> list[str]:
    return NUMBER_PATTERN.findall(text)


def _extract_titlecase_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for token in WORD_PATTERN.findall(text):
        if any(char.isalpha() for char in token) and token[:1].isupper() and token.lower() != token.upper():
            tokens.append(token)
    return tokens


def _terminal_punctuation(text: str) -> str:
    stripped = text.rstrip()
    if not stripped:
        return ""
    last = stripped[-1]
    return last if last in TERMINAL_PUNCTUATION else ""


def _word_tokens(text: str) -> list[str]:
    return WORD_PATTERN.findall(text)


def _levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            substitution_cost = 0 if left_char == right_char else 1
            current.append(
                min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def _slugify_candidate_label(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "-", str(value).strip()).strip("-").lower()
    return slug or "phase7-candidate"
