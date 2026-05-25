from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from aihub.phase7 import materialize_vpcd_local_aimet_candidate_bundle
from model_bundle.manifest import ModelBundleManifest


@dataclass(frozen=True)
class Phase8VpcdLaneSpec:
    candidate_label: str
    encoder_sequence: int
    decoder_sequence: int
    policy_mode: str


def list_phase8_vpcd_lane_specs() -> tuple[Phase8VpcdLaneSpec, ...]:
    return (
        Phase8VpcdLaneSpec("VPCD-A0-control-1024x128-L2", 1024, 128, "decoder_expanded"),
        Phase8VpcdLaneSpec("VPCD-A1-512x64-L2", 512, 64, "decoder_expanded"),
        Phase8VpcdLaneSpec("VPCD-A2-384x64-L2", 384, 64, "decoder_expanded"),
        Phase8VpcdLaneSpec("VPCD-A3-256x48-L2", 256, 48, "decoder_expanded"),
        Phase8VpcdLaneSpec("VPCD-A4-384x64-L1", 384, 64, "local_quality_parity"),
    )


def resolve_phase8_vpcd_lane_spec(candidate_label: str) -> Phase8VpcdLaneSpec:
    normalized = str(candidate_label).strip().lower()
    for spec in list_phase8_vpcd_lane_specs():
        if spec.candidate_label.lower() == normalized:
            return spec
    raise KeyError(f"Unknown Phase 8 VPCD lane: {candidate_label!r}")


def write_phase8_vpcd_override_manifest(
    *,
    control_manifest_path: str | Path,
    lane_spec: Phase8VpcdLaneSpec,
    output_dir: str | Path,
) -> Path:
    control_path = Path(control_manifest_path).resolve()
    control_manifest = ModelBundleManifest.from_path(control_path)
    if control_manifest.project != "vpcd":
        raise ValueError(f"Expected a VPCD control manifest, got: {control_manifest.project!r}")

    metadata = dict(control_manifest.metadata)
    metadata["max_source_length"] = int(lane_spec.encoder_sequence)
    metadata["max_decode_length"] = int(lane_spec.decoder_sequence)
    metadata["fixed_input_shapes"] = {
        "model": {
            "input_ids": [1, int(lane_spec.encoder_sequence)],
            "attention_mask": [1, int(lane_spec.encoder_sequence)],
            "decoder_input_ids": [1, int(lane_spec.decoder_sequence)],
            "decoder_attention_mask": [1, int(lane_spec.decoder_sequence)],
        }
    }
    metadata["phase8_candidate"] = {
        "candidate_label": lane_spec.candidate_label,
        "encoder_sequence": int(lane_spec.encoder_sequence),
        "decoder_sequence": int(lane_spec.decoder_sequence),
        "policy_mode": lane_spec.policy_mode,
        "control_manifest_path": control_path.as_posix(),
    }

    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{lane_spec.candidate_label.lower().replace('/', '-').replace(' ', '-').replace('_', '-')}.bundle_manifest.json"

    manifest = ModelBundleManifest(
        bundle_version=control_manifest.bundle_version,
        project=control_manifest.project,
        model_family=control_manifest.model_family,
        model_name=control_manifest.model_name,
        model_variant=control_manifest.model_variant,
        asset_namespace=control_manifest.asset_namespace,
        runtime_kind=control_manifest.runtime_kind,
        artifacts=dict(control_manifest.artifacts),
        fixtures=dict(control_manifest.fixtures),
        metadata=metadata,
    )
    manifest.write_json(output_path)
    return output_path


def materialize_phase8_vpcd_candidate_bundle(
    *,
    lane_spec: Phase8VpcdLaneSpec,
    control_bundle_root: str | Path,
    quantize_report_path: str | Path,
    output_root: str | Path,
) -> Path:
    control_bundle = Path(control_bundle_root).resolve()
    control_manifest = ModelBundleManifest.from_path(control_bundle / "bundle_manifest.json")

    output_dir = materialize_vpcd_local_aimet_candidate_bundle(
        candidate_label=lane_spec.candidate_label,
        control_bundle_root=control_bundle,
        quantize_report_path=quantize_report_path,
        output_root=output_root,
    )

    candidate_manifest = ModelBundleManifest.from_path(output_dir / "bundle_manifest.json")
    metadata = dict(candidate_manifest.metadata)
    metadata["max_source_length"] = int(lane_spec.encoder_sequence)
    metadata["max_decode_length"] = int(lane_spec.decoder_sequence)
    metadata["fixed_input_shapes"] = {
        "model": {
            "input_ids": [1, int(lane_spec.encoder_sequence)],
            "attention_mask": [1, int(lane_spec.encoder_sequence)],
            "decoder_input_ids": [1, int(lane_spec.decoder_sequence)],
            "decoder_attention_mask": [1, int(lane_spec.decoder_sequence)],
        }
    }
    metadata["phase8_candidate"] = {
        "candidate_label": lane_spec.candidate_label,
        "encoder_sequence": int(lane_spec.encoder_sequence),
        "decoder_sequence": int(lane_spec.decoder_sequence),
        "policy_mode": lane_spec.policy_mode,
        "control_manifest_path": (control_bundle / "bundle_manifest.json").as_posix(),
    }
    quantization = dict(metadata.get("quantization") or {})
    quantization["phase8_lane"] = lane_spec.candidate_label
    metadata["quantization"] = quantization

    manifest = ModelBundleManifest(
        bundle_version=control_manifest.bundle_version,
        project=control_manifest.project,
        model_family=control_manifest.model_family,
        model_name=control_manifest.model_name,
        model_variant=lane_spec.candidate_label.lower().replace("/", "-").replace(" ", "-").replace("_", "-"),
        asset_namespace=f"{control_manifest.asset_namespace}/phase8/{lane_spec.candidate_label.lower().replace('/', '-').replace(' ', '-').replace('_', '-')}",
        runtime_kind=control_manifest.runtime_kind,
        artifacts=dict(candidate_manifest.artifacts),
        fixtures=dict(candidate_manifest.fixtures),
        metadata=metadata,
    )
    manifest.write_json(output_dir / "bundle_manifest.json")
    return output_dir


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 8 VPCD lane helpers.")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--control-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", action="store_true")
    return parser


def cli(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    spec = resolve_phase8_vpcd_lane_spec(args.candidate)
    output_path = write_phase8_vpcd_override_manifest(
        control_manifest_path=args.control_manifest,
        lane_spec=spec,
        output_dir=args.output_dir,
    )

    if args.json:
        payload = {
            "candidate": asdict(spec),
            "override_manifest_path": output_path.as_posix(),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(output_path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
