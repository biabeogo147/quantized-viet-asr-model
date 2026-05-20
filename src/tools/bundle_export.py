from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from quantize.vpcd_bundle import (
    DEFAULT_ASSET_NAMESPACE as VPCD_DEFAULT_ASSET_NAMESPACE,
    DEFAULT_MODEL_DIR as VPCD_DEFAULT_MODEL_DIR,
    DEFAULT_MODEL_VARIANT as VPCD_DEFAULT_MODEL_VARIANT,
    DEFAULT_OUTPUT_DIR as VPCD_DEFAULT_OUTPUT_DIR,
    export_bundle as VPCD_EXPORT_BUNDLE,
)
from quantize.zipformer_bundle import (
    DEFAULT_ASSET_NAMESPACE as ZIPFORMER_DEFAULT_ASSET_NAMESPACE,
    DEFAULT_MODEL_DIR as ZIPFORMER_DEFAULT_MODEL_DIR,
    DEFAULT_VARIANT as ZIPFORMER_DEFAULT_MODEL_VARIANT,
    DEFAULT_OUTPUT_DIR as ZIPFORMER_DEFAULT_OUTPUT_DIR,
    export_bundle as ZIPFORMER_EXPORT_BUNDLE,
)


def _defaults_for(project: str) -> dict[str, object]:
    if project == "vpcd":
        return {
            "model_dir": VPCD_DEFAULT_MODEL_DIR,
            "output_dir": VPCD_DEFAULT_OUTPUT_DIR,
            "asset_namespace": VPCD_DEFAULT_ASSET_NAMESPACE,
            "model_variant": VPCD_DEFAULT_MODEL_VARIANT,
        }
    if project == "zipformer":
        return {
            "model_dir": ZIPFORMER_DEFAULT_MODEL_DIR,
            "output_dir": ZIPFORMER_DEFAULT_OUTPUT_DIR,
            "asset_namespace": ZIPFORMER_DEFAULT_ASSET_NAMESPACE,
            "model_variant": ZIPFORMER_DEFAULT_MODEL_VARIANT,
        }
    raise ValueError(f"Unsupported project: {project}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a shared model bundle without the retired export package.")
    parser.add_argument("--project", choices=("vpcd", "zipformer"), required=True)
    parser.add_argument("--model-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--asset-namespace")
    parser.add_argument("--model-variant")
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--max-decode-length", type=int, default=128)
    return parser


def export_model_bundle(
    *,
    project: str,
    model_dir: str | Path,
    output_dir: str | Path,
    asset_namespace: str,
    model_variant: str,
    provider: str = "CPUExecutionProvider",
    max_decode_length: int = 128,
):
    resolved_model_dir = Path(model_dir)
    resolved_output_dir = Path(output_dir)

    if project == "vpcd":
        return VPCD_EXPORT_BUNDLE(
            model_dir=resolved_model_dir,
            output_dir=resolved_output_dir,
            model_variant=model_variant,
            asset_namespace=asset_namespace,
            max_decode_length=max_decode_length,
        )

    if project == "zipformer":
        return ZIPFORMER_EXPORT_BUNDLE(
            model_dir=resolved_model_dir,
            output_dir=resolved_output_dir,
            asset_namespace=asset_namespace,
            provider=provider,
            model_variant=model_variant,
        )

    raise ValueError(f"Unsupported project: {project}")


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argument_parser().parse_args(argv)
    defaults = _defaults_for(args.project)
    model_dir = Path(args.model_dir) if args.model_dir else Path(defaults["model_dir"])
    output_dir = Path(args.output_dir) if args.output_dir else Path(defaults["output_dir"])
    asset_namespace = args.asset_namespace or str(defaults["asset_namespace"])
    model_variant = args.model_variant or str(defaults["model_variant"])

    manifest = export_model_bundle(
        project=args.project,
        model_dir=model_dir,
        output_dir=output_dir,
        asset_namespace=asset_namespace,
        model_variant=model_variant,
        provider=args.provider,
        max_decode_length=args.max_decode_length,
    )

    print("Bundle export complete.")
    print("Project        :", args.project)
    print("Model dir      :", model_dir)
    print("Output dir     :", output_dir)
    print("Manifest       :", output_dir / "bundle_manifest.json")
    print("Asset namespace:", manifest.asset_namespace)
    print("Variant        :", manifest.model_variant)


if __name__ == "__main__":
    main()
