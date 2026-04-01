import argparse
from pathlib import Path

from android_bundle.exporter import export_android_bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the Android punctuation bundle for vietnamese-punc-cap-denorm-v1."
    )
    parser.add_argument(
        "--model-dir",
        default="assets/vietnamese-punc-cap-denorm-v1",
        help="Model directory containing tokenizer/config files and ONNX variants.",
    )
    parser.add_argument(
        "--output-dir",
        default="build/android_bundle/vpcd",
        help="Output directory for the standardized Android bundle.",
    )
    parser.add_argument(
        "--model-variant",
        default="vpcd_balanced",
        help="Model variant name without extension, or an explicit .onnx filename stem.",
    )
    parser.add_argument(
        "--asset-namespace",
        default="models/punctuation/vpcd",
        help="Namespace used by Android asset delivery.",
    )
    parser.add_argument(
        "--max-decode-length",
        type=int,
        default=128,
        help="Maximum decode length used for golden sample generation.",
    )
    args = parser.parse_args()

    manifest = export_android_bundle(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        model_variant=args.model_variant,
        asset_namespace=args.asset_namespace,
        max_decode_length=args.max_decode_length,
    )

    output_dir = Path(args.output_dir).resolve()
    print("Android bundle exported.")
    print("Output dir :", output_dir)
    print("Model file :", output_dir / manifest.to_dict()["model_file"])
    print("Manifest   :", output_dir / "bundle_manifest.json")


if __name__ == "__main__":
    main()
