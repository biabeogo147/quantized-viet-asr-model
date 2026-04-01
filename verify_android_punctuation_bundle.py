import argparse
from pathlib import Path

from android_bundle.verifier import verify_exported_tokenizer_bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify exported Android tokenizer bundle parity for vietnamese-punc-cap-denorm-v1."
    )
    parser.add_argument(
        "--model-dir",
        default="assets/vietnamese-punc-cap-denorm-v1",
        help="Model directory containing the original Hugging Face tokenizer files.",
    )
    parser.add_argument(
        "--bundle-dir",
        default="build/android_bundle/vpcd",
        help="Directory containing bundle_manifest.json and exported tokenizer assets.",
    )
    args = parser.parse_args()

    encode_verified, decode_verified = verify_exported_tokenizer_bundle(
        model_dir=args.model_dir,
        bundle_dir=args.bundle_dir,
    )

    bundle_dir = Path(args.bundle_dir).resolve()
    print("Tokenizer bundle verification passed.")
    print("Bundle dir      :", bundle_dir)
    print("Encode samples  :", encode_verified)
    print("Decode samples  :", decode_verified)


if __name__ == "__main__":
    main()
