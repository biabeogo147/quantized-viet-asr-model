from __future__ import annotations

import argparse
from pathlib import Path

import sentencepiece as spm


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate tokens.txt from a SentencePiece model.")
    parser.add_argument(
        "--bpe-model",
        default="assets/zipformer/bpe.model",
        help="Path to the SentencePiece model file.",
    )
    parser.add_argument(
        "--output",
        default="assets/zipformer/tokens.txt",
        help="Path to the generated tokens.txt file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    bpe_model_path = Path(args.bpe_model)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sp = spm.SentencePieceProcessor(model_file=str(bpe_model_path))
    with output_path.open("w", encoding="utf-8") as handle:
        for index in range(sp.get_piece_size()):
            handle.write(f"{sp.id_to_piece(index)} {index}\n")

    print(f"Wrote {sp.get_piece_size()} tokens to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
