from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TokenizerArtifacts:
    encode: Path
    decode: Path
    to_model_ids: Path
    from_model_ids: Path


def export_tokenizer(model_dir: str | Path, output_dir: str | Path) -> TokenizerArtifacts:
    """Export tokenizer encode/decode graphs and bidirectional ID maps.

    Args:
        model_dir: Local Hugging Face tokenizer directory.
        output_dir: Directory receiving ONNX graphs and JSON maps.

    Returns:
        Paths to the four CPU tokenizer artifacts.

    Raises:
        RuntimeError: If ONNX Runtime Extensions does not emit both graphs.
    """
    import onnx
    from onnxruntime_extensions import gen_processing_models
    from transformers import AutoTokenizer

    source = Path(model_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(source.as_posix(), local_files_only=True)
    with _ortx_class_alias(tokenizer):
        graphs = gen_processing_models(
            tokenizer,
            pre_kwargs={"fairseq": True},
            post_kwargs={"fairseq": True},
        )
    if len(graphs) < 2:
        raise RuntimeError("Tokenizer export did not produce encode and decode graphs")
    encode = destination / "tokenizer.encode.onnx"
    decode = destination / "tokenizer.decode.onnx"
    onnx.save_model(graphs[0], encode)
    onnx.save_model(graphs[1], decode)
    tokenizer_to_model, model_to_tokenizer = _build_id_maps(tokenizer)
    to_model = destination / "tokenizer.to-model-ids.json"
    from_model = destination / "tokenizer.from-model-ids.json"
    to_model.write_text(json.dumps(tokenizer_to_model, separators=(",", ":")) + "\n", encoding="utf-8")
    from_model.write_text(json.dumps(model_to_tokenizer, separators=(",", ":")) + "\n", encoding="utf-8")
    return TokenizerArtifacts(encode, decode, to_model, from_model)


@contextmanager
def _ortx_class_alias(tokenizer):
    """Temporarily expose the tokenizer class name expected by ORT Extensions.

    Args:
        tokenizer: Loaded tokenizer whose class name is temporarily changed.

    Yields:
        Control while the tokenizer class uses the compatible alias.
    """
    tokenizer_class = tokenizer.__class__
    original = tokenizer_class.__name__
    tokenizer_class.__name__ = "XLMRobertaTokenizer"
    try:
        yield
    finally:
        tokenizer_class.__name__ = original


def _build_id_maps(tokenizer) -> tuple[list[int], list[int]]:
    """Build reversible SentencePiece-to-fairseq token ID bridge tables.

    Args:
        tokenizer: Fairseq-compatible tokenizer with a SentencePiece model.

    Returns:
        Tokenizer-to-model and model-to-tokenizer ID arrays.
    """
    sentencepiece = tokenizer.sp_model
    tokenizer_to_model = [tokenizer.unk_token_id] * (sentencepiece.get_piece_size() + 1)
    tokenizer_to_model[:4] = [
        tokenizer.cls_token_id,
        tokenizer.pad_token_id,
        tokenizer.sep_token_id,
        tokenizer.unk_token_id,
    ]
    specials = {
        tokenizer.cls_token_id,
        tokenizer.pad_token_id,
        tokenizer.sep_token_id,
        tokenizer.unk_token_id,
    }
    for token, model_id in tokenizer.fairseq_tokens_to_ids.items():
        if model_id not in specials and token not in tokenizer.all_special_tokens:
            piece_id = sentencepiece.piece_to_id(token)
            if piece_id >= 0:
                tokenizer_to_model[piece_id + 1] = model_id
    model_to_tokenizer = [tokenizer.unk_token_id] * len(tokenizer.fairseq_tokens_to_ids)
    model_to_tokenizer[tokenizer.cls_token_id] = 0
    model_to_tokenizer[tokenizer.pad_token_id] = 1
    model_to_tokenizer[tokenizer.sep_token_id] = 2
    model_to_tokenizer[tokenizer.unk_token_id] = 3
    for token, model_id in tokenizer.fairseq_tokens_to_ids.items():
        if model_id not in specials:
            piece_id = sentencepiece.piece_to_id(token)
            if piece_id >= 0:
                model_to_tokenizer[model_id] = piece_id + 1
    return tokenizer_to_model, model_to_tokenizer
