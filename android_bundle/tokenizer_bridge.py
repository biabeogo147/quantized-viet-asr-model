from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TokenizerIdBridge:
    tokenizer_to_model_ids: list[int]
    model_to_tokenizer_ids: list[int]

    def write_files(
        self,
        *,
        tokenizer_to_model_path: str | Path,
        model_to_tokenizer_path: str | Path,
    ) -> tuple[str, str]:
        tokenizer_to_model_file = Path(tokenizer_to_model_path)
        model_to_tokenizer_file = Path(model_to_tokenizer_path)
        tokenizer_to_model_file.write_text(
            json.dumps(self.tokenizer_to_model_ids, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        model_to_tokenizer_file.write_text(
            json.dumps(self.model_to_tokenizer_ids, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        return (tokenizer_to_model_file.name, model_to_tokenizer_file.name)


def build_ort_tokenizer_id_bridge(tokenizer: object) -> TokenizerIdBridge:
    sp_model = tokenizer.sp_model
    tokenizer_to_model_ids = [tokenizer.unk_token_id] * (sp_model.get_piece_size() + 1)
    tokenizer_to_model_ids[0] = tokenizer.cls_token_id
    tokenizer_to_model_ids[1] = tokenizer.pad_token_id
    tokenizer_to_model_ids[2] = tokenizer.sep_token_id
    tokenizer_to_model_ids[3] = tokenizer.unk_token_id

    special_model_ids = {
        tokenizer.cls_token_id,
        tokenizer.pad_token_id,
        tokenizer.sep_token_id,
        tokenizer.unk_token_id,
    }

    for token, model_id in tokenizer.fairseq_tokens_to_ids.items():
        if model_id in special_model_ids or token in tokenizer.all_special_tokens:
            continue
        sp_id = sp_model.piece_to_id(token)
        if sp_id >= 0:
            tokenizer_to_model_ids[sp_id + 1] = model_id

    model_to_tokenizer_ids = [tokenizer.unk_token_id] * len(tokenizer.fairseq_tokens_to_ids)
    model_to_tokenizer_ids[tokenizer.cls_token_id] = 0
    model_to_tokenizer_ids[tokenizer.pad_token_id] = 1
    model_to_tokenizer_ids[tokenizer.sep_token_id] = 2
    model_to_tokenizer_ids[tokenizer.unk_token_id] = 3

    for token, model_id in tokenizer.fairseq_tokens_to_ids.items():
        if model_id in special_model_ids:
            continue
        sp_id = sp_model.piece_to_id(token)
        if sp_id >= 0:
            model_to_tokenizer_ids[model_id] = sp_id + 1

    return TokenizerIdBridge(
        tokenizer_to_model_ids=tokenizer_to_model_ids,
        model_to_tokenizer_ids=model_to_tokenizer_ids,
    )
