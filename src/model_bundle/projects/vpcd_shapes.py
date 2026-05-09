from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class VpcdModelInputShapes:
    input_ids: tuple[int, int]
    attention_mask: tuple[int, int]
    decoder_input_ids: tuple[int, int]
    decoder_attention_mask: tuple[int, int]

    @property
    def encoder_sequence(self) -> int:
        return self.input_ids[1]

    @property
    def decoder_sequence(self) -> int:
        return self.decoder_input_ids[1]


def resolve_vpcd_model_input_shapes(metadata: dict[str, Any]) -> VpcdModelInputShapes | None:
    fixed_input_shapes = metadata.get('fixed_input_shapes') if isinstance(metadata, dict) else None
    if not isinstance(fixed_input_shapes, dict):
        return None

    model = fixed_input_shapes.get('model')
    if not isinstance(model, dict):
        return None

    def shape2(name: str) -> tuple[int, int]:
        value = model.get(name)
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f'VPCD fixed shape for {name} must be [batch, sequence]')
        batch, sequence = int(value[0]), int(value[1])
        if batch != 1 or sequence <= 0:
            raise ValueError(f'Unsupported VPCD fixed shape for {name}: {value}')
        return batch, sequence

    shapes = VpcdModelInputShapes(
        input_ids=shape2('input_ids'),
        attention_mask=shape2('attention_mask'),
        decoder_input_ids=shape2('decoder_input_ids'),
        decoder_attention_mask=shape2('decoder_attention_mask'),
    )
    if shapes.input_ids != shapes.attention_mask:
        raise ValueError('VPCD input_ids and attention_mask fixed shapes must match')
    if shapes.decoder_input_ids != shapes.decoder_attention_mask:
        raise ValueError('VPCD decoder_input_ids and decoder_attention_mask fixed shapes must match')
    return shapes


def pad_token_row(values: Sequence[int] | np.ndarray, *, target_length: int, pad_value: int) -> np.ndarray:
    flattened = np.asarray(values, dtype=np.int64).reshape(-1)
    if flattened.size > target_length:
        raise ValueError(f'Input length {flattened.size} exceeds fixed target length {target_length}')

    row = np.full((1, target_length), int(pad_value), dtype=np.int64)
    row[0, : flattened.size] = flattened
    return row


def attention_mask_for_length(*, actual_length: int, target_length: int) -> np.ndarray:
    if actual_length > target_length:
        raise ValueError(f'Input length {actual_length} exceeds fixed target length {target_length}')
    if actual_length < 0:
        raise ValueError('Input length must be non-negative')

    row = np.zeros((1, target_length), dtype=np.int64)
    row[0, :actual_length] = 1
    return row
