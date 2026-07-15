from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from model_pipeline.models.vpcd.quantization import (
    CalibrationBatch,
    build_encoder_matmul_policy,
    build_matmul_only_aimet_config,
    load_calibration_batches,
    pad_calibration_batch,
    write_calibration_batches,
)


def test_matmul_only_aimet_config_disables_per_channel_and_bias() -> None:
    """Verify the AIMET config enables only canonical tensor quantization behavior.

    Returns:
        None.
    """
    config = build_matmul_only_aimet_config()

    assert list(config["op_type"]) == ["MatMul"]
    assert config["defaults"]["per_channel_quantization"] == "False"
    assert config["params"]["bias"]["is_quantized"] == "False"
    assert config["op_type"]["MatMul"]["params"]["weight"]["is_quantized"] == "True"


def test_encoder_policy_excludes_decoder_and_lm_head_matmuls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify encoder policy disables every decoder and language-head MatMul.

    Args:
        monkeypatch: Pytest fixture replacing graph inventory with canonical test data.

    Returns:
        None.
    """
    from model_pipeline.models.vpcd import quantization
    from model_pipeline.models.vpcd.graph import VpcdMatmulInventory

    inventory = VpcdMatmulInventory(
        encoder=("/encoder/a/MatMul", "/encoder/b/MatMul"),
        decoder=("/decoder/a/MatMul",),
        lm_head=("/lm_head/MatMul",),
        other=(),
    )
    monkeypatch.setattr(quantization, "inspect_vpcd_matmuls", lambda _path: inventory)

    policy = build_encoder_matmul_policy("model.onnx", require_canonical_counts=False)

    assert policy["quantize_op_names"] == ["/encoder/a/MatMul", "/encoder/b/MatMul"]
    assert policy["disable_op_names"] == ["/decoder/a/MatMul", "/lm_head/MatMul"]
    assert policy["coverage"] == {"quantized": 2, "total_matmul": 4}
    assert "policy_mode" not in policy


def test_encoder_policy_rejects_noncanonical_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify policy creation fails when graph coverage differs from 96/168/1.

    Args:
        monkeypatch: Pytest fixture replacing graph inventory with invalid test data.

    Returns:
        None.
    """
    from model_pipeline.models.vpcd import quantization
    from model_pipeline.models.vpcd.graph import VpcdMatmulInventory

    monkeypatch.setattr(
        quantization,
        "inspect_vpcd_matmuls",
        lambda _path: VpcdMatmulInventory(("encoder",), (), (), ()),
    )

    with pytest.raises(ValueError, match="96/168/1"):
        build_encoder_matmul_policy("model.onnx")


def test_calibration_is_padded_to_a4_and_round_trips(tmp_path: Path) -> None:
    """Verify A4 padding and calibration serialization preserve ordered arrays.

    Args:
        tmp_path: Isolated directory for calibration package files.

    Returns:
        None.
    """
    batch = CalibrationBatch(
        inputs={
            "input_ids": np.asarray([[5, 6]], dtype=np.int64),
            "attention_mask": np.asarray([[1, 1]], dtype=np.int64),
            "decoder_input_ids": np.asarray([[0]], dtype=np.int64),
            "decoder_attention_mask": np.asarray([[1]], dtype=np.int64),
        }
    )

    padded = pad_calibration_batch(batch, pad_token_id=1)
    manifest_path = write_calibration_batches([padded], tmp_path)
    restored = load_calibration_batches(tmp_path)

    assert padded.inputs["input_ids"].shape == (1, 384)
    assert padded.inputs["decoder_input_ids"].shape == (1, 64)
    assert padded.inputs["input_ids"][0, 2] == 1
    assert padded.inputs["attention_mask"][0, 2] == 0
    assert np.array_equal(restored[0]["input_ids"], padded.inputs["input_ids"])
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["input_order"] == list(batch.inputs)
