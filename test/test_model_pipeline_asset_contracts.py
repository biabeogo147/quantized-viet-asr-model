from __future__ import annotations

from pathlib import Path

import pytest

from model_pipeline.core import sha256_file
from model_pipeline.models.vpcd import inspect_vpcd_matmuls
from model_pipeline.models.zipformer import ZIPFORMER_GRAPH_CONTRACT
from model_pipeline.models.zipformer.graph import graph_matmul_count


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.asset_contract
def test_zipformer_source_bytes_and_graph_inventory() -> None:
    """Verify Zipformer source checksums and observed MatMul counts.

    Returns:
        None.
    """
    model_dir = ROOT / "assets" / "zipformer"
    expected = {
        "encoder": ("encoder-epoch-20-avg-1.onnx", "b0daa9842a1f39d146e57d6e951edc8910ddd234cbb00e9b5015a5280a5ba221"),
        "decoder": ("decoder-epoch-20-avg-1.onnx", "cf2aa385b82c9d5d40cd29c3188af52d0249b3b78f0d4b7eb84ad502d50c7e7f"),
        "joiner": ("joiner-epoch-20-avg-1.onnx", "d861afe55f7ff43c90069cad0a5d07261a408be5c7fd2aac8c84b1f3225da021"),
    }
    if not model_dir.is_dir():
        pytest.skip("Zipformer model assets are not materialized")

    counts = {}
    for role, (file_name, checksum) in expected.items():
        path = model_dir / file_name
        assert sha256_file(path) == checksum
        counts[role] = graph_matmul_count(path)

    assert counts == ZIPFORMER_GRAPH_CONTRACT.matmul_by_component


@pytest.mark.asset_contract
def test_vpcd_source_bytes_and_96_168_1_inventory() -> None:
    """Verify VPCD source checksum and canonical MatMul scope inventory.

    Returns:
        None.
    """
    path = ROOT / "assets" / "vietnamese-punc-cap-denorm-v1" / "onnx" / "model.fp32.onnx"
    if not path.is_file():
        pytest.skip("VPCD model asset is not materialized")

    assert sha256_file(path) == "e9b517c185501d56d0b32c9a2a35e6e736de82e12f4457e586a75e0a7adbb8b4"
    inventory = inspect_vpcd_matmuls(path)

    assert inventory.counts == {
        "encoder": 96,
        "decoder": 168,
        "lm_head": 1,
        "other": 0,
        "total": 265,
    }
    assert len(inventory.quantized_names) == 96
