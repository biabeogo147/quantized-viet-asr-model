from __future__ import annotations

from model_pipeline.models import get_recipe
from model_pipeline.models.vpcd import classify_vpcd_matmul_name
from model_pipeline.models.zipformer import (
    BOOLEAN_MASK_SLICE_NODES,
    BOOLEAN_MASK_UNSQUEEZE_NODES,
    ZIPFORMER_GRAPH_CONTRACT,
)


def test_zipformer_production_records_no_local_quantization() -> None:
    """Verify Zipformer production explicitly skips local quantization.

    Returns:
        None.
    """
    recipe = get_recipe("zipformer", "production")

    assert recipe.artifact.artifact_id == (
        "zipformer__q-none-fp32-fp32-none__s-enc1x2009x80-dec1x2-join1x512"
        "__c-aihub-qnn-htp-encoder"
    )
    assert recipe.parameters["prepare_scope"] == "encoder"
    assert recipe.parameters["compile_scope"] == "encoder"
    assert recipe.parameters["quantize_action"] == "explicit-skip"
    assert recipe.parameters["execution_targets"] == {
        "encoder": "qnn-htp",
        "decoder": "cpu",
        "joiner": "cpu",
        "tokens": "cpu",
    }


def test_zipformer_graph_contract_preserves_observed_truth() -> None:
    """Verify Zipformer graph constants match retained model evidence.

    Returns:
        None.
    """
    assert ZIPFORMER_GRAPH_CONTRACT.matmul_by_component == {
        "encoder": 278,
        "decoder": 0,
        "joiner": 0,
    }
    assert len(BOOLEAN_MASK_SLICE_NODES) == 3
    assert len(BOOLEAN_MASK_UNSQUEEZE_NODES) == 3


def test_vpcd_production_is_only_aimet_encoder_matmul_a4() -> None:
    """Verify VPCD production remains the single A4 encoder-MatMul recipe.

    Returns:
        None.
    """
    recipe = get_recipe("vpcd", "production")

    assert recipe.artifact.artifact_id == (
        "vpcd__q-aimet-int8-int16-encoder-matmul__s-src1x384-dec1x64"
        "__c-aihub-qnn-htp-model"
    )
    assert recipe.parameters["quantization_engine"] == "aimet-onnx"
    assert recipe.parameters["quant_scheme"] == "min-max"
    assert recipe.parameters["per_channel"] is False
    assert recipe.parameters["op_types"] == ["MatMul"]
    assert recipe.parameters["matmul_contract"] == {
        "encoder": 96,
        "decoder": 168,
        "lm_head": 1,
        "quantized": 96,
    }
    assert recipe.parameters["execution_targets"] == {
        "model": "qnn-htp",
        "tokenizer_encode": "cpu",
        "tokenizer_decode": "cpu",
        "autoregressive_loop": "cpu",
    }
    assert recipe.parameters["truncate_64bit_io"] is True


def test_vpcd_matmul_classifier_has_no_policy_lanes() -> None:
    """Verify MatMul classification exposes scopes rather than rollout policies.

    Returns:
        None.
    """
    assert classify_vpcd_matmul_name("/encoder/layer.0/self_attn/q_proj/MatMul") == "encoder"
    assert classify_vpcd_matmul_name("/decoder/layer.0/self_attn/q_proj/MatMul") == "decoder"
    assert classify_vpcd_matmul_name("/lm_head/MatMul") == "lm_head"
    assert classify_vpcd_matmul_name("/unrelated/MatMul") == "other"


def test_fp32_profiles_are_explicit_controls() -> None:
    """Verify both FP32 profiles explicitly skip quantization and compilation.

    Returns:
        None.
    """
    for model in ("zipformer", "vpcd"):
        recipe = get_recipe(model, "fp32")
        assert recipe.artifact.quantization.engine == "none"
        assert recipe.artifact.compilation.compiler == "none"
        assert recipe.parameters["quantize_action"] == "explicit-skip"
