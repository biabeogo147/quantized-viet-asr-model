from __future__ import annotations

from model_pipeline.core import ArtifactSpec, CompileSpec, QuantizationSpec, RecipeSpec


SHAPE_SLUG = "src1x384-dec1x64"
COMPONENTS = ("model", "tokenizer_encode", "tokenizer_decode", "autoregressive_loop")


def vpcd_recipe(profile: str) -> RecipeSpec:
    """Build the canonical fixed-shape VPCD control or production recipe.

    Args:
        profile: Either `fp32` control or `production` AIMET/AI Hub execution.

    Returns:
        The validated A4 VPCD recipe.

    Raises:
        ValueError: If the profile is unsupported.
    """
    if profile not in {"fp32", "production"}:
        raise ValueError(f"Unsupported VPCD profile: {profile!r}")
    production = profile == "production"
    quantization = (
        QuantizationSpec("aimet", "int8", "int16", "encoder-matmul")
        if production
        else QuantizationSpec("none", "fp32", "fp32", "none")
    )
    compilation = (
        CompileSpec("aihub", "qnn-htp", "model")
        if production
        else CompileSpec("none", "cpu", "none")
    )
    return RecipeSpec(
        artifact=ArtifactSpec("vpcd", quantization, SHAPE_SLUG, compilation),
        profile=profile,
        components=COMPONENTS,
        parameters={
            "fixed_input_shapes": {
                "input_ids": [1, 384],
                "attention_mask": [1, 384],
                "decoder_input_ids": [1, 64],
                "decoder_attention_mask": [1, 64],
            },
            "quantize_action": "aimet" if production else "explicit-skip",
            "quantization_engine": "aimet-onnx" if production else "none",
            "weight_dtype": "int8" if production else "fp32",
            "activation_dtype": "int16" if production else "fp32",
            "quant_scheme": "min-max" if production else "none",
            "per_channel": False,
            "op_types": ["MatMul"] if production else [],
            "matmul_contract": {"encoder": 96, "decoder": 168, "lm_head": 1, "quantized": 96},
            "truncate_64bit_io": production,
            "execution_targets": {
                "model": "qnn-htp" if production else "cpu",
                "tokenizer_encode": "cpu",
                "tokenizer_decode": "cpu",
                "autoregressive_loop": "cpu",
            },
            "runtime_metadata": {
                "model_family": "bartpho-seq2seq",
                "model_name": "tourmii/vietnamese-punc-cap-denorm-v1",
                "model_variant": profile,
                "runtime_kind": "text-seq2seq",
                "pad_token_id": 1,
                "eos_token_id": 2,
                "decoder_start_token_id": 2,
                "max_source_length": 384,
                "max_decode_length": 64,
                "input_text_case": "lower",
                "fixed_input_shapes": {
                    "model": {
                        "input_ids": [1, 384],
                        "attention_mask": [1, 384],
                        "decoder_input_ids": [1, 64],
                        "decoder_attention_mask": [1, 64],
                    }
                },
            },
        },
    )
