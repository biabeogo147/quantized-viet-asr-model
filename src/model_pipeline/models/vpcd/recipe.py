from __future__ import annotations

from model_pipeline.core import ArtifactSpec, CompileSpec, QuantizationSpec, RecipeSpec


SHAPE_SLUG = "src1x384-dec1x64"
COMPONENTS = ("model", "tokenizer_encode", "tokenizer_decode", "autoregressive_loop")


def vpcd_recipe(configuration: str) -> RecipeSpec:
    """Build a fixed-shape VPCD configuration.

    Args:
        configuration: Descriptive precision, quantizer, scope, and compile selection.

    Returns:
        The validated fixed-shape VPCD recipe.

    Raises:
        ValueError: If the configuration is unsupported.
    """
    if configuration not in {"fp32-fixed-shape", "aimet-int8-int16-encoder-matmul"}:
        raise ValueError(f"Unsupported VPCD configuration: {configuration!r}")
    requires_quantization = configuration == "aimet-int8-int16-encoder-matmul"
    requires_aihub_compile = requires_quantization
    quantization = (
        QuantizationSpec("aimet", "int8", "int16", "encoder-matmul")
        if requires_quantization
        else QuantizationSpec("none", "fp32", "fp32", "none")
    )
    compilation = (
        CompileSpec("aihub", "qnn-htp", "model")
        if requires_aihub_compile
        else CompileSpec("none", "cpu", "none")
    )
    return RecipeSpec(
        artifact=ArtifactSpec("vpcd", quantization, SHAPE_SLUG, compilation),
        configuration=configuration,
        components=COMPONENTS,
        parameters={
            "fixed_input_shapes": {
                "input_ids": [1, 384],
                "attention_mask": [1, 384],
                "decoder_input_ids": [1, 64],
                "decoder_attention_mask": [1, 64],
            },
            "quantize_action": "aimet" if requires_quantization else "explicit-skip",
            "quantization_engine": "aimet-onnx" if requires_quantization else "none",
            "weight_dtype": "int8" if requires_quantization else "fp32",
            "activation_dtype": "int16" if requires_quantization else "fp32",
            "quant_scheme": "min-max" if requires_quantization else "none",
            "per_channel": False,
            "op_types": ["MatMul"] if requires_quantization else [],
            "matmul_contract": {"encoder": 96, "decoder": 168, "lm_head": 1, "quantized": 96},
            "truncate_64bit_io": requires_aihub_compile,
            "execution_targets": {
                "model": "qnn-htp" if requires_aihub_compile else "cpu",
                "tokenizer_encode": "cpu",
                "tokenizer_decode": "cpu",
                "autoregressive_loop": "cpu",
            },
            "runtime_metadata": {
                "model_family": "bartpho-seq2seq",
                "model_name": "tourmii/vietnamese-punc-cap-denorm-v1",
                "configuration": configuration,
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
