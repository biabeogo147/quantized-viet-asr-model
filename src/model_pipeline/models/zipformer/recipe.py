from __future__ import annotations

from model_pipeline.core import ArtifactSpec, CompileSpec, QuantizationSpec, RecipeSpec


SHAPE_SLUG = "enc1x2009x80-dec1x2-join1x512"
COMPONENTS = ("encoder", "decoder", "joiner", "tokens")


def zipformer_recipe(configuration: str) -> RecipeSpec:
    """Build a fixed-shape Zipformer configuration.

    Args:
        configuration: Descriptive precision, quantizer, scope, and compile selection.

    Returns:
        The validated fixed-shape Zipformer recipe.

    Raises:
        ValueError: If the configuration is unsupported.
    """
    supported_configurations = {
        "fp32-fixed-shape",
        "fp32-fixed-shape-aihub-encoder",
        "ortqnn-uint8-uint16-encoder-matmul",
        "aimet-int8-int16-encoder-matmul",
    }
    if configuration not in supported_configurations:
        raise ValueError(f"Unsupported Zipformer configuration: {configuration!r}")
    requires_quantization = configuration in {
        "ortqnn-uint8-uint16-encoder-matmul",
        "aimet-int8-int16-encoder-matmul",
    }
    requires_aihub_compile = configuration != "fp32-fixed-shape"
    if configuration == "ortqnn-uint8-uint16-encoder-matmul":
        quantization = QuantizationSpec("ortqnn", "uint8", "uint16", "encoder-matmul")
        quantize_action = "ortqnn"
    elif configuration == "aimet-int8-int16-encoder-matmul":
        quantization = QuantizationSpec("aimet", "int8", "int16", "encoder-matmul")
        quantize_action = "aimet"
    else:
        quantization = QuantizationSpec("none", "fp32", "fp32", "none")
        quantize_action = "explicit-skip"
    compilation = (
        CompileSpec("aihub", "qnn-htp", "encoder")
        if requires_aihub_compile
        else CompileSpec("none", "cpu", "none")
    )
    artifact = ArtifactSpec(
        model="zipformer",
        quantization=quantization,
        shape=SHAPE_SLUG,
        compilation=compilation,
    )
    return RecipeSpec(
        artifact=artifact,
        configuration=configuration,
        components=COMPONENTS,
        parameters={
            "fixed_input_shapes": {
                "encoder": {"x": [1, 2009, 80], "x_lens": [1]},
                "decoder": {"y": [1, 2]},
                "joiner": {"encoder_out": [1, 512], "decoder_out": [1, 512]},
            },
            "prepare_scope": "encoder" if requires_aihub_compile else "all-components",
            "quantize_action": quantize_action,
            "quantization_engine": quantization.engine,
            "weight_dtype": quantization.weight,
            "activation_dtype": quantization.activation,
            "quant_scheme": "min-max" if requires_quantization else "none",
            "per_channel": False,
            "op_types": ["MatMul"] if requires_quantization else [],
            "compile_scope": "encoder" if requires_aihub_compile else "none",
            "boolean_mask_rewrite": {"slice_count": 3, "unsqueeze_count": 3},
            "matmul_contract": {"encoder": 278, "decoder": 0, "joiner": 0},
            "execution_targets": {
                "encoder": "qnn-htp" if requires_aihub_compile else "cpu",
                "decoder": "cpu",
                "joiner": "cpu",
                "tokens": "cpu",
            },
            "runtime_metadata": {
                "model_family": "zipformer-rnnt",
                "model_name": "zipformer",
                "configuration": configuration,
                "runtime_kind": "rnnt-greedy",
                "sample_rate": 16000,
                "feature_dim": 80,
                "blank_id": 0,
                "context_size": 2,
                "fixed_encoder_frames": 2009,
                "fixed_input_shapes": {
                    "encoder": {"x": [1, 2009, 80], "x_lens": [1]},
                    "decoder": {"y": [1, 2]},
                    "joiner": {"encoder_out": [1, 512], "decoder_out": [1, 512]},
                },
            },
        },
    )
