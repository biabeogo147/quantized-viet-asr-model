from __future__ import annotations

from model_pipeline.core import ArtifactSpec, CompileSpec, QuantizationSpec, RecipeSpec


SHAPE_SLUG = "enc1x2009x80-dec1x2-join1x512"
COMPONENTS = ("encoder", "decoder", "joiner", "tokens")


def zipformer_recipe(profile: str) -> RecipeSpec:
    """Build the canonical Zipformer control or production recipe.

    Args:
        profile: Either `fp32` control or `production` encoder compilation.

    Returns:
        The validated fixed-shape Zipformer recipe.

    Raises:
        ValueError: If the profile is unsupported.
    """
    if profile not in {"fp32", "production"}:
        raise ValueError(f"Unsupported Zipformer profile: {profile!r}")
    production = profile == "production"
    compilation = (
        CompileSpec("aihub", "qnn-htp", "encoder")
        if production
        else CompileSpec("none", "cpu", "none")
    )
    artifact = ArtifactSpec(
        model="zipformer",
        quantization=QuantizationSpec("none", "fp32", "fp32", "none"),
        shape=SHAPE_SLUG,
        compilation=compilation,
    )
    return RecipeSpec(
        artifact=artifact,
        profile=profile,
        components=COMPONENTS,
        parameters={
            "fixed_input_shapes": {
                "encoder": {"x": [1, 2009, 80], "x_lens": [1]},
                "decoder": {"y": [1, 2]},
                "joiner": {"encoder_out": [1, 512], "decoder_out": [1, 512]},
            },
            "prepare_scope": "encoder" if production else "all-components",
            "quantize_action": "explicit-skip",
            "compile_scope": "encoder" if production else "none",
            "boolean_mask_rewrite": {"slice_count": 3, "unsqueeze_count": 3},
            "matmul_contract": {"encoder": 278, "decoder": 0, "joiner": 0},
            "execution_targets": {
                "encoder": "qnn-htp" if production else "cpu",
                "decoder": "cpu",
                "joiner": "cpu",
                "tokens": "cpu",
            },
            "runtime_metadata": {
                "model_family": "zipformer-rnnt",
                "model_name": "zipformer",
                "model_variant": profile,
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
