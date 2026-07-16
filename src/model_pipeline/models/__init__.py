from __future__ import annotations

from model_pipeline.core import RecipeSpec
from model_pipeline.models.vpcd import vpcd_recipe
from model_pipeline.models.zipformer import zipformer_recipe


def get_recipe(model: str, configuration: str) -> RecipeSpec:
    """Resolve the canonical recipe for a supported model configuration.

    Args:
        model: Canonical model family name.
        configuration: Requested descriptive model configuration.

    Returns:
        The validated model recipe.

    Raises:
        ValueError: If the model or configuration is unsupported.
    """
    factories = {"zipformer": zipformer_recipe, "vpcd": vpcd_recipe}
    try:
        factory = factories[model]
    except KeyError as exc:
        raise ValueError(f"Unsupported model: {model!r}") from exc
    return factory(configuration)


__all__ = ["get_recipe"]
