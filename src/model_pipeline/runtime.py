from __future__ import annotations

import json
import os
from pathlib import Path

from model_pipeline.core import RecipeSpec, Stage
from model_pipeline.integrations.aihub import EvidenceStore, QualcommAiHubClient
from model_pipeline.models.vpcd.adapter import VpcdAdapter
from model_pipeline.models.zipformer.adapter import ZipformerAdapter
from model_pipeline.pipeline import ModelPipeline


def run_pipeline(
    *,
    recipe: RecipeSpec,
    repo_root: Path,
    build_root: Path,
    through: str,
    android_destination: Path | None,
    device: str | None,
) -> int:
    """Resolve concrete adapters and execute a pipeline from CLI inputs.

    Args:
        recipe: Canonical model/profile recipe.
        repo_root: Repository root used to resolve source assets.
        build_root: Absolute or repository-relative pipeline output directory.
        through: Final pipeline stage requested by the caller.
        android_destination: Optional Android bundle synchronization directory.
        device: Optional Qualcomm AI Hub device name required for compilation.

    Returns:
        Zero after the pipeline result has been printed as JSON.

    Raises:
        ValueError: If AI Hub compilation is requested without a device.
    """
    resolved_repo = repo_root.resolve()
    resolved_build = (resolved_repo / build_root).resolve() if not build_root.is_absolute() else build_root.resolve()
    adapter = (
        ZipformerAdapter(resolved_repo)
        if recipe.artifact.model == "zipformer"
        else VpcdAdapter(resolved_repo)
    )
    needs_aihub = (
        recipe.artifact.compilation.compiler == "aihub"
        and Stage.ordered().index(Stage(through)) >= Stage.ordered().index(Stage.COMPILE)
    )
    client = None
    if needs_aihub:
        if not device:
            raise ValueError("--device is required when running through the AI Hub compile stage")
        client = QualcommAiHubClient(
            device_name=device,
            api_token=os.environ.get("QAI_HUB_API_TOKEN"),
            qairt_version=os.environ.get("QAIRT_VERSION"),
        )
        client.authenticate()
    pipeline = ModelPipeline(
        build_root=resolved_build,
        evidence_store=EvidenceStore(resolved_build / "aihub-evidence"),
        aihub_client=client,
    )
    result = pipeline.run(
        recipe=recipe,
        adapter=adapter,
        through=through,
        android_destination=android_destination,
    )
    print(
        json.dumps(
            {
                "artifact_id": recipe.artifact.artifact_id,
                "validation": result.validation.status,
                "bundle": result.bundle.bundle_dir.as_posix() if result.bundle else None,
                "resumed_stages": list(result.resumed_stages),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
