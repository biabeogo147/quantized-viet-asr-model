from __future__ import annotations

from pathlib import Path

from model_bundle.verifier import verify_model_bundle


def evaluate_bundle_against_model_dir(
    *,
    project: str,
    model_dir: str | Path,
    bundle_dir: str | Path,
    provider: str = 'CPUExecutionProvider',
):
    return verify_model_bundle(
        project=project,
        model_dir=model_dir,
        bundle_dir=bundle_dir,
        provider=provider,
    )


def evaluate_candidate_bundle(
    *,
    project: str,
    reference_bundle: str | Path,
    candidate_bundle: str | Path,
    provider: str = 'CPUExecutionProvider',
):
    return verify_model_bundle(
        project=project,
        reference_bundle=reference_bundle,
        candidate_bundle=candidate_bundle,
        provider=provider,
    )
