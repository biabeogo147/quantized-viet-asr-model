from __future__ import annotations

from pathlib import Path

from model_bundle.verifier import verify_model_bundle


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
