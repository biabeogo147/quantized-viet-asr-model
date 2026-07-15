from __future__ import annotations

import json

import pytest

from model_pipeline.cli import main


@pytest.mark.parametrize("model", ["zipformer", "vpcd"])
@pytest.mark.parametrize("profile", ["fp32", "production"])
def test_cli_dry_run_all_model_profile_combinations(model: str, profile: str, capsys) -> None:
    """Verify every public model/profile combination supports deterministic dry-run.

    Args:
        model: Parameterized canonical model family.
        profile: Parameterized control or production profile.
        capsys: Pytest capture fixture for emitted JSON.

    Returns:
        None.
    """
    exit_code = main(
        [
            "run",
            "--model",
            model,
            "--profile",
            profile,
            "--through",
            "sync",
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["model"] == model
    assert payload["profile"] == profile
    assert payload["stages"] == [
        "source",
        "prepare",
        "quantize",
        "validate",
        "compile",
        "package",
        "sync",
    ]
    assert "artifact_id" in payload
    assert payload["actions"]["quantize"] in {"aimet", "explicit-skip"}
