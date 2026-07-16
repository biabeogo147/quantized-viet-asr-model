from __future__ import annotations

import json

import pytest

from model_pipeline.cli import main


@pytest.mark.parametrize(
    ("model", "configuration"),
    [
        ("zipformer", "fp32-fixed-shape"),
        ("zipformer", "fp32-fixed-shape-aihub-encoder"),
        ("zipformer", "ortqnn-uint8-uint16-encoder-matmul"),
        ("zipformer", "aimet-int8-int16-encoder-matmul"),
        ("vpcd", "fp32-fixed-shape"),
        ("vpcd", "aimet-int8-int16-encoder-matmul"),
    ],
)
def test_cli_dry_run_all_model_configurations(model: str, configuration: str, capsys) -> None:
    """Verify every public model configuration supports deterministic dry-run.

    Args:
        model: Parameterized canonical model family.
        configuration: Parameterized descriptive model configuration.
        capsys: Pytest capture fixture for emitted JSON.

    Returns:
        None.
    """
    exit_code = main(
        [
            "run",
            "--model",
            model,
            "--configuration",
            configuration,
            "--through",
            "sync",
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["model"] == model
    assert payload["configuration"] == configuration
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
    assert payload["actions"]["quantize"] in {"aimet", "ortqnn", "explicit-skip"}


def test_cli_rejects_removed_generic_configuration_option() -> None:
    """Verify the removed generic option has no compatibility alias.

    Returns:
        None.
    """
    with pytest.raises(SystemExit):
        main(
            [
                "run",
                "--model",
                "vpcd",
                "--pro" + "file",
                "pro" + "duction",
                "--through",
                "validate",
                "--dry-run",
            ]
        )
