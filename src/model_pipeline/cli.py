from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from model_pipeline.core import Stage
from model_pipeline.models import get_recipe


def build_parser() -> argparse.ArgumentParser:
    """Build the single public command-line parser for model pipelines.

    Returns:
        The configured parser for canonical `run` commands.
    """
    parser = argparse.ArgumentParser(prog="python -m model_pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run the canonical model artifact pipeline")
    run.add_argument("--model", choices=("zipformer", "vpcd"), required=True)
    run.add_argument("--profile", choices=("fp32", "production"), required=True)
    run.add_argument(
        "--through",
        choices=tuple(stage.value for stage in Stage if stage != Stage.SOURCE),
        required=True,
    )
    run.add_argument("--repo-root", default=".")
    run.add_argument("--build-root", default="build/model-pipeline")
    run.add_argument("--android-destination")
    run.add_argument("--device")
    run.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and dry-run or execute the selected recipe.

    Args:
        argv: Optional explicit argument sequence; process arguments are used when omitted.

    Returns:
        Zero when the requested command completes successfully.
    """
    args = build_parser().parse_args(argv)
    recipe = get_recipe(args.model, args.profile)
    stages = [
        stage.value
        for stage in Stage.ordered()[: Stage.ordered().index(Stage(args.through)) + 1]
    ]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "profile": args.profile,
                    "artifact_id": recipe.artifact.artifact_id,
                    "stages": stages,
                    "actions": {
                        "prepare": recipe.parameters.get("prepare_scope", "fixed-shape"),
                        "quantize": recipe.parameters["quantize_action"],
                        "compile": recipe.parameters.get("compile_scope", recipe.artifact.compilation.scope),
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    from model_pipeline.runtime import run_pipeline

    return run_pipeline(
        recipe=recipe,
        repo_root=Path(args.repo_root),
        build_root=Path(args.build_root),
        through=args.through,
        android_destination=Path(args.android_destination) if args.android_destination else None,
        device=args.device,
    )
