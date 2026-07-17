from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from model_pipeline.core import Stage
from model_pipeline.models import get_recipe
from model_pipeline.benchmarks import BENCHMARK_CONFIGURATIONS


def build_parser() -> argparse.ArgumentParser:
    """Build the single public command-line parser for model pipelines.

    Returns:
        The configured parser for canonical `run` commands.
    """
    parser = argparse.ArgumentParser(prog="python -m model_pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run the canonical model artifact pipeline")
    run.add_argument("--model", choices=("zipformer", "vpcd"), required=True)
    run.add_argument(
        "--configuration",
        choices=(
            "fp32-fixed-shape",
            "fp32-fixed-shape-aihub-encoder",
            "ortqnn-uint8-uint16-encoder-matmul",
            "aimet-int8-int16-encoder-matmul",
        ),
        required=True,
    )
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
    payload = commands.add_parser(
        "android-benchmark-payload",
        help="Materialize a checksummed Android CPU/NPU benchmark payload",
    )
    payload.add_argument("--model", choices=("zipformer", "vpcd"), required=True)
    payload.add_argument("--output", required=True)
    payload.add_argument("--build-root", default="build/model-pipeline")
    payload.add_argument("--dry-run", action="store_true")
    report = commands.add_parser(
        "android-benchmark-report",
        help="Aggregate Android benchmark result JSON files",
    )
    report.add_argument("--results-root", required=True)
    report.add_argument("--output", required=True)
    report.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and dry-run or execute the selected recipe.

    Args:
        argv: Optional explicit argument sequence; process arguments are used when omitted.

    Returns:
        Zero when the requested command completes successfully.
    """
    args = build_parser().parse_args(argv)
    if args.command == "android-benchmark-payload":
        payload = {
            "model": args.model,
            "configurations": list(BENCHMARK_CONFIGURATIONS),
            "output": Path(args.output).as_posix(),
            "build_root": Path(args.build_root).as_posix(),
            "writes": not args.dry_run,
            "cloud_calls": False,
        }
        if args.dry_run:
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        from model_pipeline.benchmarks.runtime import materialize_from_pipeline_build

        manifest = materialize_from_pipeline_build(
            model=args.model,
            build_root=Path(args.build_root),
            output_dir=Path(args.output),
        )
        payload["manifest"] = manifest.as_posix()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "android-benchmark-report":
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "results_root": Path(args.results_root).as_posix(),
                        "output": Path(args.output).as_posix(),
                        "writes": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        from model_pipeline.benchmarks.runtime import aggregate_result_directory

        comparison = aggregate_result_directory(
            results_root=Path(args.results_root),
            output_dir=Path(args.output),
        )
        print(json.dumps(comparison, indent=2, sort_keys=True))
        return 0
    recipe = get_recipe(args.model, args.configuration)
    stages = [
        stage.value
        for stage in Stage.ordered()[: Stage.ordered().index(Stage(args.through)) + 1]
    ]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "configuration": args.configuration,
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
