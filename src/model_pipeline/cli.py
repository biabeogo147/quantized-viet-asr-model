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
    repository = commands.add_parser(
        "android-model-repository",
        help="Materialize the canonical manifest-v2 Android model repository",
    )
    repository.add_argument("--destination", required=True)
    repository.add_argument("--build-root", default="build/android-integration")
    repository.add_argument("--dry-run", action="store_true")
    report = commands.add_parser(
        "android-benchmark-report",
        help="Aggregate Android benchmark result JSON files",
    )
    report.add_argument("--results-root", required=True)
    report.add_argument("--output", required=True)
    report.add_argument("--dry-run", action="store_true")
    benchmark = commands.add_parser(
        "benchmark-vlsp",
        help="Reproduce the VLSP local, compile, and hosted benchmark protocol",
    )
    benchmark.add_argument("--model", choices=("zipformer", "vpcd", "all"), required=True)
    benchmark.add_argument("--dataset-root", required=True)
    benchmark.add_argument("--build-root", default="build/vlsp-benchmark")
    benchmark.add_argument("--providers", default="cpu,cuda")
    benchmark.add_argument("--through", choices=("local", "compile", "hosted"), required=True)
    benchmark.add_argument("--submit-cloud", action="store_true")
    benchmark.add_argument("--device")
    benchmark.add_argument("--qairt-version")
    benchmark.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and dry-run or execute the selected recipe.

    Args:
        argv: Optional explicit argument sequence; process arguments are used when omitted.

    Returns:
        Zero when the requested command completes successfully.
    """
    args = build_parser().parse_args(argv)
    if args.command == "benchmark-vlsp":
        from model_pipeline.benchmarks.vlsp import (
            VlspBenchmarkRequest,
            build_benchmark_plan,
            parse_provider_list,
        )

        request = VlspBenchmarkRequest(
            model=args.model,
            dataset_root=Path(args.dataset_root),
            build_root=Path(args.build_root),
            providers=parse_provider_list(args.providers),
            through=args.through,
            submit_cloud=args.submit_cloud,
            device=args.device,
            qairt_version=args.qairt_version,
        )
        payload = build_benchmark_plan(request)
        if args.dry_run:
            payload["writes"] = False
            payload["cloud_calls"] = False
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        from model_pipeline.benchmarks.vlsp import run_vlsp_benchmark

        result = run_vlsp_benchmark(request, repo_root=Path.cwd())
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "android-model-repository":
        artifact_ids = [
            {
                "model": model,
                "configuration": configuration,
                "artifact_id": get_recipe(model, configuration).artifact.artifact_id,
            }
            for model in ("zipformer", "vpcd")
            for configuration in (
                "fp32-fixed-shape",
                "aimet-int8-int16-encoder-matmul",
            )
        ]
        payload = {
            "artifact_ids": artifact_ids,
            "destination": Path(args.destination).as_posix(),
            "build_root": Path(args.build_root).as_posix(),
            "writes": not args.dry_run,
            "cloud_calls": False,
        }
        if args.dry_run:
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        from model_pipeline.integrations.android.repository_runtime import (
            materialize_canonical_repository,
        )

        result = materialize_canonical_repository(
            repo_root=Path.cwd(),
            build_root=Path(args.build_root),
            destination=Path(args.destination),
        )
        payload["index"] = result.index_path.as_posix()
        payload["repository_checksum"] = result.repository_checksum
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
