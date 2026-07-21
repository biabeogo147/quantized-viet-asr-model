from __future__ import annotations

import json
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from model_pipeline.core import sha256_file, sha256_path, stable_digest
from model_pipeline.models import get_recipe


BENCHMARK_STAGES = ("dataset", "local", "compile", "hosted")
SUPPORTED_BENCHMARK_MODELS = ("zipformer", "vpcd")
SUPPORTED_LOCAL_PROVIDERS = ("cpu", "cuda")
HOSTED_INPUT_LIMIT_PER_MODEL = 5


@dataclass(frozen=True)
class VlspBenchmarkRequest:
    """Describe one reproducible VLSP benchmark execution request."""

    model: str
    dataset_root: Path
    build_root: Path
    providers: tuple[str, ...]
    through: str
    submit_cloud: bool = False
    device: str | None = None
    qairt_version: str | None = None

    def __post_init__(self) -> None:
        """Validate model, provider, stop-point, and cloud-safety fields.

        Returns:
            None.

        Raises:
            ValueError: If the request is unsupported or could consume cloud implicitly.
        """
        if self.model not in {*SUPPORTED_BENCHMARK_MODELS, "all"}:
            raise ValueError(f"Unsupported benchmark model: {self.model!r}")
        if self.through not in BENCHMARK_STAGES[1:]:
            raise ValueError(f"Unsupported benchmark stop point: {self.through!r}")
        if not self.providers or "cpu" not in self.providers:
            raise ValueError("VLSP benchmark providers must include cpu")
        unsupported = set(self.providers) - set(SUPPORTED_LOCAL_PROVIDERS)
        if unsupported:
            raise ValueError(f"Unsupported local benchmark providers: {sorted(unsupported)!r}")
        if len(set(self.providers)) != len(self.providers):
            raise ValueError("Local benchmark providers must be unique")
        if self.through in {"compile", "hosted"}:
            if not self.submit_cloud:
                raise ValueError("--submit-cloud is required for compile and hosted stages")
            if not self.device or not self.qairt_version:
                raise ValueError("Cloud stages require --device and --qairt-version")

    @property
    def models(self) -> tuple[str, ...]:
        """Return concrete model families in deterministic execution order.

        Returns:
            Zipformer and/or VPCD model names selected by the request.
        """
        return SUPPORTED_BENCHMARK_MODELS if self.model == "all" else (self.model,)

    @property
    def stages(self) -> tuple[str, ...]:
        """Return cumulative benchmark stages through the requested stop point.

        Returns:
            Ordered dataset, local, and optional cloud stages.
        """
        index = BENCHMARK_STAGES.index(self.through)
        return BENCHMARK_STAGES[: index + 1]


def build_benchmark_plan(request: VlspBenchmarkRequest) -> dict[str, Any]:
    """Build a portable side-effect-free description of benchmark work.

    Args:
        request: Validated VLSP benchmark request.

    Returns:
        JSON-compatible model identities, stages, dataset counts, and cloud limits.
    """
    artifacts = {
        model: {
            "fp32": get_recipe(model, "fp32-fixed-shape").artifact.artifact_id,
            "quantized": get_recipe(
                model,
                "aimet-int8-int16-encoder-matmul",
            ).artifact.artifact_id,
        }
        for model in request.models
    }
    return {
        "models": list(request.models),
        "artifacts": artifacts,
        "stages": list(request.stages),
        "dataset": {"calibration_count": 24, "evaluation_count": 100},
        "providers": list(request.providers),
        "build_root": request.build_root.as_posix(),
        "dataset_root": request.dataset_root.as_posix(),
        "cloud": {
            "enabled": request.submit_cloud,
            "device": request.device,
            "qairt_version": request.qairt_version,
            "hosted_input_limit_per_model": HOSTED_INPUT_LIMIT_PER_MODEL,
        },
        "writes": True,
        "cloud_calls": request.through in {"compile", "hosted"},
    }


def parse_provider_list(value: str) -> tuple[str, ...]:
    """Parse a comma-separated provider list into normalized names.

    Args:
        value: Comma-separated `cpu` and optional `cuda` identifiers.

    Returns:
        Lowercase provider identifiers in request order.

    Raises:
        ValueError: If the list is empty.
    """
    providers = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not providers:
        raise ValueError("--providers must contain at least cpu")
    return providers


class VlspBenchmarkBackend(Protocol):
    """Declare model- and provider-specific benchmark operations."""

    def materialize_dataset(
        self,
        *,
        dataset_root: Path,
        output_dir: Path,
    ) -> Mapping[str, Any]:
        """Materialize the canonical VLSP split.

        Args:
            dataset_root: Source parquet shard directory.
            output_dir: Step-owned output directory.

        Returns:
            Portable dataset evidence.
        """

    def local_input_digest(
        self,
        *,
        model: str,
        dataset: Mapping[str, Any],
        providers: tuple[str, ...],
    ) -> str:
        """Hash model sources, dataset identity, recipe, and provider request.

        Args:
            model: Canonical model family.
            dataset: Materialized dataset evidence.
            providers: Requested local providers.

        Returns:
            Digest controlling safe local-step resume.
        """

    def run_local(
        self,
        *,
        model: str,
        dataset: Mapping[str, Any],
        providers: tuple[str, ...],
        output_dir: Path,
    ) -> Mapping[str, Any]:
        """Prepare, quantize, export QDQ, and evaluate one model locally.

        Args:
            model: Canonical model family.
            dataset: Materialized dataset evidence.
            providers: Requested CPU and optional CUDA providers.
            output_dir: Step-owned output directory.

        Returns:
            Graph, provider, metric, quality, and compile-source evidence.
        """

    def compile(
        self,
        *,
        model: str,
        local: Mapping[str, Any],
        output_dir: Path,
    ) -> Mapping[str, Any]:
        """Compile the canonical AIMET package and validate its target.

        Args:
            model: Canonical model family.
            local: Passing local benchmark evidence.
            output_dir: Step-owned output directory.

        Returns:
            Checksum-keyed compile and downloaded-package evidence.
        """

    def hosted_validate(
        self,
        *,
        model: str,
        local: Mapping[str, Any],
        compiled: Mapping[str, Any],
        input_limit: int,
        output_dir: Path,
    ) -> Mapping[str, Any]:
        """Validate a bounded set of compiled model inputs on AI Hub.

        Args:
            model: Canonical model family.
            local: Passing local benchmark evidence.
            compiled: Validated compile evidence.
            input_limit: Maximum hosted inputs allowed for this model.
            output_dir: Step-owned output directory.

        Returns:
            Hosted job, output checksum, and parity evidence.
        """


class BenchmarkStepRunner:
    """Resume benchmark steps only from matching inputs and verified outputs."""

    def __init__(self, build_root: str | Path):
        """Initialize deterministic benchmark step storage.

        Args:
            build_root: Root directory receiving step state and evidence.

        Returns:
            None.
        """
        self.build_root = Path(build_root)

    def run(
        self,
        *,
        name: str,
        input_digest: str,
        execute: Callable[[Path], Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Run or resume one evidence-producing benchmark step.

        Args:
            name: Slash-separated deterministic step identity.
            input_digest: Digest of every input affecting the step.
            execute: Callback writing evidence inside the allocated directory.

        Returns:
            JSON-compatible step result.

        Raises:
            ValueError: If a step result cannot be serialized.
        """
        step_dir = self.build_root / Path(name)
        result_path = step_dir / "result.json"
        state_path = step_dir / "state.json"
        state = _read_json(state_path)
        if _step_cache_matches(
            state,
            input_digest,
            step_dir,
            result_path,
            self.build_root,
        ):
            cached = _read_json(result_path)
            if cached is not None:
                return cached
        if step_dir.exists():
            shutil.rmtree(step_dir)
        step_dir.mkdir(parents=True, exist_ok=True)
        result = dict(execute(step_dir))
        external_artifacts = dict(result.pop("_resume_artifacts", {}))
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        files = {
            path.relative_to(step_dir).as_posix(): sha256_file(path)
            for path in sorted(item for item in step_dir.rglob("*") if item.is_file())
            if path != state_path
        }
        state_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "input_digest": input_digest,
                    "files": files,
                    "external_artifacts": external_artifacts,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return result


def evaluate_zipformer_quality_gate(
    *,
    fp32_cer: float,
    qdq_cer: float,
    fp32_wer: float,
    qdq_wer: float,
    empty_outputs: int,
    collapse_outputs: int,
) -> dict[str, Any]:
    """Evaluate Zipformer transcript regression against documented limits.

    Args:
        fp32_cer: FP32 corpus character error rate as a zero-to-one ratio.
        qdq_cer: QDQ corpus character error rate as a zero-to-one ratio.
        fp32_wer: FP32 corpus word error rate as a zero-to-one ratio.
        qdq_wer: QDQ corpus word error rate as a zero-to-one ratio.
        empty_outputs: Number of empty QDQ transcripts.
        collapse_outputs: Number of repetition-collapsed QDQ transcripts.

    Returns:
        Metric deltas, invalid-output counts, and pass status.
    """
    cer_increase = (float(qdq_cer) - float(fp32_cer)) * 100.0
    wer_increase = (float(qdq_wer) - float(fp32_wer)) * 100.0
    passed = (
        cer_increase <= 1.0
        and wer_increase <= 2.0
        and int(empty_outputs) == 0
        and int(collapse_outputs) == 0
    )
    return {
        "passed": passed,
        "cer_increase_percentage_points": cer_increase,
        "wer_increase_percentage_points": wer_increase,
        "empty_outputs": int(empty_outputs),
        "collapse_outputs": int(collapse_outputs),
    }


def evaluate_vpcd_quality_gate(
    *,
    sample_count: int,
    exact_output_matches: int,
    first_five_top1_matches: int,
    first_five_step_count: int,
    early_eos_count: int,
    collapse_count: int,
) -> dict[str, Any]:
    """Evaluate VPCD parity against documented local acceptance limits.

    Args:
        sample_count: Number of held-out restored outputs.
        exact_output_matches: Full-output matches against FP32.
        first_five_top1_matches: Matching top-one tokens in the first five steps.
        first_five_step_count: Total first-five token comparisons.
        early_eos_count: Quantized outputs ending earlier than FP32.
        collapse_count: Quantized punctuation-collapse detections.

    Returns:
        Parity counts and pass status.
    """
    minimum_exact = int(sample_count * 0.95)
    passed = (
        sample_count > 0
        and exact_output_matches >= minimum_exact
        and first_five_top1_matches == first_five_step_count == sample_count * 5
        and early_eos_count == 0
        and collapse_count == 0
    )
    return {
        "passed": passed,
        "sample_count": int(sample_count),
        "minimum_exact_output_matches": minimum_exact,
        "exact_output_matches": int(exact_output_matches),
        "first_five_top1_matches": int(first_five_top1_matches),
        "first_five_step_count": int(first_five_step_count),
        "early_eos_count": int(early_eos_count),
        "collapse_count": int(collapse_count),
    }


def run_vlsp_benchmark(
    request: VlspBenchmarkRequest,
    *,
    repo_root: str | Path,
    backend: VlspBenchmarkBackend | None = None,
) -> dict[str, Any]:
    """Execute or resume the requested VLSP benchmark workflow.

    Args:
        request: Validated benchmark request and cumulative stop point.
        repo_root: Repository root used for model and integration resolution.
        backend: Optional injected operations for deterministic integration tests.

    Returns:
        Aggregate status and per-model machine-readable evidence.

    Raises:
        RuntimeError: If any local or hosted quality gate fails.
    """
    if backend is None:
        from model_pipeline.benchmarks.vlsp_runtime import ProductionVlspBenchmarkBackend

        backend = ProductionVlspBenchmarkBackend(request=request, repo_root=Path(repo_root))
    request.build_root.mkdir(parents=True, exist_ok=True)
    environment = _environment_payload()
    environment_path = request.build_root / "environment.json"
    environment_path.write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    runner = BenchmarkStepRunner(request.build_root)
    dataset = runner.run(
        name="dataset",
        input_digest=_dataset_source_digest(request.dataset_root),
        execute=lambda output: backend.materialize_dataset(
            dataset_root=request.dataset_root,
            output_dir=output,
        ),
    )
    model_results: dict[str, Any] = {}
    for model in request.models:
        recipe = get_recipe(model, "aimet-int8-int16-encoder-matmul")
        local_digest_method = getattr(backend, "local_input_digest", None)
        local_input_digest = (
            local_digest_method(model=model, dataset=dataset, providers=request.providers)
            if callable(local_digest_method)
            else stable_digest(
                {
                    "dataset": dataset,
                    "recipe": recipe.digest,
                    "providers": list(request.providers),
                    "environment": environment,
                }
            )
        )

        def execute_local(output: Path, selected: str = model) -> Mapping[str, Any]:
            """Run one local model benchmark and record its exact input identity.

            Args:
                output: Step-owned local evidence directory.
                selected: Model family captured for this loop iteration.

            Returns:
                Backend evidence augmented with the local input digest.
            """
            result = dict(
                backend.run_local(
                    model=selected,
                    dataset=dataset,
                    providers=request.providers,
                    output_dir=output,
                )
            )
            result["input_digest"] = local_input_digest
            return result

        local = runner.run(
            name=f"{model}/local",
            input_digest=stable_digest(
                {"backend": local_input_digest, "environment": environment}
            ),
            execute=execute_local,
        )
        if local.get("quality_passed") is not True:
            raise RuntimeError(f"{model} local benchmark quality gate failed")
        result: dict[str, Any] = {"local": local}
        compiled: Mapping[str, Any] | None = None
        if "compile" in request.stages:
            compiled = runner.run(
                name=f"{model}/compile",
                input_digest=stable_digest(
                    {
                        "local": local,
                        "device": request.device,
                        "qairt_version": request.qairt_version,
                    }
                ),
                execute=lambda output, selected=model: backend.compile(
                    model=selected,
                    local=local,
                    output_dir=output,
                ),
            )
            result["compile"] = compiled
        if "hosted" in request.stages:
            assert compiled is not None
            hosted = runner.run(
                name=f"{model}/hosted",
                input_digest=stable_digest(
                    {
                        "local": local,
                        "compiled": compiled,
                        "input_limit": HOSTED_INPUT_LIMIT_PER_MODEL,
                    }
                ),
                execute=lambda output, selected=model: backend.hosted_validate(
                    model=selected,
                    local=local,
                    compiled=compiled,
                    input_limit=HOSTED_INPUT_LIMIT_PER_MODEL,
                    output_dir=output,
                ),
            )
            if (
                hosted.get("input_count") != HOSTED_INPUT_LIMIT_PER_MODEL
                or hosted.get("quality_passed") is not True
            ):
                raise RuntimeError(f"{model} hosted benchmark quality gate failed")
            result["hosted"] = hosted
        model_results[model] = result
    comparison = {
        "schema_version": 1,
        "status": "passed",
        "stages": list(request.stages),
        "environment_checksum": sha256_file(environment_path),
        "models": model_results,
    }
    (request.build_root / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return comparison


def _environment_payload() -> dict[str, Any]:
    """Describe portable runtime versions and registered ONNX providers.

    Returns:
        JSON-compatible environment evidence without machine-local paths.
    """
    try:
        import onnxruntime as ort

        onnx_runtime: dict[str, Any] = {
            "version": ort.__version__,
            "available_providers": list(ort.get_available_providers()),
        }
    except ImportError:
        onnx_runtime = {"version": "unavailable", "available_providers": []}
    return {
        "python_version": platform.python_version(),
        "operating_system": platform.system(),
        "machine": platform.machine(),
        "onnx_runtime": onnx_runtime,
        "python_implementation": sys.implementation.name,
    }


def _dataset_source_digest(dataset_root: Path) -> str:
    """Hash the two deterministic VLSP source shards used by the protocol.

    Args:
        dataset_root: Directory containing ordered parquet shards.

    Returns:
        Content digest for the first two shards, or a path digest before discovery.
    """
    shards = sorted(Path(dataset_root).glob("*.parquet"))
    if len(shards) < 2:
        return stable_digest({"dataset_root": Path(dataset_root).as_posix()})
    return stable_digest({path.name: sha256_file(path) for path in shards[:2]})


def _read_json(path: Path) -> dict[str, Any] | None:
    """Read one JSON object while treating corruption as a cache miss.

    Args:
        path: JSON file path.

    Returns:
        Decoded mapping or `None` when unavailable or invalid.
    """
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return dict(payload) if isinstance(payload, dict) else None


def _step_cache_matches(
    state: Mapping[str, Any] | None,
    input_digest: str,
    step_dir: Path,
    result_path: Path,
    build_root: Path,
) -> bool:
    """Validate cached step inputs and every tracked evidence file checksum.

    Args:
        state: Decoded cached state if available.
        input_digest: Expected current input digest.
        step_dir: Directory containing cached evidence.
        result_path: Required serialized result path.
        build_root: Root used to resolve external generated artifacts.

    Returns:
        `True` only when inputs and every output byte still match.
    """
    if not state or state.get("input_digest") != input_digest or not result_path.is_file():
        return False
    files = state.get("files")
    if not isinstance(files, dict) or not files:
        return False
    files_match = all(
        (step_dir / relative).is_file()
        and sha256_file(step_dir / relative) == checksum
        for relative, checksum in files.items()
    )
    if not files_match:
        return False
    external = state.get("external_artifacts") or {}
    if not isinstance(external, dict):
        return False
    root = build_root.resolve()
    for relative, checksum in external.items():
        path = (root / str(relative)).resolve()
        if path != root and root not in path.parents:
            return False
        if not path.exists() or sha256_path(path) != checksum:
            return False
    return True
