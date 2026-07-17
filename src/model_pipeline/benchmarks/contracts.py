"""Shared configuration and execution-order contracts for Android benchmarks."""

from __future__ import annotations

from dataclasses import dataclass


FP32_CPU_CONFIGURATION = "fp32-fixed-shape-onnxruntime-cpu"
QDQ_CPU_CONFIGURATION = "aimet-int8-int16-encoder-matmul-onnxruntime-cpu"
NPU_CONFIGURATION = "aimet-int8-int16-encoder-matmul-aihub-qnn-htp"
BENCHMARK_CONFIGURATIONS = (
    FP32_CPU_CONFIGURATION,
    QDQ_CPU_CONFIGURATION,
    NPU_CONFIGURATION,
)


@dataclass(frozen=True)
class BenchmarkRun:
    """Describe one fresh-process benchmark execution."""

    round_index: int
    ordinal: int
    configuration: str
    provider: str


def balanced_run_schedule() -> tuple[BenchmarkRun, ...]:
    """Build the three-round Latin-square benchmark order.

    Returns:
        Nine executions where every configuration occupies every ordinal once.
    """
    providers = {
        FP32_CPU_CONFIGURATION: "onnxruntime-cpu",
        QDQ_CPU_CONFIGURATION: "onnxruntime-cpu",
        NPU_CONFIGURATION: "qnn-htp",
    }
    runs: list[BenchmarkRun] = []
    for round_offset in range(3):
        ordered = BENCHMARK_CONFIGURATIONS[round_offset:] + BENCHMARK_CONFIGURATIONS[:round_offset]
        for ordinal, configuration in enumerate(ordered, start=1):
            runs.append(
                BenchmarkRun(
                    round_index=round_offset + 1,
                    ordinal=ordinal,
                    configuration=configuration,
                    provider=providers[configuration],
                )
            )
    return tuple(runs)
