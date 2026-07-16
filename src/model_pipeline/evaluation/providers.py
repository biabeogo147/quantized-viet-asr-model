from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class OrtProfileSummary:
    node_counts: dict[str, int]
    cuda_executed: bool

    def to_dict(self) -> dict[str, object]:
        """Serialize ONNX Runtime provider-placement evidence.

        Returns:
            Provider node counts and the CUDA execution flag.
        """
        return asdict(self)


@dataclass(frozen=True)
class OrtProviderSelection:
    available_providers: tuple[str, ...]
    selected_providers: tuple[str, ...]
    cuda_requested: bool

    def to_dict(self) -> dict[str, object]:
        """Serialize provider availability and requested session order.

        Returns:
            JSON-compatible provider selection fields.
        """
        return asdict(self)


def select_ort_providers(
    available_providers: Sequence[str],
    *,
    prefer_cuda: bool,
) -> OrtProviderSelection:
    """Choose CUDA with CPU fallback only when CUDA is registered.

    Args:
        available_providers: Providers registered by ONNX Runtime.
        prefer_cuda: Whether the caller requests CUDA execution.

    Returns:
        Available providers and deterministic session provider order.
    """
    available = tuple(str(provider) for provider in available_providers)
    if prefer_cuda and "CUDAExecutionProvider" in available:
        selected = ("CUDAExecutionProvider", "CPUExecutionProvider")
    else:
        selected = ("CPUExecutionProvider",)
    return OrtProviderSelection(available, selected, prefer_cuda)


def create_profiled_ort_session(
    model_path: str | Path,
    *,
    prefer_cuda: bool,
):
    """Create an ONNX Runtime session with node profiling enabled.

    Args:
        model_path: ONNX model to load.
        prefer_cuda: Whether to request CUDA before CPU fallback.

    Returns:
        Session and provider-selection evidence.
    """
    import onnxruntime as ort

    selection = select_ort_providers(ort.get_available_providers(), prefer_cuda=prefer_cuda)
    options = ort.SessionOptions()
    options.enable_profiling = True
    session = ort.InferenceSession(
        Path(model_path).resolve().as_posix(),
        sess_options=options,
        providers=list(selection.selected_providers),
    )
    return session, selection


def summarize_ort_profile(profile_path: str | Path) -> OrtProfileSummary:
    """Count executed nodes by provider from an ONNX Runtime profile.

    Args:
        profile_path: JSON profile returned by `InferenceSession.end_profiling`.

    Returns:
        Sorted provider node counts and whether any node executed on CUDA.

    Raises:
        ValueError: If the profile root is not an event list.
    """
    events = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    if not isinstance(events, list):
        raise ValueError("ONNX Runtime profile must contain an event list")
    counts: dict[str, int] = {}
    for event in events:
        if event.get("cat") != "Node":
            continue
        provider = str(event.get("args", {}).get("provider") or "UnknownExecutionProvider")
        counts[provider] = counts.get(provider, 0) + 1
    ordered_counts = dict(sorted(counts.items()))
    return OrtProfileSummary(
        node_counts=ordered_counts,
        cuda_executed=ordered_counts.get("CUDAExecutionProvider", 0) > 0,
    )
