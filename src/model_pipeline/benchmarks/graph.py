"""Benchmark-only graph validation that preserves canonical model scope."""

from __future__ import annotations

from pathlib import Path

from model_pipeline.models.vpcd.graph import inspect_vpcd_matmuls
from model_pipeline.models.vpcd.quantization import inspect_encoder_matmul_aimet_encodings
from model_pipeline.models.zipformer.quantization import inspect_zipformer_qdq_coverage


def validate_benchmark_qdq(
    model: str,
    qdq_model_path: str | Path,
    encodings_path: str | Path,
) -> dict[str, object]:
    """Validate QDQ coverage without changing the production artifact package.

    Args:
        model: Canonical model family selecting its graph contract.
        qdq_model_path: Benchmark-only QDQ ONNX graph.
        encodings_path: Exact AIMET encodings used to export the graph.

    Returns:
        Normalized graph counts and encoder-scope validity.

    Raises:
        ValueError: If graph counts, coverage, precision, or scope differ.
    """
    if model == "zipformer":
        inventory = inspect_zipformer_qdq_coverage(qdq_model_path)
        scope = (
            inventory.matmul_count == 278
            and inventory.quantized_matmul_count == 278
            and not inventory.unquantized_matmul_names
        )
        if not scope:
            raise ValueError(
                "Zipformer benchmark QDQ must cover all 278 encoder MatMul nodes"
            )
        return {
            "matmul": inventory.matmul_count,
            "quantized_matmul": inventory.quantized_matmul_count,
            "scope": True,
        }
    if model == "vpcd":
        inventory = inspect_vpcd_matmuls(qdq_model_path)
        counts = inventory.counts
        encodings = inspect_encoder_matmul_aimet_encodings(encodings_path)
        scope = (
            (counts["encoder"], counts["decoder"], counts["lm_head"], counts["other"])
            == (96, 168, 1, 0)
            and encodings["activation_count"] == 168
            and encodings["parameter_count"] == 72
            and encodings["activation_contract"] is True
            and encodings["parameter_contract"] is True
            and not encodings["non_encoder_names"]
        )
        if not scope:
            raise ValueError(
                "VPCD benchmark QDQ must preserve 96/168/1 MatMul and encoder-only encodings"
            )
        return {
            "encoder_matmul": counts["encoder"],
            "decoder_matmul": counts["decoder"],
            "language_model_head_matmul": counts["lm_head"],
            "scope": True,
        }
    raise ValueError(f"Unsupported benchmark model: {model!r}")
