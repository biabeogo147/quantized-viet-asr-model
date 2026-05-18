from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import onnxruntime as ort

from model_bundle.fixtures import read_jsonl
from model_bundle.manifest import ModelBundleManifest
from model_bundle.projects._vpcd_support import BundleOnnxRuntime
from model_bundle.projects.zipformer import (
    BundleAcousticRuntime,
    ModelDirAcousticRuntime,
    decode_encoder_frames_greedy,
    prepare_encoder_inputs,
    trim_encoder_frames,
)
from quantize.calibration import greedy_decode_ids, load_decoder_start_token_id, resolve_ort_providers
from tools.aihub_option1_pilots import (
    Option1RuntimeConfig,
    build_vpcd_fixed_shape_inputs,
    build_job_options,
    build_vpcd_input_specs,
    build_zipformer_encoder_input_specs,
    coerce_inputs_for_compiled_model,
    resolve_downloaded_quantized_model_path,
    resolve_vpcd_fp32_source_model_path,
    resolve_vpcd_model_dir,
    resolve_target_model_id,
    resolve_vpcd_pilot_source,
    resolve_zipformer_encoder_pilot_source,
    summarize_vpcd_step_logits,
)
from transformers import AutoTokenizer

ZIPFORMER_PHASE2_PILOT = "zipformer_encoder_option1"
ZIPFORMER_PHASE3_PILOT = "zipformer_hybrid_option1"
VPCD_PHASE2_PILOT = "vpcd_option1"
VPCD_PHASE3_PILOT = "vpcd_hybrid_option1"
VPCD_TEACHER_FORCED_PILOT = "vpcd_teacher_forced_option1"
VPCD_QUANTIZED_TEACHER_FORCED_PILOT = "vpcd_quantized_teacher_forced_option1"
DEFAULT_ZIPFORMER_MAX_SAMPLES = 2
DEFAULT_VPCD_MAX_SAMPLES = 4


@dataclass(frozen=True)
class ResolvedCompiledTarget:
    compile_pilot_name: str
    target_model_id: str
    compile_record_path: Path | None
    run_label: str | None
    explicit_override: bool


def resolve_compiled_target_reference(
    *,
    runtime_config: Option1RuntimeConfig,
    compile_pilot_name: str,
    explicit_target_model_id: str | None = None,
    run_label: str | None = None,
) -> ResolvedCompiledTarget:
    normalized_explicit_id = _normalize_optional_string(explicit_target_model_id)
    target_model_id = resolve_target_model_id(
        pilot_name=compile_pilot_name,
        runtime_config=runtime_config,
        explicit_target_model_id=normalized_explicit_id,
        run_label=run_label,
    )
    compile_record_path = None
    if normalized_explicit_id is None:
        compile_record_path = _resolve_phase2_compile_record_path(
            runtime_config=runtime_config,
            compile_pilot_name=compile_pilot_name,
            run_label=run_label,
        )
    return ResolvedCompiledTarget(
        compile_pilot_name=compile_pilot_name,
        target_model_id=target_model_id,
        compile_record_path=compile_record_path,
        run_label=_normalize_optional_string(run_label),
        explicit_override=normalized_explicit_id is not None,
    )


def normalize_compiled_output_tensors(
    output_tensors: Mapping[str, object] | Sequence[object],
) -> dict[str, np.ndarray]:
    normalized: dict[str, np.ndarray] = {}
    if isinstance(output_tensors, Mapping):
        for name in sorted(str(key) for key in output_tensors.keys()):
            normalized[name] = _unwrap_single_output_tensor(name, output_tensors[name])
        return normalized
    if isinstance(output_tensors, Sequence) and not isinstance(output_tensors, (bytes, bytearray, str)):
        for index, value in enumerate(output_tensors):
            normalized[f"output_{index}"] = np.asarray(value)
        return normalized
    raise TypeError(f"Unsupported compiled output type: {type(output_tensors)!r}")


def run_compiled_inference(
    *,
    target_reference: ResolvedCompiledTarget,
    runtime_config: Option1RuntimeConfig,
    inputs: Mapping[str, np.ndarray] | Mapping[str, list[np.ndarray]],
    input_specs: Mapping[str, tuple[tuple[int, ...], str]] | None,
    inference_runner: Callable[..., object] | None = None,
    inference_name: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    coerced_inputs = coerce_inputs_for_compiled_model(dict(inputs), input_specs=input_specs)
    wrapped_inputs = _wrap_inference_inputs(coerced_inputs)
    runner = inference_runner or _submit_live_compiled_inference

    started = time.perf_counter()
    runner_result = runner(
        target_model_id=target_reference.target_model_id,
        runtime_config=runtime_config,
        inputs=wrapped_inputs,
        inference_name=inference_name,
    )
    elapsed_seconds = round(time.perf_counter() - started, 6)

    raw_outputs: object
    raw_job_metadata: object | None
    if isinstance(runner_result, tuple) and len(runner_result) == 2:
        raw_outputs, raw_job_metadata = runner_result
    else:
        raw_outputs, raw_job_metadata = runner_result, None

    return normalize_compiled_output_tensors(raw_outputs), {
        "target_model_id": target_reference.target_model_id,
        "compile_record_path": (
            target_reference.compile_record_path.as_posix() if target_reference.compile_record_path is not None else None
        ),
        "elapsed_seconds": elapsed_seconds,
        "job": _normalize_job_metadata(raw_job_metadata),
        "inference_name": inference_name,
    }


def run_zipformer_hybrid_evaluation(
    *,
    runtime_config: Option1RuntimeConfig,
    run_label: str | None = None,
    explicit_target_model_id: str | None = None,
    max_samples: int = DEFAULT_ZIPFORMER_MAX_SAMPLES,
    inference_runner: Callable[..., object] | None = None,
    bundle_runtime: object | None = None,
    feature_loader: Callable[..., np.ndarray] | None = None,
) -> dict[str, Any]:
    source = resolve_zipformer_encoder_pilot_source(runtime_config.repo_root)
    target_reference = resolve_compiled_target_reference(
        runtime_config=runtime_config,
        compile_pilot_name=ZIPFORMER_PHASE2_PILOT,
        explicit_target_model_id=explicit_target_model_id,
        run_label=run_label,
    )
    runtime = bundle_runtime or BundleAcousticRuntime.from_manifest_path(
        source.bundle_manifest_path,
        provider="CPUExecutionProvider",
    )
    input_specs = build_zipformer_encoder_input_specs(source)
    feature_reader = feature_loader or ModelDirAcousticRuntime._load_features
    sample_rows = _load_zipformer_evaluation_rows(
        bundle_manifest_path=source.bundle_manifest_path,
        repo_root=source.repo_root,
        sample_manifest_path=source.sample_manifest_path,
        max_samples=max_samples,
    )

    results: list[dict[str, Any]] = []
    for sample in sample_rows:
        sample_id = str(sample["sample_id"])
        audio_path = _resolve_repo_relative_path(source.repo_root, sample["audio_path"])
        features = feature_reader(
            audio_path,
            sample_rate=int(getattr(runtime, "sample_rate", source.sample_rate)),
            feature_dim=int(getattr(runtime, "feature_dim", source.feature_dim)),
        )
        encoder_inputs = prepare_encoder_inputs(
            features,
            fixed_encoder_frames=_normalize_optional_int(getattr(runtime, "fixed_encoder_frames", None)),
        )

        encoder_outputs, inference_metadata = run_compiled_inference(
            target_reference=target_reference,
            runtime_config=runtime_config,
            inputs=encoder_inputs,
            input_specs=input_specs,
            inference_runner=inference_runner,
            inference_name=f"bkmeeting-zipformer-hybrid-{sample_id}",
        )
        trimmed_frames = trim_encoder_frames(
            _resolve_output_array(encoder_outputs, preferred_names=("output_0",))[0].astype(np.float32, copy=False),
            _resolve_output_array(encoder_outputs, preferred_names=("output_1",), allow_missing=True),
        )

        decode_started = time.perf_counter()
        decode_result = decode_encoder_frames_greedy(
            frames=trimmed_frames,
            decoder_session=getattr(runtime, "decoder_sess"),
            joiner_session=getattr(runtime, "joiner_sess"),
            tokens_table=list(getattr(runtime, "tokens_table")),
            blank_id=int(getattr(runtime, "blank_id")),
            context_size=int(getattr(runtime, "context_size")),
        )
        decode_seconds = round(time.perf_counter() - decode_started, 6)

        expected_text = str(sample.get("expected_text", ""))
        expected_available = bool(expected_text)
        results.append(
            {
                "sample_id": sample_id,
                "audio_path": str(sample["audio_path"]),
                "text": str(decode_result["text"]),
                "expected_text": expected_text,
                "expected_available": expected_available,
                "matches_expected": (str(decode_result["text"]) == expected_text) if expected_available else None,
                "num_tokens": int(decode_result["num_tokens"]),
                "token_ids": [int(token_id) for token_id in decode_result["token_ids"]],
                "cloud_inference_seconds": float(inference_metadata["elapsed_seconds"]),
                "decode_seconds": decode_seconds,
                "job": inference_metadata["job"],
            }
        )

    summary = _summarize_match_results(results)
    record_path = write_hybrid_run_record(
        pilot_name=ZIPFORMER_PHASE3_PILOT,
        runtime_config=runtime_config,
        target_reference=target_reference,
        sample_results=results,
        run_label=run_label,
    )
    return {
        "pilot_name": ZIPFORMER_PHASE3_PILOT,
        "target_reference": target_reference,
        "results": results,
        "summary": summary,
        "record_path": record_path,
    }


def run_vpcd_hybrid_evaluation(
    *,
    runtime_config: Option1RuntimeConfig,
    run_label: str | None = None,
    explicit_target_model_id: str | None = None,
    max_samples: int = DEFAULT_VPCD_MAX_SAMPLES,
    max_decode_steps: int | None = None,
    inference_runner: Callable[..., object] | None = None,
    bundle_runtime: BundleOnnxRuntime | None = None,
) -> dict[str, Any]:
    source = resolve_vpcd_pilot_source(runtime_config.repo_root)
    target_reference = resolve_compiled_target_reference(
        runtime_config=runtime_config,
        compile_pilot_name=VPCD_PHASE2_PILOT,
        explicit_target_model_id=explicit_target_model_id,
        run_label=run_label,
    )
    runtime = bundle_runtime or BundleOnnxRuntime.from_manifest_path(
        source.bundle_manifest_path,
        provider="CPUExecutionProvider",
    )
    input_specs = build_vpcd_input_specs(source)
    sample_rows = read_jsonl(source.golden_samples_path)[: max(0, int(max_samples))]
    decode_step_limit = max(1, min(int(max_decode_steps), int(source.decoder_sequence))) if max_decode_steps is not None else int(source.decoder_sequence)

    results: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(sample_rows):
        step_jobs: list[dict[str, Any]] = []

        def step_runner(feeds: dict[str, np.ndarray]) -> np.ndarray:
            outputs, inference_metadata = run_compiled_inference(
                target_reference=target_reference,
                runtime_config=runtime_config,
                inputs=feeds,
                input_specs=input_specs,
                inference_runner=inference_runner,
                inference_name=f"bkmeeting-vpcd-hybrid-s{sample_index}-step-{len(step_jobs) + 1}",
            )
            step_jobs.append(inference_metadata)
            return _resolve_output_array(outputs, preferred_names=("output_0",))

        decode_started = time.perf_counter()
        restored = runtime.restore_with_model_step(
            str(sample["raw_text"]),
            step_runner,
            max_length=decode_step_limit,
        )
        decode_seconds = round(time.perf_counter() - decode_started, 6)
        output_text = str(restored["text"])
        expected_text = str(sample.get("expected_output", ""))
        results.append(
            {
                "sample_index": int(sample_index),
                "raw_text": str(sample["raw_text"]),
                "text": output_text,
                "expected_text": expected_text,
                "expected_available": bool(expected_text),
                "matches_expected": output_text == expected_text,
                "decode_step_limit": int(decode_step_limit),
                "decode_steps": int(restored["decode_steps"]),
                "generated_ids": [int(token_id) for token_id in np.asarray(restored["generated_ids"]).tolist()],
                "golden_input_ids": [int(token_id) for token_id in np.asarray(sample.get("input_ids", []), dtype=np.int64).tolist()],
                "input_ids_fixture_available": "input_ids" in sample,
                "cloud_inference_seconds": round(sum(float(job["elapsed_seconds"]) for job in step_jobs), 6),
                "decode_seconds": decode_seconds,
                "jobs": [job["job"] for job in step_jobs],
            }
        )

    summary = _summarize_match_results(results)
    record_path = write_hybrid_run_record(
        pilot_name=VPCD_PHASE3_PILOT,
        runtime_config=runtime_config,
        target_reference=target_reference,
        sample_results=results,
        run_label=run_label,
    )
    return {
        "pilot_name": VPCD_PHASE3_PILOT,
        "target_reference": target_reference,
        "results": results,
        "summary": summary,
        "record_path": record_path,
        "decode_step_limit": int(decode_step_limit),
    }


def run_vpcd_teacher_forced_diagnostics(
    *,
    runtime_config: Option1RuntimeConfig,
    run_label: str | None = None,
    explicit_target_model_id: str | None = None,
    compile_pilot_name: str = VPCD_PHASE2_PILOT,
    sample_index: int = 0,
    max_decode_steps: int | None = None,
    top_k: int = 5,
    inference_runner: Callable[..., object] | None = None,
    cpu_model_step_runner: Callable[[dict[str, np.ndarray]], object] | None = None,
    decode_ids_fn: Callable[[str], tuple[dict[str, np.ndarray], list[int]]] | None = None,
) -> dict[str, Any]:
    source = resolve_vpcd_pilot_source(runtime_config.repo_root)
    sample_rows = read_jsonl(source.golden_samples_path)
    if sample_index < 0 or sample_index >= len(sample_rows):
        raise IndexError(f"VPCD sample_index out of range: {sample_index}")

    target_reference = resolve_compiled_target_reference(
        runtime_config=runtime_config,
        compile_pilot_name=compile_pilot_name,
        explicit_target_model_id=explicit_target_model_id,
        run_label=run_label,
    )
    input_specs = build_vpcd_input_specs(source)
    sample = sample_rows[sample_index]

    reference_stats: dict[str, Any] = {
        "requested_provider": "cpu",
        "session_providers": "injected" if cpu_model_step_runner is not None else None,
        "fp32_model_path": None,
        "model_dir": None,
        "compile_pilot_name": target_reference.compile_pilot_name,
        "source_strategy": None,
        "quantize_stage": None,
        "compile_compatibility": {},
    }
    reference_stats.update(_load_compile_record_context(target_reference))
    if decode_ids_fn is None or cpu_model_step_runner is None:
        resolved_fp32_model_path = resolve_vpcd_fp32_source_model_path(source)
        if resolved_fp32_model_path is None:
            raise FileNotFoundError("Could not resolve a VPCD FP32 ONNX source model for teacher-forced diagnostics.")
        fp32_session = ort.InferenceSession(
            resolved_fp32_model_path.as_posix(),
            providers=resolve_ort_providers("cpu"),
        )
        reference_stats["fp32_model_path"] = resolved_fp32_model_path.as_posix()
        reference_stats["session_providers"] = ",".join(fp32_session.get_providers())

        if decode_ids_fn is None:
            resolved_model_dir = resolve_vpcd_model_dir(source)
            if resolved_model_dir is None:
                raise FileNotFoundError("Could not resolve a VPCD model directory for teacher-forced diagnostics.")
            tokenizer = AutoTokenizer.from_pretrained(resolved_model_dir, local_files_only=True)
            decoder_start_token_id = load_decoder_start_token_id(resolved_model_dir, tokenizer)
            reference_stats["model_dir"] = resolved_model_dir.as_posix()

            def _decode_ids_with_fp32(text: str) -> tuple[dict[str, np.ndarray], list[int]]:
                return greedy_decode_ids(
                    session=fp32_session,
                    tokenizer=tokenizer,
                    text=text,
                    decoder_start_token_id=decoder_start_token_id,
                    max_generation_length=max(1, int(max_decode_steps or source.decoder_sequence)),
                )

            decode_ids_fn = _decode_ids_with_fp32

        if cpu_model_step_runner is None:
            cpu_model_step_runner = lambda feeds: fp32_session.run(None, feeds)[0]

    encoder_inputs, decoded_ids = decode_ids_fn(str(sample["raw_text"]))
    normalized_decoded_ids = [int(token_id) for token_id in decoded_ids]
    if not normalized_decoded_ids:
        raise ValueError("decode_ids_fn returned an empty decoded_ids sequence.")
    if normalized_decoded_ids[0] != int(source.decoder_start_token_id):
        raise ValueError("decoded_ids must start with source.decoder_start_token_id.")

    encoder_input_ids = np.asarray(encoder_inputs["input_ids"], dtype=np.int64).reshape(-1)
    encoder_attention_mask = np.asarray(encoder_inputs["attention_mask"], dtype=np.int64).reshape(-1)
    available_steps = max(0, len(normalized_decoded_ids) - 1)
    requested_steps = int(source.decoder_sequence) if max_decode_steps is None else max(1, int(max_decode_steps))
    decode_step_limit = min(int(source.decoder_sequence), requested_steps, available_steps)

    step_results: list[dict[str, Any]] = []
    inference_jobs: list[dict[str, Any]] = []
    decode_started = time.perf_counter()
    for step_index in range(1, decode_step_limit + 1):
        prefix_ids = normalized_decoded_ids[:step_index]
        expected_next_token_id = normalized_decoded_ids[step_index] if step_index < len(normalized_decoded_ids) else None
        feeds = build_vpcd_fixed_shape_inputs(
            source,
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            decoder_prefix=prefix_ids,
        )
        cpu_logits = np.asarray(cpu_model_step_runner(feeds))
        cpu_summary = summarize_vpcd_step_logits(cpu_logits, feeds["decoder_attention_mask"], top_k=top_k)
        cpu_argmax_token_id = BundleOnnxRuntime._argmax_token_at(cpu_logits, cpu_summary["active_index"])

        cloud_outputs, inference_metadata = run_compiled_inference(
            target_reference=target_reference,
            runtime_config=runtime_config,
            inputs=feeds,
            input_specs=input_specs,
            inference_runner=inference_runner,
            inference_name=f"bkmeeting-vpcd-teacher-forced-s{sample_index}-step-{step_index}",
        )
        inference_jobs.append(inference_metadata)
        cloud_logits = _resolve_output_array(cloud_outputs, preferred_names=("output_0",))
        cloud_summary = summarize_vpcd_step_logits(cloud_logits, feeds["decoder_attention_mask"], top_k=top_k)
        cloud_argmax_token_id = BundleOnnxRuntime._argmax_token_at(cloud_logits, cloud_summary["active_index"])

        step_results.append(
            {
                "step_index": int(step_index),
                "decoder_prefix_ids": [int(token_id) for token_id in prefix_ids],
                "expected_next_token_id": int(expected_next_token_id) if expected_next_token_id is not None else None,
                "active_index": int(cpu_summary["active_index"]),
                "cpu_top_tokens": list(cpu_summary["top_tokens"]),
                "cloud_top_tokens": list(cloud_summary["top_tokens"]),
                "cpu_argmax_token_id": int(cpu_argmax_token_id),
                "cloud_argmax_token_id": int(cloud_argmax_token_id),
                "cpu_matches_expected_next_token": (
                    int(cpu_argmax_token_id) == int(expected_next_token_id) if expected_next_token_id is not None else None
                ),
                "cloud_matches_expected_next_token": (
                    int(cloud_argmax_token_id) == int(expected_next_token_id) if expected_next_token_id is not None else None
                ),
                "matches_cpu_argmax": int(cpu_argmax_token_id) == int(cloud_argmax_token_id),
                "job_id": inference_metadata["job"].get("job_id"),
                "job_url": inference_metadata["job"].get("url"),
            }
        )

    decode_seconds = round(time.perf_counter() - decode_started, 6)
    sample_result = {
        "sample_index": int(sample_index),
        "raw_text": str(sample["raw_text"]),
        "expected_text": str(sample.get("expected_output", "")),
        "expected_available": bool(sample.get("expected_output")),
        "matches_expected": None,
        "decode_step_limit": int(decode_step_limit),
        "available_teacher_forced_steps": int(available_steps),
        "gold_decoder_ids": [int(token_id) for token_id in normalized_decoded_ids],
        "encoder_input_ids": [int(token_id) for token_id in encoder_input_ids.tolist()],
        "cloud_inference_seconds": round(sum(float(job["elapsed_seconds"]) for job in inference_jobs), 6),
        "decode_seconds": decode_seconds,
        "jobs": [job["job"] for job in inference_jobs],
        "reference_stats": reference_stats,
        "steps": step_results,
    }
    record_path = write_hybrid_run_record(
        pilot_name=VPCD_TEACHER_FORCED_PILOT,
        runtime_config=runtime_config,
        target_reference=target_reference,
        sample_results=[sample_result],
        run_label=run_label,
    )
    summary = _summarize_match_results([sample_result])
    return {
        "pilot_name": VPCD_TEACHER_FORCED_PILOT,
        "target_reference": target_reference,
        "results": [sample_result],
        "steps": step_results,
        "summary": summary,
        "record_path": record_path,
        "decode_step_limit": int(decode_step_limit),
    }


def run_vpcd_quantized_teacher_forced_diagnostics(
    *,
    runtime_config: Option1RuntimeConfig,
    run_label: str | None = None,
    sample_index: int = 0,
    max_decode_steps: int | None = None,
    top_k: int = 5,
    explicit_quantized_model_path: str | Path | None = None,
    cpu_model_step_runner: Callable[[dict[str, np.ndarray]], object] | None = None,
    quantized_model_step_runner: Callable[[dict[str, np.ndarray]], object] | None = None,
    decode_ids_fn: Callable[[str], tuple[dict[str, np.ndarray], list[int]]] | None = None,
) -> dict[str, Any]:
    source = resolve_vpcd_pilot_source(runtime_config.repo_root)
    sample_rows = read_jsonl(source.golden_samples_path)
    if sample_index < 0 or sample_index >= len(sample_rows):
        raise IndexError(f"VPCD sample_index out of range: {sample_index}")

    sample = sample_rows[sample_index]
    reference_stats: dict[str, Any] = {
        "requested_provider": "cpu",
        "session_providers": "injected" if cpu_model_step_runner is not None else None,
        "fp32_model_path": None,
        "model_dir": None,
        "quantized_model_path": None,
        "quantized_session_providers": "injected" if quantized_model_step_runner is not None else None,
    }
    if decode_ids_fn is None or cpu_model_step_runner is None:
        resolved_fp32_model_path = resolve_vpcd_fp32_source_model_path(source)
        if resolved_fp32_model_path is None:
            raise FileNotFoundError("Could not resolve a VPCD FP32 ONNX source model for quantized teacher-forced diagnostics.")
        fp32_session = ort.InferenceSession(
            resolved_fp32_model_path.as_posix(),
            providers=resolve_ort_providers("cpu"),
        )
        reference_stats["fp32_model_path"] = resolved_fp32_model_path.as_posix()
        reference_stats["session_providers"] = ",".join(fp32_session.get_providers())

        if decode_ids_fn is None:
            resolved_model_dir = resolve_vpcd_model_dir(source)
            if resolved_model_dir is None:
                raise FileNotFoundError("Could not resolve a VPCD model directory for quantized teacher-forced diagnostics.")
            tokenizer = AutoTokenizer.from_pretrained(resolved_model_dir, local_files_only=True)
            decoder_start_token_id = load_decoder_start_token_id(resolved_model_dir, tokenizer)
            reference_stats["model_dir"] = resolved_model_dir.as_posix()

            def _decode_ids_with_fp32(text: str) -> tuple[dict[str, np.ndarray], list[int]]:
                return greedy_decode_ids(
                    session=fp32_session,
                    tokenizer=tokenizer,
                    text=text,
                    decoder_start_token_id=decoder_start_token_id,
                    max_generation_length=max(1, int(max_decode_steps or source.decoder_sequence)),
                )

            decode_ids_fn = _decode_ids_with_fp32

        if cpu_model_step_runner is None:
            cpu_model_step_runner = lambda feeds: fp32_session.run(None, feeds)[0]

    if quantized_model_step_runner is None:
        resolved_quantized_model_path = resolve_downloaded_quantized_model_path(
            pilot_name=VPCD_PHASE2_PILOT,
            runtime_config=runtime_config,
            explicit_quantized_model_path=explicit_quantized_model_path,
            run_label=run_label,
        )
        quantized_session = ort.InferenceSession(
            resolved_quantized_model_path.as_posix(),
            providers=resolve_ort_providers("cpu"),
        )
        reference_stats["quantized_model_path"] = resolved_quantized_model_path.as_posix()
        reference_stats["quantized_session_providers"] = ",".join(quantized_session.get_providers())
        quantized_model_step_runner = lambda feeds: quantized_session.run(None, feeds)[0]
    elif explicit_quantized_model_path is not None:
        reference_stats["quantized_model_path"] = Path(explicit_quantized_model_path).resolve().as_posix()

    encoder_inputs, decoded_ids = decode_ids_fn(str(sample["raw_text"]))
    normalized_decoded_ids = [int(token_id) for token_id in decoded_ids]
    if not normalized_decoded_ids:
        raise ValueError("decode_ids_fn returned an empty decoded_ids sequence.")
    if normalized_decoded_ids[0] != int(source.decoder_start_token_id):
        raise ValueError("decoded_ids must start with source.decoder_start_token_id.")

    encoder_input_ids = np.asarray(encoder_inputs["input_ids"], dtype=np.int64).reshape(-1)
    encoder_attention_mask = np.asarray(encoder_inputs["attention_mask"], dtype=np.int64).reshape(-1)
    available_steps = max(0, len(normalized_decoded_ids) - 1)
    requested_steps = int(source.decoder_sequence) if max_decode_steps is None else max(1, int(max_decode_steps))
    decode_step_limit = min(int(source.decoder_sequence), requested_steps, available_steps)

    step_results: list[dict[str, Any]] = []
    quantized_inference_seconds = 0.0
    decode_started = time.perf_counter()
    for step_index in range(1, decode_step_limit + 1):
        prefix_ids = normalized_decoded_ids[:step_index]
        expected_next_token_id = normalized_decoded_ids[step_index] if step_index < len(normalized_decoded_ids) else None
        feeds = build_vpcd_fixed_shape_inputs(
            source,
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            decoder_prefix=prefix_ids,
        )
        cpu_logits = np.asarray(cpu_model_step_runner(feeds))
        cpu_summary = summarize_vpcd_step_logits(cpu_logits, feeds["decoder_attention_mask"], top_k=top_k)
        cpu_argmax_token_id = BundleOnnxRuntime._argmax_token_at(cpu_logits, cpu_summary["active_index"])

        quantized_started = time.perf_counter()
        quantized_logits = np.asarray(quantized_model_step_runner(feeds))
        quantized_inference_seconds += time.perf_counter() - quantized_started
        quantized_summary = summarize_vpcd_step_logits(quantized_logits, feeds["decoder_attention_mask"], top_k=top_k)
        quantized_argmax_token_id = BundleOnnxRuntime._argmax_token_at(quantized_logits, quantized_summary["active_index"])

        step_results.append(
            {
                "step_index": int(step_index),
                "decoder_prefix_ids": [int(token_id) for token_id in prefix_ids],
                "expected_next_token_id": int(expected_next_token_id) if expected_next_token_id is not None else None,
                "active_index": int(cpu_summary["active_index"]),
                "cpu_top_tokens": list(cpu_summary["top_tokens"]),
                "quantized_top_tokens": list(quantized_summary["top_tokens"]),
                "cpu_argmax_token_id": int(cpu_argmax_token_id),
                "quantized_argmax_token_id": int(quantized_argmax_token_id),
                "cpu_matches_expected_next_token": (
                    int(cpu_argmax_token_id) == int(expected_next_token_id) if expected_next_token_id is not None else None
                ),
                "quantized_matches_expected_next_token": (
                    int(quantized_argmax_token_id) == int(expected_next_token_id) if expected_next_token_id is not None else None
                ),
                "matches_fp32_argmax": int(cpu_argmax_token_id) == int(quantized_argmax_token_id),
            }
        )

    decode_seconds = round(time.perf_counter() - decode_started, 6)
    sample_result = {
        "sample_index": int(sample_index),
        "raw_text": str(sample["raw_text"]),
        "expected_text": str(sample.get("expected_output", "")),
        "expected_available": bool(sample.get("expected_output")),
        "matches_expected": None,
        "decode_step_limit": int(decode_step_limit),
        "available_teacher_forced_steps": int(available_steps),
        "gold_decoder_ids": [int(token_id) for token_id in normalized_decoded_ids],
        "encoder_input_ids": [int(token_id) for token_id in encoder_input_ids.tolist()],
        "cloud_inference_seconds": None,
        "quantized_inference_seconds": round(float(quantized_inference_seconds), 6),
        "decode_seconds": decode_seconds,
        "jobs": [],
        "reference_stats": reference_stats,
        "steps": step_results,
    }
    synthetic_reference = ResolvedCompiledTarget(
        compile_pilot_name=VPCD_PHASE2_PILOT,
        target_model_id=reference_stats["quantized_model_path"] or "local-quantized-vpcd",
        compile_record_path=None,
        run_label=_normalize_optional_string(run_label),
        explicit_override=explicit_quantized_model_path is not None,
    )
    record_path = write_hybrid_run_record(
        pilot_name=VPCD_QUANTIZED_TEACHER_FORCED_PILOT,
        runtime_config=runtime_config,
        target_reference=synthetic_reference,
        sample_results=[sample_result],
        run_label=run_label,
    )
    summary = _summarize_match_results([sample_result])
    return {
        "pilot_name": VPCD_QUANTIZED_TEACHER_FORCED_PILOT,
        "target_reference": synthetic_reference,
        "results": [sample_result],
        "steps": step_results,
        "summary": summary,
        "record_path": record_path,
        "decode_step_limit": int(decode_step_limit),
    }


def _load_compile_record_context(target_reference: ResolvedCompiledTarget) -> dict[str, Any]:
    compile_record_path = target_reference.compile_record_path
    if compile_record_path is None or not compile_record_path.exists():
        return {}

    payload = json.loads(compile_record_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        return {}

    compatibility = payload.get("compatibility")
    return {
        "source_strategy": _normalize_optional_string(payload.get("source_strategy")),
        "quantize_stage": _normalize_optional_string(payload.get("quantize_stage")),
        "compile_compatibility": dict(compatibility) if isinstance(compatibility, Mapping) else {},
    }


def write_hybrid_run_record(
    *,
    pilot_name: str,
    runtime_config: Option1RuntimeConfig,
    target_reference: ResolvedCompiledTarget,
    sample_results: list[dict[str, Any]],
    run_label: str | None = None,
    output_path: str | Path | None = None,
) -> Path:
    record_path = _resolve_hybrid_record_path(
        runtime_config=runtime_config,
        pilot_name=pilot_name,
        run_label=run_label,
        output_path=output_path,
    )
    summary = _summarize_match_results(sample_results)
    payload = {
        "record_kind": "hybrid_run",
        "pilot_name": pilot_name,
        "device_name": runtime_config.device_name,
        "qairt_version": runtime_config.qairt_version,
        "compute_unit": runtime_config.compute_unit,
        "target_model_id": target_reference.target_model_id,
        "compile_pilot_name": target_reference.compile_pilot_name,
        "compile_record_path": (
            target_reference.compile_record_path.as_posix() if target_reference.compile_record_path is not None else None
        ),
        "explicit_target_model_override": bool(target_reference.explicit_override),
        "run_label": _normalize_optional_string(run_label) or "latest",
        "summary": summary,
        "latency_summary": _summarize_latency_fields(sample_results),
        "sample_results": [_json_safe_dict(row) for row in sample_results],
        "record_path": record_path.as_posix(),
        "created_at_utc": _utc_now_isoformat(),
    }
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return record_path


def _submit_live_compiled_inference(
    *,
    target_model_id: str,
    runtime_config: Option1RuntimeConfig,
    inputs: dict[str, list[np.ndarray]],
    inference_name: str | None,
) -> tuple[Mapping[str, object], dict[str, Any]]:
    import qai_hub as hub

    target_model = hub.get_model(target_model_id)
    inference_job = hub.submit_inference_job(
        model=target_model,
        device=hub.Device(runtime_config.device_name),
        inputs=inputs,
        options=build_job_options(
            compute_unit=runtime_config.compute_unit,
            qairt_version=runtime_config.qairt_version,
        ),
        name=inference_name or f"compiled-inference-{target_model_id}",
    )
    return inference_job.download_output_data(), {
        "job_id": getattr(inference_job, "job_id", None),
        "url": getattr(inference_job, "url", None),
        "status": getattr(inference_job, "status", None),
    }


def _resolve_phase2_compile_record_path(
    *,
    runtime_config: Option1RuntimeConfig,
    compile_pilot_name: str,
    run_label: str | None,
) -> Path:
    normalized_label = _normalize_record_label(run_label or "latest")
    return (runtime_config.pilot_record_dir(compile_pilot_name) / f"compile-run-{normalized_label}.json").resolve()


def _resolve_hybrid_record_path(
    *,
    runtime_config: Option1RuntimeConfig,
    pilot_name: str,
    run_label: str | None,
    output_path: str | Path | None,
) -> Path:
    if output_path is not None:
        resolved = Path(output_path).resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved
    normalized_label = _normalize_record_label(run_label or "latest")
    record_dir = runtime_config.pilot_record_dir(pilot_name)
    record_dir.mkdir(parents=True, exist_ok=True)
    return (record_dir / f"hybrid-run-{normalized_label}.json").resolve()


def _load_zipformer_expected_texts(bundle_manifest_path: Path) -> dict[str, str]:
    manifest = ModelBundleManifest.from_path(bundle_manifest_path)
    expected_name = manifest.fixtures.get("expected_outputs")
    if not expected_name:
        return {}
    expected_path = bundle_manifest_path.parent / expected_name
    return {
        str(row["sample_id"]): str(row.get("text", ""))
        for row in read_jsonl(expected_path)
    }


def _load_zipformer_evaluation_rows(
    *,
    bundle_manifest_path: Path,
    repo_root: Path,
    sample_manifest_path: Path,
    max_samples: int,
) -> list[dict[str, Any]]:
    manifest = ModelBundleManifest.from_path(bundle_manifest_path)
    expected_name = manifest.fixtures.get("expected_outputs")
    if expected_name:
        expected_rows = read_jsonl(bundle_manifest_path.parent / expected_name)
        resolved_rows: list[dict[str, Any]] = []
        for row in expected_rows:
            audio_path = row.get("audio_path")
            if not audio_path:
                continue
            resolved_audio_path = _resolve_repo_relative_path(repo_root, audio_path)
            if not resolved_audio_path.exists():
                continue
            resolved_rows.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "audio_path": str(audio_path),
                    "expected_text": str(row.get("text", "")),
                }
            )
        if resolved_rows:
            return resolved_rows[: max(0, int(max_samples))]

    expected_by_sample = _load_zipformer_expected_texts(bundle_manifest_path)
    fallback_rows: list[dict[str, Any]] = []
    for row in read_jsonl(sample_manifest_path)[: max(0, int(max_samples))]:
        fallback_rows.append(
            {
                "sample_id": str(row["sample_id"]),
                "audio_path": str(row["audio_path"]),
                "expected_text": expected_by_sample.get(str(row["sample_id"]), ""),
            }
        )
    return fallback_rows


def _wrap_inference_inputs(inputs: Mapping[str, np.ndarray] | Mapping[str, list[np.ndarray]]) -> dict[str, list[np.ndarray]]:
    wrapped: dict[str, list[np.ndarray]] = {}
    for name, value in inputs.items():
        if isinstance(value, list):
            wrapped[name] = [np.asarray(item) for item in value]
        else:
            wrapped[name] = [np.asarray(value)]
    return wrapped


def _unwrap_single_output_tensor(name: str, value: object) -> np.ndarray:
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"Expected exactly one tensor for output '{name}', got {len(value)}")
        return np.asarray(value[0])
    return np.asarray(value)


def _resolve_output_array(
    outputs: Mapping[str, np.ndarray],
    *,
    preferred_names: Sequence[str],
    allow_missing: bool = False,
) -> np.ndarray | None:
    for name in preferred_names:
        if name in outputs:
            return np.asarray(outputs[name])
    if allow_missing:
        return None
    if not outputs:
        raise ValueError("Compiled inference did not return any outputs.")
    first_name = sorted(outputs.keys())[0]
    return np.asarray(outputs[first_name])


def _summarize_match_results(sample_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sample_count = len(sample_results)
    comparable_rows = [row for row in sample_results if row.get("matches_expected") is not None]
    comparable_samples = len(comparable_rows)
    matched = sum(1 for row in comparable_rows if bool(row.get("matches_expected")))
    mismatched = sum(1 for row in comparable_rows if row.get("matches_expected") is False)
    mismatch_items = [
        row.get("sample_id", row.get("sample_index"))
        for row in comparable_rows
        if row.get("matches_expected") is False
    ]
    unavailable_items = [
        row.get("sample_id", row.get("sample_index"))
        for row in sample_results
        if row.get("matches_expected") is None
    ]
    return {
        "sample_count": sample_count,
        "comparable_samples": comparable_samples,
        "matched_samples": matched,
        "mismatched_samples": mismatched,
        "mismatch_items": mismatch_items,
        "comparison_unavailable_samples": sample_count - comparable_samples,
        "comparison_unavailable_items": unavailable_items,
    }


def _summarize_latency_fields(sample_results: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    cloud_values = [float(row["cloud_inference_seconds"]) for row in sample_results if row.get("cloud_inference_seconds") is not None]
    decode_values = [float(row["decode_seconds"]) for row in sample_results if row.get("decode_seconds") is not None]
    return {
        "average_cloud_inference_seconds": round(sum(cloud_values) / len(cloud_values), 6) if cloud_values else None,
        "average_decode_seconds": round(sum(decode_values) / len(decode_values), 6) if decode_values else None,
    }


def _normalize_optional_string(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _normalize_optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _normalize_record_label(value: str) -> str:
    normalized = "".join(char if char.isalnum() or char in ("-", "_") else "-" for char in str(value).strip())
    collapsed = "-".join(part for part in normalized.split("-") if part)
    return collapsed or "run"


def _normalize_job_metadata(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(key): value[key] for key in value}
    return {
        key: getattr(value, key)
        for key in ("job_id", "url", "status")
        if getattr(value, key, None) is not None
    } or None


def _json_safe_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_safe(item) for key, item in value.items()}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _json_safe_dict(value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _utc_now_isoformat() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _resolve_repo_relative_path(repo_root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (repo_root / path)
