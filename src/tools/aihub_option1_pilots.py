from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from shutil import copy2
from typing import Any, Callable, Mapping

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

from model_bundle.fixtures import AudioSampleFixture, read_jsonl
from model_bundle.manifest import ModelBundleManifest
from model_bundle.projects.vpcd_shapes import (
    attention_mask_for_length,
    pad_token_row,
    resolve_vpcd_model_input_shapes,
)
from model_bundle.projects.zipformer import ModelDirAcousticRuntime, prepare_encoder_inputs, resolve_fixed_encoder_frames
from quantize.calibration import (
    greedy_decode_ids,
    iter_calibration_texts,
    load_decoder_start_token_id,
    resolve_ort_providers,
)
from quantize.projects.vpcd import (
    DEFAULT_PRESET as VPCD_DEFAULT_PRESET,
    build_vpcd_aihub_quantize_recipe,
    inspect_vpcd_qdq_compile_candidate,
    resolve_vpcd_aihub_quantize_dtype_names as resolve_vpcd_quantize_dtype_names_from_preset,
)
from quantize.fixed_shapes import freeze_model_inputs
from tools.paths import resolve_repo_path
from transformers import AutoTokenizer

DEFAULT_TARGET_RUNTIME = "precompiled_qnn_onnx"
DEFAULT_COMPUTE_UNIT = "npu"
InputSpecs = dict[str, tuple[tuple[int, ...], str]]
MS_QDQ_OP_TYPES = {"QuantizeLinear", "DequantizeLinear"}
ZIPFORMER_BOOL_SLICE_NODE_NAMES = (
    "/encoder/Slice_1",
    "/encoder/Slice_3",
    "/encoder/Slice_5",
)
ZIPFORMER_BOOL_UNSQUEEZE_NODE_NAMES = (
    "/encoder/1/encoder/0/self_attn_weights/Unsqueeze_15",
    "/encoder/2/encoder/0/self_attn_weights/Unsqueeze_15",
    "/encoder/3/encoder/0/self_attn_weights/Unsqueeze_15",
)


@dataclass(frozen=True)
class ZipformerEncoderPilotSource:
    repo_root: Path
    source_model_path: Path
    bundle_manifest_path: Path
    sample_manifest_path: Path
    fixed_encoder_frames: int
    sample_rate: int
    feature_dim: int


@dataclass(frozen=True)
class VpcdPilotSource:
    repo_root: Path
    bundle_manifest_path: Path
    model_path: Path
    golden_samples_path: Path
    encoder_sequence: int
    decoder_sequence: int
    pad_token_id: int
    eos_token_id: int
    decoder_start_token_id: int
    input_text_case: str
    is_quantized_source: bool


@dataclass(frozen=True)
class PreparedVpcdOption1Source:
    prepared_model_path: Path
    is_quantized_source: bool
    source_strategy: str
    source_kind: str
    packaging_kind: str
    packaging_path: Path
    transformation_kind: str
    report: dict[str, Any] | None = None
    graph_report: dict[str, Any] | None = None

    def __iter__(self):
        yield self.prepared_model_path
        yield self.is_quantized_source


@dataclass(frozen=True)
class Option1RuntimeConfig:
    repo_root: Path
    device_name: str
    qairt_version: str | None
    compute_unit: str
    artifact_root: Path
    record_root: Path

    def pilot_artifact_dir(self, pilot_name: str) -> Path:
        return (self.artifact_root / _normalize_record_label(pilot_name)).resolve()

    def pilot_record_dir(self, pilot_name: str) -> Path:
        return (self.record_root / _normalize_record_label(pilot_name)).resolve()


def load_env_file(
    env_path: str | Path | None = None,
    *,
    repo_root: str | Path | None = None,
    override: bool = False,
) -> dict[str, str]:
    resolved_env_path = (
        Path(env_path).resolve()
        if env_path is not None
        else (_resolve_repo_root(repo_root) / ".env").resolve()
    )
    if not resolved_env_path.exists():
        return {}

    loaded: dict[str, str] = {}
    for raw_line in resolved_env_path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        key, separator, value = stripped.partition("=")
        if separator != "=":
            continue
        normalized_key = key.strip()
        if not normalized_key:
            continue
        normalized_value = _strip_optional_quotes(value.strip())
        loaded[normalized_key] = normalized_value
        if override or normalized_key not in os.environ:
            os.environ[normalized_key] = normalized_value
    return loaded


def resolve_qai_hub_api_token(
    *,
    repo_root: str | Path | None = None,
    env_path: str | Path | None = None,
    env_var_name: str = "QAI_HUB_API_TOKEN",
    override: bool = False,
) -> str | None:
    load_env_file(env_path, repo_root=repo_root, override=override)
    return _normalize_optional_string(os.environ.get(env_var_name))


def build_option1_runtime_config(
    *,
    device_name: str,
    qairt_version: str | None = None,
    compute_unit: str = DEFAULT_COMPUTE_UNIT,
    repo_root: str | Path | None = None,
    artifact_root: str | Path | None = None,
    record_root: str | Path | None = None,
) -> Option1RuntimeConfig:
    resolved_repo_root = _resolve_repo_root(repo_root)
    normalized_device_name = (device_name or "").strip()
    if not normalized_device_name:
        raise ValueError("device_name must not be empty.")

    normalized_qairt_version = _normalize_optional_string(qairt_version)
    normalized_compute_unit = _normalize_optional_string(compute_unit) or DEFAULT_COMPUTE_UNIT
    resolved_artifact_root = (
        Path(artifact_root).resolve()
        if artifact_root is not None
        else (resolved_repo_root / "build" / "aihub").resolve()
    )
    resolved_record_root = (
        Path(record_root).resolve()
        if record_root is not None
        else (resolved_artifact_root / "records").resolve()
    )
    resolved_artifact_root.mkdir(parents=True, exist_ok=True)
    resolved_record_root.mkdir(parents=True, exist_ok=True)

    return Option1RuntimeConfig(
        repo_root=resolved_repo_root,
        device_name=normalized_device_name,
        qairt_version=normalized_qairt_version,
        compute_unit=normalized_compute_unit,
        artifact_root=resolved_artifact_root,
        record_root=resolved_record_root,
    )


def requires_truncate_64bit_io(input_specs: InputSpecs | None) -> bool:
    if not input_specs:
        return False
    for _name, spec in input_specs.items():
        if not isinstance(spec, (list, tuple)) or len(spec) < 2:
            continue
        dtype = str(spec[1]).strip().lower()
        if dtype == "int64":
            return True
    return False


def build_compile_options(
    *,
    qairt_version: str | None = None,
    input_specs: InputSpecs | None = None,
) -> str:
    options = [f"--target_runtime {DEFAULT_TARGET_RUNTIME}"]
    if requires_truncate_64bit_io(input_specs):
        options.append("--truncate_64bit_io")
    if qairt_version and qairt_version.strip():
        options.append(f"--qairt_version {qairt_version.strip()}")
    return " ".join(options)


def build_job_options(*, compute_unit: str = DEFAULT_COMPUTE_UNIT, qairt_version: str | None = None) -> str:
    options = []
    normalized_compute_unit = (compute_unit or "").strip()
    if normalized_compute_unit:
        options.append(f"--compute_unit {normalized_compute_unit}")
    if qairt_version and qairt_version.strip():
        options.append(f"--qairt_version {qairt_version.strip()}")
    return " ".join(options)


def resolve_zipformer_encoder_pilot_source(repo_root: str | Path | None = None) -> ZipformerEncoderPilotSource:
    root = _resolve_repo_root(repo_root)
    manifest_path = _first_existing_path(
        [
            root / "build" / "model_bundle" / "zipformer" / "qnn_u16u8" / "bundle_manifest.json",
            root / "build" / "model_bundle" / "zipformer" / "fp32" / "bundle_manifest.json",
            root / "build" / "zipformer" / "bundle_manifest.json",
        ]
    )
    if manifest_path is None:
        raise FileNotFoundError("Could not resolve a Zipformer bundle manifest for the AI Hub pilot.")

    manifest = ModelBundleManifest.from_path(manifest_path)
    if manifest.project != "zipformer":
        raise ValueError(f"Expected a zipformer manifest, got: {manifest.project}")

    bundle_dir = manifest_path.parent
    source_model_path = _first_existing_path(
        [
            root / "build" / "quantize" / "zipformer" / "qnn_u16u8" / "fixed_shapes" / "encoder.fixed.onnx",
            bundle_dir / "artifacts" / "fixed_shapes" / "encoder.fixed.onnx",
            root / "build" / "zipformer" / "artifacts" / "fixed_shapes" / "encoder.fixed.onnx",
            bundle_dir / manifest.artifacts["encoder"],
        ]
    )
    if source_model_path is None:
        raise FileNotFoundError("Could not resolve a Zipformer encoder source model for the AI Hub pilot.")

    sample_manifest_name = manifest.fixtures.get("sample_manifest")
    if not sample_manifest_name:
        raise ValueError("Zipformer manifest does not include a sample_manifest fixture.")
    sample_manifest_path = bundle_dir / sample_manifest_name

    fixed_encoder_frames = resolve_fixed_encoder_frames(manifest.metadata)
    if fixed_encoder_frames is None:
        raise ValueError("Zipformer manifest does not expose fixed encoder frames for the AI Hub pilot.")

    return ZipformerEncoderPilotSource(
        repo_root=root,
        source_model_path=source_model_path,
        bundle_manifest_path=manifest_path,
        sample_manifest_path=sample_manifest_path,
        fixed_encoder_frames=int(fixed_encoder_frames),
        sample_rate=int(manifest.metadata.get("sample_rate", 16000)),
        feature_dim=int(manifest.metadata.get("feature_dim", 80)),
    )


def build_zipformer_encoder_input_specs(source: ZipformerEncoderPilotSource) -> dict[str, tuple[tuple[int, ...], str]]:
    return {
        "x": ((1, int(source.fixed_encoder_frames), int(source.feature_dim)), "float32"),
        "x_lens": ((1,), "int64"),
    }


def build_zipformer_encoder_calibration_entries(
    source: ZipformerEncoderPilotSource,
    *,
    max_samples: int | None = None,
    feature_loader: Callable[..., np.ndarray] | None = None,
) -> dict[str, list[np.ndarray]]:
    fixtures = _read_audio_fixtures(source.sample_manifest_path)
    if max_samples is not None:
        fixtures = fixtures[: max(0, int(max_samples))]
    dataset = {"x": [], "x_lens": []}
    loader = feature_loader or _default_zipformer_feature_loader
    for fixture in fixtures:
        audio_path = source.repo_root / fixture.audio_path
        features = loader(
            audio_path,
            sample_rate=int(source.sample_rate),
            feature_dim=int(source.feature_dim),
        )
        encoder_inputs = prepare_encoder_inputs(features, fixed_encoder_frames=int(source.fixed_encoder_frames))
        dataset["x"].append(np.asarray(encoder_inputs["x"], dtype=np.float32))
        dataset["x_lens"].append(np.asarray(encoder_inputs["x_lens"], dtype=np.int64))
    return dataset


def build_zipformer_encoder_inference_entries(
    source: ZipformerEncoderPilotSource,
    *,
    sample_id: str | None = None,
    feature_loader: Callable[..., np.ndarray] | None = None,
) -> dict[str, list[np.ndarray]]:
    fixtures = _read_audio_fixtures(source.sample_manifest_path)
    if sample_id is not None:
        fixtures = [fixture for fixture in fixtures if fixture.sample_id == sample_id]
        if not fixtures:
            raise ValueError(f"Zipformer sample_id not found in sample manifest: {sample_id}")
    else:
        fixtures = fixtures[:1]

    dataset = {"x": [], "x_lens": []}
    loader = feature_loader or _default_zipformer_feature_loader
    for fixture in fixtures:
        audio_path = source.repo_root / fixture.audio_path
        features = loader(
            audio_path,
            sample_rate=int(source.sample_rate),
            feature_dim=int(source.feature_dim),
        )
        encoder_inputs = prepare_encoder_inputs(features, fixed_encoder_frames=int(source.fixed_encoder_frames))
        dataset["x"].append(np.asarray(encoder_inputs["x"], dtype=np.float32))
        dataset["x_lens"].append(np.asarray(encoder_inputs["x_lens"], dtype=np.int64))
    return dataset


def strip_model_io_value_info_conflicts(
    model: onnx.ModelProto,
    *,
    extra_names: set[str] | None = None,
) -> None:
    io_names = {value.name for value in model.graph.input} | {value.name for value in model.graph.output}
    if extra_names:
        io_names.update(extra_names)
    kept = [value for value in model.graph.value_info if value.name not in io_names]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept)


def rewrite_zipformer_bool_mask_slices_for_htp(model: onnx.ModelProto) -> None:
    slice_names = set(ZIPFORMER_BOOL_SLICE_NODE_NAMES)
    unsqueeze_names = set(ZIPFORMER_BOOL_UNSQUEEZE_NODE_NAMES)
    shared_cast_name = "/GreaterOrEqual_output_0_u8_cast"
    shared_cast_output = "/GreaterOrEqual_output_0_u8"
    stale_value_info_names: set[str] = set()
    new_nodes: list[onnx.NodeProto] = []
    shared_cast_inserted = False

    for node in model.graph.node:
        if node.name in slice_names and not shared_cast_inserted:
            new_nodes.append(
                helper.make_node(
                    "Cast",
                    ["/GreaterOrEqual_output_0"],
                    [shared_cast_output],
                    name=shared_cast_name,
                    to=TensorProto.UINT8,
                )
            )
            shared_cast_inserted = True

        if node.name in slice_names:
            node.input[0] = shared_cast_output
            stale_value_info_names.add(node.output[0])

        if node.name in unsqueeze_names:
            original_output = node.output[0]
            temp_output = f"{original_output}_u8"
            stale_value_info_names.add(temp_output)
            node.output[0] = temp_output
            new_nodes.append(node)
            new_nodes.append(
                helper.make_node(
                    "Cast",
                    [temp_output],
                    [original_output],
                    name=f"{node.name}_cast_bool",
                    to=TensorProto.BOOL,
                )
            )
            continue

        new_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    strip_model_io_value_info_conflicts(model, extra_names=stale_value_info_names)


def prepare_zipformer_encoder_option1_source_model(
    source: ZipformerEncoderPilotSource,
    *,
    output_path: str | Path | None = None,
) -> Path:
    prepared_output_path = Path(output_path).resolve() if output_path is not None else (
        source.repo_root / "build" / "aihub" / "zipformer_encoder_option1" / "encoder.aihub.option1.onnx"
    ).resolve()
    work_dir = prepared_output_path.parent
    work_dir.mkdir(parents=True, exist_ok=True)

    optimized_model_path = work_dir / "encoder.fixed.optimized.onnx"
    symshape_model_path = work_dir / "encoder.fixed.optimized.symshape.onnx"

    _optimize_onnx_model_for_aihub(source.source_model_path, optimized_model_path)
    _run_symbolic_shape_inference(optimized_model_path, symshape_model_path)

    model = onnx.load(symshape_model_path.as_posix())
    strip_model_io_value_info_conflicts(model)
    rewrite_zipformer_bool_mask_slices_for_htp(model)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, prepared_output_path.as_posix())
    return prepared_output_path


def resolve_vpcd_pilot_source(repo_root: str | Path | None = None) -> VpcdPilotSource:
    root = _resolve_repo_root(repo_root)
    manifest_path = _first_existing_path(
        [
            root / "build" / "model_bundle" / "vpcd" / "qnn_fixed_1024x128" / "bundle_manifest.json",
            root / "build" / "model_bundle" / "vpcd" / "qnn_fixed_8x4" / "bundle_manifest.json",
        ]
    )
    if manifest_path is None:
        matches = sorted((root / "build" / "model_bundle" / "vpcd").glob("qnn_fixed_*/*bundle_manifest.json"))
        manifest_path = matches[0] if matches else None
    if manifest_path is None:
        raise FileNotFoundError("Could not resolve a fixed-shape VPCD bundle manifest for the AI Hub pilot.")

    manifest = ModelBundleManifest.from_path(manifest_path)
    if manifest.project != "vpcd":
        raise ValueError(f"Expected a vpcd manifest, got: {manifest.project}")

    shapes = resolve_vpcd_model_input_shapes(manifest.metadata)
    if shapes is None:
        raise ValueError("VPCD manifest does not expose fixed input shapes for the AI Hub pilot.")

    golden_samples_name = manifest.fixtures.get("golden_samples")
    if not golden_samples_name:
        raise ValueError("VPCD manifest does not include golden_samples.")

    quantization = manifest.metadata.get("quantization", {}) if isinstance(manifest.metadata, dict) else {}
    return VpcdPilotSource(
        repo_root=root,
        bundle_manifest_path=manifest_path,
        model_path=manifest_path.parent / manifest.artifacts["model"],
        golden_samples_path=manifest_path.parent / golden_samples_name,
        encoder_sequence=int(shapes.encoder_sequence),
        decoder_sequence=int(shapes.decoder_sequence),
        pad_token_id=int(manifest.metadata.get("pad_token_id", 1)),
        eos_token_id=int(manifest.metadata.get("eos_token_id", 2)),
        decoder_start_token_id=int(manifest.metadata.get("decoder_start_token_id", 2)),
        input_text_case=str(manifest.metadata.get("input_text_case", "")),
        is_quantized_source=str(quantization.get("format", "")).strip().upper() == "QDQ",
    )


def build_vpcd_input_specs(source: VpcdPilotSource) -> dict[str, tuple[tuple[int, ...], str]]:
    return {
        "input_ids": ((1, int(source.encoder_sequence)), "int64"),
        "attention_mask": ((1, int(source.encoder_sequence)), "int64"),
        "decoder_input_ids": ((1, int(source.decoder_sequence)), "int64"),
        "decoder_attention_mask": ((1, int(source.decoder_sequence)), "int64"),
    }


def build_vpcd_fixed_shape_inputs(
    source: VpcdPilotSource,
    *,
    input_ids: np.ndarray | list[int],
    decoder_prefix: list[int] | np.ndarray | None = None,
    attention_mask: np.ndarray | list[int] | None = None,
) -> dict[str, np.ndarray]:
    model_ids = np.asarray(input_ids, dtype=np.int64).reshape(-1)
    prefix = np.asarray(
        [int(source.decoder_start_token_id)] if decoder_prefix is None else decoder_prefix,
        dtype=np.int64,
    ).reshape(-1)
    if prefix.size == 0:
        raise ValueError("decoder_prefix must contain at least one token.")

    if attention_mask is None:
        actual_encoder_length = int(model_ids.size)
    else:
        normalized_attention_mask = np.asarray(attention_mask, dtype=np.int64).reshape(-1)
        actual_encoder_length = int(normalized_attention_mask.sum())
        if actual_encoder_length <= 0:
            actual_encoder_length = int(model_ids.size)

    return {
        "input_ids": pad_token_row(
            model_ids,
            target_length=int(source.encoder_sequence),
            pad_value=int(source.pad_token_id),
        ),
        "attention_mask": attention_mask_for_length(
            actual_length=actual_encoder_length,
            target_length=int(source.encoder_sequence),
        ),
        "decoder_input_ids": pad_token_row(
            prefix,
            target_length=int(source.decoder_sequence),
            pad_value=int(source.pad_token_id),
        ),
        "decoder_attention_mask": attention_mask_for_length(
            actual_length=int(prefix.size),
            target_length=int(source.decoder_sequence),
        ),
    }


def build_vpcd_single_step_inputs(
    source: VpcdPilotSource,
    *,
    sample_index: int = 0,
    decoder_prefix: list[int] | np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    golden_samples = read_jsonl(source.golden_samples_path)
    if not golden_samples:
        raise ValueError("VPCD golden sample file is empty.")
    if sample_index < 0 or sample_index >= len(golden_samples):
        raise IndexError(f"VPCD sample_index out of range: {sample_index}")

    sample = golden_samples[sample_index]
    return build_vpcd_fixed_shape_inputs(
        source,
        input_ids=np.asarray(sample["input_ids"], dtype=np.int64),
        decoder_prefix=decoder_prefix,
    )


def build_vpcd_single_step_calibration_entries(
    source: VpcdPilotSource,
    *,
    max_samples: int | None = None,
) -> dict[str, list[np.ndarray]]:
    golden_samples = read_jsonl(source.golden_samples_path)
    if max_samples is not None:
        golden_samples = golden_samples[: max(0, int(max_samples))]

    dataset = {
        "input_ids": [],
        "attention_mask": [],
        "decoder_input_ids": [],
        "decoder_attention_mask": [],
    }
    for index, _sample in enumerate(golden_samples):
        inputs = build_vpcd_single_step_inputs(source, sample_index=index)
        for name, value in inputs.items():
            dataset[name].append(np.asarray(value, dtype=np.int64))
    return dataset


def resolve_vpcd_model_dir(source: VpcdPilotSource) -> Path | None:
    resolved_fp32_path = resolve_vpcd_fp32_source_model_path(source)
    if resolved_fp32_path is not None and resolved_fp32_path.parent.name == "onnx":
        candidate = resolved_fp32_path.parent.parent
        if candidate.exists():
            return candidate.resolve()
    candidate = source.repo_root / "assets" / "vietnamese-punc-cap-denorm-v1"
    if candidate.exists():
        return candidate.resolve()
    return None


def resolve_vpcd_calibration_text_path(source: VpcdPilotSource) -> Path | None:
    return _first_existing_path(
        [
            source.repo_root / "build" / "calibration" / "vlsp2020" / "vpcd_transcriptions.txt",
        ]
    )


def resolve_vpcd_aihub_quantize_dtype_names(source: VpcdPilotSource) -> dict[str, str]:
    manifest = ModelBundleManifest.from_path(source.bundle_manifest_path)
    metadata = manifest.metadata if isinstance(manifest.metadata, dict) else {}
    quantization = metadata.get("quantization", {}) if isinstance(metadata, dict) else {}
    preset = _normalize_optional_string(str(quantization.get("preset", ""))) or VPCD_DEFAULT_PRESET
    return resolve_vpcd_quantize_dtype_names_from_preset(preset=preset)


def build_vpcd_autoregressive_calibration_entries(
    source: VpcdPilotSource,
    *,
    fp32_model_path: str | Path | None = None,
    model_dir: str | Path | None = None,
    calibration_source_path: str | Path | None = None,
    max_samples: int | None = None,
    max_generation_length: int | None = None,
    ort_provider: str = "cpu",
    decode_ids_fn: Callable[[str], tuple[dict[str, np.ndarray], list[int]]] | None = None,
) -> tuple[dict[str, list[np.ndarray]], dict[str, Any]]:
    if decode_ids_fn is None:
        resolved_fp32_model_path = (
            Path(fp32_model_path).resolve()
            if fp32_model_path is not None
            else resolve_vpcd_fp32_source_model_path(source)
        )
        if resolved_fp32_model_path is None:
            raise FileNotFoundError("Could not resolve a VPCD FP32 ONNX source model for autoregressive calibration.")

        resolved_model_dir = (
            Path(model_dir).resolve()
            if model_dir is not None
            else resolve_vpcd_model_dir(source)
        )
        if resolved_model_dir is None:
            raise FileNotFoundError("Could not resolve a VPCD model directory for autoregressive calibration.")

        resolved_calibration_source_path = (
            Path(calibration_source_path).resolve()
            if calibration_source_path is not None
            else resolve_vpcd_calibration_text_path(source)
        )
        manifest = ModelBundleManifest.from_path(source.bundle_manifest_path)
        metadata = manifest.metadata if isinstance(manifest.metadata, dict) else {}
        quantization = metadata.get("quantization", {}) if isinstance(metadata, dict) else {}
        preset = _normalize_optional_string(str(quantization.get("preset", ""))) or VPCD_DEFAULT_PRESET
        if resolved_calibration_source_path is not None:
            recipe = build_vpcd_aihub_quantize_recipe(
                model_dir=resolved_model_dir,
                fp32_onnx_path=resolved_fp32_model_path,
                calibration_source_path=resolved_calibration_source_path,
                preset=preset,
                max_calibration_samples=max(0, int(max_samples)) if max_samples is not None else 128,
                max_generation_length=max(1, int(max_generation_length or source.decoder_sequence)),
                ort_provider=ort_provider,
                fixed_input_shapes={
                    "input_ids": (1, int(source.encoder_sequence)),
                    "attention_mask": (1, int(source.encoder_sequence)),
                    "decoder_input_ids": (1, int(source.decoder_sequence)),
                    "decoder_attention_mask": (1, int(source.decoder_sequence)),
                },
                pad_token_id=int(source.pad_token_id),
            )
            return recipe.calibration_dataset, dict(recipe.calibration_stats)

        tokenizer = AutoTokenizer.from_pretrained(resolved_model_dir, local_files_only=True)
        decoder_start_token_id = load_decoder_start_token_id(resolved_model_dir, tokenizer)
        session = ort.InferenceSession(
            os.fspath(resolved_fp32_model_path),
            providers=resolve_ort_providers(ort_provider),
        )

        def _decode_ids_with_fp32(text: str) -> tuple[dict[str, np.ndarray], list[int]]:
            return greedy_decode_ids(
                session=session,
                tokenizer=tokenizer,
                text=text,
                decoder_start_token_id=decoder_start_token_id,
                max_generation_length=max(1, int(max_generation_length or source.decoder_sequence)),
            )

        decode_ids_fn = _decode_ids_with_fp32
        stats_fallback = {
            "session_providers": ",".join(session.get_providers()),
            "fp32_model_path": resolved_fp32_model_path.as_posix(),
            "model_dir": resolved_model_dir.as_posix(),
        }
    else:
        stats_fallback = {}

    sample_limit = None if max_samples is None else max(0, int(max_samples))
    normalized_generation_length = max(1, int(max_generation_length or source.decoder_sequence))
    dataset = {
        "input_ids": [],
        "attention_mask": [],
        "decoder_input_ids": [],
        "decoder_attention_mask": [],
    }
    stats: dict[str, Any] = {
        "strategy": "autoregressive_fp32",
        "text_samples": 0,
        "records": 0,
        "max_encoder_len": 0,
        "max_decoder_prefix_len": 0,
        "requested_provider": ort_provider,
        "session_providers": "injected",
        "calibration_source_path": None,
        "fp32_model_path": None,
        "model_dir": None,
        "quantize_preset": None,
        "activation_type": None,
        "weight_type": None,
    }
    stats.update(stats_fallback)

    texts: list[str] = []
    if calibration_source_path is not None:
        resolved_calibration_source_path = Path(calibration_source_path).resolve()
        stats["calibration_source_path"] = resolved_calibration_source_path.as_posix()
        limit = sample_limit if sample_limit is not None else 2**31 - 1
        texts.extend(iter_calibration_texts(resolved_calibration_source_path, max_samples=limit))
    else:
        resolved_calibration_source_path = resolve_vpcd_calibration_text_path(source)
        if resolved_calibration_source_path is not None:
            stats["calibration_source_path"] = resolved_calibration_source_path.as_posix()
            limit = sample_limit if sample_limit is not None else 2**31 - 1
            texts.extend(iter_calibration_texts(resolved_calibration_source_path, max_samples=limit))
        else:
            for sample in read_jsonl(source.golden_samples_path):
                raw_text = str(sample.get("raw_text", "")).strip()
                if raw_text:
                    texts.append(raw_text)
                    if sample_limit is not None and len(texts) >= sample_limit:
                        break

    if sample_limit is not None:
        texts = texts[:sample_limit]
    if not texts:
        raise ValueError("Could not resolve any VPCD calibration texts for autoregressive calibration.")

    for text in texts:
        encoder_inputs, decoded_ids = decode_ids_fn(text)
        normalized_decoded_ids = [int(token_id) for token_id in decoded_ids]
        if not normalized_decoded_ids:
            raise ValueError("decode_ids_fn returned an empty decoded_ids sequence.")
        if normalized_decoded_ids[0] != int(source.decoder_start_token_id):
            raise ValueError("decoded_ids must start with source.decoder_start_token_id.")

        encoder_input_ids = np.asarray(encoder_inputs["input_ids"], dtype=np.int64).reshape(-1)
        encoder_attention_mask = np.asarray(encoder_inputs["attention_mask"], dtype=np.int64).reshape(-1)
        if len(normalized_decoded_ids) == 1:
            prefix_lengths = [1]
        else:
            prefix_lengths = list(range(1, min(len(normalized_decoded_ids), int(source.decoder_sequence) + 1)))

        for prefix_length in prefix_lengths:
            inputs = build_vpcd_fixed_shape_inputs(
                source,
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
                decoder_prefix=normalized_decoded_ids[:prefix_length],
            )
            for name, value in inputs.items():
                dataset[name].append(np.asarray(value, dtype=np.int64))
            stats["records"] = int(stats["records"]) + 1
            stats["max_decoder_prefix_len"] = max(int(stats["max_decoder_prefix_len"]), int(prefix_length))

        stats["text_samples"] = int(stats["text_samples"]) + 1
        stats["max_encoder_len"] = max(int(stats["max_encoder_len"]), int(encoder_attention_mask.sum()))

    return dataset, stats


def wrap_single_inference_inputs(inputs: dict[str, np.ndarray]) -> dict[str, list[np.ndarray]]:
    return {name: [value] for name, value in inputs.items()}


def resolve_vpcd_fp32_source_model_path(source: VpcdPilotSource) -> Path | None:
    return _first_existing_path(
        [
            source.repo_root / "assets" / "vietnamese-punc-cap-denorm-v1" / "onnx" / "model.fp32.onnx",
            source.repo_root / "build" / "export" / "vpcd" / "onnx" / "model.fp32.onnx",
        ]
    )


def rewrite_aihub_compatible_qdq_domains(model: onnx.ModelProto) -> None:
    touched_ms_qdq = False
    for node in model.graph.node:
        if node.domain == "com.microsoft" and node.op_type in MS_QDQ_OP_TYPES:
            node.domain = ""
            touched_ms_qdq = True

    if not touched_ms_qdq:
        return

    has_remaining_ms_domain_nodes = any(node.domain == "com.microsoft" for node in model.graph.node)
    preserved_opsets = [
        opset
        for opset in model.opset_import
        if opset.domain != "com.microsoft" or has_remaining_ms_domain_nodes
    ]
    del model.opset_import[:]
    model.opset_import.extend(preserved_opsets)


def _package_aihub_onnx_upload(model_path: Path) -> tuple[str, Path]:
    resolved_model_path = model_path.resolve()
    if resolved_model_path.is_dir():
        return "onnx_dir", resolved_model_path

    external_data_path = resolved_model_path.with_suffix(f"{resolved_model_path.suffix}.data")
    if not external_data_path.exists():
        return "onnx_file", resolved_model_path

    package_dir = (resolved_model_path.parent / f"{resolved_model_path.stem}.upload.onnx").resolve()
    package_dir.mkdir(parents=True, exist_ok=True)
    packaged_model_path = package_dir / resolved_model_path.name
    packaged_data_path = package_dir / external_data_path.name
    copy2(resolved_model_path, packaged_model_path)
    copy2(external_data_path, packaged_data_path)
    return "onnx_dir", package_dir


def _copy_onnx_artifact(source_model_path: Path, destination_model_path: Path) -> None:
    source_path = source_model_path.resolve()
    destination_path = destination_model_path.resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    copy2(source_path, destination_path)

    source_data_path = source_path.with_suffix(f"{source_path.suffix}.data")
    if source_data_path.exists():
        destination_data_path = destination_path.with_suffix(f"{destination_path.suffix}.data")
        copy2(source_data_path, destination_data_path)


def _build_local_qdq_compile_candidate_report(
    *,
    graph_report: Mapping[str, Any],
    packaging_kind: str,
    packaging_path: Path,
    transformation_kind: str,
) -> dict[str, Any]:
    graph_readiness = str(graph_report.get("aihub_compile_readiness", "unsafe"))
    packaging_ready = bool(graph_report.get("packaging_ready", False))
    compile_candidate_readiness = "unsafe"
    if packaging_ready:
        compile_candidate_readiness = "ready" if graph_readiness == "ready" else "experimental"
    return {
        "aihub_compile_readiness": compile_candidate_readiness,
        "graph_aihub_compile_readiness": graph_readiness,
        "packaging_kind": packaging_kind,
        "packaging_path": packaging_path.as_posix(),
        "transformation_kind": transformation_kind,
        "graph_report": dict(graph_report),
        "readiness_flags": list(graph_report.get("readiness_flags", [])),
    }


def prepare_vpcd_option1_source_model(
    source: VpcdPilotSource,
    *,
    output_path: str | Path | None = None,
    strategy: str | None = None,
) -> PreparedVpcdOption1Source:
    normalized_strategy = _normalize_optional_string(strategy)
    if normalized_strategy is None:
        normalized_strategy = "prefer_fp32_fixed"
    prepared_output_path = Path(output_path).resolve() if output_path is not None else (
        source.repo_root / "build" / "aihub" / "vpcd_fp32_fixed" / "model.fp32.fixed.onnx"
    ).resolve()
    if normalized_strategy in {"direct_qdq_sanitized", "local_qdq_compile_candidate"}:
        raw_graph_report = inspect_vpcd_qdq_compile_candidate(source.model_path)
        preserve_as_is_for_compile_probe = (
            normalized_strategy == "local_qdq_compile_candidate"
            and (
                bool(raw_graph_report.get("uses_uint16_qdq"))
                or bool(raw_graph_report.get("uses_int16_qdq"))
            )
            and int(dict(raw_graph_report.get("opsets", {})).get("main", 0)) < 21
        )
        prepared_output_path.parent.mkdir(parents=True, exist_ok=True)
        transformation_kind = "as_is"
        if preserve_as_is_for_compile_probe:
            _copy_onnx_artifact(source.model_path, prepared_output_path)
        else:
            model = onnx.load(source.model_path.as_posix())
            rewritten_domains = {
                (node.name, node.op_type): node.domain
                for node in model.graph.node
                if node.op_type in MS_QDQ_OP_TYPES
            }
            rewrite_aihub_compatible_qdq_domains(model)
            rewritten_domain_count = 0
            for node in model.graph.node:
                if node.op_type not in MS_QDQ_OP_TYPES:
                    continue
                previous_domain = rewritten_domains.get((node.name, node.op_type))
                if previous_domain == "com.microsoft" and node.domain == "":
                    rewritten_domain_count += 1
            if rewritten_domain_count > 0:
                transformation_kind = "domain_rewritten"
            onnx.checker.check_model(model, full_check=True)
            onnx.save(model, prepared_output_path.as_posix())
        packaging_kind, packaging_path = _package_aihub_onnx_upload(prepared_output_path)
        graph_report = inspect_vpcd_qdq_compile_candidate(prepared_output_path)
        report: dict[str, Any] | None = None
        if normalized_strategy == "local_qdq_compile_candidate":
            report = _build_local_qdq_compile_candidate_report(
                graph_report=graph_report,
                packaging_kind=packaging_kind,
                packaging_path=packaging_path,
                transformation_kind=transformation_kind,
            )
        return PreparedVpcdOption1Source(
            prepared_model_path=prepared_output_path,
            is_quantized_source=True,
            source_strategy=normalized_strategy,
            source_kind="local_qdq",
            packaging_kind=packaging_kind,
            packaging_path=packaging_path,
            transformation_kind=transformation_kind,
            report=report,
            graph_report=graph_report,
        )

    if normalized_strategy != "prefer_fp32_fixed":
        raise ValueError(f"Unsupported VPCD Option 1 source strategy: {normalized_strategy}")

    fp32_source_path = resolve_vpcd_fp32_source_model_path(source)
    if fp32_source_path is None:
        raise FileNotFoundError(
            "Could not resolve a VPCD FP32 ONNX source model for the FP32 -> AI Hub quantize debug lane."
        )

    input_shapes = {name: spec[0] for name, spec in build_vpcd_input_specs(source).items()}
    freeze_model_inputs(fp32_source_path, prepared_output_path, input_shapes)
    return PreparedVpcdOption1Source(
        prepared_model_path=prepared_output_path,
        is_quantized_source=False,
        source_strategy=normalized_strategy,
        source_kind="fp32_fixed",
        packaging_kind="onnx_file",
        packaging_path=prepared_output_path,
        transformation_kind="fixed_shape_freeze",
        report=None,
        graph_report=None,
    )


def summarize_tensor_outputs(
    output_tensors: Mapping[str, np.ndarray | list[np.ndarray]],
) -> dict[str, list[dict[str, Any]]]:
    summary: dict[str, list[dict[str, Any]]] = {}
    for name, value in output_tensors.items():
        values = value if isinstance(value, list) else [value]
        summary[name] = []
        for item in values:
            array = np.asarray(item)
            summary[name].append(
                {
                    "shape": [int(dim) for dim in array.shape],
                    "dtype": str(array.dtype),
                }
            )
    return summary


def compare_output_tensors(
    reference_outputs: Mapping[str, np.ndarray | list[np.ndarray]],
    candidate_outputs: Mapping[str, np.ndarray | list[np.ndarray]],
    *,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> dict[str, dict[str, Any]]:
    aligned_reference = _normalize_output_tensor_mapping(reference_outputs)
    aligned_candidate = _normalize_output_tensor_mapping(candidate_outputs)
    summary: dict[str, dict[str, Any]] = {}
    shared_slots = [slot for slot in aligned_reference if slot in aligned_candidate]
    for slot in shared_slots:
        reference_array = np.asarray(aligned_reference[slot][0])
        candidate_array = np.asarray(aligned_candidate[slot][0])
        shape_match = tuple(reference_array.shape) == tuple(candidate_array.shape)
        stats: dict[str, Any] = {
            "reference_dtype": str(reference_array.dtype),
            "candidate_dtype": str(candidate_array.dtype),
            "reference_shape": [int(dim) for dim in reference_array.shape],
            "candidate_shape": [int(dim) for dim in candidate_array.shape],
            "shape_match": shape_match,
            "allclose": False,
            "max_abs_diff": None,
            "mean_abs_diff": None,
        }
        if shape_match:
            difference = np.abs(reference_array.astype(np.float64) - candidate_array.astype(np.float64))
            stats["max_abs_diff"] = float(difference.max()) if difference.size else 0.0
            stats["mean_abs_diff"] = float(difference.mean()) if difference.size else 0.0
            stats["allclose"] = bool(np.allclose(reference_array, candidate_array, atol=atol, rtol=rtol))
        summary[slot] = stats
    return summary


def summarize_vpcd_step_logits(
    logits: np.ndarray,
    decoder_attention_mask: np.ndarray,
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    logits_array = np.asarray(logits)
    if logits_array.ndim < 2:
        raise ValueError(f"Expected logits with at least 2 dimensions, got shape {logits_array.shape}")

    sequence_logits = logits_array.reshape(-1, logits_array.shape[-2], logits_array.shape[-1])[0]
    active_index = resolve_active_decoder_position(decoder_attention_mask)
    scores = np.asarray(sequence_logits[active_index], dtype=np.float64).reshape(-1)
    k = max(1, min(int(top_k), int(scores.size)))
    top_indices = np.argsort(scores)[::-1][:k]
    return {
        "active_index": active_index,
        "top_tokens": [
            {
                "token_id": int(token_id),
                "score": float(scores[token_id]),
            }
            for token_id in top_indices
        ],
    }


def resolve_active_decoder_position(decoder_attention_mask: np.ndarray) -> int:
    attention_mask = np.asarray(decoder_attention_mask)
    if attention_mask.ndim == 0:
        raise ValueError("decoder_attention_mask must not be scalar.")
    flattened = attention_mask.reshape(attention_mask.shape[0], -1)[0]
    active_count = int(flattened.astype(np.int64).sum())
    if active_count <= 0:
        raise ValueError("decoder_attention_mask must contain at least one active position.")
    return active_count - 1


def write_prepared_artifact_record(
    *,
    pilot_name: str,
    runtime_config: Option1RuntimeConfig,
    source_model_path: str | Path,
    prepared_model_path: str | Path,
    input_specs: InputSpecs | None,
    compile_options: str,
    source_strategy: str | None = None,
    source_kind: str | None = None,
    packaging_kind: str | None = None,
    packaging_path: str | Path | None = None,
    compatibility: Mapping[str, Any] | None = None,
    run_label: str | None = None,
    output_path: str | Path | None = None,
) -> Path:
    resolved_source_model_path = Path(source_model_path).resolve()
    resolved_prepared_model_path = Path(prepared_model_path).resolve()
    resolved_packaging_path = Path(packaging_path).resolve() if packaging_path is not None else None
    record_path = _resolve_record_path(
        runtime_config=runtime_config,
        pilot_name=pilot_name,
        record_kind="prepared-artifact",
        run_label=run_label,
        output_path=output_path,
    )
    payload = {
        "record_kind": "prepared_artifact",
        "pilot_name": pilot_name,
        "device_name": runtime_config.device_name,
        "qairt_version": runtime_config.qairt_version,
        "compute_unit": runtime_config.compute_unit,
        "compile_options": compile_options,
        "input_specs": _serialize_input_specs(input_specs),
        "source_model": _build_file_metadata(resolved_source_model_path),
        "prepared_model": _build_file_metadata(resolved_prepared_model_path),
        "source_strategy": _normalize_optional_string(source_strategy),
        "source_kind": _normalize_optional_string(source_kind),
        "packaging_kind": _normalize_optional_string(packaging_kind),
        "packaging_path": resolved_packaging_path.as_posix() if resolved_packaging_path is not None else None,
        "compatibility": dict(compatibility or {}),
        "record_path": record_path.as_posix(),
        "created_at_utc": _utc_now_isoformat(),
    }
    return _write_json_record(record_path, payload)


def write_compile_run_record(
    *,
    pilot_name: str,
    runtime_config: Option1RuntimeConfig,
    compile_options: str,
    compile_job: Any = None,
    target_model: Any = None,
    source_strategy: str | None = None,
    quantize_stage: str | None = None,
    compatibility: Mapping[str, Any] | None = None,
    run_label: str | None = None,
    output_path: str | Path | None = None,
) -> Path:
    record_path = _resolve_record_path(
        runtime_config=runtime_config,
        pilot_name=pilot_name,
        record_kind="compile-run",
        run_label=run_label,
        output_path=output_path,
    )
    payload = {
        "record_kind": "compile_run",
        "pilot_name": pilot_name,
        "device_name": runtime_config.device_name,
        "qairt_version": runtime_config.qairt_version,
        "compute_unit": runtime_config.compute_unit,
        "compile_options": compile_options,
        "source_strategy": _normalize_optional_string(source_strategy),
        "quantize_stage": _normalize_optional_string(quantize_stage),
        "compatibility": dict(compatibility or {}),
        "jobs": {
            "compile": _extract_job_metadata(compile_job),
        },
        "target_model": _extract_model_metadata(target_model),
        "record_path": record_path.as_posix(),
        "created_at_utc": _utc_now_isoformat(),
    }
    return _write_json_record(record_path, payload)


def write_quantize_run_record(
    *,
    pilot_name: str,
    runtime_config: Option1RuntimeConfig,
    quantize_job: Any = None,
    target_model: Any = None,
    quantized_model_path: str | Path,
    weights_dtype_name: str,
    activations_dtype_name: str,
    quantize_options: str = "",
    calibration_stats: Mapping[str, Any] | None = None,
    run_label: str | None = None,
    output_path: str | Path | None = None,
) -> Path:
    resolved_quantized_model_path = Path(quantized_model_path).resolve()
    record_path = _resolve_record_path(
        runtime_config=runtime_config,
        pilot_name=pilot_name,
        record_kind="quantize-run",
        run_label=run_label,
        output_path=output_path,
    )
    payload = {
        "record_kind": "quantize_run",
        "pilot_name": pilot_name,
        "device_name": runtime_config.device_name,
        "qairt_version": runtime_config.qairt_version,
        "compute_unit": runtime_config.compute_unit,
        "weights_dtype_name": str(weights_dtype_name),
        "activations_dtype_name": str(activations_dtype_name),
        "quantize_options": str(quantize_options or ""),
        "jobs": {
            "quantize": _extract_job_metadata(quantize_job),
        },
        "target_model": _extract_model_metadata(target_model),
        "quantized_model": _build_file_metadata(resolved_quantized_model_path),
        "calibration": dict(calibration_stats or {}),
        "record_path": record_path.as_posix(),
        "created_at_utc": _utc_now_isoformat(),
    }
    return _write_json_record(record_path, payload)


def resolve_downloaded_quantized_model_path(
    *,
    pilot_name: str,
    runtime_config: Option1RuntimeConfig,
    explicit_quantized_model_path: str | Path | None = None,
    run_label: str | None = None,
) -> Path:
    if explicit_quantized_model_path is not None:
        return Path(explicit_quantized_model_path).resolve()

    record_path = _resolve_record_path(
        runtime_config=runtime_config,
        pilot_name=pilot_name,
        record_kind="quantize-run",
        run_label=run_label,
        output_path=None,
    )
    if not record_path.exists():
        raise FileNotFoundError(
            f"Could not resolve a quantize-run record for pilot '{pilot_name}' at: {record_path}"
        )

    payload = json.loads(record_path.read_text(encoding="utf-8"))
    quantized_model = payload.get("quantized_model") if isinstance(payload, Mapping) else None
    quantized_model_path = None
    if isinstance(quantized_model, Mapping):
        quantized_model_path = _normalize_optional_string(quantized_model.get("path"))
    if not quantized_model_path:
        raise ValueError(f"Quantize-run record does not include a quantized model path: {record_path}")
    return Path(quantized_model_path).resolve()


def download_quantized_target_model(
    *,
    quantize_job: Any,
    output_path: str | Path,
) -> Path:
    resolved_output_path = Path(output_path).resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    download_result = quantize_job.download_target_model(filename=resolved_output_path.as_posix())
    if isinstance(download_result, (str, Path)):
        resolved_output_path = Path(download_result).resolve()
    if not resolved_output_path.exists():
        raise FileNotFoundError(f"Quantized target model was not downloaded to: {resolved_output_path}")
    return resolved_output_path


def resolve_target_model_id(
    *,
    pilot_name: str,
    runtime_config: Option1RuntimeConfig,
    explicit_target_model_id: str | None = None,
    run_label: str | None = None,
) -> str:
    normalized_explicit_id = _normalize_optional_string(explicit_target_model_id)
    if normalized_explicit_id:
        return normalized_explicit_id

    record_path = _resolve_record_path(
        runtime_config=runtime_config,
        pilot_name=pilot_name,
        record_kind="compile-run",
        run_label=run_label,
        output_path=None,
    )
    if not record_path.exists():
        raise FileNotFoundError(
            f"Could not resolve a compile-run record for pilot '{pilot_name}' at: {record_path}"
        )

    payload = json.loads(record_path.read_text(encoding="utf-8"))
    target_model = payload.get("target_model") if isinstance(payload, Mapping) else None
    target_model_id = None
    if isinstance(target_model, Mapping):
        target_model_id = _normalize_optional_string(target_model.get("model_id"))
    if not target_model_id:
        raise ValueError(f"Compile-run record does not include a target model id: {record_path}")
    return target_model_id


def write_live_run_record(
    *,
    pilot_name: str,
    runtime_config: Option1RuntimeConfig,
    compile_options: str,
    job_options: str,
    compile_job: Any = None,
    profile_job: Any = None,
    inference_job: Any = None,
    output_tensors: Mapping[str, np.ndarray | list[np.ndarray]] | None = None,
    profile_path: str | Path | None = None,
    run_label: str | None = None,
    output_path: str | Path | None = None,
) -> Path:
    record_path = _resolve_record_path(
        runtime_config=runtime_config,
        pilot_name=pilot_name,
        record_kind="live-run",
        run_label=run_label,
        output_path=output_path,
    )
    resolved_profile_path = Path(profile_path).resolve() if profile_path is not None else None
    payload = {
        "record_kind": "live_run",
        "pilot_name": pilot_name,
        "device_name": runtime_config.device_name,
        "qairt_version": runtime_config.qairt_version,
        "compute_unit": runtime_config.compute_unit,
        "compile_options": compile_options,
        "job_options": job_options,
        "jobs": {
            "compile": _extract_job_metadata(compile_job),
            "profile": _extract_job_metadata(profile_job),
            "inference": _extract_job_metadata(inference_job),
        },
        "output_tensors": summarize_tensor_outputs(output_tensors or {}),
        "profile_artifact": _build_file_metadata(resolved_profile_path) if resolved_profile_path is not None else None,
        "record_path": record_path.as_posix(),
        "created_at_utc": _utc_now_isoformat(),
    }
    return _write_json_record(record_path, payload)


def coerce_inputs_for_compiled_model(
    inputs: dict[str, np.ndarray] | dict[str, list[np.ndarray]],
    *,
    input_specs: InputSpecs | None,
) -> dict[str, np.ndarray] | dict[str, list[np.ndarray]]:
    if not inputs or not input_specs or not requires_truncate_64bit_io(input_specs):
        return inputs

    coerced: dict[str, np.ndarray] | dict[str, list[np.ndarray]] = {}
    for name, value in inputs.items():
        spec = input_specs.get(name)
        dtype = str(spec[1]).strip().lower() if spec and len(spec) >= 2 else ""
        if dtype != "int64":
            coerced[name] = value
            continue
        if isinstance(value, list):
            coerced[name] = [np.asarray(item, dtype=np.int32) for item in value]
        else:
            coerced[name] = np.asarray(value, dtype=np.int32)
    return coerced


def _optimize_onnx_model_for_aihub(source_model_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    session_options.optimized_model_filepath = output_path.as_posix()
    ort.InferenceSession(source_model_path.as_posix(), sess_options=session_options, providers=["CPUExecutionProvider"])
    return output_path.resolve()


def _run_symbolic_shape_inference(source_model_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(source_model_path.as_posix())
    inferred = SymbolicShapeInference.infer_shapes(model, auto_merge=True, guess_output_rank=True, verbose=0)
    onnx.save(inferred, output_path.as_posix())
    return output_path.resolve()


def _resolve_repo_root(repo_root: str | Path | None) -> Path:
    if repo_root is not None:
        return Path(repo_root).resolve()
    return resolve_repo_path(".", anchor=__file__)


def _first_existing_path(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _read_audio_fixtures(sample_manifest_path: Path) -> list[AudioSampleFixture]:
    return [AudioSampleFixture.from_dict(row) for row in read_jsonl(sample_manifest_path)]


def _default_zipformer_feature_loader(audio_path: Path, *, sample_rate: int, feature_dim: int) -> np.ndarray:
    return ModelDirAcousticRuntime._load_features(
        audio_path,
        sample_rate=int(sample_rate),
        feature_dim=int(feature_dim),
    )


def _normalize_optional_string(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _normalize_record_label(value: str) -> str:
    normalized = "".join(char if char.isalnum() or char in ("-", "_") else "-" for char in str(value).strip())
    collapsed = "-".join(part for part in normalized.split("-") if part)
    return collapsed or "run"


def _serialize_input_specs(input_specs: InputSpecs | None) -> dict[str, dict[str, Any]]:
    if not input_specs:
        return {}
    serialized: dict[str, dict[str, Any]] = {}
    for name, spec in input_specs.items():
        shape = spec[0] if len(spec) >= 1 else ()
        dtype = spec[1] if len(spec) >= 2 else ""
        serialized[name] = {
            "shape": [int(dim) for dim in shape],
            "dtype": str(dtype),
        }
    return serialized


def _hash_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_file_metadata(path: Path) -> dict[str, Any]:
    resolved_path = Path(path).resolve()
    return {
        "path": resolved_path.as_posix(),
        "size_bytes": int(resolved_path.stat().st_size),
        "sha256": _hash_file_sha256(resolved_path),
    }


def _resolve_record_path(
    *,
    runtime_config: Option1RuntimeConfig,
    pilot_name: str,
    record_kind: str,
    run_label: str | None,
    output_path: str | Path | None,
) -> Path:
    if output_path is not None:
        resolved = Path(output_path).resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved

    pilot_record_dir = runtime_config.pilot_record_dir(pilot_name)
    pilot_record_dir.mkdir(parents=True, exist_ok=True)
    normalized_kind = _normalize_record_label(record_kind)
    normalized_label = _normalize_record_label(run_label or "latest")
    return (pilot_record_dir / f"{normalized_kind}-{normalized_label}.json").resolve()


def _extract_job_metadata(job: Any) -> dict[str, Any] | None:
    if job is None:
        return None
    if isinstance(job, Mapping):
        metadata = {key: job[key] for key in ("job_id", "url", "status") if key in job and job[key] is not None}
        return metadata or None

    metadata = {}
    for attr_name in ("job_id", "url", "status"):
        value = getattr(job, attr_name, None)
        if value is not None:
            metadata[attr_name] = value
    return metadata or None


def _extract_model_metadata(model: Any) -> dict[str, Any] | None:
    if model is None:
        return None
    if isinstance(model, Mapping):
        metadata = {
            key: model[key]
            for key in ("model_id", "url", "name")
            if key in model and model[key] is not None
        }
        return metadata or None

    metadata = {}
    for attr_name in ("model_id", "url", "name"):
        value = getattr(model, attr_name, None)
        if value is not None:
            metadata[attr_name] = value
    return metadata or None


def _utc_now_isoformat() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_json_record(path: Path, payload: Mapping[str, Any]) -> Path:
    resolved_path = Path(path).resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return resolved_path


def _normalize_output_tensor_mapping(
    output_tensors: Mapping[str, np.ndarray | list[np.ndarray]],
) -> dict[str, list[np.ndarray]]:
    normalized: dict[str, list[np.ndarray]] = {}
    for index, (_name, value) in enumerate(output_tensors.items()):
        values = value if isinstance(value, list) else [value]
        normalized[f"output_{index}"] = [np.asarray(item) for item in values]
    return normalized


def _strip_optional_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
