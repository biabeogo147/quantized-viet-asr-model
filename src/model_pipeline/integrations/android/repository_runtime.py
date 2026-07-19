"""Resolve exact retained sources for the canonical Android model repository."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from model_pipeline.core import ArtifactSpec, sha256_file
from model_pipeline.integrations.android.repository import (
    AndroidArtifactInput,
    AndroidComponentInput,
    ModelRepositoryResult,
    materialize_model_repository,
)
CANONICAL_COMPILED_CHECKSUMS = {
    "zipformer": "8568fdc6902679c5eda866c7ea5ce82a203a2d79a628c8d89d838e353539415d",
    "vpcd": "c2886b67e06461ddb9d8ee311afa7ef7bf4c48dc17fc9b27b5f26102a2384cb4",
}
CANONICAL_ARTIFACT_IDS = {
    ("zipformer", "fp32-fixed-shape"):
        "zipformer__q-none-fp32-fp32-none__s-enc1x2009x80-dec1x2-join1x512__c-none-cpu-none",
    ("zipformer", "aimet-int8-int16-encoder-matmul"):
        "zipformer__q-aimet-int8-int16-encoder-matmul__s-enc1x2009x80-dec1x2-join1x512__c-aihub-qnn-htp-encoder",
    ("vpcd", "fp32-fixed-shape"):
        "vpcd__q-none-fp32-fp32-none__s-src1x384-dec1x64__c-none-cpu-none",
    ("vpcd", "aimet-int8-int16-encoder-matmul"):
        "vpcd__q-aimet-int8-int16-encoder-matmul__s-src1x384-dec1x64__c-aihub-qnn-htp-model",
}


def materialize_canonical_repository(
    *,
    repo_root: Path,
    build_root: Path,
    destination: Path,
) -> ModelRepositoryResult:
    """Materialize the four canonical Android artifacts from retained exact sources.

    Args:
        repo_root: Quantized model repository root.
        build_root: Generated work directory for portable fixture manifests.
        destination: BKMeeting model repository destination.

    Returns:
        Promoted canonical repository result.

    Raises:
        FileNotFoundError: If retained exact sources are unavailable.
        ValueError: If retained checksums or artifact contracts differ.
    """
    retained_payload = Path(repo_root).resolve() / "build" / "qdc-benchmark" / "payload"
    artifacts = resolve_retained_repository_inputs(
        payload_root=retained_payload,
        generated_root=Path(build_root).resolve() / "repository-sources",
        expected_compiled_checksums=CANONICAL_COMPILED_CHECKSUMS,
    )
    return materialize_model_repository(artifacts=artifacts, destination=destination)


def resolve_retained_repository_inputs(
    *,
    payload_root: Path,
    generated_root: Path,
    expected_compiled_checksums: Mapping[str, str],
) -> tuple[AndroidArtifactInput, ...]:
    """Resolve retained benchmark files into canonical repository inputs.

    Args:
        payload_root: Root containing exact retained Zipformer and VPCD payloads.
        generated_root: Ignored directory receiving normalized fixture manifests.
        expected_compiled_checksums: Required post-compile ONNX checksum by model.

    Returns:
        Four artifact inputs ordered by model and configuration.

    Raises:
        FileNotFoundError: If a retained manifest or component is missing.
        ValueError: If identity, checksum, role, or fixture contracts differ.
    """
    root = Path(payload_root).resolve()
    generated = Path(generated_root).resolve()
    generated.mkdir(parents=True, exist_ok=True)
    artifacts: list[AndroidArtifactInput] = []
    for model in ("zipformer", "vpcd"):
        model_root = root / model
        manifest_path = model_root / "benchmark-manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("artifact_id") != CANONICAL_ARTIFACT_IDS[
            (model, "aimet-int8-int16-encoder-matmul")
        ]:
            raise ValueError(f"{model} retained artifact ID does not match the canonical recipe")
        sources = _validated_component_sources(model_root, payload)
        compiled_checksum = sha256_file(sources["compiled_model"])
        if compiled_checksum != expected_compiled_checksums[model]:
            raise ValueError(
                f"{model} compiled ONNX checksum mismatch: "
                f"expected {expected_compiled_checksums[model]}, got {compiled_checksum}"
            )
        fixtures = _normalized_fixtures(
            model=model,
            model_root=model_root,
            payload=payload,
            generated_root=generated,
        )
        artifacts.extend(
            _artifact_inputs(
                model=model,
                sources=sources,
                fixtures=fixtures,
            )
        )
    return tuple(artifacts)


def _validated_component_sources(
    model_root: Path,
    payload: Mapping[str, object],
) -> dict[str, Path]:
    """Validate retained component rows and resolve their source files.

    Args:
        model_root: Retained payload directory for one model.
        payload: Parsed retained payload manifest.

    Returns:
        Component roles mapped to checksummed source files.

    Raises:
        FileNotFoundError: If a declared source file is absent.
        ValueError: If a component checksum or role is invalid.
    """
    sources: dict[str, Path] = {}
    for row in payload["components"]:
        role = str(row["role"])
        if role in sources:
            raise ValueError(f"Duplicate retained component role: {role!r}")
        relative = Path(str(row["file"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Retained component file must be relative")
        source = (model_root / relative).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        checksum = sha256_file(source)
        if checksum != row["checksum"]:
            raise ValueError(f"Retained component checksum mismatch for {role!r}")
        sources[role] = source
    required = {"fp32_model", "compiled_model", "compiled_external_data"}
    required.update(
        {"decoder", "joiner", "tokens"}
        if model_root.name == "zipformer"
        else {
            "tokenizer_encode",
            "tokenizer_decode",
            "tokenizer_to_model_id_map",
            "model_to_tokenizer_id_map",
            "autoregressive_loop",
        }
    )
    missing = sorted(required - set(sources))
    if missing:
        raise ValueError(f"Missing retained component roles: {missing!r}")
    return sources


def _normalized_fixtures(
    *,
    model: str,
    model_root: Path,
    payload: Mapping[str, object],
    generated_root: Path,
) -> dict[str, Path]:
    """Normalize retained fixtures under repository-relative paths.

    Args:
        model: Canonical model family.
        model_root: Retained model payload root.
        payload: Parsed retained payload manifest.
        generated_root: Work directory for normalized fixture metadata.

    Returns:
        Repository-relative fixture paths mapped to source files.

    Raises:
        FileNotFoundError: If a fixture input is absent.
        ValueError: If a fixture checksum differs.
    """
    fixtures: dict[str, Path] = {}
    fixture_rows = json.loads(json.dumps(list(payload["fixtures"])))
    for row in fixture_rows:
        for tensor in dict(row["inputs"]).values():
            relative = Path(str(tensor["file"]))
            source = (model_root / relative).resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            if sha256_file(source) != tensor["checksum"]:
                raise ValueError(f"{model} fixture checksum mismatch: {relative.as_posix()}")
            repository_relative = f"fixtures/{model}/{relative.relative_to('fixtures').as_posix()}"
            fixtures[repository_relative] = source
            tensor["file"] = repository_relative
    manifest_source = generated_root / model / "fixture-manifest.json"
    manifest_source.parent.mkdir(parents=True, exist_ok=True)
    manifest_source.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": model,
                "fixtures": fixture_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    fixtures[f"fixtures/{model}/fixture-manifest.json"] = manifest_source
    return fixtures


def _artifact_inputs(
    *,
    model: str,
    sources: Mapping[str, Path],
    fixtures: Mapping[str, Path],
) -> tuple[AndroidArtifactInput, AndroidArtifactInput]:
    """Build FP32 and NPU artifact inputs for one model family.

    Args:
        model: Canonical model family.
        sources: Verified retained component sources.
        fixtures: Shared checksummed fixture sources.

    Returns:
        FP32 CPU and post-compile QNN HTP artifact inputs.
    """
    fp32_artifact = ArtifactSpec.parse(
        CANONICAL_ARTIFACT_IDS[(model, "fp32-fixed-shape")]
    )
    npu_artifact = ArtifactSpec.parse(
        CANONICAL_ARTIFACT_IDS[(model, "aimet-int8-int16-encoder-matmul")]
    )
    fp32_components = _components(model=model, sources=sources, npu=False)
    npu_components = _components(model=model, sources=sources, npu=True)
    source_checksums = {
        role: sha256_file(source)
        for role, source in sorted(sources.items())
        if role != "qdq_model" and not role.startswith("qdq_")
    }
    graph_checks = (
        {"encoder_matmul_count": 278, "graph_contract": True}
        if model == "zipformer"
        else {
            "encoder_matmul_count": 96,
            "decoder_matmul_count": 168,
            "language_model_head_matmul_count": 1,
            "graph_contract": True,
        }
    )
    return (
        AndroidArtifactInput(
            artifact=fp32_artifact,
            configuration="fp32-fixed-shape",
            representation="onnx-fp32-fixed-shape",
            execution_target="cpu",
            build_surfaces=("benchmark", "cpuCompat"),
            components=fp32_components,
            fixtures=dict(fixtures),
            source_checksums=source_checksums,
            validation_checks=graph_checks,
            runtime_metadata=_runtime_metadata(model=model, npu=False),
        ),
        AndroidArtifactInput(
            artifact=npu_artifact,
            configuration="aimet-int8-int16-encoder-matmul",
            representation="onnx-epcontext-external-binary",
            execution_target="qnn-htp",
            build_surfaces=("benchmark", "qnnOfficialArm64"),
            components=npu_components,
            fixtures=dict(fixtures),
            source_checksums=source_checksums,
            validation_checks={**graph_checks, "strict_htp_evidence_retained": True},
            runtime_metadata=_runtime_metadata(model=model, npu=True),
        ),
    )


def _components(
    *,
    model: str,
    sources: Mapping[str, Path],
    npu: bool,
) -> tuple[AndroidComponentInput, ...]:
    """Map retained roles to canonical repository component paths.

    Args:
        model: Canonical model family.
        sources: Verified retained component sources.
        npu: Whether to select the post-compile NPU representation.

    Returns:
        Canonical repository component inputs.
    """
    configuration = (
        "aimet-int8-int16-encoder-matmul" if npu else "fp32-fixed-shape"
    )
    runtime_dir = "qnn-htp" if npu else "cpu"
    model_role = "encoder" if model == "zipformer" else "model"
    rows = [
        AndroidComponentInput(
            role=model_role,
            source=sources["compiled_model" if npu else "fp32_model"],
            relative_file=(
                f"artifacts/{model}/{configuration}/{runtime_dir}/{model_role}.onnx"
            ),
            format="onnx-epcontext" if npu else "onnx",
            precision="int8/int16" if npu else "fp32",
            input_shapes=(
                {"x": [1, 2009, 80], "x_lens": [1]}
                if model == "zipformer"
                else {
                    "input_ids": [1, 384],
                    "attention_mask": [1, 384],
                    "decoder_input_ids": [1, 64],
                    "decoder_attention_mask": [1, 64],
                }
            ),
            quantization_engine="aimet" if npu else "none",
            quantization_scope="encoder-matmul" if npu else "none",
            execution_target="qnn-htp" if npu else "cpu",
        )
    ]
    if npu:
        rows.append(
            AndroidComponentInput(
                role=f"{model_role}_external_data",
                source=sources["compiled_external_data"],
                relative_file=(
                    f"artifacts/{model}/{configuration}/{runtime_dir}/model.bin"
                ),
                format="qnn-context-binary",
                precision="int8/int16",
                input_shapes={},
                quantization_engine="aimet",
                quantization_scope="encoder-matmul",
                execution_target="qnn-htp",
            )
        )
    support_roles = (
        ("decoder", "joiner", "tokens")
        if model == "zipformer"
        else (
            "tokenizer_encode",
            "tokenizer_decode",
            "tokenizer_to_model_id_map",
            "model_to_tokenizer_id_map",
            "autoregressive_loop",
        )
    )
    support_names = {
        "decoder": "decoder.onnx",
        "joiner": "joiner.onnx",
        "tokens": "tokens.txt",
        "tokenizer_encode": "tokenizer.encode.onnx",
        "tokenizer_decode": "tokenizer.decode.onnx",
        "tokenizer_to_model_id_map": "tokenizer.to_model_id_map.json",
        "model_to_tokenizer_id_map": "tokenizer.from_model_id_map.json",
        "autoregressive_loop": "autoregressive-loop.json",
    }
    for role in support_roles:
        source = sources[role]
        rows.append(
            AndroidComponentInput(
                role=role,
                source=source,
                relative_file=(
                    f"artifacts/{model}/shared-fp32-cpu/{support_names[role]}"
                ),
                format=_support_format(source),
                precision="fp32" if source.suffix == ".onnx" else "not-applicable",
                input_shapes={},
                quantization_engine="none",
                quantization_scope="none",
                execution_target="cpu",
            )
        )
    return tuple(rows)


def _support_format(source: Path) -> str:
    """Infer a portable support-component format from its suffix.

    Args:
        source: Support component source file.

    Returns:
        Canonical format label.
    """
    if source.suffix == ".onnx":
        return "onnx"
    if source.suffix == ".json":
        return "json"
    return "text"


def _io_contract(*, model: str, npu: bool) -> dict[str, object]:
    """Build component I/O dtype truth for Android tensor preparation.

    Args:
        model: Canonical model family.
        npu: Whether the compiled artifact truncates 64-bit integer inputs.

    Returns:
        AI Hub-compatible I/O metadata embedded in manifest v2.
    """
    if model == "zipformer":
        inputs = [
            {
                "name": "x",
                "shape": [1, 2009, 80],
                "dtype": "float32",
                "source_dtype": "float32",
            },
            {
                "name": "x_lens",
                "shape": [1],
                "dtype": "int32" if npu else "int64",
                "source_dtype": "int64",
            },
        ]
    else:
        inputs = [
            {
                "name": name,
                "shape": shape,
                "dtype": "int32" if npu else "int64",
                "source_dtype": "int64",
            }
            for name, shape in (
                ("input_ids", [1, 384]),
                ("attention_mask", [1, 384]),
                ("decoder_input_ids", [1, 64]),
                ("decoder_attention_mask", [1, 64]),
            )
        ]
    return {
        "target_runtime": "qnn-htp" if npu else "onnxruntime-cpu",
        "inputs": inputs,
        "outputs": [],
        "special_handling": ["truncate_64bit_io"] if npu else [],
    }


def _runtime_metadata(*, model: str, npu: bool) -> dict[str, object]:
    """Build Android preprocessing metadata shared by CPU and NPU surfaces.

    Args:
        model: Canonical model family.
        npu: Whether the selected representation is compiled for QNN HTP.

    Returns:
        Manifest-v2 runtime metadata for tensor and tokenizer preparation.
    """
    metadata: dict[str, object] = {
        "io_contract": _io_contract(model=model, npu=npu),
    }
    if model == "zipformer":
        metadata["fixed_encoder_frames"] = 2009
    else:
        metadata.update(
            {
                "pad_token_id": 1,
                "eos_token_id": 2,
                "decoder_start_token_id": 2,
                "max_source_length": 384,
                "input_text_case": "lower",
            }
        )
    return metadata
