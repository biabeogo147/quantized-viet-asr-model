import json
from pathlib import Path
import zipfile

import pytest

from model_bundle.manifest import ModelBundleManifest


def _init_repo_root(repo_root: Path) -> None:
    (repo_root / "src").mkdir(parents=True, exist_ok=True)
    (repo_root / "assets").mkdir(parents=True, exist_ok=True)
    (repo_root / "test").mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text("[project]\nname = 'python-model-test'\nversion = '0.0.0'\n", encoding="utf-8")


def _write_zipformer_source_bundle(repo_root: Path) -> Path:
    bundle_dir = repo_root / "build" / "model_bundle" / "zipformer" / "qnn_u16u8"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    manifest = ModelBundleManifest(
        bundle_version=1,
        project="zipformer",
        model_family="zipformer-rnnt",
        model_name="zipformer/qnn_u16u8",
        model_variant="qnn_u16u8",
        asset_namespace="models/asr/zipformer/qnn_u16u8",
        runtime_kind="rnnt_greedy",
        artifacts={
            "encoder": "encoder.onnx",
            "decoder": "decoder.onnx",
            "joiner": "joiner.onnx",
            "tokens": "tokens.txt",
        },
        fixtures={
            "sample_manifest": "sample_manifest.jsonl",
            "expected_outputs": "expected_outputs.jsonl",
        },
        metadata={
            "sample_rate": 16000,
            "feature_dim": 80,
            "blank_id": 0,
            "context_size": 2,
            "fixed_encoder_frames": 2009,
            "quantization": {
                "format": "QDQ",
                "activation_type": "quint16",
                "weight_type": "quint8",
                "preset": "zipformer_sd8g2_balanced",
                "fixed_shapes": True,
            },
        },
    )
    manifest.write_json(bundle_dir / "bundle_manifest.json")
    for file_name in (
        "encoder.onnx",
        "decoder.onnx",
        "joiner.onnx",
        "tokens.txt",
        "sample_manifest.jsonl",
        "expected_outputs.jsonl",
    ):
        (bundle_dir / file_name).write_text(file_name, encoding="utf-8")
    return bundle_dir


def _write_vpcd_source_bundle(repo_root: Path) -> Path:
    bundle_dir = repo_root / "build" / "model_bundle" / "vpcd" / "qnn_fixed_1024x128"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    manifest = ModelBundleManifest(
        bundle_version=1,
        project="vpcd",
        model_family="bartpho-seq2seq",
        model_name="tourmii/vietnamese-punc-cap-denorm-v1",
        model_variant="vpcd_balanced_fixed_1024x128",
        asset_namespace="models/punctuation/vpcd/qnn_fixed_1024x128",
        runtime_kind="text_seq2seq",
        artifacts={
            "model": "model.mobile.onnx",
            "tokenizer_encode": "tokenizer.encode.onnx",
            "tokenizer_decode": "tokenizer.decode.onnx",
            "tokenizer_to_model_id_map": "tokenizer.to_model_id_map.json",
            "model_to_tokenizer_id_map": "tokenizer.from_model_id_map.json",
        },
        fixtures={"golden_samples": "golden_samples.jsonl"},
        metadata={
            "pad_token_id": 1,
            "eos_token_id": 2,
            "decoder_start_token_id": 2,
            "max_source_length": 1024,
            "max_decode_length": 128,
            "input_text_case": "lower",
            "quantization": {
                "format": "QDQ",
                "activation_type": "quint16",
                "weight_type": "quint8",
                "preset": "sd8g2_balanced",
                "fixed_shapes": True,
            },
            "qnn_readiness": {
                "target_backend": "qnn_htp",
                "model_session_candidate": True,
                "tokenizer_policy": "cpu_only_first_slice",
                "requires_fixed_shapes": True,
                "fixed_shapes_ready": True,
            },
        },
    )
    manifest.write_json(bundle_dir / "bundle_manifest.json")
    for file_name in (
        "model.mobile.onnx",
        "tokenizer.encode.onnx",
        "tokenizer.decode.onnx",
        "tokenizer.to_model_id_map.json",
        "tokenizer.from_model_id_map.json",
        "golden_samples.jsonl",
    ):
        (bundle_dir / file_name).write_text(file_name, encoding="utf-8")
    return bundle_dir


def _write_compiled_zip(zip_path: Path, *, payload_dir: str, onnx_bytes: bytes, bin_bytes: bytes, extra_entries: dict[str, bytes] | None = None) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(f"{payload_dir}/model.onnx", onnx_bytes)
        archive.writestr(f"{payload_dir}/model.bin", bin_bytes)
        for name, data in (extra_entries or {}).items():
            archive.writestr(name, data)


def _write_deployment_package(
    repo_root: Path,
    *,
    project: str,
    run_label: str,
    source_bundle_dir: Path,
    target_model_id: str,
    device_name: str = "Samsung Galaxy S24 (Family)",
) -> Path:
    package_dir = repo_root / "build" / "aihub" / "deploy" / project / run_label
    download_dir = package_dir / "download"
    package_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)

    zip_name = "encoder.precompiled_qnn_onnx.onnx.onnx.zip" if project == "zipformer" else "model.precompiled_qnn_onnx.onnx.onnx.zip"
    zip_path = download_dir / zip_name
    _write_compiled_zip(
        zip_path,
        payload_dir=f"job_{project}_optimized_onnx",
        onnx_bytes=f"{project}-compiled-onnx".encode("utf-8"),
        bin_bytes=f"{project}-compiled-bin".encode("utf-8"),
    )

    io_contract_path = package_dir / "io_contract.json"
    if project == "zipformer":
        io_contract_payload = {
            "target_runtime": "precompiled_qnn_onnx",
            "inputs": [
                {"name": "x", "shape": [1, 2009, 80], "dtype": "float32", "source_dtype": "float32"},
                {"name": "x_lens", "shape": [1], "dtype": "int32", "source_dtype": "int64"},
            ],
            "outputs": [{"name": "output_0", "shape": [1, 501, 512], "dtype": "float32"}],
            "special_handling": ["truncate_64bit_io required"],
        }
    else:
        io_contract_payload = {
            "target_runtime": "precompiled_qnn_onnx",
            "inputs": [
                {"name": "input_ids", "shape": [1, 1024], "dtype": "int32", "source_dtype": "int64"},
                {"name": "attention_mask", "shape": [1, 1024], "dtype": "int32", "source_dtype": "int64"},
            ],
            "outputs": [{"name": "output_0", "shape": [1, 128, 40030], "dtype": "float32"}],
            "special_handling": ["truncate_64bit_io required"],
        }
    io_contract_path.write_text(json.dumps(io_contract_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    deployment_manifest_path = package_dir / "deployment_manifest.json"
    deployment_manifest_path.write_text(
        json.dumps(
            {
                "project": project,
                "run_label": run_label,
                "target_model_id": target_model_id,
                "target_runtime": "precompiled_qnn_onnx",
                "device_name": device_name,
                "qairt_version": "2.46.0",
                "compile_options": "--target_runtime precompiled_qnn_onnx --truncate_64bit_io",
                "downloaded_artifact": {
                    "path": zip_path.as_posix(),
                    "size_bytes": int(zip_path.stat().st_size),
                },
                "source_bundle_manifest": (source_bundle_dir / "bundle_manifest.json").as_posix(),
                "io_contract_path": io_contract_path.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return package_dir


def test_materialize_android_bundle_synthesizes_zipformer_bundle_from_deployment_package(tmp_path):
    from aihub.android_bundle import materialize_android_bundle

    repo_root = tmp_path / "repo"
    _init_repo_root(repo_root)
    source_bundle_dir = _write_zipformer_source_bundle(repo_root)
    package_dir = _write_deployment_package(
        repo_root,
        project="zipformer",
        run_label="unit-step6",
        source_bundle_dir=source_bundle_dir,
        target_model_id="zip-target-1",
    )

    result = materialize_android_bundle(deployment_package_dir=package_dir)

    manifest = ModelBundleManifest.from_path(result.manifest_path)
    assert result.bundle_dir == (repo_root / "build" / "aihub" / "android_bundle" / "zipformer" / "unit-step6").resolve()
    assert (result.bundle_dir / "encoder.onnx").read_bytes() == b"zipformer-compiled-onnx"
    assert (result.bundle_dir / "model.bin").read_bytes() == b"zipformer-compiled-bin"
    assert (result.bundle_dir / "decoder.onnx").read_text(encoding="utf-8") == "decoder.onnx"
    assert (result.bundle_dir / "joiner.onnx").read_text(encoding="utf-8") == "joiner.onnx"
    assert (result.bundle_dir / "tokens.txt").read_text(encoding="utf-8") == "tokens.txt"
    assert (result.bundle_dir / "sample_manifest.jsonl").exists()
    assert (result.bundle_dir / "expected_outputs.jsonl").exists()
    assert (result.bundle_dir / "io_contract.json").exists()
    assert manifest.model_name == "zipformer/precompiled_qnn_onnx"
    assert manifest.model_variant == "precompiled_qnn_onnx"
    assert manifest.asset_namespace == "models/asr/zipformer/precompiled_qnn_onnx"
    assert manifest.runtime_kind == "onnx"
    assert manifest.artifacts["encoder"] == "encoder.onnx"
    assert manifest.artifacts["encoder_external_data"] == "model.bin"
    assert manifest.artifacts["io_contract"] == "io_contract.json"
    assert manifest.metadata["quantization"]["fixed_shapes"] is True
    assert manifest.metadata["aihub"]["run_label"] == "unit-step6"
    assert manifest.metadata["aihub"]["target_model_id"] == "zip-target-1"
    assert manifest.metadata["aihub"]["io_contract_artifact"] == "io_contract"
    assert manifest.metadata["aihub"]["components"]["encoder"]["target_runtime"] == "precompiled_qnn_onnx"
    assert manifest.metadata["aihub"]["components"]["decoder"]["target_runtime"] == "cpu_onnx"


def test_materialize_android_bundle_synthesizes_vpcd_bundle_from_deployment_package(tmp_path):
    from aihub.android_bundle import materialize_android_bundle

    repo_root = tmp_path / "repo"
    _init_repo_root(repo_root)
    source_bundle_dir = _write_vpcd_source_bundle(repo_root)
    package_dir = _write_deployment_package(
        repo_root,
        project="vpcd",
        run_label="unit-step6",
        source_bundle_dir=source_bundle_dir,
        target_model_id="vpcd-target-1",
    )

    result = materialize_android_bundle(deployment_package_dir=package_dir)

    manifest = ModelBundleManifest.from_path(result.manifest_path)
    assert (result.bundle_dir / "model.mobile.onnx").read_bytes() == b"vpcd-compiled-onnx"
    assert (result.bundle_dir / "model.bin").read_bytes() == b"vpcd-compiled-bin"
    assert (result.bundle_dir / "tokenizer.encode.onnx").read_text(encoding="utf-8") == "tokenizer.encode.onnx"
    assert (result.bundle_dir / "tokenizer.decode.onnx").read_text(encoding="utf-8") == "tokenizer.decode.onnx"
    assert (result.bundle_dir / "tokenizer.to_model_id_map.json").read_text(encoding="utf-8") == "tokenizer.to_model_id_map.json"
    assert (result.bundle_dir / "tokenizer.from_model_id_map.json").read_text(encoding="utf-8") == "tokenizer.from_model_id_map.json"
    assert (result.bundle_dir / "golden_samples.jsonl").exists()
    assert manifest.model_name == "tourmii/vietnamese-punc-cap-denorm-v1"
    assert manifest.model_variant == "precompiled_qnn_onnx"
    assert manifest.asset_namespace == "models/punctuation/vpcd/precompiled_qnn_onnx"
    assert manifest.runtime_kind == "onnx"
    assert manifest.artifacts["model"] == "model.mobile.onnx"
    assert manifest.artifacts["model_external_data"] == "model.bin"
    assert manifest.artifacts["io_contract"] == "io_contract.json"
    assert manifest.metadata["aihub"]["components"]["model"]["target_runtime"] == "precompiled_qnn_onnx"
    assert manifest.metadata["aihub"]["components"]["tokenizer_encode"]["target_runtime"] == "cpu_onnx"


def test_materialize_android_bundle_rejects_compiled_zip_with_unexpected_payload(tmp_path):
    from aihub.android_bundle import materialize_android_bundle

    repo_root = tmp_path / "repo"
    _init_repo_root(repo_root)
    source_bundle_dir = _write_zipformer_source_bundle(repo_root)
    package_dir = _write_deployment_package(
        repo_root,
        project="zipformer",
        run_label="unit-step6",
        source_bundle_dir=source_bundle_dir,
        target_model_id="zip-target-1",
    )
    zip_path = package_dir / "download" / "encoder.precompiled_qnn_onnx.onnx.onnx.zip"
    _write_compiled_zip(
        zip_path,
        payload_dir="job_bad_optimized_onnx",
        onnx_bytes=b"zipformer-compiled-onnx",
        bin_bytes=b"zipformer-compiled-bin",
        extra_entries={"job_bad_optimized_onnx/extra.onnx": b"extra"},
    )

    with pytest.raises(ValueError, match="exactly one compiled ONNX file"):
        materialize_android_bundle(deployment_package_dir=package_dir)
