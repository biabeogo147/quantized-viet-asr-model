import json
from pathlib import Path

import numpy as np

from model_bundle.manifest import ModelBundleManifest


def _init_repo_root(repo_root: Path) -> None:
    (repo_root / "src").mkdir(parents=True, exist_ok=True)
    (repo_root / "assets").mkdir(parents=True, exist_ok=True)
    (repo_root / "test").mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text("[project]\nname = 'python-model-test'\nversion = '0.0.0'\n", encoding="utf-8")


def _write_zipformer_bundle(repo_root: Path) -> Path:
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
        },
    )
    manifest.write_json(bundle_dir / "bundle_manifest.json")
    for file_name in ("encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt", "sample_manifest.jsonl", "expected_outputs.jsonl"):
        (bundle_dir / file_name).write_text(file_name, encoding="utf-8")
    fixed_encoder = repo_root / "build" / "quantize" / "zipformer" / "qnn_u16u8" / "fixed_shapes" / "encoder.fixed.onnx"
    fixed_encoder.parent.mkdir(parents=True, exist_ok=True)
    fixed_encoder.write_bytes(b"encoder-fixed")
    return bundle_dir


def _write_vpcd_bundle(repo_root: Path) -> Path:
    bundle_dir = repo_root / "build" / "model_bundle" / "vpcd" / "qnn_fixed_1024x128"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    manifest = ModelBundleManifest(
        bundle_version=1,
        project="vpcd",
        model_family="bartpho-seq2seq",
        model_name="tourmii/vietnamese-punc-cap-denorm-v1",
        model_variant="qnn_fixed_1024x128",
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
            "fixed_input_shapes": {
                "model": {
                    "input_ids": [1, 1024],
                    "attention_mask": [1, 1024],
                    "decoder_input_ids": [1, 128],
                    "decoder_attention_mask": [1, 128],
                }
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


def _write_phase8_vpcd_bundle(repo_root: Path) -> Path:
    bundle_dir = repo_root / "build" / "phase8" / "candidate-bundles" / "vpcd-a4-384x64-l1"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    manifest = ModelBundleManifest(
        bundle_version=1,
        project="vpcd",
        model_family="bartpho-seq2seq",
        model_name="tourmii/vietnamese-punc-cap-denorm-v1",
        model_variant="vpcd-a4-384x64-l1",
        asset_namespace="models/punctuation/vpcd/phase8/vpcd-a4-384x64-l1",
        runtime_kind="text_seq2seq",
        artifacts={
            "model": "model.option1.qdq.onnx",
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
            "max_source_length": 384,
            "max_decode_length": 64,
            "input_text_case": "lower",
            "fixed_input_shapes": {
                "model": {
                    "input_ids": [1, 384],
                    "attention_mask": [1, 384],
                    "decoder_input_ids": [1, 64],
                    "decoder_attention_mask": [1, 64],
                }
            },
        },
    )
    manifest.write_json(bundle_dir / "bundle_manifest.json")
    for file_name in (
        "model.option1.qdq.onnx",
        "tokenizer.encode.onnx",
        "tokenizer.decode.onnx",
        "tokenizer.to_model_id_map.json",
        "tokenizer.from_model_id_map.json",
        "golden_samples.jsonl",
    ):
        (bundle_dir / file_name).write_text(file_name, encoding="utf-8")
    return bundle_dir


class FakeJob:
    def __init__(self, job_id: str, url: str, status: str) -> None:
        self.job_id = job_id
        self.url = url
        self.status = status


class FakeTargetModel:
    def __init__(self, model_id: str, url: str, name: str, payload: bytes = b"compiled-target") -> None:
        self.model_id = model_id
        self.url = url
        self.name = name
        self.payload = payload
        self.download_calls: list[str] = []

    def download(self, filename: str) -> str:
        self.download_calls.append(filename)
        Path(filename).write_bytes(self.payload)
        return filename


def _build_zipformer_records(repo_root: Path, *, run_label: str = "unit-deploy"):
    from aihub.evaluation import ResolvedCompiledModel, write_evaluation_record
    from aihub.session import (
        build_runtime_config,
        write_compile_run_record,
        write_live_run_record,
        write_prepared_artifact_record,
    )

    _write_zipformer_bundle(repo_root)
    config = build_runtime_config(
        device_name="Samsung Galaxy S24 (Family)",
        qairt_version="2.46.0",
        repo_root=repo_root,
    )
    source_model_path = repo_root / "build" / "quantize" / "zipformer" / "qnn_u16u8" / "fixed_shapes" / "encoder.fixed.onnx"
    prepared_model_path = repo_root / "build" / "aihub" / "zipformer_encoder_option1" / "encoder.aihub.option1.onnx"
    prepared_model_path.parent.mkdir(parents=True, exist_ok=True)
    prepared_model_path.write_bytes(b"prepared-zipformer")

    prepared_record_path = write_prepared_artifact_record(
        pilot_name="zipformer_encoder_option1",
        runtime_config=config,
        source_model_path=source_model_path,
        prepared_model_path=prepared_model_path,
        input_specs={
            "x": ((1, 2009, 80), "float32"),
            "x_lens": ((1,), "int64"),
        },
        compile_options="--target_runtime precompiled_qnn_onnx --truncate_64bit_io --qairt_version 2.46.0",
        run_label=run_label,
    )
    compile_record_path = write_compile_run_record(
        pilot_name="zipformer_encoder_option1",
        runtime_config=config,
        compile_options="--target_runtime precompiled_qnn_onnx --truncate_64bit_io --qairt_version 2.46.0",
        compile_job=FakeJob("compile-z", "https://aihub/jobs/compile-z", "SUCCESS"),
        target_model=FakeTargetModel("zip-model-1", "https://aihub/models/zip-model-1", "zipformer-target"),
        run_label=run_label,
    )
    live_record_path = write_live_run_record(
        pilot_name="zipformer_encoder_option1",
        runtime_config=config,
        compile_options="--target_runtime precompiled_qnn_onnx --truncate_64bit_io --qairt_version 2.46.0",
        job_options="--compute_unit npu --qairt_version 2.46.0",
        compile_job=FakeJob("compile-z", "https://aihub/jobs/compile-z", "SUCCESS"),
        inference_job=FakeJob("infer-z", "https://aihub/jobs/infer-z", "SUCCESS"),
        output_tensors={
            "output_0": [np.zeros((1, 501, 512), dtype=np.float32)],
            "output_1": [np.asarray([501], dtype=np.int32)],
        },
        run_label=run_label,
    )
    hybrid_record_path = write_evaluation_record(
        pilot_name="zipformer_hybrid_option1",
        runtime_config=config,
        target_reference=ResolvedCompiledModel(
            compile_pilot_name="zipformer_encoder_option1",
            target_model_id="zip-model-1",
            compile_record_path=compile_record_path,
            run_label=run_label,
            explicit_override=False,
        ),
        sample_results=[
            {
                "sample_id": "sample-1",
                "audio_path": "assets/speech/sample-1.wav",
                "text": "xin chao",
                "expected_text": "xin chao",
                "expected_available": True,
                "matches_expected": True,
                "cloud_inference_seconds": 0.12,
                "decode_seconds": 0.03,
            }
        ],
        run_label=run_label,
    )
    return config, {
        "prepared_record_path": prepared_record_path,
        "compile_record_path": compile_record_path,
        "live_record_path": live_record_path,
        "hybrid_record_path": hybrid_record_path,
    }


def _build_vpcd_records(repo_root: Path, *, run_label: str = "unit-deploy"):
    from aihub.evaluation import ResolvedCompiledModel, write_evaluation_record
    from aihub.session import (
        build_runtime_config,
        write_compile_run_record,
        write_live_run_record,
        write_prepared_artifact_record,
    )

    bundle_dir = _write_vpcd_bundle(repo_root)
    config = build_runtime_config(
        device_name="Samsung Galaxy S24 (Family)",
        qairt_version="2.46.0",
        repo_root=repo_root,
    )
    source_model_path = bundle_dir / "model.mobile.onnx"
    prepared_model_path = repo_root / "build" / "quantize" / "vpcd" / "local_aimet" / "model.fp32.fixed.onnx"
    prepared_model_path.parent.mkdir(parents=True, exist_ok=True)
    prepared_model_path.write_bytes(b"prepared-vpcd")
    packaging_path = repo_root / "build" / "quantize" / "vpcd" / "local_aimet" / "model.option1.aimet"
    packaging_path.mkdir(parents=True, exist_ok=True)

    prepared_record_path = write_prepared_artifact_record(
        pilot_name="vpcd_option1_local_aimet",
        runtime_config=config,
        source_model_path=source_model_path,
        prepared_model_path=prepared_model_path,
        input_specs={
            "input_ids": ((1, 1024), "int64"),
            "attention_mask": ((1, 1024), "int64"),
            "decoder_input_ids": ((1, 128), "int64"),
            "decoder_attention_mask": ((1, 128), "int64"),
        },
        compile_options="--target_runtime precompiled_qnn_onnx --truncate_64bit_io --qairt_version 2.46.0",
        source_strategy="local_aimet_compile_candidate",
        source_kind="local_aimet",
        packaging_kind="aimet_dir",
        packaging_path=packaging_path,
        compatibility={"aihub_compile_readiness": "experimental"},
        run_label=run_label,
    )
    compile_record_path = write_compile_run_record(
        pilot_name="vpcd_option1_local_aimet",
        runtime_config=config,
        compile_options="--target_runtime precompiled_qnn_onnx --truncate_64bit_io --qairt_version 2.46.0",
        compile_job=FakeJob("compile-v", "https://aihub/jobs/compile-v", "SUCCESS"),
        target_model=FakeTargetModel("vpcd-model-1", "https://aihub/models/vpcd-model-1", "vpcd-target"),
        source_strategy="local_aimet_compile_candidate",
        quantize_stage="local_aimet",
        compatibility={"aihub_compile_readiness": "experimental"},
        run_label=run_label,
    )
    live_record_path = write_live_run_record(
        pilot_name="vpcd_option1_local_aimet",
        runtime_config=config,
        compile_options="--target_runtime precompiled_qnn_onnx --truncate_64bit_io --qairt_version 2.46.0",
        job_options="--compute_unit npu --qairt_version 2.46.0",
        compile_job=FakeJob("compile-v", "https://aihub/jobs/compile-v", "SUCCESS"),
        inference_job=FakeJob("infer-v", "https://aihub/jobs/infer-v", "SUCCESS"),
        output_tensors={
            "output_0": [np.zeros((1, 128, 256), dtype=np.float32)],
        },
        run_label=run_label,
    )
    hybrid_record_path = write_evaluation_record(
        pilot_name="vpcd_hybrid_option1",
        runtime_config=config,
        target_reference=ResolvedCompiledModel(
            compile_pilot_name="vpcd_option1_local_aimet",
            target_model_id="vpcd-model-1",
            compile_record_path=compile_record_path,
            run_label=run_label,
            explicit_override=False,
        ),
        sample_results=[
            {
                "sample_index": 0,
                "raw_text": "xin chao",
                "text": "Xin chao.",
                "expected_text": "Xin chao.",
                "matches_expected": True,
                "generated_ids": [2, 10, 11],
                "cloud_inference_seconds": 0.21,
                "decode_seconds": 0.04,
            }
        ],
        run_label=run_label,
    )
    return config, {
        "prepared_record_path": prepared_record_path,
        "compile_record_path": compile_record_path,
        "live_record_path": live_record_path,
        "hybrid_record_path": hybrid_record_path,
    }


def test_resolve_deployment_inputs_reads_retained_zipformer_records(tmp_path):
    from aihub.deployment import resolve_deployment_inputs

    repo_root = tmp_path / "repo"
    _init_repo_root(repo_root)
    config, record_paths = _build_zipformer_records(repo_root)

    resolved = resolve_deployment_inputs(
        runtime_config=config,
        project="zipformer",
        run_label="unit-deploy",
    )

    assert resolved.layout.project == "zipformer"
    assert resolved.layout.compile_record_group == "zipformer_encoder_option1"
    assert resolved.target_model_id == "zip-model-1"
    assert resolved.prepared_record_path == record_paths["prepared_record_path"]
    assert resolved.compile_record_path == record_paths["compile_record_path"]
    assert resolved.live_record_path == record_paths["live_record_path"]
    assert resolved.hybrid_record_path == record_paths["hybrid_record_path"]
    assert resolved.source_bundle_manifest_path == (
        repo_root / "build" / "model_bundle" / "zipformer" / "qnn_u16u8" / "bundle_manifest.json"
    ).resolve()


def test_resolve_deployment_inputs_prefers_explicit_source_bundle_manifest_for_vpcd(tmp_path):
    from aihub.deployment import resolve_deployment_inputs

    repo_root = tmp_path / "repo"
    _init_repo_root(repo_root)
    config, _record_paths = _build_vpcd_records(repo_root)
    phase8_bundle_dir = _write_phase8_vpcd_bundle(repo_root)

    resolved = resolve_deployment_inputs(
        runtime_config=config,
        project="vpcd",
        run_label="unit-deploy",
        source_bundle_manifest_path=phase8_bundle_dir / "bundle_manifest.json",
    )

    assert resolved.source_bundle_manifest_path == (phase8_bundle_dir / "bundle_manifest.json").resolve()


def test_materialize_deployment_package_downloads_artifact_and_writes_io_contract_for_vpcd(tmp_path):
    from aihub.deployment import materialize_deployment_package

    repo_root = tmp_path / "repo"
    _init_repo_root(repo_root)
    config, _record_paths = _build_vpcd_records(repo_root)
    fake_model = FakeTargetModel("vpcd-model-1", "https://aihub/models/vpcd-model-1", "vpcd-target")

    result = materialize_deployment_package(
        runtime_config=config,
        project="vpcd",
        run_label="unit-deploy",
        target_model_resolver=lambda target_model_id: fake_model,
    )

    manifest_payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    io_contract_payload = json.loads(result.io_contract_path.read_text(encoding="utf-8"))

    assert result.package_dir == (
        repo_root / "build" / "aihub" / "deploy" / "vpcd" / "unit-deploy"
    ).resolve()
    assert result.manifest_path.name == "deployment_manifest.json"
    assert result.downloaded_artifact_path.exists()
    assert fake_model.download_calls == [result.downloaded_artifact_path.as_posix()]
    assert manifest_payload["project"] == "vpcd"
    assert manifest_payload["target_model_id"] == "vpcd-model-1"
    assert manifest_payload["target_runtime"] == "precompiled_qnn_onnx"
    assert manifest_payload["downloaded_artifact"]["path"] == result.downloaded_artifact_path.as_posix()
    assert manifest_payload["special_handling"] == ["truncate_64bit_io required"]
    assert io_contract_payload["target_runtime"] == "precompiled_qnn_onnx"
    assert io_contract_payload["special_handling"] == ["truncate_64bit_io required"]
    assert io_contract_payload["inputs"][0]["source_dtype"] == "int64"
    assert io_contract_payload["inputs"][0]["dtype"] == "int32"
    assert io_contract_payload["deployment_notes"] == [
        "Model session runs on the compiled target artifact.",
        "Tokenizer encode and tokenizer decode remain CPU-side artifacts.",
    ]
    assert (result.package_dir / "evidence" / "compile-run-unit-deploy.json").exists()
    assert (result.package_dir / "evidence" / "deployment-download-unit-deploy.json").exists()


def test_materialize_deployment_package_uses_explicit_phase8_vpcd_bundle(tmp_path):
    from aihub.deployment import materialize_deployment_package

    repo_root = tmp_path / "repo"
    _init_repo_root(repo_root)
    config, _record_paths = _build_vpcd_records(repo_root)
    phase8_bundle_dir = _write_phase8_vpcd_bundle(repo_root)
    fake_model = FakeTargetModel("vpcd-model-1", "https://aihub/models/vpcd-model-1", "vpcd-target")

    result = materialize_deployment_package(
        runtime_config=config,
        project="vpcd",
        run_label="unit-deploy",
        source_bundle_manifest_path=phase8_bundle_dir / "bundle_manifest.json",
        target_model_resolver=lambda target_model_id: fake_model,
    )

    manifest_payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["source_bundle_manifest"] == (
        phase8_bundle_dir / "bundle_manifest.json"
    ).resolve().as_posix()


def test_deployment_cli_dry_run_resolves_all_projects_without_materializing_packages(tmp_path, capsys):
    from aihub.deployment import main

    repo_root = tmp_path / "repo"
    _init_repo_root(repo_root)
    config, _ = _build_zipformer_records(repo_root, run_label="unit-deploy")
    _build_vpcd_records(repo_root, run_label="unit-deploy")

    exit_code = main(
        [
            "--project",
            "all",
            "--run-label",
            "unit-deploy",
            "--repo-root",
            repo_root.as_posix(),
            "--device-name",
            config.device_name,
            "--qairt-version",
            config.qairt_version or "",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "zipformer" in captured.out
    assert "vpcd" in captured.out
    assert "zip-model-1" in captured.out
    assert "vpcd-model-1" in captured.out
    assert not (repo_root / "build" / "aihub" / "deploy" / "zipformer" / "unit-deploy").exists()

