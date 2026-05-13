import json
from pathlib import Path

import numpy as np

from model_bundle.manifest import ModelBundleManifest
from model_bundle.projects._vpcd_support import BundleOnnxRuntime


def _init_repo_root(repo_root: Path) -> None:
    (repo_root / "src").mkdir(parents=True, exist_ok=True)
    (repo_root / "assets").mkdir(parents=True, exist_ok=True)
    (repo_root / "test").mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text("[project]\nname = 'python-model-test'\nversion = '0.0.0'\n", encoding="utf-8")


def _write_zipformer_bundle(bundle_dir: Path, *, fixed_encoder_frames: int = 128, feature_dim: int = 80) -> None:
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
            "feature_dim": feature_dim,
            "blank_id": 0,
            "context_size": 2,
            "fixed_input_shapes": {
                "encoder": {"x": [1, fixed_encoder_frames, feature_dim], "x_lens": [1]},
                "decoder": {"y": [1, 2]},
                "joiner": {"encoder_out": [1, 512], "decoder_out": [1, 512]},
            },
            "fixed_encoder_frames": fixed_encoder_frames,
            "quantization": {
                "format": "QDQ",
                "activation_type": "quint16",
                "weight_type": "quint8",
                "fixed_shapes": True,
            },
        },
    )
    manifest.write_json(bundle_dir / "bundle_manifest.json")
    (bundle_dir / "sample_manifest.jsonl").write_text(
        json.dumps({"sample_id": "sample-1", "audio_path": "assets/speech/sample-1.wav"}) + "\n",
        encoding="utf-8",
    )
    (bundle_dir / "expected_outputs.jsonl").write_text(
        json.dumps({"sample_id": "sample-1", "audio_path": "assets/speech/sample-1.wav", "text": "xin chao"}) + "\n",
        encoding="utf-8",
    )


def _write_vpcd_bundle(bundle_dir: Path, *, encoder_sequence: int = 8, decoder_sequence: int = 4) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    manifest = ModelBundleManifest(
        bundle_version=1,
        project="vpcd",
        model_family="bartpho-seq2seq",
        model_name="tourmii/vietnamese-punc-cap-denorm-v1",
        model_variant="vpcd_balanced_fixed_8x4",
        asset_namespace="models/punctuation/vpcd/qnn_fixed_8x4",
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
            "max_source_length": encoder_sequence,
            "max_decode_length": decoder_sequence,
            "input_text_case": "lower",
            "fixed_input_shapes": {
                "model": {
                    "input_ids": [1, encoder_sequence],
                    "attention_mask": [1, encoder_sequence],
                    "decoder_input_ids": [1, decoder_sequence],
                    "decoder_attention_mask": [1, decoder_sequence],
                }
            },
            "quantization": {
                "format": "QDQ",
                "activation_type": "quint16",
                "weight_type": "quint8",
                "fixed_shapes": True,
            },
        },
    )
    manifest.write_json(bundle_dir / "bundle_manifest.json")
    (bundle_dir / "golden_samples.jsonl").write_text(
        json.dumps(
            {
                "raw_text": "xin chao",
                "input_ids": [0, 11, 12, 2],
                "expected_output": "Xin chao.",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def test_resolve_target_or_inference_adapter_prefers_explicit_id_and_normalizes_outputs(tmp_path):
    from tools.aihub_option1_hybrid_pipeline import (
        resolve_compiled_target_reference,
        run_compiled_inference,
    )
    from tools.aihub_option1_pilots import build_option1_runtime_config

    repo_root = tmp_path / "repo"
    _init_repo_root(repo_root)
    runtime_config = build_option1_runtime_config(
        device_name="Samsung Galaxy S24",
        repo_root=repo_root,
    )

    resolved = resolve_compiled_target_reference(
        runtime_config=runtime_config,
        compile_pilot_name="zipformer_encoder_option1",
        explicit_target_model_id="model-explicit",
        run_label="phase3",
    )

    seen: dict[str, object] = {}

    def fake_runner(*, target_model_id: str, runtime_config: object, inputs: dict[str, list[np.ndarray]], inference_name: str | None):
        seen["target_model_id"] = target_model_id
        seen["runtime_config"] = runtime_config
        seen["inputs"] = inputs
        seen["inference_name"] = inference_name
        return (
            {
                "output_1": [np.asarray([2], dtype=np.int32)],
                "output_0": [np.asarray([[1.0, 2.0]], dtype=np.float32)],
            },
            {"job_id": "job-1", "url": "https://example/jobs/job-1"},
        )

    outputs, metadata = run_compiled_inference(
        target_reference=resolved,
        runtime_config=runtime_config,
        inputs={
            "x": np.zeros((1, 3, 2), dtype=np.float32),
            "x_lens": np.asarray([2], dtype=np.int64),
        },
        input_specs={
            "x": ((1, 3, 2), "float32"),
            "x_lens": ((1,), "int64"),
        },
        inference_runner=fake_runner,
        inference_name="zipformer-hybrid-sample-1",
    )

    assert resolved.target_model_id == "model-explicit"
    assert resolved.compile_record_path is None
    assert seen["target_model_id"] == "model-explicit"
    assert seen["inputs"]["x_lens"][0].dtype == np.int32
    assert list(outputs) == ["output_0", "output_1"]
    assert outputs["output_0"].shape == (1, 2)
    assert outputs["output_1"].tolist() == [2]
    assert metadata["job"]["job_id"] == "job-1"
    assert metadata["target_model_id"] == "model-explicit"


def test_resolve_target_or_inference_adapter_reads_compile_record_when_override_missing(tmp_path):
    from tools.aihub_option1_hybrid_pipeline import resolve_compiled_target_reference
    from tools.aihub_option1_pilots import build_option1_runtime_config, write_compile_run_record

    repo_root = tmp_path / "repo"
    _init_repo_root(repo_root)
    runtime_config = build_option1_runtime_config(
        device_name="Samsung Galaxy S24",
        repo_root=repo_root,
    )
    compile_record_path = write_compile_run_record(
        pilot_name="zipformer_encoder_option1",
        runtime_config=runtime_config,
        compile_options="--target_runtime precompiled_qnn_onnx --truncate_64bit_io",
        target_model={"model_id": "model-from-record", "url": "https://example/models/model-from-record"},
        run_label="phase3",
    )

    resolved = resolve_compiled_target_reference(
        runtime_config=runtime_config,
        compile_pilot_name="zipformer_encoder_option1",
        run_label="phase3",
    )

    assert resolved.target_model_id == "model-from-record"
    assert resolved.compile_record_path == compile_record_path


def test_zipformer_hybrid_runner_decodes_expected_text(tmp_path):
    from tools.aihub_option1_hybrid_pipeline import run_zipformer_hybrid_evaluation
    from tools.aihub_option1_pilots import build_option1_runtime_config, write_compile_run_record

    repo_root = tmp_path / "repo"
    _init_repo_root(repo_root)
    bundle_dir = repo_root / "build" / "model_bundle" / "zipformer" / "qnn_u16u8"
    _write_zipformer_bundle(bundle_dir, fixed_encoder_frames=3, feature_dim=2)
    fixed_encoder = repo_root / "build" / "quantize" / "zipformer" / "qnn_u16u8" / "fixed_shapes" / "encoder.fixed.onnx"
    fixed_encoder.parent.mkdir(parents=True, exist_ok=True)
    fixed_encoder.write_bytes(b"encoder")

    runtime_config = build_option1_runtime_config(
        device_name="Samsung Galaxy S24",
        repo_root=repo_root,
    )
    write_compile_run_record(
        pilot_name="zipformer_encoder_option1",
        runtime_config=runtime_config,
        compile_options="--target_runtime precompiled_qnn_onnx --truncate_64bit_io",
        target_model={"model_id": "zipformer-target", "url": "https://example/models/zipformer-target"},
        run_label="phase3",
    )

    class FakeDecoderSession:
        def __init__(self) -> None:
            self.inputs: list[dict[str, object]] = []

        def run(self, _outputs: object, feeds: dict[str, object]) -> list[object]:
            self.inputs.append(feeds)
            return [np.asarray([[0.25, 0.5]], dtype=np.float32)]

    class FakeJoinerSession:
        def __init__(self) -> None:
            self.responses = [
                np.asarray([[0.0, 9.0, 0.0]], dtype=np.float32),
                np.asarray([[9.0, 0.0, 0.0]], dtype=np.float32),
                np.asarray([[0.0, 0.0, 9.0]], dtype=np.float32),
                np.asarray([[9.0, 0.0, 0.0]], dtype=np.float32),
            ]

        def run(self, _outputs: object, feeds: dict[str, object]) -> list[object]:
            return [self.responses.pop(0)]

    class FakeBundleRuntime:
        sample_rate = 16000
        feature_dim = 2
        fixed_encoder_frames = 3
        blank_id = 0
        context_size = 2
        tokens_table = ["<blk>", "xin", " chao"]
        decoder_sess = FakeDecoderSession()
        joiner_sess = FakeJoinerSession()

    seen: dict[str, object] = {}

    def fake_feature_loader(audio_path, *, sample_rate: int, feature_dim: int):
        seen["audio_path"] = audio_path
        seen["sample_rate"] = sample_rate
        seen["feature_dim"] = feature_dim
        return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    def fake_inference_runner(*, target_model_id: str, runtime_config: object, inputs: dict[str, list[np.ndarray]], inference_name: str | None):
        seen["target_model_id"] = target_model_id
        seen["inputs"] = inputs
        return (
            {
                "output_0": [np.asarray([[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]], dtype=np.float32)],
                "output_1": [np.asarray([2], dtype=np.int32)],
            },
            {"job_id": "zip-job", "url": "https://example/jobs/zip-job"},
        )

    report = run_zipformer_hybrid_evaluation(
        runtime_config=runtime_config,
        run_label="phase3",
        max_samples=1,
        inference_runner=fake_inference_runner,
        bundle_runtime=FakeBundleRuntime(),
        feature_loader=fake_feature_loader,
    )

    assert report["target_reference"].target_model_id == "zipformer-target"
    assert report["summary"]["matched_samples"] == 1
    assert report["results"][0]["sample_id"] == "sample-1"
    assert report["results"][0]["text"] == "xin chao"
    assert report["results"][0]["expected_text"] == "xin chao"
    assert report["results"][0]["matches_expected"] is True
    assert report["results"][0]["num_tokens"] == 2
    assert seen["target_model_id"] == "zipformer-target"
    assert seen["inputs"]["x_lens"][0].dtype == np.int32
    assert seen["audio_path"] == repo_root / "assets" / "speech" / "sample-1.wav"


def test_zipformer_hybrid_runner_prefers_expected_output_fixture_audio_when_available(tmp_path):
    from tools.aihub_option1_hybrid_pipeline import run_zipformer_hybrid_evaluation
    from tools.aihub_option1_pilots import build_option1_runtime_config, write_compile_run_record

    repo_root = tmp_path / "repo"
    _init_repo_root(repo_root)
    bundle_dir = repo_root / "build" / "model_bundle" / "zipformer" / "qnn_u16u8"
    _write_zipformer_bundle(bundle_dir, fixed_encoder_frames=3, feature_dim=2)
    fixed_encoder = repo_root / "build" / "quantize" / "zipformer" / "qnn_u16u8" / "fixed_shapes" / "encoder.fixed.onnx"
    fixed_encoder.parent.mkdir(parents=True, exist_ok=True)
    fixed_encoder.write_bytes(b"encoder")
    (bundle_dir / "sample_manifest.jsonl").write_text(
        json.dumps({"sample_id": "audio-1", "audio_path": "build/calibration/audio-1.wav"}) + "\n",
        encoding="utf-8",
    )
    expected_audio = repo_root / "assets" / "speech" / "sample-1.wav"
    expected_audio.parent.mkdir(parents=True, exist_ok=True)
    expected_audio.write_bytes(b"wav")

    runtime_config = build_option1_runtime_config(
        device_name="Samsung Galaxy S24",
        repo_root=repo_root,
    )
    write_compile_run_record(
        pilot_name="zipformer_encoder_option1",
        runtime_config=runtime_config,
        compile_options="--target_runtime precompiled_qnn_onnx --truncate_64bit_io",
        target_model={"model_id": "zipformer-target", "url": "https://example/models/zipformer-target"},
        run_label="phase3",
    )

    class FakeDecoderSession:
        def run(self, _outputs: object, feeds: dict[str, object]) -> list[object]:
            return [np.asarray([[0.25, 0.5]], dtype=np.float32)]

    class FakeJoinerSession:
        def __init__(self) -> None:
            self.responses = [
                np.asarray([[0.0, 9.0, 0.0]], dtype=np.float32),
                np.asarray([[9.0, 0.0, 0.0]], dtype=np.float32),
                np.asarray([[0.0, 0.0, 9.0]], dtype=np.float32),
                np.asarray([[9.0, 0.0, 0.0]], dtype=np.float32),
            ]

        def run(self, _outputs: object, feeds: dict[str, object]) -> list[object]:
            return [self.responses.pop(0)]

    class FakeBundleRuntime:
        sample_rate = 16000
        feature_dim = 2
        fixed_encoder_frames = 3
        blank_id = 0
        context_size = 2
        tokens_table = ["<blk>", "xin", " chao"]
        decoder_sess = FakeDecoderSession()
        joiner_sess = FakeJoinerSession()

    seen: dict[str, object] = {}

    def fake_feature_loader(audio_path, *, sample_rate: int, feature_dim: int):
        seen["audio_path"] = audio_path
        return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    def fake_inference_runner(*, target_model_id: str, runtime_config: object, inputs: dict[str, list[np.ndarray]], inference_name: str | None):
        return (
            {
                "output_0": [np.asarray([[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]], dtype=np.float32)],
                "output_1": [np.asarray([2], dtype=np.int32)],
            },
            {"job_id": "zip-job", "url": "https://example/jobs/zip-job"},
        )

    report = run_zipformer_hybrid_evaluation(
        runtime_config=runtime_config,
        run_label="phase3",
        max_samples=1,
        inference_runner=fake_inference_runner,
        bundle_runtime=FakeBundleRuntime(),
        feature_loader=fake_feature_loader,
    )

    assert seen["audio_path"] == expected_audio
    assert report["results"][0]["sample_id"] == "sample-1"
    assert report["results"][0]["expected_available"] is True
    assert report["summary"]["comparable_samples"] == 1


def test_vpcd_hybrid_runner_restores_expected_text(tmp_path):
    from tools.aihub_option1_hybrid_pipeline import run_vpcd_hybrid_evaluation
    from tools.aihub_option1_pilots import build_option1_runtime_config, write_compile_run_record

    repo_root = tmp_path / "repo"
    _init_repo_root(repo_root)
    bundle_dir = repo_root / "build" / "model_bundle" / "vpcd" / "qnn_fixed_1024x128"
    _write_vpcd_bundle(bundle_dir, encoder_sequence=1024, decoder_sequence=128)

    runtime_config = build_option1_runtime_config(
        device_name="Samsung Galaxy S24",
        repo_root=repo_root,
    )
    write_compile_run_record(
        pilot_name="vpcd_option1",
        runtime_config=runtime_config,
        compile_options="--target_runtime precompiled_qnn_onnx --truncate_64bit_io",
        target_model={"model_id": "vpcd-target", "url": "https://example/models/vpcd-target"},
        run_label="phase3",
    )

    manifest = ModelBundleManifest.from_path(bundle_dir / "bundle_manifest.json")

    class FakeSession:
        def __init__(self, responses):
            self.responses = list(responses)
            self.inputs: list[dict[str, object]] = []

        def run(self, _outputs, feeds):
            self.inputs.append(feeds)
            return [self.responses.pop(0)]

    encode_session = FakeSession([np.asarray([[0, 4, 2]], dtype=np.int64)])
    decode_session = FakeSession([np.asarray(["Xin chao."], dtype=object)])
    model_step_outputs = []

    first_logits = np.zeros((1, 4, 7), dtype=np.float32)
    first_logits[0, 0, 5] = 9.0
    second_logits = np.zeros((1, 4, 7), dtype=np.float32)
    second_logits[0, 1, 2] = 9.0
    model_step_outputs.extend([first_logits, second_logits])
    seen_inputs: list[dict[str, list[np.ndarray]]] = []

    def fake_inference_runner(*, target_model_id: str, runtime_config: object, inputs: dict[str, list[np.ndarray]], inference_name: str | None):
        seen_inputs.append(inputs)
        return (
            {"output_0": [model_step_outputs.pop(0)]},
            {"job_id": f"vpcd-job-{len(seen_inputs)}", "url": f"https://example/jobs/vpcd-job-{len(seen_inputs)}"},
        )

    runtime = BundleOnnxRuntime(
        manifest=manifest,
        model_session=object(),
        encode_session=encode_session,
        decode_session=decode_session,
        tokenizer_to_model_ids=np.asarray([0, 1, 2, 3, 11], dtype=np.int64),
        model_to_tokenizer_ids=np.asarray([0, 1, 2, 3, 4, 5, 6], dtype=np.int64),
    )

    report = run_vpcd_hybrid_evaluation(
        runtime_config=runtime_config,
        run_label="phase3",
        max_samples=1,
        inference_runner=fake_inference_runner,
        bundle_runtime=runtime,
    )

    assert report["target_reference"].target_model_id == "vpcd-target"
    assert report["summary"]["matched_samples"] == 1
    assert report["results"][0]["raw_text"] == "xin chao"
    assert report["results"][0]["text"] == "Xin chao."
    assert report["results"][0]["expected_text"] == "Xin chao."
    assert report["results"][0]["matches_expected"] is True
    assert report["results"][0]["decode_steps"] == 2
    assert len(seen_inputs) == 2
    assert seen_inputs[0]["decoder_input_ids"][0].dtype == np.int32


def test_vpcd_hybrid_runner_passes_decode_step_limit_to_bundle_runtime(tmp_path):
    from tools.aihub_option1_hybrid_pipeline import run_vpcd_hybrid_evaluation
    from tools.aihub_option1_pilots import build_option1_runtime_config, write_compile_run_record

    repo_root = tmp_path / "repo"
    _init_repo_root(repo_root)
    bundle_dir = repo_root / "build" / "model_bundle" / "vpcd" / "qnn_fixed_1024x128"
    _write_vpcd_bundle(bundle_dir, encoder_sequence=1024, decoder_sequence=128)

    runtime_config = build_option1_runtime_config(
        device_name="Samsung Galaxy S24",
        repo_root=repo_root,
    )
    write_compile_run_record(
        pilot_name="vpcd_option1",
        runtime_config=runtime_config,
        compile_options="--target_runtime precompiled_qnn_onnx --truncate_64bit_io",
        target_model={"model_id": "vpcd-target", "url": "https://example/models/vpcd-target"},
        run_label="phase3",
    )

    seen: dict[str, object] = {}

    class FakeRuntime:
        def restore_with_model_step(self, text: str, model_step_runner, *, max_length: int = 128):
            seen["text"] = text
            seen["max_length"] = max_length
            return {
                "text": "Xin chao.",
                "decode_steps": max_length,
                "generated_ids": np.asarray([5] * max_length, dtype=np.int64),
            }

    report = run_vpcd_hybrid_evaluation(
        runtime_config=runtime_config,
        run_label="phase3",
        max_samples=1,
        max_decode_steps=5,
        bundle_runtime=FakeRuntime(),
    )

    assert seen["text"] == "xin chao"
    assert seen["max_length"] == 5
    assert report["results"][0]["decode_steps"] == 5


def test_vpcd_teacher_forced_diagnostics_records_cpu_and_cloud_step_summaries(tmp_path):
    from tools.aihub_option1_hybrid_pipeline import run_vpcd_teacher_forced_diagnostics
    from tools.aihub_option1_pilots import build_option1_runtime_config, write_compile_run_record

    repo_root = tmp_path / "repo"
    _init_repo_root(repo_root)
    bundle_dir = repo_root / "build" / "model_bundle" / "vpcd" / "qnn_fixed_1024x128"
    _write_vpcd_bundle(bundle_dir, encoder_sequence=1024, decoder_sequence=128)

    runtime_config = build_option1_runtime_config(
        device_name="Samsung Galaxy S24",
        repo_root=repo_root,
    )
    write_compile_run_record(
        pilot_name="vpcd_option1",
        runtime_config=runtime_config,
        compile_options="--target_runtime precompiled_qnn_onnx --truncate_64bit_io",
        target_model={"model_id": "vpcd-target", "url": "https://example/models/vpcd-target"},
        run_label="phase3",
    )

    cpu_step_outputs: list[np.ndarray] = []
    first_cpu_logits = np.zeros((1, 4, 7), dtype=np.float32)
    first_cpu_logits[0, 0, 5] = 9.0
    second_cpu_logits = np.zeros((1, 4, 7), dtype=np.float32)
    second_cpu_logits[0, 1, 6] = 9.0
    cpu_step_outputs.extend([first_cpu_logits, second_cpu_logits])

    cloud_step_outputs: list[np.ndarray] = []
    first_cloud_logits = np.zeros((1, 4, 7), dtype=np.float32)
    first_cloud_logits[0, 0, 5] = 8.0
    second_cloud_logits = np.zeros((1, 4, 7), dtype=np.float32)
    second_cloud_logits[0, 1, 4] = 8.0
    cloud_step_outputs.extend([first_cloud_logits, second_cloud_logits])

    seen_inputs: list[dict[str, list[np.ndarray]]] = []

    def fake_decode_ids(text: str) -> tuple[dict[str, np.ndarray], list[int]]:
        assert text == "xin chao"
        return (
            {
                "input_ids": np.asarray([[0, 11, 12, 2]], dtype=np.int64),
                "attention_mask": np.asarray([[1, 1, 1, 1]], dtype=np.int64),
            },
            [2, 5, 6],
        )

    def fake_cpu_model_step_runner(feeds: dict[str, np.ndarray]) -> np.ndarray:
        return cpu_step_outputs.pop(0)

    def fake_inference_runner(*, target_model_id: str, runtime_config: object, inputs: dict[str, list[np.ndarray]], inference_name: str | None):
        seen_inputs.append(inputs)
        return (
            {"output_0": [cloud_step_outputs.pop(0)]},
            {"job_id": f"teacher-job-{len(seen_inputs)}", "url": f"https://example/jobs/teacher-job-{len(seen_inputs)}"},
        )

    report = run_vpcd_teacher_forced_diagnostics(
        runtime_config=runtime_config,
        run_label="phase3",
        sample_index=0,
        max_decode_steps=2,
        inference_runner=fake_inference_runner,
        cpu_model_step_runner=fake_cpu_model_step_runner,
        decode_ids_fn=fake_decode_ids,
    )

    assert report["decode_step_limit"] == 2
    assert len(report["steps"]) == 2
    assert seen_inputs[0]["decoder_input_ids"][0].dtype == np.int32

    first_step = report["steps"][0]
    assert first_step["step_index"] == 1
    assert first_step["decoder_prefix_ids"] == [2]
    assert first_step["expected_next_token_id"] == 5
    assert first_step["cpu_argmax_token_id"] == 5
    assert first_step["cloud_argmax_token_id"] == 5
    assert first_step["job_id"] == "teacher-job-1"
    assert "cloud_top_tokens" in first_step

    second_step = report["steps"][1]
    assert second_step["step_index"] == 2
    assert second_step["decoder_prefix_ids"] == [2, 5]
    assert second_step["expected_next_token_id"] == 6
    assert second_step["cpu_argmax_token_id"] == 6
    assert second_step["cloud_argmax_token_id"] == 4
    assert second_step["matches_cpu_argmax"] is False


def test_hybrid_record_writer_persists_sample_results_and_summary(tmp_path):
    from tools.aihub_option1_hybrid_pipeline import (
        ResolvedCompiledTarget,
        write_hybrid_run_record,
    )
    from tools.aihub_option1_pilots import build_option1_runtime_config

    repo_root = tmp_path / "repo"
    _init_repo_root(repo_root)
    runtime_config = build_option1_runtime_config(
        device_name="Samsung Galaxy S24",
        repo_root=repo_root,
    )
    compile_record_path = runtime_config.pilot_record_dir("zipformer_encoder_option1") / "compile-run-phase3.json"
    compile_record_path.parent.mkdir(parents=True, exist_ok=True)
    compile_record_path.write_text("{}", encoding="utf-8")

    record_path = write_hybrid_run_record(
        pilot_name="zipformer_hybrid_option1",
        runtime_config=runtime_config,
        target_reference=ResolvedCompiledTarget(
            compile_pilot_name="zipformer_encoder_option1",
            target_model_id="zipformer-target",
            compile_record_path=compile_record_path,
            run_label="phase3",
            explicit_override=False,
        ),
        sample_results=[
            {
                "sample_id": "sample-1",
                "text": "xin chao",
                "expected_text": "xin chao",
                "matches_expected": True,
                "cloud_inference_seconds": 0.12,
                "decode_seconds": 0.03,
            },
            {
                "sample_id": "sample-2",
                "text": "xin chao",
                "expected_text": "xin chao.",
                "matches_expected": False,
                "cloud_inference_seconds": 0.22,
                "decode_seconds": 0.04,
            },
        ],
        run_label="phase3",
    )

    payload = __import__("json").loads(record_path.read_text(encoding="utf-8"))

    assert payload["target_model_id"] == "zipformer-target"
    assert payload["compile_record_path"] == compile_record_path.as_posix()
    assert payload["summary"]["sample_count"] == 2
    assert payload["summary"]["comparable_samples"] == 2
    assert payload["summary"]["matched_samples"] == 1
    assert payload["summary"]["mismatched_samples"] == 1
    assert payload["latency_summary"]["average_cloud_inference_seconds"] == 0.17
    assert payload["latency_summary"]["average_decode_seconds"] == 0.035
