from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from model_pipeline.evaluation import (
    VlspEvaluationSample,
    classify_text_output,
    compute_transcript_metrics,
    evaluate_vpcd_output,
    summarize_latency,
    summarize_ort_profile,
    write_evaluation_json,
    write_sample_jsonl,
)
from model_pipeline.evaluation.vlsp100 import (
    evaluate_vpcd_samples,
    evaluate_zipformer_samples,
    summarize_vpcd_parity,
)
from model_pipeline.models.vpcd.runtime import VpcdInferenceResult, VpcdLocalRuntime
from model_pipeline.models.zipformer.runtime import (
    ZipformerInferenceResult,
    ZipformerLocalRuntime,
    extract_zipformer_features,
)


def test_transcript_metrics_use_normalized_character_and_word_errors() -> None:
    """Verify transcript metrics normalize case, punctuation, and whitespace.

    Returns:
        None.
    """
    metrics = compute_transcript_metrics(
        references=["Xin chào, thế giới!", "một hai ba"],
        hypotheses=["xin chào thế giới", "một hai bốn"],
    )

    assert metrics.exact_matches == 1
    assert metrics.sample_count == 2
    assert metrics.word_errors == 1
    assert metrics.reference_words == 7
    assert metrics.character_errors == 2
    assert metrics.reference_characters == 22


def test_text_output_classification_detects_empty_and_repetition_collapse() -> None:
    """Verify invalid text-output states are classified without subjective labels.

    Returns:
        None.
    """
    assert classify_text_output("").empty is True
    assert classify_text_output("xin xin xin xin xin").repetition_collapse is True
    assert classify_text_output("xin chào mọi người").valid is True


def test_vpcd_output_metrics_record_prefix_and_full_output_parity() -> None:
    """Verify VPCD metrics separate decoder-prefix and restored-output parity.

    Returns:
        None.
    """
    result = evaluate_vpcd_output(
        fp32_output="Xin chào!",
        quantized_output="Xin chào!",
        fp32_top1=[2, 9, 4, 7, 3],
        quantized_top1=[2, 9, 4, 8, 3],
        eos_token_id=3,
    )

    assert result.exact_output_match is True
    assert result.character_edit_distance == 0
    assert result.first_five_top1_matches == 4
    assert result.first_five_step_count == 5
    assert result.early_eos is False
    assert result.punctuation_collapse is False


def test_ort_profile_attributes_cuda_only_when_nodes_executed(tmp_path: Path) -> None:
    """Verify CUDA attribution requires profiler evidence for executed nodes.

    Args:
        tmp_path: Isolated directory for synthetic ONNX Runtime profiles.

    Returns:
        None.
    """
    cpu_profile = tmp_path / "cpu.json"
    cpu_profile.write_text(
        json.dumps([{"cat": "Node", "args": {"provider": "CPUExecutionProvider"}}]),
        encoding="utf-8",
    )
    mixed_profile = tmp_path / "mixed.json"
    mixed_profile.write_text(
        json.dumps(
            [
                {"cat": "Node", "args": {"provider": "CUDAExecutionProvider"}},
                {"cat": "Node", "args": {"provider": "CPUExecutionProvider"}},
                {"cat": "Session", "args": {}},
            ]
        ),
        encoding="utf-8",
    )

    cpu = summarize_ort_profile(cpu_profile)
    mixed = summarize_ort_profile(mixed_profile)

    assert cpu.cuda_executed is False
    assert cpu.node_counts == {"CPUExecutionProvider": 1}
    assert mixed.cuda_executed is True
    assert mixed.node_counts == {"CPUExecutionProvider": 1, "CUDAExecutionProvider": 1}


def test_latency_and_machine_reports_are_deterministic(tmp_path: Path) -> None:
    """Verify latency summaries and JSON/JSONL evidence are deterministic.

    Args:
        tmp_path: Isolated report output directory.

    Returns:
        None.
    """
    latency = summarize_latency([4.0, 1.0, 3.0, 2.0])
    first_json = write_evaluation_json(tmp_path / "one.json", {"latency": latency.to_dict(), "model": "vpcd"})
    second_json = write_evaluation_json(tmp_path / "two.json", {"model": "vpcd", "latency": latency.to_dict()})
    first_jsonl = write_sample_jsonl(tmp_path / "one.jsonl", [{"b": 2, "a": 1}])
    second_jsonl = write_sample_jsonl(tmp_path / "two.jsonl", [{"a": 1, "b": 2}])

    assert latency.median_ms == 2.5
    assert latency.p95_ms == 4.0
    assert first_json.read_bytes() == second_json.read_bytes()
    assert first_jsonl.read_bytes() == second_jsonl.read_bytes()


def test_zipformer_runtime_decodes_encoder_frames_with_cpu_host_components() -> None:
    """Verify Zipformer runtime combines encoder output with FP32 decoder and joiner.

    Returns:
        None.
    """
    class EncoderSession:
        def run(self, _outputs, _feeds):
            """Return two deterministic encoder frames.

            Args:
                _outputs: Unused requested output names.
                _feeds: Unused encoder input mapping.

            Returns:
                Encoder frames and their valid length.
            """
            return [np.asarray([[[1.0], [2.0]]], dtype=np.float32), np.asarray([2])]

    class DecoderSession:
        def run(self, _outputs, _feeds):
            """Return a deterministic decoder embedding.

            Args:
                _outputs: Unused requested output names.
                _feeds: Unused decoder input mapping.

            Returns:
                One batch-one decoder embedding.
            """
            return [np.asarray([[[0.0]]], dtype=np.float32)]

    class JoinerSession:
        def __init__(self):
            """Initialize the one-token emission state.

            Returns:
                None.
            """
            self.emitted = False

        def run(self, _outputs, feeds):
            """Map the first encoder frame to token one and the second to blank.

            Args:
                _outputs: Unused requested output names.
                feeds: Joiner inputs containing the current encoder frame.

            Returns:
                Token logits for blank and one lexical token.
            """
            encoder_value = float(np.asarray(feeds["encoder_out"]).reshape(-1)[0])
            should_emit = encoder_value == 1.0 and not self.emitted
            self.emitted = self.emitted or should_emit
            logits = [0.0, 2.0] if should_emit else [2.0, 0.0]
            return [np.asarray([logits], dtype=np.float32)]

    runtime = ZipformerLocalRuntime(
        encoder_session=EncoderSession(),
        decoder_session=DecoderSession(),
        joiner_session=JoinerSession(),
        token_table={0: "<blk>", 1: "▁xin"},
        feature_extractor=lambda _waveform, _sample_rate: np.ones((2, 80), dtype=np.float32),
        fixed_encoder_frames=2,
    )

    result = runtime.transcribe(np.ones(3200, dtype=np.float32), sample_rate=16_000)

    assert result.transcript == "xin"
    assert result.token_ids == (1,)
    assert result.encoder_execution_target == "configured-onnx-runtime-provider"
    assert result.decoder_execution_target == "cpu"
    assert result.joiner_execution_target == "cpu"


def test_zipformer_runtime_emits_multiple_tokens_from_one_encoder_frame() -> None:
    """Verify recurrent neural network transducer decoding loops until blank per frame.

    Returns:
        None.
    """
    class EncoderSession:
        def run(self, _outputs, _feeds):
            """Return one deterministic encoder frame.

            Args:
                _outputs: Unused requested output names.
                _feeds: Unused encoder input mapping.

            Returns:
                One encoder frame and valid length one.
            """
            return [np.asarray([[[1.0]]], dtype=np.float32), np.asarray([1])]

    class DecoderSession:
        def run(self, _outputs, _feeds):
            """Return a deterministic decoder embedding.

            Args:
                _outputs: Unused requested output names.
                _feeds: Unused decoder input mapping.

            Returns:
                One decoder embedding.
            """
            return [np.asarray([[[0.0]]], dtype=np.float32)]

    class JoinerSession:
        def __init__(self):
            """Initialize the deterministic token sequence.

            Returns:
                None.
            """
            self.calls = 0

        def run(self, _outputs, _feeds):
            """Emit tokens one and two before blank on the same frame.

            Args:
                _outputs: Unused requested output names.
                _feeds: Unused joiner input mapping.

            Returns:
                Logits selecting the next deterministic token.
            """
            token = (1, 2, 0)[self.calls]
            self.calls += 1
            logits = np.zeros((1, 3), dtype=np.float32)
            logits[0, token] = 1.0
            return [logits]

    runtime = ZipformerLocalRuntime(
        encoder_session=EncoderSession(),
        decoder_session=DecoderSession(),
        joiner_session=JoinerSession(),
        token_table={0: "<blk>", 1: "\u2581xin", 2: "\u2581chao"},
        feature_extractor=lambda _waveform, _sample_rate: np.ones((1, 80), dtype=np.float32),
        fixed_encoder_frames=1,
    )

    result = runtime.decode_encoder_outputs(
        np.asarray([[[1.0]]], dtype=np.float32),
        encoded_length=1,
    )

    assert result.token_ids == (1, 2)
    assert result.transcript == "xin chao"


def test_zipformer_feature_extractor_matches_log_mel_frame_contract() -> None:
    """Verify Zipformer features use centered log-Mel frames on normalized audio.

    Returns:
        None.
    """
    features = extract_zipformer_features(np.zeros(3200, dtype=np.float32), 16_000)

    assert features.shape == (21, 80)
    assert np.allclose(features, np.log(1.0e-10), atol=1.0e-5)


def test_vpcd_runtime_runs_fixed_shape_autoregressive_decode() -> None:
    """Verify VPCD runtime pads fixed inputs and performs greedy host decoding.

    Returns:
        None.
    """
    class ModelSession:
        def run(self, _outputs, feeds):
            """Return the next deterministic token at the active decoder position.

            Args:
                _outputs: Unused requested output names.
                feeds: Fixed-shape source and decoder arrays.

            Returns:
                Fixed-shape logits with one selected token.
            """
            active_length = int(np.asarray(feeds["decoder_attention_mask"]).sum())
            next_tokens = {1: 5, 2: 6, 3: 2}
            logits = np.zeros((1, 4, 8), dtype=np.float32)
            logits[0, active_length - 1, next_tokens[active_length]] = 1.0
            return [logits]

    runtime = VpcdLocalRuntime(
        model_session=ModelSession(),
        encode_text=lambda _text: (np.asarray([7, 8]), np.asarray([1, 1])),
        decode_tokens=lambda token_ids: "restored:" + ",".join(str(token) for token in token_ids),
        source_length=4,
        decoder_length=4,
        pad_token_id=1,
        decoder_start_token_id=2,
        eos_token_id=2,
    )

    result = runtime.restore("input text")

    assert result.output_text == "restored:5,6"
    assert result.top1_token_ids == (5, 6, 2)
    assert result.model_execution_target == "configured-onnx-runtime-provider"
    assert result.tokenizer_execution_target == "cpu"
    assert result.autoregressive_execution_target == "cpu"


def test_vlsp_zipformer_runner_records_reference_output_and_latency() -> None:
    """Verify reusable VLSP Zipformer evaluation emits deterministic sample records.

    Returns:
        None.
    """
    class Runtime:
        def transcribe(self, waveform, *, sample_rate):
            """Return one deterministic transcript for a loaded waveform.

            Args:
                waveform: Loaded mono waveform.
                sample_rate: Loaded sample rate in hertz.

            Returns:
                Deterministic Zipformer inference result.
            """
            assert waveform.shape == (4,)
            assert sample_rate == 16_000
            return ZipformerInferenceResult("xin chao", (1, 2), 3.5)

    samples = (VlspEvaluationSample("evaluation-1", Path("one.wav"), "xin chao"),)

    records = evaluate_zipformer_samples(
        Runtime(),
        samples,
        audio_loader=lambda _path: (np.ones(4, dtype=np.float32), 16_000),
    )

    assert records == (
        {
            "sample_id": "evaluation-1",
            "reference": "xin chao",
            "transcript": "xin chao",
            "token_ids": [1, 2],
            "latency_ms": 3.5,
            "empty": False,
            "repetition_collapse": False,
        },
    )


def test_vlsp_vpcd_runner_summarizes_full_and_prefix_parity() -> None:
    """Verify VPCD VLSP evaluation aggregates exact and first-five-step parity.

    Returns:
        None.
    """
    class Runtime:
        def __init__(self, output: str, tokens: tuple[int, ...]):
            """Initialize deterministic VPCD output fields.

            Args:
                output: Restored output text.
                tokens: Generated top-1 token IDs.

            Returns:
                None.
            """
            self.output = output
            self.tokens = tokens

        def restore(self, _text):
            """Return the configured deterministic VPCD output.

            Args:
                _text: Source text accepted for runtime compatibility.

            Returns:
                Deterministic VPCD inference result.
            """
            return VpcdInferenceResult(self.output, self.tokens, 4.0)

    samples = (VlspEvaluationSample("evaluation-1", Path("one.wav"), "xin chao"),)
    fp32 = evaluate_vpcd_samples(Runtime("Xin chao!", (4, 5, 2)), samples)
    quantized = evaluate_vpcd_samples(Runtime("Xin chao!", (4, 5, 2)), samples)

    summary, records = summarize_vpcd_parity(fp32, quantized, eos_token_id=2)

    assert summary["exact_output_matches"] == 1
    assert summary["first_five_top1_matches"] == 3
    assert summary["first_five_step_count"] == 3
    assert records[0]["exact_output_match"] is True
