from __future__ import annotations

from pathlib import Path
import shutil

from model_bundle.fixtures import AudioExpectedOutput, AudioSampleFixture, read_jsonl, serialize_jsonl
from model_bundle.manifest import ModelBundleManifest
from model_bundle.zipformer_runtime import BundleAcousticRuntime, ModelDirAcousticRuntime
from tools.bundle_paths import resolve_bundle_dir
from tools.paths import resolve_repo_path


DEFAULT_MODEL_DIR = Path("assets") / "zipformer"
DEFAULT_OUTPUT_DIR = resolve_bundle_dir("zipformer", "fp32")
DEFAULT_ASSET_NAMESPACE = "models/asr/zipformer/fp32"
DEFAULT_VARIANT = "fp32"
DEFAULT_AUDIO_FIXTURES = [
    AudioSampleFixture(sample_id="sample-1", audio_path="assets/speech/sample-1.mp3"),
    AudioSampleFixture(sample_id="sample-2", audio_path="assets/speech/sample-2.wav"),
]


def _build_expected_outputs(
    runtime: ModelDirAcousticRuntime,
    fixtures: list[AudioSampleFixture],
    workspace_root: Path,
) -> list[AudioExpectedOutput]:
    outputs: list[AudioExpectedOutput] = []
    for fixture in fixtures:
        text = runtime.transcribe(workspace_root / fixture.audio_path)["text"]
        outputs.append(AudioExpectedOutput(sample_id=fixture.sample_id, audio_path=fixture.audio_path, text=text))
    return outputs


def export_bundle(
    *,
    model_dir: Path,
    output_dir: Path,
    asset_namespace: str = DEFAULT_ASSET_NAMESPACE,
    provider: str = "CPUExecutionProvider",
    sample_fixtures: list[AudioSampleFixture] | None = None,
    expected_outputs: list[AudioExpectedOutput] | None = None,
    component_paths: dict[str, str | Path] | None = None,
    model_variant: str = DEFAULT_VARIANT,
    extra_metadata: dict | None = None,
) -> ModelBundleManifest:
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = {key: Path(value) for key, value in (component_paths or {}).items()}
    source_encoder = resolved.get("encoder", model_dir / "encoder-epoch-20-avg-1.onnx")
    source_decoder = resolved.get("decoder", model_dir / "decoder-epoch-20-avg-1.onnx")
    source_joiner = resolved.get("joiner", model_dir / "joiner-epoch-20-avg-1.onnx")
    source_tokens = resolved.get("tokens", model_dir / "tokens.txt")

    encoder_out = output_dir / "encoder.onnx"
    decoder_out = output_dir / "decoder.onnx"
    joiner_out = output_dir / "joiner.onnx"
    tokens_out = output_dir / "tokens.txt"
    shutil.copy2(source_encoder, encoder_out)
    shutil.copy2(source_decoder, decoder_out)
    shutil.copy2(source_joiner, joiner_out)
    shutil.copy2(source_tokens, tokens_out)

    fixtures = sample_fixtures or list(DEFAULT_AUDIO_FIXTURES)
    expected = expected_outputs
    if expected is None:
        runtime = ModelDirAcousticRuntime(
            model_dir=model_dir,
            provider=provider,
            component_paths={
                "encoder": source_encoder,
                "decoder": source_decoder,
                "joiner": source_joiner,
                "tokens": source_tokens,
            },
        )
        expected = _build_expected_outputs(runtime, fixtures, resolve_repo_path(".", anchor=__file__))

    sample_manifest_path = output_dir / "sample_manifest.jsonl"
    expected_outputs_path = output_dir / "expected_outputs.jsonl"
    sample_manifest_path.write_text(serialize_jsonl(fixtures), encoding="utf-8")
    expected_outputs_path.write_text(serialize_jsonl(expected), encoding="utf-8")

    metadata = {
        "sample_rate": 16000,
        "feature_dim": 80,
        "blank_id": 0,
        "context_size": 2,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    manifest = ModelBundleManifest(
        bundle_version=1,
        project="zipformer",
        model_family="zipformer-rnnt",
        model_name=f"zipformer/{model_variant}",
        model_variant=model_variant,
        asset_namespace=asset_namespace,
        runtime_kind="rnnt_greedy",
        artifacts={
            "encoder": encoder_out.name,
            "decoder": decoder_out.name,
            "joiner": joiner_out.name,
            "tokens": tokens_out.name,
        },
        fixtures={
            "sample_manifest": sample_manifest_path.name,
            "expected_outputs": expected_outputs_path.name,
        },
        metadata=metadata,
    )
    manifest.write_json(output_dir / "bundle_manifest.json")
    return manifest


def verify_bundle(
    *,
    model_dir: Path | None = None,
    bundle_dir: Path | None = None,
    reference_bundle: Path | None = None,
    candidate_bundle: Path | None = None,
    provider: str = "CPUExecutionProvider",
) -> dict:
    mismatches: list[dict] = []

    if reference_bundle is not None and candidate_bundle is not None:
        reference_manifest = ModelBundleManifest.from_path(reference_bundle / "bundle_manifest.json")
        sample_rows = [
            AudioSampleFixture.from_dict(row)
            for row in read_jsonl(reference_bundle / reference_manifest.fixtures["sample_manifest"])
        ]
        expected_rows = {
            row["sample_id"]: AudioExpectedOutput.from_dict(row)
            for row in read_jsonl(reference_bundle / reference_manifest.fixtures["expected_outputs"])
        }
        reference_runtime = BundleAcousticRuntime.from_manifest_path(
            reference_bundle / "bundle_manifest.json", provider=provider
        )
        candidate_runtime = BundleAcousticRuntime.from_manifest_path(
            candidate_bundle / "bundle_manifest.json", provider=provider
        )
        workspace_root = resolve_repo_path(".", anchor=__file__)
        for fixture in sample_rows:
            audio_path = workspace_root / fixture.audio_path
            reference_text = reference_runtime.transcribe(audio_path)["text"]
            candidate_text = candidate_runtime.transcribe(audio_path)["text"]
            expected_text = expected_rows[fixture.sample_id].text if fixture.sample_id in expected_rows else reference_text
            if candidate_text != reference_text:
                mismatches.append(
                    {
                        "sample_id": fixture.sample_id,
                        "expected_text": expected_text,
                        "reference_text": reference_text,
                        "candidate_text": candidate_text,
                    }
                )
        return {
            "project": "zipformer",
            "passed": not mismatches,
            "checked_samples": len(sample_rows),
            "mismatches": mismatches,
            "bundle_dir": str(candidate_bundle),
        }

    if model_dir is None or bundle_dir is None:
        raise ValueError("zipformer verification requires either model_dir+bundle_dir or reference_bundle+candidate_bundle")

    manifest = ModelBundleManifest.from_path(bundle_dir / "bundle_manifest.json")
    sample_rows = [
        AudioSampleFixture.from_dict(row)
        for row in read_jsonl(bundle_dir / manifest.fixtures["sample_manifest"])
    ]
    expected_rows = {
        row["sample_id"]: AudioExpectedOutput.from_dict(row)
        for row in read_jsonl(bundle_dir / manifest.fixtures["expected_outputs"])
    }
    model_runtime = ModelDirAcousticRuntime(model_dir=model_dir, provider=provider)
    bundle_runtime = BundleAcousticRuntime.from_manifest_path(bundle_dir / "bundle_manifest.json", provider=provider)
    workspace_root = resolve_repo_path(".", anchor=__file__)
    for fixture in sample_rows:
        audio_path = workspace_root / fixture.audio_path
        model_text = model_runtime.transcribe(audio_path)["text"]
        bundle_text = bundle_runtime.transcribe(audio_path)["text"]
        expected_text = expected_rows[fixture.sample_id].text if fixture.sample_id in expected_rows else model_text
        if bundle_text != model_text:
            mismatches.append(
                {
                    "sample_id": fixture.sample_id,
                    "expected_text": expected_text,
                    "model_dir_text": model_text,
                    "bundle_text": bundle_text,
                }
            )
    return {
        "project": "zipformer",
        "passed": not mismatches,
        "checked_samples": len(sample_rows),
        "mismatches": mismatches,
        "bundle_dir": str(bundle_dir),
    }
