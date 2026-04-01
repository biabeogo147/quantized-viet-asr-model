from __future__ import annotations

import json
import shutil
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from android_bundle.golden_samples import GoldenSample, serialize_golden_samples
from android_bundle.manifest import AndroidBundleManifest
from android_bundle.tokenizer_bridge import build_ort_tokenizer_id_bridge


DEFAULT_ASSET_NAMESPACE = "models/punctuation/vpcd"
DEFAULT_MODEL_VARIANT = "vpcd_balanced"

@dataclass(frozen=True)
class TokenizerExportArtifacts:
    encode_file_name: str
    decode_file_name: str
    tokenizer_to_model_id_map_file_name: str
    model_to_tokenizer_id_map_file_name: str


TokenizerExporter = Callable[[str, str], TokenizerExportArtifacts]
GoldenSampleBuilder = Callable[..., list[GoldenSample]]


def ensure_local_vendor_path() -> None:
    vendor_dir = Path(__file__).resolve().parent.parent / "_vendor"
    if vendor_dir.exists():
        vendor_path = str(vendor_dir)
        if vendor_path not in sys.path:
            sys.path.insert(0, vendor_path)


def resolve_variant_onnx_path(model_dir: str, model_variant: str) -> Path:
    variant_file = model_variant if model_variant.endswith(".onnx") else f"{model_variant}.onnx"
    return Path(model_dir) / "onnx" / variant_file


@contextmanager
def bartpho_tokenizer_ortx_alias(tokenizer: object):
    tokenizer_class = tokenizer.__class__
    original_name = tokenizer_class.__name__
    tokenizer_class.__name__ = "XLMRobertaTokenizer"
    try:
        yield
    finally:
        tokenizer_class.__name__ = original_name


def default_tokenizer_exporter(model_dir: str, bundle_dir: str) -> TokenizerExportArtifacts:
    ensure_local_vendor_path()
    try:
        import onnx
        from onnxruntime_extensions import gen_processing_models
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Tokenizer export requires onnx, transformers, and onnxruntime-extensions."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    with bartpho_tokenizer_ortx_alias(tokenizer):
        processing_models = gen_processing_models(
            tokenizer,
            pre_kwargs={"fairseq": True},
            post_kwargs={"fairseq": True},
        )
    if len(processing_models) < 2:
        raise RuntimeError("gen_processing_models did not return both encode and decode graphs.")

    bundle_path = Path(bundle_dir)
    encode_path = bundle_path / AndroidBundleManifest.DEFAULT_TOKENIZER_ENCODE_FILE
    decode_path = bundle_path / AndroidBundleManifest.DEFAULT_TOKENIZER_DECODE_FILE
    tokenizer_to_model_id_map_path = (
        bundle_path / AndroidBundleManifest.DEFAULT_TOKENIZER_TO_MODEL_ID_MAP_FILE
    )
    model_to_tokenizer_id_map_path = (
        bundle_path / AndroidBundleManifest.DEFAULT_MODEL_TO_TOKENIZER_ID_MAP_FILE
    )
    onnx.save_model(processing_models[0], encode_path)
    onnx.save_model(processing_models[1], decode_path)
    bridge = build_ort_tokenizer_id_bridge(tokenizer)
    bridge.write_files(
        tokenizer_to_model_path=tokenizer_to_model_id_map_path,
        model_to_tokenizer_path=model_to_tokenizer_id_map_path,
    )
    return TokenizerExportArtifacts(
        encode_file_name=encode_path.name,
        decode_file_name=decode_path.name,
        tokenizer_to_model_id_map_file_name=tokenizer_to_model_id_map_path.name,
        model_to_tokenizer_id_map_file_name=model_to_tokenizer_id_map_path.name,
    )


def default_golden_sample_builder(
    *,
    model_dir: str,
    onnx_path: str,
    max_decode_length: int,
) -> list[GoldenSample]:
    from test.test_punctuation_onnx import DEFAULT_TEXTS, VietnamesePuncCapDenormOnnx

    model = VietnamesePuncCapDenormOnnx(model_dir=model_dir, onnx_path=onnx_path)
    samples: list[GoldenSample] = []
    for text in DEFAULT_TEXTS:
        encoded = model.tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=512,
        )
        samples.append(
            GoldenSample(
                raw_text=text,
                input_ids=encoded["input_ids"][0].astype(int).tolist(),
                expected_output=model.restore(text, max_length=max_decode_length),
            )
        )
    return samples


def export_android_bundle(
    *,
    model_dir: str,
    output_dir: str,
    model_variant: str = DEFAULT_MODEL_VARIANT,
    asset_namespace: str = DEFAULT_ASSET_NAMESPACE,
    tokenizer_exporter: TokenizerExporter | None = None,
    golden_sample_builder: GoldenSampleBuilder | None = None,
    max_decode_length: int = 128,
) -> AndroidBundleManifest:
    model_root = Path(model_dir)
    bundle_dir = Path(output_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    source_onnx_path = resolve_variant_onnx_path(str(model_root), model_variant)
    if not source_onnx_path.exists():
        raise FileNotFoundError(f"Khong tim thay ONNX variant: {source_onnx_path}")

    model_output_path = bundle_dir / AndroidBundleManifest.DEFAULT_MODEL_FILE
    shutil.copy2(source_onnx_path, model_output_path)

    tokenizer_export = tokenizer_exporter or default_tokenizer_exporter
    golden_builder = golden_sample_builder or default_golden_sample_builder
    tokenizer_artifacts = tokenizer_export(str(model_root), str(bundle_dir))

    samples = golden_builder(
        model_dir=str(model_root),
        onnx_path=str(model_output_path),
        max_decode_length=max_decode_length,
    )
    (bundle_dir / AndroidBundleManifest.DEFAULT_GOLDEN_SAMPLES_FILE).write_text(
        serialize_golden_samples(samples),
        encoding="utf-8",
    )

    manifest = AndroidBundleManifest(
        bundle_version=1,
        model_name="tourmii/vietnamese-punc-cap-denorm-v1",
        model_variant=model_variant,
        asset_namespace=asset_namespace,
        model_file=model_output_path.name,
        tokenizer_encode_file=tokenizer_artifacts.encode_file_name,
        tokenizer_decode_file=tokenizer_artifacts.decode_file_name,
        tokenizer_to_model_id_map_file=tokenizer_artifacts.tokenizer_to_model_id_map_file_name,
        model_to_tokenizer_id_map_file=tokenizer_artifacts.model_to_tokenizer_id_map_file_name,
        golden_samples_file=AndroidBundleManifest.DEFAULT_GOLDEN_SAMPLES_FILE,
        pad_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=2,
        max_source_length=1024,
        max_decode_length=max_decode_length,
    )

    (bundle_dir / "bundle_manifest.json").write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
