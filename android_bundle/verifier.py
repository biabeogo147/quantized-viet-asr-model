from __future__ import annotations

import json
from pathlib import Path

from android_bundle.exporter import bartpho_tokenizer_ortx_alias, ensure_local_vendor_path
from android_bundle.manifest import AndroidBundleManifest


def load_json_array(path: str | Path) -> list[int]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def iter_golden_samples(bundle_dir: str | Path) -> list[dict[str, object]]:
    bundle_path = Path(bundle_dir)
    manifest = AndroidBundleManifest.from_path(bundle_path / "bundle_manifest.json")
    golden_samples_path = bundle_path / manifest.golden_samples_file
    samples: list[dict[str, object]] = []
    for line in golden_samples_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            samples.append(json.loads(line))
    return samples


def verify_exported_tokenizer_bundle(
    *,
    model_dir: str,
    bundle_dir: str,
) -> tuple[int, int]:
    ensure_local_vendor_path()
    import numpy as np
    import onnxruntime as ort
    from onnxruntime_extensions import get_library_path
    from transformers import AutoTokenizer

    bundle_path = Path(bundle_dir)
    manifest = AndroidBundleManifest.from_path(bundle_path / "bundle_manifest.json")
    tokenizer_to_model_ids = np.asarray(
        load_json_array(bundle_path / manifest.tokenizer_to_model_id_map_file),
        dtype=np.int64,
    )
    model_to_tokenizer_ids = np.asarray(
        load_json_array(bundle_path / manifest.model_to_tokenizer_id_map_file),
        dtype=np.int64,
    )

    session_options = ort.SessionOptions()
    session_options.register_custom_ops_library(get_library_path())
    encode_session = ort.InferenceSession(
        str(bundle_path / manifest.tokenizer_encode_file),
        session_options,
        providers=["CPUExecutionProvider"],
    )
    decode_session = ort.InferenceSession(
        str(bundle_path / manifest.tokenizer_decode_file),
        session_options,
        providers=["CPUExecutionProvider"],
    )

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    encode_verified = 0
    decode_verified = 0
    for sample in iter_golden_samples(bundle_path):
        raw_text = str(sample["raw_text"])
        expected_input_ids = np.asarray(sample["input_ids"], dtype=np.int64)
        ort_tokenizer_ids = np.asarray(
            encode_session.run(None, {"inputs": np.asarray([raw_text], dtype=object)})[0],
            dtype=np.int64,
        )
        bridged_model_ids = tokenizer_to_model_ids[ort_tokenizer_ids]
        if not np.array_equal(bridged_model_ids, expected_input_ids):
            raise AssertionError(
                f"Tokenizer encode mismatch for '{raw_text}': "
                f"{bridged_model_ids.tolist()} != {expected_input_ids.tolist()}"
            )
        encode_verified += 1

        bridged_tokenizer_ids = model_to_tokenizer_ids[expected_input_ids]
        decoded_text = decode_session.run(None, {"ids": bridged_tokenizer_ids})[0].tolist()[0]
        with bartpho_tokenizer_ortx_alias(tokenizer):
            expected_decoded_text = tokenizer.decode(expected_input_ids.tolist(), skip_special_tokens=True)
        if decoded_text != expected_decoded_text:
            raise AssertionError(
                f"Tokenizer decode mismatch for '{raw_text}': {decoded_text!r} != {expected_decoded_text!r}"
            )
        decode_verified += 1

    return encode_verified, decode_verified
