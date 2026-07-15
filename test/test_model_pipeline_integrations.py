from __future__ import annotations

import json
import zipfile
from pathlib import Path

from model_pipeline.core import ArtifactSpec
from model_pipeline.integrations.aihub import (
    CompileRequest,
    EvidenceStore,
    FakeAiHubClient,
    compile_or_reuse,
)
from model_pipeline.integrations.android import (
    LEGACY_NAMESPACE_COMPATIBILITY,
    materialize_bundle,
)


VPCD_ID = (
    "vpcd__q-aimet-int8-int16-encoder-matmul__s-src1x384-dec1x64"
    "__c-aihub-qnn-htp-model"
)


def test_aihub_compile_reuses_record_by_input_checksum(tmp_path: Path) -> None:
    """Verify checksum-identical compile inputs reuse stored AI Hub evidence.

    Args:
        tmp_path: Isolated evidence and output directory.

    Returns:
        None.
    """
    source = tmp_path / "model.onnx"
    source.write_bytes(b"aimet-package")
    store = EvidenceStore(tmp_path / "evidence")
    client = FakeAiHubClient(compiled_bytes=b"ep-context")
    request = CompileRequest(
        artifact=ArtifactSpec.parse(VPCD_ID),
        component="model",
        source_path=source,
        input_shapes={"input_ids": [1, 384], "decoder_input_ids": [1, 64]},
        truncate_64bit_io=True,
    )

    first = compile_or_reuse(request, client=client, evidence_store=store, output_dir=tmp_path / "one")
    second = compile_or_reuse(request, client=client, evidence_store=store, output_dir=tmp_path / "two")

    assert first.reused is False
    assert second.reused is True
    assert client.submit_count == 1
    assert first.output_checksum == second.output_checksum
    assert second.output_path.read_bytes() == b"ep-context"
    assert store.resolve(request.source_checksum).artifact_id == VPCD_ID


def test_aihub_checksum_change_forces_new_compile(tmp_path: Path) -> None:
    """Verify changed compile-package bytes force a new hosted submission.

    Args:
        tmp_path: Isolated evidence and output directory.

    Returns:
        None.
    """
    source = tmp_path / "model.onnx"
    source.write_bytes(b"v1")
    store = EvidenceStore(tmp_path / "evidence")
    client = FakeAiHubClient()
    request = CompileRequest(ArtifactSpec.parse(VPCD_ID), "model", source, {}, True)
    compile_or_reuse(request, client=client, evidence_store=store, output_dir=tmp_path / "one")
    source.write_bytes(b"v2")
    changed = CompileRequest(ArtifactSpec.parse(VPCD_ID), "model", source, {}, True)

    compile_or_reuse(changed, client=client, evidence_store=store, output_dir=tmp_path / "two")

    assert client.submit_count == 2


def test_android_bundle_is_deterministic_and_metadata_driven(tmp_path: Path) -> None:
    """Verify Android bundle checksums and component targets are deterministic.

    Args:
        tmp_path: Isolated bundle output directory.

    Returns:
        None.
    """
    model = tmp_path / "model.onnx"
    tokenizer = tmp_path / "tokenizer.onnx"
    model.write_bytes(b"compiled")
    tokenizer.write_bytes(b"cpu")
    external_data = tmp_path / "model.bin"
    external_data.write_bytes(b"external")
    components = {
        "model": (model, "qnn-htp", "onnx-epcontext"),
        "model_external_data": (external_data, "qnn-htp", "onnx-external-data"),
        "tokenizer_encode": (tokenizer, "cpu", "onnx"),
    }

    one = materialize_bundle(
        artifact=ArtifactSpec.parse(VPCD_ID), components=components, output_dir=tmp_path / "one"
    )
    two = materialize_bundle(
        artifact=ArtifactSpec.parse(VPCD_ID), components=components, output_dir=tmp_path / "two"
    )

    one_manifest = json.loads(one.manifest_path.read_text(encoding="utf-8"))
    two_manifest = json.loads(two.manifest_path.read_text(encoding="utf-8"))
    assert one_manifest == two_manifest
    assert one_manifest["components"][0]["execution_target"] == "qnn-htp"
    assert one_manifest["components"][1]["file"] == "model.bin"
    assert one_manifest["components"][1]["quantization_scope"] == "encoder-matmul"
    assert one_manifest["components"][2]["execution_target"] == "cpu"
    assert one.bundle_checksum == two.bundle_checksum


def test_android_compatibility_mapping_is_not_an_artifact_identity() -> None:
    """Verify the retained Android namespace cannot parse as an artifact ID.

    Returns:
        None.
    """
    assert len(LEGACY_NAMESPACE_COMPATIBILITY) == 1
    assert next(iter(LEGACY_NAMESPACE_COMPATIBILITY.values())) == "vpcd-runtime-default"
    for legacy_name in LEGACY_NAMESPACE_COMPATIBILITY:
        try:
            ArtifactSpec.parse(legacy_name)
        except ValueError:
            pass
        else:
            raise AssertionError("compatibility namespace must not parse as an artifact ID")


def test_aihub_zip_download_caches_onnx_and_external_data_as_one_package(tmp_path: Path) -> None:
    """Verify AI Hub ZIP outputs preserve ONNX and external data as one package.

    Args:
        tmp_path: Isolated download, evidence, and output directory.

    Returns:
        None.
    """
    class ZipClient(FakeAiHubClient):
        def download(self, job_id: str, output_path: Path) -> Path:
            """Create a provider-style ZIP containing model and external data.

            Args:
                job_id: Fake compile job identifier.
                output_path: Requested raw download destination.

            Returns:
                Path to the generated ZIP archive.
            """
            del job_id
            archive = output_path.with_suffix(".zip")
            archive.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("nested/model.onnx", b"ep-context")
                bundle.writestr("nested/model.bin", b"context-binary")
            return archive

    source = tmp_path / "source.onnx"
    source.write_bytes(b"source")
    request = CompileRequest(ArtifactSpec.parse(VPCD_ID), "model", source, {}, True)
    store = EvidenceStore(tmp_path / "evidence")

    first = compile_or_reuse(request, client=ZipClient(), evidence_store=store, output_dir=tmp_path / "one")
    second = compile_or_reuse(request, client=FakeAiHubClient(), evidence_store=store, output_dir=tmp_path / "two")

    assert first.output_path.read_bytes() == b"ep-context"
    assert [path.name for path in first.support_files] == ["model.bin"]
    assert first.support_files[0].read_bytes() == b"context-binary"
    assert second.reused is True
    assert second.output_path.read_bytes() == b"ep-context"
    assert [path.name for path in second.support_files] == ["model.bin"]
