from __future__ import annotations

import json
import sys
from types import SimpleNamespace
import zipfile
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from model_pipeline.core import ArtifactSpec
from model_pipeline.integrations.aihub import (
    CompiledModelContract,
    CompileRequest,
    EvidenceStore,
    FakeAiHubClient,
    HostedInferenceStore,
    QualcommAiHubClient,
    compile_or_reuse,
    run_hosted_inputs,
    validate_compiled_model,
)
from model_pipeline.integrations import android
from model_pipeline.integrations.android import materialize_bundle


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


def test_android_integration_exports_no_historical_namespace_alias() -> None:
    """Verify Android integration exposes no historical namespace alias.

    Returns:
        None.
    """
    assert not hasattr(android, "LEGACY_" + "NAMESPACE_COMPATIBILITY")


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


def test_hosted_inference_rejects_more_than_five_inputs_before_submission(tmp_path: Path) -> None:
    """Verify the hosted input quota is enforced before provider submission.

    Args:
        tmp_path: Isolated hosted-evidence directory.

    Returns:
        None.
    """
    client = FakeAiHubClient()
    inputs = tuple({"input": np.asarray([index], dtype=np.float32)} for index in range(6))

    with pytest.raises(ValueError, match="at most 5"):
        run_hosted_inputs(
            artifact=ArtifactSpec.parse(VPCD_ID),
            compile_job_id="fake-compile-job",
            inputs=inputs,
            client=client,
            evidence_store=HostedInferenceStore(tmp_path),
        )

    assert client.live_run_count == 0


def test_hosted_inference_records_five_inputs_by_content_checksum(tmp_path: Path) -> None:
    """Verify five hosted results are keyed by deterministic input checksums.

    Args:
        tmp_path: Isolated hosted-evidence directory.

    Returns:
        None.
    """
    client = FakeAiHubClient()
    store = HostedInferenceStore(tmp_path)
    inputs = tuple({"input": np.asarray([index], dtype=np.float32)} for index in range(5))

    results = run_hosted_inputs(
        artifact=ArtifactSpec.parse(VPCD_ID),
        compile_job_id="fake-compile-job",
        inputs=inputs,
        client=client,
        evidence_store=store,
    )

    assert client.live_run_count == 5
    assert len({result.evidence.input_checksum for result in results}) == 5
    assert all(store.resolve(result.evidence.input_checksum) == result.evidence for result in results)
    assert all(result.evidence.output_checksum for result in results)


def test_hosted_inference_reuses_saved_outputs_without_new_submission(tmp_path: Path) -> None:
    """Verify successful hosted inputs resume without consuming cloud quota again.

    Args:
        tmp_path: Isolated hosted evidence and output store.

    Returns:
        None.
    """
    client = FakeAiHubClient()
    store = HostedInferenceStore(tmp_path)
    inputs = tuple({"input": np.asarray([index], dtype=np.float32)} for index in range(5))

    first = run_hosted_inputs(
        artifact=ArtifactSpec.parse(VPCD_ID),
        compile_job_id="fake-compile-job",
        inputs=inputs,
        client=client,
        evidence_store=store,
    )
    second = run_hosted_inputs(
        artifact=ArtifactSpec.parse(VPCD_ID),
        compile_job_id="fake-compile-job",
        inputs=inputs,
        client=client,
        evidence_store=store,
    )

    assert client.live_run_count == 5
    assert [result.evidence for result in second] == [result.evidence for result in first]
    assert [result.outputs for result in second] == [result.outputs for result in first]


def test_qualcomm_client_reconnects_compile_job_for_live_run(monkeypatch) -> None:
    """Verify hosted inference resolves a target from a prior compile process.

    Args:
        monkeypatch: Pytest fixture replacing the Qualcomm AI Hub SDK module.

    Returns:
        None.
    """

    class Target:
        """Represent one resolved hosted target model."""

    class CompileJob:
        def get_target_model(self):
            """Return the compiled target model.

            Returns:
                Resolved fake target.
            """
            return Target()

    class InferenceJob:
        job_id = "hosted-job"

        def download_output_data(self):
            """Return deterministic hosted output tensors.

            Returns:
                Named fake output tensors.
            """
            return {"output": [np.asarray([3.0], dtype=np.float32)]}

    calls: dict[str, object] = {}

    def submit_inference_job(**kwargs):
        """Capture a fake hosted inference submission.

        Args:
            kwargs: Hosted model, device, input, option, and name fields.

        Returns:
            Fake completed inference job.
        """
        calls.update(kwargs)
        return InferenceJob()

    fake_hub = SimpleNamespace(
        get_job=lambda job_id: CompileJob(),
        Device=lambda name: name,
        submit_inference_job=submit_inference_job,
    )
    monkeypatch.setitem(sys.modules, "qai_hub", fake_hub)
    client = QualcommAiHubClient(device_name="Samsung Galaxy S23 (Family)")

    result = client.live_run(
        job_id="prior-compile-job",
        inputs={"input": [np.asarray([1.0], dtype=np.float32)]},
    )

    assert result["job_id"] == "hosted-job"
    assert isinstance(calls["model"], Target)
    assert calls["options"] == "--compute_unit npu"


def test_qualcomm_client_reconnects_compile_job_while_waiting(monkeypatch) -> None:
    """Verify compile waiting can resume after the submitting process exits.

    Args:
        monkeypatch: Pytest fixture replacing the Qualcomm AI Hub SDK module.

    Returns:
        None.
    """
    class Target:
        model_id = "compiled-target"

    class Status:
        code = "SUCCESS"
        message = "complete"

    class CompileJob:
        def get_target_model(self):
            """Return the completed target model.

            Returns:
                Fake compiled target model.
            """
            return Target()

        def get_status(self):
            """Return the successful compile status.

            Returns:
                Fake successful status object.
            """
            return Status()

    fake_hub = SimpleNamespace(get_job=lambda job_id: CompileJob())
    monkeypatch.setitem(sys.modules, "qai_hub", fake_hub)
    client = QualcommAiHubClient(device_name="Samsung Galaxy S23 (Family)")

    result = client.wait("prior-compile-job")

    assert result == {
        "job_id": "prior-compile-job",
        "status": "success",
        "message": "complete",
        "target_model_id": "compiled-target",
    }


def test_compiled_model_validator_requires_epcontext_and_expected_io(tmp_path: Path) -> None:
    """Verify downloaded ONNX packages expose EPContext and target I/O dtypes.

    Args:
        tmp_path: Isolated compiled-model directory.

    Returns:
        None.
    """
    model_path = tmp_path / "model.onnx"
    graph = helper.make_graph(
        [helper.make_node("EPContext", ["input_ids"], ["logits"], domain="com.microsoft")],
        "compiled",
        [helper.make_tensor_value_info("input_ids", TensorProto.INT32, [1, 384])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 64, 10])],
    )
    onnx.save(helper.make_model(graph), model_path)
    contract = CompiledModelContract(
        artifact=ArtifactSpec.parse(VPCD_ID),
        input_dtypes={"input_ids": "int32"},
        output_dtypes={"logits": "float32"},
        requires_int64_to_int32=True,
    )

    evidence = validate_compiled_model(model_path, contract)

    assert evidence.has_ep_context is True
    assert evidence.execution_target == "qnn-htp"
    assert evidence.quantization_scope == "encoder-matmul"
    assert evidence.input_dtypes == {"input_ids": "int32"}
    assert evidence.output_dtypes == {"logits": "float32"}


def test_compiled_model_validator_rejects_untransformed_int64_input(tmp_path: Path) -> None:
    """Verify VPCD downloaded models cannot retain int64 target inputs.

    Args:
        tmp_path: Isolated compiled-model directory.

    Returns:
        None.
    """
    model_path = tmp_path / "model.onnx"
    graph = helper.make_graph(
        [helper.make_node("EPContext", ["input_ids"], ["logits"], domain="com.microsoft")],
        "compiled",
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, 384])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 64, 10])],
    )
    onnx.save(helper.make_model(graph), model_path)

    with pytest.raises(ValueError, match="int64-to-int32"):
        validate_compiled_model(
            model_path,
            CompiledModelContract(
                artifact=ArtifactSpec.parse(VPCD_ID),
                input_dtypes={"input_ids": "int32"},
                output_dtypes={"logits": "float32"},
                requires_int64_to_int32=True,
            ),
        )
