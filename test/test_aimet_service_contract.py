from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from model_pipeline.models import get_recipe
from model_pipeline.models.aimet_service import (
    AimetServiceClient,
    _enable_only_allowlisted_ops,
    export_qdq_with_aimet,
)
from model_pipeline.models.vpcd.adapter import VpcdAdapter
from model_pipeline.models.vpcd.quantization import CalibrationBatch


def test_generic_aimet_client_translates_only_repository_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify generic AIMET export requests contain portable container paths.

    Args:
        tmp_path: Isolated host repository root.
        monkeypatch: Pytest fixture replacing the HTTP request.

    Returns:
        None.
    """
    paths = {}
    for name in ("model.onnx", "calibration", "output", "config.json", "policy.json"):
        path = tmp_path / name
        if "." in name:
            path.write_text("{}", encoding="utf-8")
        else:
            path.mkdir()
        paths[name] = path
    captured: dict[str, object] = {}
    client = AimetServiceClient(repo_root=tmp_path)

    def fake_request(endpoint, payload=None, timeout=30):
        """Capture one generic service request.

        Args:
            endpoint: Requested HTTP endpoint.
            payload: JSON-compatible request body.
            timeout: Request timeout in seconds.

        Returns:
            Fake service response.
        """
        captured.update({"endpoint": endpoint, "payload": payload, "timeout": timeout})
        return {"outputs": {}}

    monkeypatch.setattr(client, "_request", fake_request)

    client.export(
        fp32_model_path=paths["model.onnx"],
        calibration_dir=paths["calibration"],
        output_dir=paths["output"],
        config_path=paths["config.json"],
        policy_path=paths["policy.json"],
    )

    assert captured["endpoint"] == "/export"
    assert captured["timeout"] == 7200
    assert captured["payload"] == {
        "fp32_model_path": "/workspace/model.onnx",
        "calibration_dir": "/workspace/calibration",
        "output_dir": "/workspace/output",
        "config_path": "/workspace/config.json",
        "policy_path": "/workspace/policy.json",
    }


def test_generic_aimet_client_exports_benchmark_qdq_with_strict_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify QDQ export sends every provenance input through portable paths.

    Args:
        tmp_path: Isolated repository root containing fake AIMET inputs.
        monkeypatch: Pytest fixture replacing the HTTP request.

    Returns:
        None.
    """
    paths = {}
    for name in ("model.onnx", "model.encodings", "config.json", "policy.json", "qdq"):
        path = tmp_path / name
        if name == "qdq":
            path.mkdir()
        else:
            path.write_text("{}", encoding="utf-8")
        paths[name] = path
    captured: dict[str, object] = {}
    client = AimetServiceClient(repo_root=tmp_path)

    def fake_request(endpoint, payload=None, timeout=30):
        """Capture one QDQ service request.

        Args:
            endpoint: Requested HTTP endpoint.
            payload: JSON-compatible request body.
            timeout: Request timeout in seconds.

        Returns:
            Fake QDQ service response.
        """
        captured.update({"endpoint": endpoint, "payload": payload, "timeout": timeout})
        return {"outputs": {"model": "model.qdq.onnx"}}

    monkeypatch.setattr(client, "_request", fake_request)

    client.export_qdq(
        fp32_model_path=paths["model.onnx"],
        encodings_path=paths["model.encodings"],
        output_dir=paths["qdq"],
        config_path=paths["config.json"],
        policy_path=paths["policy.json"],
    )

    assert captured == {
        "endpoint": "/export-qdq",
        "timeout": 7200,
        "payload": {
            "fp32_model_path": "/workspace/model.onnx",
            "encodings_path": "/workspace/model.encodings",
            "output_dir": "/workspace/qdq",
            "config_path": "/workspace/config.json",
            "policy_path": "/workspace/policy.json",
        },
    }


def test_qdq_export_loads_encodings_strictly_and_uses_signed_activations(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify benchmark QDQ export reuses exact encodings without calibration.

    Args:
        tmp_path: Isolated fake AIMET package and QDQ output root.
        monkeypatch: Pytest fixture replacing AIMET and ONNX dependencies.

    Returns:
        None.
    """
    import sys
    from types import SimpleNamespace

    fp32 = tmp_path / "model.onnx"
    encodings = tmp_path / "model.encodings"
    config = tmp_path / "config.json"
    policy = tmp_path / "policy.json"
    for path in (fp32, encodings, config):
        path.write_text("{}", encoding="utf-8")
    policy.write_text(
        json.dumps(
            {
                "quantizer_selection": "operator-name-allowlist",
                "quantize_op_names": [],
                "symmetric_activation_encodings": True,
            }
        ),
        encoding="utf-8",
    )
    calls: dict[str, object] = {}

    class FakeSimulation:
        def __init__(self, model, **kwargs) -> None:
            """Capture simulation construction.

            Args:
                model: Fake loaded ONNX model.
                kwargs: Quantization simulation settings.

            Returns:
                None.
            """
            calls["simulation"] = {"model": model, **kwargs}
            self.qc_quantize_op_dict = {}
            self.connected_graph = SimpleNamespace(get_all_ops=lambda: {})

        def to_onnx_qdq(self, **kwargs):
            """Return a fake QDQ graph and record export settings.

            Args:
                kwargs: QDQ conversion settings.

            Returns:
                Fake converted graph.
            """
            calls["qdq"] = kwargs
            return "qdq-graph"

    def fake_load_encodings(simulation, path, **kwargs) -> None:
        """Capture strict encoding restoration.

        Args:
            simulation: Fake AIMET simulation instance.
            path: Encoding file path.
            kwargs: Strict restoration settings.

        Returns:
            None.
        """
        calls["load_encodings"] = {"simulation": simulation, "path": path, **kwargs}

    def fake_save_model(model, path, **kwargs) -> None:
        """Materialize fake QDQ bytes.

        Args:
            model: Fake converted graph.
            path: Destination ONNX path.
            kwargs: ONNX save settings.

        Returns:
            None.
        """
        calls["save"] = {"model": model, "path": path, **kwargs}
        Path(path).write_bytes(b"qdq")

    monkeypatch.setitem(
        sys.modules,
        "onnx",
        SimpleNamespace(load=lambda path: "fp32-graph", save_model=fake_save_model),
    )
    monkeypatch.setitem(
        sys.modules,
        "aimet_common.defs",
        SimpleNamespace(QuantScheme=SimpleNamespace(min_max="min-max")),
    )
    monkeypatch.setitem(
        sys.modules,
        "aimet_onnx.quantsim",
        SimpleNamespace(
            QuantizationSimModel=FakeSimulation,
            load_encodings_to_sim=fake_load_encodings,
        ),
    )

    outputs = export_qdq_with_aimet(
        fp32_model_path=fp32,
        encodings_path=encodings,
        output_dir=tmp_path / "qdq",
        config_path=config,
        policy=json.loads(policy.read_text(encoding="utf-8")),
    )

    assert calls["load_encodings"] == {
        "simulation": calls["load_encodings"]["simulation"],
        "path": encodings.resolve().as_posix(),
        "strict": True,
        "disable_missing_quantizers": True,
    }
    assert calls["qdq"] == {
        "prequantize_constants": True,
        "force_activation_as": "signed",
    }
    assert outputs["model"].read_bytes() == b"qdq"


def test_operator_allowlist_enables_only_selected_tensor_quantizers() -> None:
    """Verify policy selection enables target tensors and disables all others.

    Returns:
        None.
    """

    class Quantizer:
        def __init__(self) -> None:
            """Initialize one enabled fake tensor quantizer.

            Returns:
                None.
            """
            self.enabled = True
            self.use_symmetric_encodings = False

    class Product:
        def __init__(self, name: str) -> None:
            """Initialize a fake connected-graph tensor.

            Args:
                name: Tensor name used to resolve its quantizer.

            Returns:
                None.
            """
            self.name = name

    class Operation:
        def __init__(self, inputs: tuple[str, ...], outputs: tuple[str, ...]) -> None:
            """Initialize a fake connected-graph operation.

            Args:
                inputs: Operation input tensor names.
                outputs: Operation output tensor names.

            Returns:
                None.
            """
            self.inputs = [Product(name) for name in inputs]
            self.outputs = [Product(name) for name in outputs]
            self.parameters = {}

    class ConnectedGraph:
        def get_all_ops(self):
            """Return target and non-target fake operations.

            Returns:
                Operations keyed by policy-visible names.
            """
            return {
                "target": Operation(("target-input", "target-weight"), ("target-output",)),
                "other": Operation(("target-output",), ("other-output",)),
            }

    class Simulation:
        def __init__(self) -> None:
            """Initialize fake simulation quantizers and graph.

            Returns:
                None.
            """
            self.connected_graph = ConnectedGraph()
            self.qc_quantize_op_dict = {
                name: Quantizer()
                for name in (
                    "unselected-model-input",
                    "target-input",
                    "target-weight",
                    "target-output",
                    "other-output",
                )
            }

    simulation = Simulation()
    result = _enable_only_allowlisted_ops(simulation, ("target",))

    assert result == {"enabled_quantizer_count": 3, "missing_op_names": []}
    assert simulation.qc_quantize_op_dict["unselected-model-input"].enabled is False
    assert simulation.qc_quantize_op_dict["target-input"].enabled is True
    assert simulation.qc_quantize_op_dict["target-weight"].enabled is True
    assert simulation.qc_quantize_op_dict["target-output"].enabled is True
    assert simulation.qc_quantize_op_dict["other-output"].enabled is False


def test_operator_allowlist_can_force_every_selected_quantizer_symmetric() -> None:
    """Verify only selected activation quantizers receive symmetric settings.

    Returns:
        None.
    """
    class Quantizer:
        def __init__(self) -> None:
            """Initialize one asymmetric fake tensor quantizer.

            Returns:
                None.
            """
            self.enabled = True
            self.use_symmetric_encodings = False

    class Product:
        def __init__(self, name: str) -> None:
            """Initialize one fake connected-graph product.

            Args:
                name: Tensor name used by the quantizer dictionary.

            Returns:
                None.
            """
            self.name = name

    class Operation:
        inputs = [Product("shared-matmul-input"), Product("matmul-weight")]
        outputs = [Product("matmul-output")]
        parameters = {"matmul-weight": object()}

    class ConnectedGraph:
        def get_all_ops(self):
            """Return one allowlisted MatMul operation.

            Returns:
                Fake connected operation mapping.
            """
            return {"encoder-matmul": Operation()}

    class Simulation:
        connected_graph = ConnectedGraph()
        qc_quantize_op_dict = {
            "shared-matmul-input": Quantizer(),
            "matmul-output": Quantizer(),
            "matmul-weight": Quantizer(),
            "unselected": Quantizer(),
        }

    simulation = Simulation()

    _enable_only_allowlisted_ops(
        simulation,
        ("encoder-matmul",),
        symmetric_encodings=True,
    )

    assert simulation.qc_quantize_op_dict["shared-matmul-input"].use_symmetric_encodings is True
    assert simulation.qc_quantize_op_dict["matmul-output"].use_symmetric_encodings is True
    assert simulation.qc_quantize_op_dict["matmul-weight"].enabled is True
    assert simulation.qc_quantize_op_dict["matmul-weight"].use_symmetric_encodings is False
    assert simulation.qc_quantize_op_dict["unselected"].use_symmetric_encodings is False


def test_vpcd_adapter_exports_384_64_encoder_matmul_package(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify VPCD adapter exports fixed-shape AIMET package and evidence.

    Args:
        tmp_path: Isolated prepared and quantized component directories.
        monkeypatch: Pytest fixture replacing model-dependent calibration and inventory.

    Returns:
        None.
    """
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    prepared = {
        "model": prepared_dir / "model.onnx",
        "tokenizer_encode": prepared_dir / "tokenizer.encode.onnx",
        "tokenizer_decode": prepared_dir / "tokenizer.decode.onnx",
        "tokenizer_to_model_id_map": prepared_dir / "to-model.json",
        "model_to_tokenizer_id_map": prepared_dir / "from-model.json",
        "autoregressive_loop": prepared_dir / "runtime.json",
    }
    for role, path in prepared.items():
        path.write_bytes(role.encode())
    batch = CalibrationBatch(
        {
            "input_ids": np.zeros((1, 384), dtype=np.int64),
            "attention_mask": np.ones((1, 384), dtype=np.int64),
            "decoder_input_ids": np.zeros((1, 64), dtype=np.int64),
            "decoder_attention_mask": np.ones((1, 64), dtype=np.int64),
        }
    )
    monkeypatch.setattr(
        "model_pipeline.models.vpcd.adapter.build_calibration_batches",
        lambda **_kwargs: ([batch], {"text_samples": 24, "batches": 24}),
    )
    monkeypatch.setattr(
        "model_pipeline.models.vpcd.adapter.build_encoder_matmul_policy",
        lambda _path: {
            "coverage": {"quantized": 96, "total_matmul": 265},
            "disable_op_names": ["node"] * 169,
        },
    )

    class FakeAimetService:
        def healthcheck(self) -> None:
            """Accept the fake service healthcheck.

            Returns:
                None.
            """
            return None

        def export(self, **kwargs):
            """Materialize a fake VPCD AIMET package.

            Args:
                kwargs: Generic AIMET export paths.

            Returns:
                Fake service output metadata.
            """
            output_dir = Path(kwargs["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "model.onnx").write_bytes(b"aimet-vpcd")
            (output_dir / "model.encodings").write_text("{}", encoding="utf-8")
            return {"outputs": {"model": "model.onnx", "encodings": "model.encodings"}}

    calibration_text = tmp_path / "transcriptions.txt"
    calibration_text.write_text("mot hai ba bon\n", encoding="utf-8")
    adapter = VpcdAdapter(
        tmp_path,
        calibration_text=calibration_text,
        aimet_service=FakeAimetService(),
    )

    outputs = adapter.quantize(
        get_recipe("vpcd", "aimet-int8-int16-encoder-matmul"),
        prepared,
        tmp_path / "quantized",
    )

    assert outputs["model"].read_bytes() == b"aimet-vpcd"
    assert outputs["encodings"].is_file()
    assert outputs["calibration_manifest"].is_file()
    assert outputs["quantization_policy"].is_file()
