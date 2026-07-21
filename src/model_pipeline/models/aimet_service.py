from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib import error, request

from model_pipeline.models.aimet import load_aimet_calibration_inputs


DEFAULT_URL = "http://127.0.0.1:18080"


class AimetServiceClient:
    def __init__(
        self,
        *,
        repo_root: str | Path,
        url: str = DEFAULT_URL,
        workspace_root: str = "/workspace",
    ):
        """Initialize host-to-container path and HTTP service settings.

        Args:
            repo_root: Host repository root mounted into the AIMET container.
            url: Base URL of the local AIMET service.
            workspace_root: Container path corresponding to the repository root.

        Returns:
            None.
        """
        self.repo_root = Path(repo_root).resolve()
        self.url = url.rstrip("/")
        self.workspace_root = workspace_root.rstrip("/")

    def healthcheck(self) -> None:
        """Require an affirmative health response from the AIMET service.

        Returns:
            None.

        Raises:
            RuntimeError: If the service is unreachable or unhealthy.
        """
        payload = self._request("/healthz")
        if payload.get("status") != "ok":
            raise RuntimeError(f"Unexpected AIMET service response: {payload!r}")

    def export(
        self,
        *,
        fp32_model_path: Path,
        calibration_dir: Path,
        output_dir: Path,
        config_path: Path,
        policy_path: Path,
    ) -> Mapping[str, Any]:
        """Request a model-independent AIMET export with portable paths.

        Args:
            fp32_model_path: Host path to a fixed-shape FP32 ONNX model.
            calibration_dir: Host path to serialized calibration inputs.
            output_dir: Host path receiving the AIMET package.
            config_path: Host path to the AIMET configuration.
            policy_path: Host path to the operator-scope policy.

        Returns:
            Decoded service response describing exported files.
        """
        return self._request(
            "/export",
            {
                "fp32_model_path": self._container_path(fp32_model_path),
                "calibration_dir": self._container_path(calibration_dir),
                "output_dir": self._container_path(output_dir),
                "config_path": self._container_path(config_path),
                "policy_path": self._container_path(policy_path),
            },
            timeout=7200,
        )

    def export_qdq(
        self,
        *,
        fp32_model_path: Path,
        encodings_path: Path,
        output_dir: Path,
        config_path: Path,
        policy_path: Path,
    ) -> Mapping[str, Any]:
        """Request benchmark-only QDQ export from exact AIMET encodings.

        Args:
            fp32_model_path: Fixed-shape FP32 model used to rebuild the simulation.
            encodings_path: Exact AIMET encoding file from the canonical package.
            output_dir: Host directory receiving the benchmark QDQ model.
            config_path: AIMET configuration used by canonical quantization.
            policy_path: Exact operator policy used by canonical quantization.

        Returns:
            Decoded service response describing exported QDQ files.
        """
        return self._request(
            "/export-qdq",
            {
                "fp32_model_path": self._container_path(fp32_model_path),
                "encodings_path": self._container_path(encodings_path),
                "output_dir": self._container_path(output_dir),
                "config_path": self._container_path(config_path),
                "policy_path": self._container_path(policy_path),
            },
            timeout=7200,
        )

    def _container_path(self, path: Path) -> str:
        """Translate a repository-contained host path into the container mount.

        Args:
            path: Host path inside the repository root.

        Returns:
            POSIX container path below the configured workspace root.

        Raises:
            ValueError: If the host path is outside the repository root.
        """
        relative = path.resolve().relative_to(self.repo_root)
        return f"{self.workspace_root}/{relative.as_posix()}"

    def _request(
        self,
        endpoint: str,
        payload: Mapping[str, Any] | None = None,
        timeout: int = 30,
    ) -> Mapping[str, Any]:
        """Send one JSON request to the local AIMET service.

        Args:
            endpoint: Service path beginning with `/`.
            payload: Optional JSON-compatible request body.
            timeout: Request timeout in seconds.

        Returns:
            Decoded JSON response mapping.

        Raises:
            RuntimeError: If the service cannot be reached.
        """
        body = json.dumps(dict(payload)).encode() if payload is not None else None
        headers = {"Content-Type": "application/json"} if body else {}
        try:
            with request.urlopen(
                request.Request(self.url + endpoint, data=body, headers=headers),
                timeout=timeout,
            ) as response:
                return json.loads(response.read().decode())
        except error.URLError as exc:
            raise RuntimeError(
                f"Could not reach AIMET service at {self.url}: {exc.reason}"
            ) from exc


def export_with_aimet(
    *,
    fp32_model_path: str | Path,
    calibration_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    policy: Mapping[str, Any],
) -> dict[str, Path]:
    """Export any fixed-shape ONNX model through AIMET QuantizationSimModel.

    Args:
        fp32_model_path: Fixed-shape FP32 ONNX model.
        calibration_dir: Serialized calibration input directory.
        output_dir: Directory receiving the AIMET package.
        config_path: MatMul-only AIMET configuration.
        policy: Operator allow/disable scope and coverage evidence.

    Returns:
        Exported model, encodings, and optional external-data paths.

    Raises:
        ValueError: If calibration is empty or policy operations are missing.
    """
    import onnx
    from aimet_common.defs import QuantScheme
    from aimet_onnx.quantsim import QuantizationSimModel

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    batches = load_aimet_calibration_inputs(calibration_dir)
    if not batches:
        raise ValueError("AIMET calibration inputs must not be empty")
    sim = QuantizationSimModel(
        onnx.load(Path(fp32_model_path).resolve().as_posix()),
        quant_scheme=QuantScheme.min_max,
        default_param_bw=8,
        default_activation_bw=16,
        config_file=Path(config_path).resolve().as_posix(),
    )
    if policy.get("quantizer_selection") == "operator-name-allowlist":
        selection = _enable_only_allowlisted_ops(
            sim,
            policy.get("quantize_op_names", ()),
            symmetric_encodings=bool(policy.get("symmetric_activation_encodings", False)),
        )
    else:
        selection = _disable_ops(sim, policy.get("disable_op_names", ()))
    if selection["missing_op_names"]:
        raise ValueError(
            f"AIMET policy nodes were not found: {selection['missing_op_names']!r}"
        )
    sim.compute_encodings(batches)
    sim.export(destination.as_posix(), "model")
    outputs = {
        "model": destination / "model.onnx",
        "encodings": destination / "model.encodings",
    }
    external_data = destination / "model.onnx.data"
    if external_data.is_file():
        outputs["external_data"] = external_data
    return outputs


def export_qdq_with_aimet(
    *,
    fp32_model_path: str | Path,
    encodings_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    policy: Mapping[str, Any],
) -> dict[str, Path]:
    """Export benchmark QDQ from the canonical fixed model and encodings.

    Args:
        fp32_model_path: Fixed-shape FP32 ONNX model used for quantization.
        encodings_path: Canonical AIMET encodings restored strictly.
        output_dir: Directory receiving the QDQ model and external data.
        config_path: Canonical AIMET quantization configuration.
        policy: Exact operation selection policy used during calibration.

    Returns:
        QDQ model and optional external-data paths.

    Raises:
        ValueError: If policy operations are missing from the rebuilt simulation.
    """
    import onnx
    from aimet_common.defs import QuantScheme
    from aimet_onnx.quantsim import QuantizationSimModel, load_encodings_to_sim

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    sim = QuantizationSimModel(
        onnx.load(Path(fp32_model_path).resolve().as_posix()),
        quant_scheme=QuantScheme.min_max,
        default_param_bw=8,
        default_activation_bw=16,
        config_file=Path(config_path).resolve().as_posix(),
    )
    if policy.get("quantizer_selection") == "operator-name-allowlist":
        selection = _enable_only_allowlisted_ops(
            sim,
            policy.get("quantize_op_names", ()),
            symmetric_encodings=bool(policy.get("symmetric_activation_encodings", False)),
        )
    else:
        selection = _disable_ops(sim, policy.get("disable_op_names", ()))
    if selection["missing_op_names"]:
        raise ValueError(
            f"AIMET policy nodes were not found: {selection['missing_op_names']!r}"
        )
    load_encodings_to_sim(
        sim,
        Path(encodings_path).resolve().as_posix(),
        strict=True,
        disable_missing_quantizers=True,
    )
    qdq_model = sim.to_onnx_qdq(
        prequantize_constants=True,
        force_activation_as="signed",
    )
    model_path = destination / "model.qdq.onnx"
    onnx.save_model(qdq_model, model_path.as_posix())
    outputs = {"model": model_path}
    external_data = destination / "model.qdq.onnx.data"
    if external_data.is_file():
        outputs["external_data"] = external_data
    return outputs


def _enable_only_allowlisted_ops(
    sim,
    op_names: Sequence[str],
    *,
    symmetric_encodings: bool = False,
) -> dict[str, object]:
    """Enable tensor quantizers associated only with allowlisted operations.

    Args:
        sim: AIMET quantization simulation model.
        op_names: Connected-graph operation names selected for quantization.
        symmetric_encodings: Whether every enabled tensor quantizer uses a
            symmetric signed range, including quantizers shared with producer ops.

    Returns:
        Enabled unique quantizer count and operation names not found.
    """
    for quantizer in sim.qc_quantize_op_dict.values():
        if quantizer is not None:
            quantizer.enabled = False
    all_ops = sim.connected_graph.get_all_ops()
    missing: list[str] = []
    enabled_quantizers: set[object] = set()
    for name in op_names:
        op = all_ops.get(str(name))
        if op is None:
            missing.append(str(name))
            continue
        connected_tensor_names = {
            product.name
            for product in (*op.inputs, *op.outputs)
            if getattr(product, "name", None)
        }
        parameter_names = {str(parameter_name) for parameter_name in op.parameters}
        activation_names = connected_tensor_names - parameter_names
        tensor_names = activation_names | parameter_names
        for tensor_name in tensor_names:
            quantizer = sim.qc_quantize_op_dict.get(tensor_name)
            if quantizer is not None:
                quantizer.enabled = True
                if symmetric_encodings and tensor_name in activation_names:
                    quantizer.use_symmetric_encodings = True
                enabled_quantizers.add(quantizer)
    return {
        "enabled_quantizer_count": len(enabled_quantizers),
        "missing_op_names": missing,
    }


def _disable_ops(sim, op_names: Sequence[str]) -> dict[str, object]:
    """Disable every enabled quantizer associated with selected operations.

    Args:
        sim: AIMET quantization simulation model.
        op_names: Connected-graph operations that must remain unquantized.

    Returns:
        Disabled quantizer count and operation names not found.
    """
    all_ops = sim.connected_graph.get_all_ops()
    missing: list[str] = []
    disabled = 0
    for name in op_names:
        op = all_ops.get(str(name))
        if op is None:
            missing.append(str(name))
            continue
        inputs, outputs, parameters = sim.get_op_quantizers(op)
        for quantizer in (*inputs, *outputs, *parameters.values()):
            if quantizer is not None and bool(getattr(quantizer, "enabled", False)):
                quantizer.enabled = False
                disabled += 1
    return {"disabled_quantizer_count": disabled, "missing_op_names": missing}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        """Suppress default local-service request logging.

        Args:
            format: Base-handler log format string.
            args: Values associated with the format string.

        Returns:
            None.
        """
        del format, args

    def do_GET(self) -> None:  # noqa: N802
        """Serve the health endpoint and reject other GET routes.

        Returns:
            None.
        """
        if self.path == "/healthz":
            self._send(200, {"status": "ok"})
        else:
            self._send(404, {"error": "not-found"})

    def do_POST(self) -> None:  # noqa: N802
        """Execute model-independent AIMET export requests.

        Returns:
            None.
        """
        if self.path not in {"/export", "/export-qdq"}:
            self._send(404, {"error": "not-found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode())
            policy = json.loads(Path(payload["policy_path"]).read_text(encoding="utf-8"))
            if self.path == "/export-qdq":
                outputs = export_qdq_with_aimet(
                    fp32_model_path=payload["fp32_model_path"],
                    encodings_path=payload["encodings_path"],
                    output_dir=payload["output_dir"],
                    config_path=payload["config_path"],
                    policy=policy,
                )
            else:
                outputs = export_with_aimet(
                    fp32_model_path=payload["fp32_model_path"],
                    calibration_dir=payload["calibration_dir"],
                    output_dir=payload["output_dir"],
                    config_path=payload["config_path"],
                    policy=policy,
                )
            self._send(200, {"outputs": {name: path.name for name, path in outputs.items()}})
        except Exception as exc:  # noqa: BLE001
            self._send(500, {"error": str(exc), "type": type(exc).__name__})

    def _send(self, status: int, payload: Mapping[str, Any]) -> None:
        """Write one JSON response with explicit content length.

        Args:
            status: HTTP response status code.
            payload: JSON-compatible response body.

        Returns:
            None.
        """
        body = json.dumps(dict(payload)).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the threaded local AIMET service until interrupted.

    Args:
        argv: Optional explicit service arguments.

    Returns:
        Zero after clean server shutdown.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
