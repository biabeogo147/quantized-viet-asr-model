from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlparse


def build_handler_class(
    *,
    export_callback: Callable[[dict[str, Any]], Mapping[str, Any]],
    version_payload: Mapping[str, Any] | None = None,
):
    resolved_version_payload = dict(version_payload or {"service": "aimet-export-service"})

    class AimetServiceHandler(BaseHTTPRequestHandler):
        server_version = "AimetExportService/1.0"

        def log_message(self, format: str, *args) -> None:  # noqa: A003 - stdlib signature
            return

        def _send_json(self, status_code: int, payload: Mapping[str, Any]) -> None:
            body = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
            self.send_response(int(status_code))
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - stdlib hook
            path = urlparse(self.path).path
            if path == "/healthz":
                self._send_json(200, {"status": "ok"})
                return
            if path == "/version":
                self._send_json(200, resolved_version_payload)
                return
            self._send_json(404, {"error": "not_found", "path": path})

        def do_POST(self) -> None:  # noqa: N802 - stdlib hook
            path = urlparse(self.path).path
            if path != "/export":
                self._send_json(404, {"error": "not_found", "path": path})
                return

            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
            try:
                payload = json.loads(raw_body.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("export payload must be a JSON object")
                report = dict(export_callback(payload))
            except Exception as exc:  # noqa: BLE001
                self._send_json(
                    500,
                    {
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                )
                return

            self._send_json(200, report)

    return AimetServiceHandler


def _default_export_callback(payload: dict[str, Any]) -> dict[str, Any]:
    from quantize.aimet import export_aimet_package

    return export_aimet_package(
        fp32_onnx_path=payload["fp32_onnx_path"],
        calibration_dir=payload["calibration_dir"],
        package_dir=payload["package_dir"],
        qdq_reference_model_path=payload["qdq_reference_model_path"],
        model_prefix=str(payload.get("model_prefix", "model.option1")),
        param_type=str(payload.get("param_type", "int8")),
        activation_type=str(payload.get("activation_type", "int8")),
        quant_scheme=str(payload.get("quant_scheme", "min_max")),
        config_file=str(payload.get("config_file", "default")),
        policy_manifest_path=payload.get("policy_manifest_path"),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve AIMET ONNX export operations over HTTP.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer(
        (str(args.host), int(args.port)),
        build_handler_class(export_callback=_default_export_callback),
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
