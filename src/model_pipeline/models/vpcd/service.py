from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib import error, request

from model_pipeline.models.vpcd.quantization import quantize_with_aimet


DEFAULT_URL = "http://127.0.0.1:18080"


class AimetServiceClient:
    def __init__(self, *, repo_root: str | Path, url: str = DEFAULT_URL, workspace_root: str = "/workspace"):
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
            RuntimeError: If the service is unreachable or returns an unexpected status.
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
        """Request a canonical AIMET export using container-relative paths.

        Args:
            fp32_model_path: Host path to the fixed-shape FP32 model.
            calibration_dir: Host path to serialized calibration batches.
            output_dir: Host path receiving the exported package.
            config_path: Host path to the AIMET configuration.
            policy_path: Host path to the encoder-only quantization policy.

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

    def _container_path(self, path: Path) -> str:
        """Translate a repository-contained host path to its container mount path.

        Args:
            path: Host path inside the repository root.

        Returns:
            POSIX container path below the configured workspace root.

        Raises:
            ValueError: If the host path is outside the repository root.
        """
        relative = path.resolve().relative_to(self.repo_root)
        return f"{self.workspace_root}/{relative.as_posix()}"

    def _request(self, endpoint: str, payload: Mapping[str, Any] | None = None, timeout: int = 30):
        """Send a JSON request to the local AIMET service.

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
                request.Request(self.url + endpoint, data=body, headers=headers), timeout=timeout
            ) as response:
                return json.loads(response.read().decode())
        except error.URLError as exc:
            raise RuntimeError(f"Could not reach AIMET service at {self.url}: {exc.reason}") from exc


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        """Suppress default HTTP request logging from the local service.

        Args:
            format: Base-handler log format string.
            args: Values associated with the format string.

        Returns:
            None.
        """
        del format, args

    def do_GET(self) -> None:  # noqa: N802
        """Serve the health endpoint and reject all other GET routes.

        Returns:
            None.
        """
        if self.path == "/healthz":
            self._send(200, {"status": "ok"})
        else:
            self._send(404, {"error": "not-found"})

    def do_POST(self) -> None:  # noqa: N802
        """Execute AIMET export requests and return structured failures.

        Returns:
            None.
        """
        if self.path != "/export":
            self._send(404, {"error": "not-found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode())
            policy = json.loads(Path(payload["policy_path"]).read_text(encoding="utf-8"))
            outputs = quantize_with_aimet(
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
        """Write one JSON HTTP response with an explicit content length.

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
    """Run the threaded local AIMET HTTP service until interrupted.

    Args:
        argv: Optional explicit service arguments.

    Returns:
        Zero after the server shuts down cleanly.
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
