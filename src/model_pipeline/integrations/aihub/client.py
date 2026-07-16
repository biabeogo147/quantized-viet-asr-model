from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol


class AiHubClient(Protocol):
    def submit_compile(
        self,
        *,
        source_path: Path,
        input_shapes: Mapping[str, list[int]],
        options: Mapping[str, Any],
    ) -> str:
        """Submit one component or model package for hosted compilation.

        Args:
            source_path: ONNX file or package directory to compile.
            input_shapes: Named fixed input shapes.
            options: Target runtime, dtype, and compiler options.

        Returns:
            Provider-specific compile job identifier.
        """
        ...

    def wait(self, job_id: str) -> Mapping[str, Any]:
        """Wait for a compile job and normalize its terminal status.

        Args:
            job_id: Compile job identifier returned by submission.

        Returns:
            Terminal status and provider evidence.
        """
        ...

    def download(self, job_id: str, output_path: Path) -> Path:
        """Download a completed compile result.

        Args:
            job_id: Successful compile job identifier.
            output_path: Preferred local destination.

        Returns:
            Path to the downloaded file or archive.
        """
        ...

    def live_run(self, *, job_id: str, inputs: Mapping[str, list[Any]]) -> Mapping[str, Any]:
        """Run one bounded hosted validation input against a compiled target.

        Args:
            job_id: Compile job whose target model should be executed.
            inputs: Named single-item input batches accepted by AI Hub.

        Returns:
            Provider inference job ID and named output tensors.
        """
        ...


@dataclass
class FakeAiHubClient:
    compiled_bytes: bytes = b"fake-ep-context"
    submit_count: int = 0
    live_run_count: int = 0

    def submit_compile(self, *, source_path: Path, input_shapes, options) -> str:
        """Record a fake compile submission for deterministic integration tests.

        Args:
            source_path: Ignored fake source path.
            input_shapes: Ignored fake input shapes.
            options: Ignored fake compile options.

        Returns:
            A monotonically numbered fake job ID.
        """
        del source_path, input_shapes, options
        self.submit_count += 1
        return f"fake-job-{self.submit_count}"

    def wait(self, job_id: str) -> Mapping[str, Any]:
        """Return a successful fake HTP compile status.

        Args:
            job_id: Fake job identifier.

        Returns:
            Normalized successful status evidence.
        """
        return {"job_id": job_id, "status": "success", "target": "qnn-htp"}

    def download(self, job_id: str, output_path: Path) -> Path:
        """Materialize deterministic fake EPContext bytes.

        Args:
            job_id: Fake job identifier accepted for protocol consistency.
            output_path: Destination file to create.

        Returns:
            The created output path.
        """
        del job_id
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(self.compiled_bytes)
        return output_path

    def live_run(self, *, job_id: str, inputs: Mapping[str, list[Any]]) -> Mapping[str, Any]:
        """Return deterministic fake outputs for one hosted validation input.

        Args:
            job_id: Fake compile job identifier.
            inputs: Named single-item input batches echoed as output data.

        Returns:
            Monotonic fake inference job ID and deterministic output tensors.
        """
        del job_id
        self.live_run_count += 1
        first_value = next(iter(inputs.values()))[0]
        return {
            "job_id": f"fake-inference-{self.live_run_count}",
            "outputs": {"output": [first_value]},
        }


class QualcommAiHubClient:
    """Thin, stateful wrapper around the official Qualcomm AI Hub SDK."""

    def __init__(self, *, device_name: str, api_token: str | None = None, qairt_version: str | None = None):
        """Initialize official AI Hub device, authentication, and compiler settings.

        Args:
            device_name: Exact Qualcomm AI Hub device name.
            api_token: Optional API token injected into the SDK environment.
            qairt_version: Optional QAIRT version forwarded to compile jobs.

        Returns:
            None.

        Raises:
            ValueError: If the device name is blank.
        """
        if not device_name.strip():
            raise ValueError("AI Hub device name is required")
        self.device_name = device_name.strip()
        self.api_token = api_token
        self.qairt_version = qairt_version
        self._jobs: dict[str, Any] = {}
        self._targets: dict[str, Any] = {}

    def authenticate(self) -> None:
        """Configure the SDK token and prove account access by listing devices.

        Returns:
            None.

        Raises:
            RuntimeError: If no API token is configured.
        """
        import os
        import qai_hub as hub

        if self.api_token:
            os.environ["QAI_HUB_API_TOKEN"] = self.api_token
        if not os.environ.get("QAI_HUB_API_TOKEN"):
            raise RuntimeError("QAI_HUB_API_TOKEN is not configured")
        hub.get_devices()

    def submit_compile(self, *, source_path: Path, input_shapes, options) -> str:
        """Submit a precompiled-QNN ONNX job through the official SDK.

        Args:
            source_path: Model file or package directory uploaded to AI Hub.
            input_shapes: Fixed input shapes by tensor name.
            options: Input dtypes and compiler flags such as 64-bit truncation.

        Returns:
            Qualcomm AI Hub compile job identifier.
        """
        import qai_hub as hub

        dtypes = dict(options.get("input_dtypes") or {})
        input_specs = {
            name: (tuple(shape), dtypes.get(name, "float32"))
            for name, shape in input_shapes.items()
        }
        compile_options = ["--target_runtime precompiled_qnn_onnx"]
        if options.get("truncate_64bit_io"):
            compile_options.append("--truncate_64bit_io")
        if self.qairt_version:
            compile_options.append(f"--qairt_version {self.qairt_version}")
        job = hub.submit_compile_job(
            model=source_path,
            device=hub.Device(self.device_name),
            input_specs=input_specs,
            options=" ".join(compile_options),
            name=f"model-pipeline-{source_path.name}",
        )
        job_id = str(job.job_id)
        self._jobs[job_id] = job
        return job_id

    def wait(self, job_id: str) -> Mapping[str, Any]:
        """Wait for a compile target and normalize provider status evidence.

        Args:
            job_id: Previously submitted compile job identifier.

        Returns:
            Success/failure status, message, and target model ID.
        """
        job = self._jobs.get(job_id)
        if job is None:
            import qai_hub as hub

            job = hub.get_job(job_id)
            self._jobs[job_id] = job
        target = job.get_target_model()
        status = job.get_status()
        if target is not None:
            self._targets[job_id] = target
        status_text = str(getattr(status, "code", status)).lower()
        success = target is not None and "fail" not in status_text
        return {
            "job_id": job_id,
            "status": "success" if success else "failed",
            "message": getattr(status, "message", None),
            "target_model_id": getattr(target, "model_id", None),
        }

    def download(self, job_id: str, output_path: Path) -> Path:
        """Download and verify a completed target-model artifact.

        Args:
            job_id: Successful compile job identifier.
            output_path: Preferred local download path.

        Returns:
            Resolved path to the downloaded artifact.

        Raises:
            FileNotFoundError: If the SDK reports success without creating a file.
        """
        target = self._targets[job_id]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            result = target.download(filename=output_path.as_posix())
        except TypeError:
            result = target.download(output_path.as_posix())
        resolved = Path(result).resolve() if isinstance(result, (str, Path)) else output_path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"AI Hub did not download the compiled model to {resolved}")
        return resolved

    def live_run(self, *, job_id: str, inputs: Mapping[str, list[Any]]) -> Mapping[str, Any]:
        """Run a strict HTP inference job against a compiled target model.

        Args:
            job_id: Compile job whose target model should be executed.
            inputs: Named inference inputs accepted by the hosted model.

        Returns:
            Inference job ID and downloaded output data.
        """
        import qai_hub as hub

        target = self._targets.get(job_id)
        if target is None:
            compile_job = hub.get_job(job_id)
            target = compile_job.get_target_model()
            if target is None:
                raise RuntimeError(
                    f"AI Hub compile job {job_id!r} has no executable target model"
                )
            self._jobs[job_id] = compile_job
            self._targets[job_id] = target
        inference = hub.submit_inference_job(
            model=target,
            device=hub.Device(self.device_name),
            inputs=dict(inputs),
            options="--compute_unit npu",
            name=f"model-pipeline-live-{job_id}",
        )
        return {
            "job_id": str(inference.job_id),
            "outputs": inference.download_output_data(),
        }
