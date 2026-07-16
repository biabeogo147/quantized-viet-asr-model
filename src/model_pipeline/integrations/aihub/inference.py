from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from model_pipeline.core import ArtifactSpec
from model_pipeline.integrations.aihub.client import AiHubClient


MAX_HOSTED_INPUTS_PER_MODEL = 5


@dataclass(frozen=True)
class HostedInferenceEvidence:
    """Identify one hosted inference result by its exact input bytes."""

    artifact_id: str
    compile_job_id: str
    input_checksum: str
    inference_job_id: str
    output_checksum: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HostedInferenceEvidence":
        """Restore hosted inference evidence from JSON-compatible fields.

        Args:
            payload: Serialized evidence fields.

        Returns:
            Immutable hosted inference evidence.
        """
        return cls(**dict(payload))


@dataclass(frozen=True)
class HostedInferenceResult:
    """Pair hosted output data with its persistent identity evidence."""

    outputs: Mapping[str, Any]
    evidence: HostedInferenceEvidence


class HostedInferenceStore:
    """Persist hosted inference records under content-derived input identities."""

    def __init__(self, root: str | Path):
        """Initialize the hosted inference evidence directory.

        Args:
            root: Directory receiving checksum-keyed JSON records.

        Returns:
            None.
        """
        self.root = Path(root)

    def resolve(self, input_checksum: str) -> HostedInferenceEvidence | None:
        """Load one hosted record by exact input checksum.

        Args:
            input_checksum: Deterministic checksum of named input tensors.

        Returns:
            Stored evidence, or `None` when no record exists.
        """
        path = self.root / f"{input_checksum}.json"
        if not path.is_file():
            return None
        return HostedInferenceEvidence.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save(self, evidence: HostedInferenceEvidence) -> Path:
        """Write one deterministic checksum-keyed hosted inference record.

        Args:
            evidence: Record to serialize.

        Returns:
            Path to the written evidence file.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{evidence.input_checksum}.json"
        path.write_text(json.dumps(asdict(evidence), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path


def checksum_named_values(values: Mapping[str, Any]) -> str:
    """Hash named tensors with stable name, dtype, shape, and byte semantics.

    Args:
        values: Named tensor-like values accepted by NumPy.

    Returns:
        Lowercase SHA-256 checksum of the complete input mapping.
    """
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        array = np.ascontiguousarray(np.asarray(value))
        _update_field(digest, name.encode("utf-8"))
        _update_field(digest, array.dtype.str.encode("ascii"))
        _update_field(digest, json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        _update_field(digest, array.tobytes(order="C"))
    return digest.hexdigest()


def run_hosted_inputs(
    *,
    artifact: ArtifactSpec,
    compile_job_id: str,
    inputs: Sequence[Mapping[str, Any]],
    client: AiHubClient,
    evidence_store: HostedInferenceStore,
) -> tuple[HostedInferenceResult, ...]:
    """Run no more than five hosted inputs and persist content-keyed evidence.

    Args:
        artifact: Compiled artifact identity being validated.
        compile_job_id: Compile job whose target model is executed.
        inputs: Independent named input mappings, one mapping per validation case.
        client: Hosted inference provider implementation.
        evidence_store: Destination for checksum-keyed result records.

    Returns:
        Hosted outputs and evidence in input order.

    Raises:
        ValueError: If the request is empty or exceeds the five-input quota.
    """
    if not inputs:
        raise ValueError("Hosted inference requires at least one input")
    if len(inputs) > MAX_HOSTED_INPUTS_PER_MODEL:
        raise ValueError(f"Hosted inference accepts at most {MAX_HOSTED_INPUTS_PER_MODEL} inputs per model")

    results: list[HostedInferenceResult] = []
    for named_input in inputs:
        input_checksum = checksum_named_values(named_input)
        response = dict(
            client.live_run(
                job_id=compile_job_id,
                inputs={name: [value] for name, value in named_input.items()},
            )
        )
        outputs = dict(response.get("outputs") or {})
        output_checksum = _checksum_output_values(outputs)
        evidence = HostedInferenceEvidence(
            artifact_id=artifact.artifact_id,
            compile_job_id=compile_job_id,
            input_checksum=input_checksum,
            inference_job_id=str(response["job_id"]),
            output_checksum=output_checksum,
        )
        evidence_store.save(evidence)
        results.append(HostedInferenceResult(outputs, evidence))
    return tuple(results)


def _checksum_output_values(values: Mapping[str, Any]) -> str:
    """Hash provider output mappings including batched tensor lists.

    Args:
        values: Named output tensor values returned by the provider.

    Returns:
        Lowercase SHA-256 checksum of the hosted outputs.
    """
    flattened: dict[str, Any] = {}
    for name, value in sorted(values.items()):
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                flattened[f"{name}[{index}]"] = item
        else:
            flattened[name] = value
    return checksum_named_values(flattened)


def _update_field(digest: Any, value: bytes) -> None:
    """Append one length-delimited value to a running digest.

    Args:
        digest: Hash object supporting `update`.
        value: Raw field bytes to append unambiguously.

    Returns:
        None.
    """
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)
