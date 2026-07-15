from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Mapping
import zipfile

from model_pipeline.core import ArtifactSpec, sha256_path
from model_pipeline.integrations.aihub.client import AiHubClient
from model_pipeline.integrations.aihub.evidence import CompilationEvidence, EvidenceStore


@dataclass(frozen=True)
class CompileRequest:
    artifact: ArtifactSpec
    component: str
    source_path: Path
    input_shapes: Mapping[str, list[int]]
    truncate_64bit_io: bool
    input_dtypes: Mapping[str, str] | None = None

    @property
    def source_checksum(self) -> str:
        """Hash the complete compile input file or package.

        Returns:
            Checksum used to resolve reusable hosted evidence.
        """
        return sha256_path(self.source_path)


@dataclass(frozen=True)
class CompileResult:
    output_path: Path
    output_checksum: str
    evidence: CompilationEvidence
    reused: bool
    support_files: tuple[Path, ...] = ()


def compile_or_reuse(
    request: CompileRequest,
    *,
    client: AiHubClient,
    evidence_store: EvidenceStore,
    output_dir: str | Path,
) -> CompileResult:
    """Reuse checksum-matched evidence or execute a new AI Hub compile job.

    Args:
        request: Artifact, component, source, and fixed I/O compile contract.
        client: AI Hub implementation used for hosted operations.
        evidence_store: Local checksum-keyed evidence and blob store.
        output_dir: Directory receiving normalized compile packages.

    Returns:
        Primary output, support files, checksum, evidence, and reuse status.

    Raises:
        RuntimeError: If the hosted compile job does not succeed.
    """
    package_dir = Path(output_dir) / request.component
    destination = package_dir / f"{request.component}.onnx"
    cached = evidence_store.resolve(request.source_checksum)
    if cached is not None and cached.artifact_id == request.artifact.artifact_id and cached.component == request.component:
        primary = evidence_store.materialize(cached, package_dir)
        support = _support_files(package_dir, primary)
        return CompileResult(primary, cached.output_checksum, cached, True, support)

    job_id = client.submit_compile(
        source_path=request.source_path,
        input_shapes=request.input_shapes,
        options={
            "target_runtime": "qnn",
            "target_device": "htp",
            "truncate_64bit_io": request.truncate_64bit_io,
            "input_dtypes": dict(request.input_dtypes or {}),
        },
    )
    status = dict(client.wait(job_id))
    if status.get("status") != "success":
        raise RuntimeError(f"AI Hub compile failed: {status!r}")
    downloaded = client.download(job_id, destination)
    output = _normalize_download(downloaded, destination, package_dir)
    support = _support_files(package_dir, output)
    checksum = sha256_path(package_dir)
    evidence = CompilationEvidence(
        artifact_id=request.artifact.artifact_id,
        component=request.component,
        input_checksum=request.source_checksum,
        output_checksum=checksum,
        job_id=job_id,
        target="qnn-htp",
        truncate_64bit_io=request.truncate_64bit_io,
        primary_file=output.relative_to(package_dir).as_posix(),
    )
    evidence_store.save(evidence, package_dir)
    return CompileResult(output, checksum, evidence, False, support)


def _normalize_download(downloaded: Path, destination: Path, package_dir: Path) -> Path:
    """Normalize raw file or ZIP downloads to one canonical ONNX package.

    Args:
        downloaded: File returned by the provider client.
        destination: Canonical primary ONNX destination.
        package_dir: Directory that owns the complete compiled package.

    Returns:
        Canonical primary ONNX path.

    Raises:
        ValueError: If an archive has duplicate names or not exactly one ONNX file.
        FileNotFoundError: If the provider output is missing.
    """
    package_dir.mkdir(parents=True, exist_ok=True)
    resolved = Path(downloaded)
    if resolved.is_file() and zipfile.is_zipfile(resolved):
        extracted: list[Path] = []
        with zipfile.ZipFile(resolved) as archive:
            for member in sorted(archive.infolist(), key=lambda row: row.filename):
                if member.is_dir():
                    continue
                name = Path(member.filename).name
                if not name:
                    continue
                output = package_dir / name
                if output in extracted:
                    raise ValueError(f"Duplicate file name in AI Hub archive: {name}")
                with archive.open(member) as source, output.open("wb") as target:
                    shutil.copyfileobj(source, target)
                extracted.append(output)
        resolved.unlink()
        onnx_files = sorted(path for path in extracted if path.suffix.lower() == ".onnx")
        if len(onnx_files) != 1:
            raise ValueError(f"Expected exactly one ONNX file in AI Hub archive, found {len(onnx_files)}")
        resolved = onnx_files[0]
    if not resolved.is_file():
        raise FileNotFoundError(f"AI Hub compiled output is missing: {resolved}")
    if resolved.resolve() != destination.resolve():
        if destination.exists():
            destination.unlink()
        shutil.move(resolved, destination)
    return destination


def _support_files(package_dir: Path, primary: Path) -> tuple[Path, ...]:
    """List every compiled package file except the primary ONNX model.

    Args:
        package_dir: Root of the normalized compiled package.
        primary: Primary ONNX model path to exclude.

    Returns:
        Sorted support-file paths, including external tensor data.
    """
    return tuple(
        path
        for path in sorted(candidate for candidate in package_dir.rglob("*") if candidate.is_file())
        if path != primary
    )
