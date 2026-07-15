from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from model_pipeline.core import sha256_path


@dataclass(frozen=True)
class CompilationEvidence:
    artifact_id: str
    component: str
    input_checksum: str
    output_checksum: str
    job_id: str
    target: str
    truncate_64bit_io: bool
    primary_file: str = "model.onnx"

    @classmethod
    def from_dict(cls, payload: dict) -> "CompilationEvidence":
        """Restore compile evidence from a serialized record.

        Args:
            payload: Mapping containing every evidence dataclass field.

        Returns:
            Immutable compilation evidence.
        """
        return cls(**payload)


class EvidenceStore:
    def __init__(self, root: str | Path):
        """Initialize record and content-addressed blob locations.

        Args:
            root: Root directory for persistent compile evidence.

        Returns:
            None.
        """
        self.root = Path(root)
        self.records = self.root / "records"
        self.blobs = self.root / "blobs"

    def resolve(self, input_checksum: str) -> CompilationEvidence | None:
        """Resolve reusable evidence only when its stored package still matches.

        Args:
            input_checksum: Checksum of the current compile input package.

        Returns:
            Verified evidence, or `None` on record/blob/checksum mismatch.
        """
        path = self.records / f"{input_checksum}.json"
        if not path.is_file():
            return None
        evidence = CompilationEvidence.from_dict(json.loads(path.read_text(encoding="utf-8")))
        blob = self.blob_path(evidence.output_checksum)
        if not blob.is_dir() or sha256_path(blob) != evidence.output_checksum:
            return None
        return evidence

    def save(self, evidence: CompilationEvidence, package_dir: str | Path) -> None:
        """Persist validated evidence and a content-addressed package copy.

        Args:
            evidence: Hosted compile evidence to serialize.
            package_dir: Complete normalized compiled package.

        Returns:
            None.

        Raises:
            ValueError: If package bytes do not match the evidence checksum.
        """
        source = Path(package_dir)
        if sha256_path(source) != evidence.output_checksum:
            raise ValueError("Compiled output does not match evidence checksum")
        self.records.mkdir(parents=True, exist_ok=True)
        self.blobs.mkdir(parents=True, exist_ok=True)
        blob = self.blob_path(evidence.output_checksum)
        if blob.exists():
            shutil.rmtree(blob)
        shutil.copytree(source, blob)
        (self.records / f"{evidence.input_checksum}.json").write_text(
            json.dumps(asdict(evidence), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def materialize(self, evidence: CompilationEvidence, destination_dir: str | Path) -> Path:
        """Copy a verified content-addressed package into a stage directory.

        Args:
            evidence: Reusable evidence identifying the stored package.
            destination_dir: Destination directory to replace.

        Returns:
            Path to the package's primary ONNX file.
        """
        output = Path(destination_dir)
        if output.exists():
            shutil.rmtree(output)
        shutil.copytree(self.blob_path(evidence.output_checksum), output)
        return output / evidence.primary_file

    def blob_path(self, checksum: str) -> Path:
        """Resolve a compiled-package checksum to its blob directory.

        Args:
            checksum: Package SHA-256 identity.

        Returns:
            Content-addressed blob directory path.
        """
        return self.blobs / checksum
