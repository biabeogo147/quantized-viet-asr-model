from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ComponentQuantizationReport:
    component: str
    input_model: str
    output_model: str
    size_mb: float
    calibration_records: int


@dataclass(frozen=True)
class QuantizationReport:
    project: str
    preset: str
    output_root: str
    bundle_output_dir: str
    sample_count: int
    trace_records: int
    components: list[ComponentQuantizationReport] = field(default_factory=list)
    evaluation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload['components'] = [asdict(component) for component in self.components]
        return payload

    def write_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        return output
