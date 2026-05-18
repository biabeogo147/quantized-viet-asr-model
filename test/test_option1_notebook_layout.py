from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_notebook(name: str) -> dict:
    return json.loads((REPO_ROOT / name).read_text(encoding="utf-8"))


def _cell_texts(notebook: dict, *, cell_type: str | None = None) -> list[str]:
    texts: list[str] = []
    for cell in notebook.get("cells", []):
        if cell_type is not None and cell.get("cell_type") != cell_type:
            continue
        texts.append("".join(cell.get("source", [])))
    return texts


def test_pilot_notebook_excludes_phase4_and_phase5_sections() -> None:
    notebook = _load_notebook("On_device_Ai_option1_pilots.ipynb")
    markdown_text = "\n".join(_cell_texts(notebook, cell_type="markdown"))

    assert "## Phase 4 Config" not in markdown_text
    assert "### Zipformer Phase 4 Benchmark And Gate" not in markdown_text
    assert "### VPCD Phase 4 Benchmark And Gate" not in markdown_text
    assert "## Phase 4 Recommendation Summary" not in markdown_text
    assert "## Phase 5 Config" not in markdown_text
    assert "### Package Zipformer Phase 5 Contract" not in markdown_text
    assert "### Package VPCD Phase 5 Contract" not in markdown_text
    assert "## Phase 5 Packaging Summary" not in markdown_text


def test_pilot_notebook_has_no_mandatory_pip_install_cell() -> None:
    notebook = _load_notebook("On_device_Ai_option1_pilots.ipynb")
    code_text = "\n".join(_cell_texts(notebook, cell_type="code"))

    assert '!pip install qai-hub "qai-hub[torch]"' not in code_text


def test_pilot_notebook_limits_vpcd_hybrid_decode_steps() -> None:
    notebook = _load_notebook("On_device_Ai_option1_pilots.ipynb")
    code_text = "\n".join(_cell_texts(notebook, cell_type="code"))

    assert "VPCD_HYBRID_MAX_STEPS = 5" in code_text
    assert "max_decode_steps=VPCD_HYBRID_MAX_STEPS" in code_text


def test_pilot_notebook_runs_vpcd_teacher_forced_debug_before_hybrid() -> None:
    notebook = _load_notebook("On_device_Ai_option1_pilots.ipynb")
    code_cells = _cell_texts(notebook, cell_type="code")
    markdown_cells = _cell_texts(notebook, cell_type="markdown")
    code_text = "\n".join(code_cells)

    assert "VPCD_TEACHER_FORCED_SAMPLE_INDEX = 0" in code_text
    assert "run_vpcd_teacher_forced_diagnostics(" in code_text
    assert "sample_index=VPCD_TEACHER_FORCED_SAMPLE_INDEX" in code_text
    assert "if vpcd_is_quantized_source:" not in code_text

    teacher_heading_index = next(index for index, text in enumerate(markdown_cells) if "### VPCD Teacher-Forced Diagnostics" in text)
    hybrid_heading_index = next(index for index, text in enumerate(markdown_cells) if "### VPCD Hybrid E2E Run" in text)
    assert teacher_heading_index < hybrid_heading_index


def test_pilot_notebook_runs_vpcd_quantized_local_teacher_forced_before_cloud_teacher_forced() -> None:
    notebook = _load_notebook("On_device_Ai_option1_pilots.ipynb")
    code_text = "\n".join(_cell_texts(notebook, cell_type="code"))
    markdown_cells = _cell_texts(notebook, cell_type="markdown")

    assert "run_vpcd_quantized_teacher_forced_diagnostics(" in code_text
    assert "vpcd quantized model path:" in code_text

    quantized_local_heading_index = next(
        index for index, text in enumerate(markdown_cells) if "### VPCD Quantized Local Teacher-Forced Diagnostics" in text
    )
    teacher_heading_index = next(index for index, text in enumerate(markdown_cells) if "### VPCD Teacher-Forced Diagnostics" in text)
    assert quantized_local_heading_index < teacher_heading_index


def test_pilot_notebook_supports_local_qdq_vpcd_source_strategy() -> None:
    notebook = _load_notebook("On_device_Ai_option1_pilots.ipynb")
    code_text = "\n".join(_cell_texts(notebook, cell_type="code"))

    assert "VPCD_SOURCE_STRATEGY = \"prefer_fp32_fixed\"" in code_text
    assert "local_qdq_compile_candidate" in code_text
    assert "Skipping AI Hub quantize for local QDQ lane" in code_text


def test_phase4_notebook_exists_with_phase4_sections() -> None:
    notebook = _load_notebook("On_device_Ai_option1_phase4_gate.ipynb")
    markdown_text = "\n".join(_cell_texts(notebook, cell_type="markdown"))

    assert "## Phase 4 Config" in markdown_text
    assert "### Zipformer Phase 4 Benchmark And Gate" in markdown_text
    assert "### VPCD Phase 4 Benchmark And Gate" in markdown_text
    assert "## Phase 4 Recommendation Summary" in markdown_text


def test_phase5_notebook_exists_with_phase5_sections() -> None:
    notebook = _load_notebook("On_device_Ai_option1_phase5_contract.ipynb")
    markdown_text = "\n".join(_cell_texts(notebook, cell_type="markdown"))
    code_text = "\n".join(_cell_texts(notebook, cell_type="code"))

    assert "## Phase 5 Config" in markdown_text
    assert "### Package Zipformer Phase 5 Contract" in markdown_text
    assert "### Package VPCD Phase 5 Contract" in markdown_text
    assert "## Phase 5 Packaging Summary" in markdown_text
    assert "PHASE5_INCLUDE_ZIPFORMER" in code_text
    assert "PHASE5_INCLUDE_VPCD" in code_text
