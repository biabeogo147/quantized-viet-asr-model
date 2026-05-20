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


def test_aihub_notebook_excludes_phase4_and_phase5_sections() -> None:
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


def test_aihub_notebook_has_no_mandatory_pip_install_cell() -> None:
    notebook = _load_notebook("On_device_Ai_option1_pilots.ipynb")
    code_text = "\n".join(_cell_texts(notebook, cell_type="code"))

    assert '!pip install qai-hub "qai-hub[torch]"' not in code_text


def test_aihub_notebook_imports_aihub_package_instead_of_tools_modules() -> None:
    notebook = _load_notebook("On_device_Ai_option1_pilots.ipynb")
    code_text = "\n".join(_cell_texts(notebook, cell_type="code"))

    assert "from aihub.session import" in code_text
    assert "from aihub.evaluation import" in code_text
    assert "from tools.aihub_" not in code_text
    assert "build_runtime_config(" in code_text
    assert "prepare_vpcd_source_model(" in code_text


def test_aihub_notebook_limits_vpcd_hybrid_decode_steps() -> None:
    notebook = _load_notebook("On_device_Ai_option1_pilots.ipynb")
    code_text = "\n".join(_cell_texts(notebook, cell_type="code"))

    assert "VPCD_HYBRID_MAX_STEPS = 5" in code_text
    assert "max_decode_steps=VPCD_HYBRID_MAX_STEPS" in code_text


def test_aihub_notebook_runs_vpcd_teacher_forced_debug_before_hybrid() -> None:
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


def test_aihub_notebook_runs_vpcd_quantized_local_teacher_forced_before_cloud_teacher_forced() -> None:
    notebook = _load_notebook("On_device_Ai_option1_pilots.ipynb")
    code_text = "\n".join(_cell_texts(notebook, cell_type="code"))
    markdown_cells = _cell_texts(notebook, cell_type="markdown")

    assert "run_vpcd_local_teacher_forced_diagnostics(" in code_text
    assert "vpcd quantized model path:" in code_text

    quantized_local_heading_index = next(
        index for index, text in enumerate(markdown_cells) if "### VPCD Quantized Local Teacher-Forced Diagnostics" in text
    )
    teacher_heading_index = next(index for index, text in enumerate(markdown_cells) if "### VPCD Teacher-Forced Diagnostics" in text)
    assert quantized_local_heading_index < teacher_heading_index


def test_aihub_notebook_limits_vpcd_source_strategies_to_supported_lanes() -> None:
    notebook = _load_notebook("On_device_Ai_option1_pilots.ipynb")
    code_text = "\n".join(_cell_texts(notebook, cell_type="code"))

    assert 'VPCD_SOURCE_STRATEGY = "local_aimet_compile_candidate"' in code_text
    assert "local_qdq_compile_candidate" not in code_text
    assert 'vpcd_source_strategy = str(VPCD_SOURCE_STRATEGY or "local_aimet_compile_candidate").strip()' in code_text
    assert "vpcd_uses_aihub_quantize" not in code_text
    assert "quantize-run-" not in code_text


def test_aihub_notebook_supports_local_aimet_vpcd_source_strategy() -> None:
    notebook = _load_notebook("On_device_Ai_option1_pilots.ipynb")
    code_text = "\n".join(_cell_texts(notebook, cell_type="code"))

    assert "local_aimet_compile_candidate" in code_text
    assert "vpcd_option1_local_aimet" in code_text
    assert 'VPCD_QUANTIZED_MODEL_PATH = None' in code_text
    assert 'VPCD_LOCAL_AIMET_OUTPUT_ROOT = Path("build/quantize/vpcd/local_aimet")' not in code_text
    assert 'VPCD_CALIBRATION_SOURCE_PATH = Path("build/calibration/vlsp2020/vpcd_transcriptions.txt")' not in code_text
    assert 'VPCD_AIMET_PARAM_TYPE = "int8"' not in code_text
    assert 'VPCD_AIMET_ACTIVATION_TYPE = "int16"' not in code_text
    assert 'VPCD_AIMET_QUANT_SCHEME = "min_max"' not in code_text
    assert 'VPCD_AIMET_CONFIG_FILE = "vpcd_matmul_only"' not in code_text
    assert 'VPCD_AIMET_POLICY_MODE = "local_quality_parity"' not in code_text
    assert 'VPCD_AIMET_SERVICE_URL = "http://127.0.0.1:18080"' not in code_text
    assert "calibration_source_path=" not in code_text
    assert "aimet_param_type=" not in code_text
    assert "aimet_activation_type=" not in code_text
    assert "aimet_quant_scheme=" not in code_text
    assert "aimet_config_file=" not in code_text
    assert "aimet_policy_mode=" not in code_text
    assert "aimet_service_url=" not in code_text
    assert 'output_path=RUNTIME_CONFIG.pilot_artifact_dir(vpcd_pilot_name) / "model.fp32.fixed.onnx"' not in code_text
    assert "max_calibration_samples=VPCD_CALIBRATION_MAX_SAMPLES" not in code_text
    assert "max_generation_length=VPCD_CALIBRATION_MAX_GENERATION_LENGTH" not in code_text
    assert "resolve_downloaded_quantized_model_path(" not in code_text


def test_aihub_notebook_treats_vpcd_local_aimet_as_prebuilt_source() -> None:
    notebook = _load_notebook("On_device_Ai_option1_pilots.ipynb")
    markdown_text = "\n".join(_cell_texts(notebook, cell_type="markdown"))

    assert "this notebook starts only after the retained local VPCD quantize artifact already exists" in markdown_text
    assert "resolves the prebuilt local AIMET artifact" in markdown_text
    assert "It does not build or export VPCD quantize artifacts locally." in markdown_text
    assert "Build the local AIMET package first with `python -m quantize --project vpcd ...`" not in markdown_text
    assert "downloaded AI Hub quantized ONNX" not in markdown_text
    assert "fixed-shape FP32 prepare locally, then AI Hub quantize, then AI Hub compile" not in markdown_text


def test_aihub_notebook_treats_zipformer_as_prebuilt_aihub_ready_source() -> None:
    notebook = _load_notebook("On_device_Ai_option1_pilots.ipynb")
    code_text = "\n".join(_cell_texts(notebook, cell_type="code"))
    markdown_text = "\n".join(_cell_texts(notebook, cell_type="markdown"))

    assert "prepare_zipformer_encoder_option1_source_model" not in code_text
    assert 'zipformer_aihub_prepared_encoder_path = Path("build/quantize/zipformer/qnn_u16u8/aihub_compile/encoder.aihub.option1.onnx")' in code_text
    assert "The notebook reads the retained AI Hub-ready encoder directly from `build/quantize`." in markdown_text


def test_aihub_notebook_treats_bounded_vpcd_truncation_as_comparison_unavailable() -> None:
    notebook = _load_notebook("On_device_Ai_option1_pilots.ipynb")
    code_text = "\n".join(_cell_texts(notebook, cell_type="code"))

    assert 'vpcd_hybrid_mismatches = [row for row in vpcd_hybrid_results if row["matches_expected"] is False]' in code_text
    assert 'vpcd_hybrid_comparison_unavailable = [row for row in vpcd_hybrid_results if row["matches_expected"] is None]' in code_text
    assert 'print("vpcd full-text comparison unavailable:")' in code_text


def test_retired_phase4_and_phase5_notebooks_are_absent() -> None:
    assert not (REPO_ROOT / "On_device_Ai_option1_phase4_gate.ipynb").exists()
    assert not (REPO_ROOT / "On_device_Ai_option1_phase5_contract.ipynb").exists()


def test_aihub_modules_live_under_dedicated_package_only() -> None:
    assert (REPO_ROOT / "src" / "aihub" / "session.py").exists()
    assert (REPO_ROOT / "src" / "aihub" / "evaluation.py").exists()
    assert (REPO_ROOT / "src" / "aihub" / "deployment.py").exists()
    assert list((REPO_ROOT / "src" / "tools").glob("aihub_*.py")) == []
