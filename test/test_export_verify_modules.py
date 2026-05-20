import pytest


def test_export_package_has_been_removed():
    import sys

    for name in list(sys.modules):
        if name == "export" or name.startswith("export."):
            sys.modules.pop(name, None)
    with pytest.raises(ModuleNotFoundError):
        __import__("export")


def test_tools_bundle_export_delegates_to_project_owned_exporters(monkeypatch, tmp_path, capsys):
    seen = {}

    class FakeManifest:
        asset_namespace = "models/asr/zipformer/fp32"
        model_variant = "fp32"

    def fake_zipformer_export_bundle(**kwargs):
        seen["kwargs"] = kwargs
        return FakeManifest()

    monkeypatch.setattr("tools.bundle_export.ZIPFORMER_EXPORT_BUNDLE", fake_zipformer_export_bundle)

    from tools.bundle_export import main

    main(
        [
            "--project",
            "zipformer",
            "--model-dir",
            "assets/zipformer",
            "--output-dir",
            str(tmp_path / "bundle"),
            "--asset-namespace",
            "models/asr/zipformer/fp32",
            "--model-variant",
            "fp32",
        ]
    )
    output = capsys.readouterr().out

    assert seen["kwargs"]["asset_namespace"] == "models/asr/zipformer/fp32"
    assert seen["kwargs"]["model_variant"] == "fp32"
    assert "Bundle export complete." in output


def test_verify_model_bundle_module_delegates_to_verify_runtime(monkeypatch, capsys):
    seen = {}

    class FakeAdapter:
        default_model_dir = "assets/model"
        default_output_dir = "build/model_bundle/project/fp32"

    def fake_resolve_bundle_project(name):
        seen["project"] = name
        return FakeAdapter()

    def fake_verify_model_bundle(**kwargs):
        seen["kwargs"] = kwargs
        return {"checked_samples": 2, "passed": True, "mismatches": []}

    monkeypatch.setattr("verify.model_bundle.resolve_bundle_project", fake_resolve_bundle_project)
    monkeypatch.setattr("verify.model_bundle.verify_model_bundle", fake_verify_model_bundle)

    from verify.model_bundle import main

    main(["--project", "zipformer"])
    output = capsys.readouterr().out

    assert seen["project"] == "zipformer"
    assert seen["kwargs"]["project"] == "zipformer"
    assert "Verification complete." in output
    assert "Passed         : True" in output


def test_tools_punctuation_onnx_command_builder_still_exists():
    from tools.punctuation_onnx import build_command

    command = build_command("python", "assets/model", "build/out", 17, 5e-5)

    assert command[:3] == ["python", "-m", "transformers.onnx"]
