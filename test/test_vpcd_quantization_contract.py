from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from model_pipeline.models.aimet import (
    build_matmul_only_aimet_config,
    load_aimet_calibration_inputs,
    write_aimet_calibration_inputs,
)
from model_pipeline.models.vpcd.quantization import (
    CalibrationBatch,
    build_encoder_matmul_policy,
    inspect_encoder_matmul_aimet_encodings,
    pad_calibration_batch,
)
from model_pipeline.models.vpcd.graph import (
    rewrite_encoder_attention_mask_boolean_casts_for_qnn,
)


def test_matmul_only_aimet_config_disables_per_channel_and_bias() -> None:
    """Verify the AIMET config enables only canonical tensor quantization behavior.

    Returns:
        None.
    """
    config = build_matmul_only_aimet_config()

    assert list(config["op_type"]) == ["MatMul"]
    assert config["defaults"]["per_channel_quantization"] == "False"
    assert config["params"]["bias"]["is_quantized"] == "False"
    assert config["op_type"]["MatMul"]["params"]["weight"]["is_quantized"] == "True"


def test_matmul_only_aimet_config_can_require_symmetric_activations_for_qnn() -> None:
    """Verify the QNN-compatible recipe requests symmetric signed activations.

    Returns:
        None.
    """
    config = build_matmul_only_aimet_config(symmetric_activations=True)

    assert config["defaults"]["ops"]["is_symmetric"] == "True"
    assert config["defaults"]["strict_symmetric"] == "False"
    assert config["defaults"]["unsigned_symmetric"] == "False"


def test_matmul_only_aimet_config_satisfies_required_model_boundaries() -> None:
    """Verify the config satisfies AIMET's required boundary initialization.

    Returns:
        None.
    """
    config = build_matmul_only_aimet_config()

    assert config["model_input"] == {"is_input_quantized": "True"}
    assert config["model_output"] == {"is_output_quantized": "True"}
    assert config["op_type"]["MatMul"]["is_input_quantized"] == "True"
    assert config["op_type"]["MatMul"]["is_output_quantized"] == "True"


def test_matmul_only_aimet_config_can_defer_operator_selection_to_policy() -> None:
    """Verify a name allowlist can avoid conflicting topology-wide config rules.

    Returns:
        None.
    """
    config = build_matmul_only_aimet_config(
        select_operators_from_policy=True,
    )

    assert config["op_type"] == {}
    assert config["model_input"] == {"is_input_quantized": "True"}
    assert config["model_output"] == {"is_output_quantized": "True"}


def test_encoder_policy_excludes_decoder_and_lm_head_matmuls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify encoder policy disables every decoder and language-head MatMul.

    Args:
        monkeypatch: Pytest fixture replacing graph inventory with canonical test data.

    Returns:
        None.
    """
    from model_pipeline.models.vpcd import quantization
    from model_pipeline.models.vpcd.graph import VpcdMatmulInventory

    inventory = VpcdMatmulInventory(
        encoder=("/encoder/a/MatMul", "/encoder/b/MatMul"),
        decoder=("/decoder/a/MatMul",),
        lm_head=("/lm_head/MatMul",),
        other=(),
    )
    monkeypatch.setattr(quantization, "inspect_vpcd_matmuls", lambda _path: inventory)

    policy = build_encoder_matmul_policy("model.onnx", require_canonical_counts=False)

    assert policy["quantize_op_names"] == ["/encoder/a/MatMul", "/encoder/b/MatMul"]
    assert policy["disable_op_names"] == ["/decoder/a/MatMul", "/lm_head/MatMul"]
    assert policy["coverage"] == {"quantized": 2, "total_matmul": 4}
    assert policy["quantizer_selection"] == "operator-name-allowlist"
    assert policy["symmetric_activation_encodings"] is True
    assert "policy_mode" not in policy


def test_encoder_policy_rejects_noncanonical_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify policy creation fails when graph coverage differs from 96/168/1.

    Args:
        monkeypatch: Pytest fixture replacing graph inventory with invalid test data.

    Returns:
        None.
    """
    from model_pipeline.models.vpcd import quantization
    from model_pipeline.models.vpcd.graph import VpcdMatmulInventory

    monkeypatch.setattr(
        quantization,
        "inspect_vpcd_matmuls",
        lambda _path: VpcdMatmulInventory(("encoder",), (), (), ()),
    )

    with pytest.raises(ValueError, match="96/168/1"):
        build_encoder_matmul_policy("model.onnx")


def test_encoding_inventory_rejects_decoder_scope_and_asymmetric_activations(
    tmp_path: Path,
) -> None:
    """Verify VPCD encoding evidence exposes dtype, symmetry, and scope violations.

    Args:
        tmp_path: Isolated directory for a synthetic AIMET encoding file.

    Returns:
        None.
    """
    path = tmp_path / "model.encodings"
    path.write_text(
        json.dumps(
            {
                "activation_encodings": [
                    {
                        "name": "/model/encoder/layers.0/input",
                        "bw": 16,
                        "dtype": "INT",
                        "is_sym": True,
                        "offset": [-32768.0],
                    },
                    {
                        "name": "/model/decoder/layers.0/input",
                        "bw": 16,
                        "dtype": "INT",
                        "is_sym": False,
                        "offset": [-30000.0],
                    },
                ],
                "param_encodings": [
                    {
                        "name": "model.encoder.layers.0.weight",
                        "bw": 8,
                        "dtype": "INT",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    inventory = inspect_encoder_matmul_aimet_encodings(path)

    assert inventory["activation_count"] == 2
    assert inventory["parameter_count"] == 1
    assert inventory["activation_contract"] is False
    assert inventory["parameter_contract"] is True
    assert inventory["non_encoder_names"] == ["/model/decoder/layers.0/input"]


def test_calibration_is_padded_to_fixed_shapes_and_round_trips(tmp_path: Path) -> None:
    """Verify fixed-shape padding and serialization preserve ordered arrays.

    Args:
        tmp_path: Isolated directory for calibration package files.

    Returns:
        None.
    """
    batch = CalibrationBatch(
        inputs={
            "input_ids": np.asarray([[5, 6]], dtype=np.int64),
            "attention_mask": np.asarray([[1, 1]], dtype=np.int64),
            "decoder_input_ids": np.asarray([[0]], dtype=np.int64),
            "decoder_attention_mask": np.asarray([[1]], dtype=np.int64),
        }
    )

    padded = pad_calibration_batch(batch, pad_token_id=1)
    manifest_path = write_aimet_calibration_inputs([padded.inputs], tmp_path)
    restored = load_aimet_calibration_inputs(tmp_path)

    assert padded.inputs["input_ids"].shape == (1, 384)
    assert padded.inputs["decoder_input_ids"].shape == (1, 64)
    assert padded.inputs["input_ids"][0, 2] == 1
    assert padded.inputs["attention_mask"][0, 2] == 0
    assert np.array_equal(restored[0]["input_ids"], padded.inputs["input_ids"])
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["input_order"] == list(batch.inputs)


def test_qnn_boolean_cast_rewrite_builds_integer_mask_comparison_without_changing_outputs(
    tmp_path: Path,
) -> None:
    """Verify the attention-mask rewrite changes only the redundant first boolean cast.

    Args:
        tmp_path: Isolated directory for the source and rewritten ONNX models.

    Returns:
        None.
    """
    import onnx
    from onnx import TensorProto, helper

    source = tmp_path / "source.onnx"
    destination = tmp_path / "rewritten.onnx"
    nodes = [
        helper.make_node(
            "Cast",
            ["attention_mask"],
            ["mask_float"],
            name="/model/encoder/Cast_1",
            to=TensorProto.FLOAT,
        ),
        helper.make_node(
            "Sub",
            ["one", "mask_float"],
            ["inverted_mask"],
            name="/model/encoder/Sub",
        ),
        helper.make_node(
            "Cast",
            ["inverted_mask"],
            ["mask_bool_1"],
            name="/model/encoder/Cast_2",
            to=TensorProto.BOOL,
        ),
        helper.make_node(
            "Cast",
            ["mask_bool_1"],
            ["mask_bool_2"],
            name="/model/encoder/Cast_3",
            to=TensorProto.BOOL,
        ),
        helper.make_node(
            "Where",
            ["mask_bool_2", "negative", "inverted_mask"],
            ["output"],
            name="/model/encoder/Where_1",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "vpcd-mask",
        [helper.make_tensor_value_info("attention_mask", TensorProto.INT32, [1, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
        [
            helper.make_tensor("one", TensorProto.FLOAT, [], [1.0]),
            helper.make_tensor("negative", TensorProto.FLOAT, [], [-3.4028235e38]),
        ],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), source)

    rewrite_count = rewrite_encoder_attention_mask_boolean_casts_for_qnn(source, destination)
    rewritten = onnx.load(destination.as_posix())
    nodes_by_name = {node.name: node for node in rewritten.graph.node}
    mask_cast = nodes_by_name["/model/encoder/AttentionMaskToInt32"]
    mask_equal = nodes_by_name["/model/encoder/AttentionMaskEqualsZero"]

    assert rewrite_count == 1
    assert mask_cast.op_type == "Cast"
    assert list(mask_cast.input) == ["attention_mask"]
    assert helper.get_attribute_value(mask_cast.attribute[0]) == TensorProto.INT32
    assert mask_equal.op_type == "Equal"
    assert list(mask_cast.output) == ["/model/encoder/AttentionMaskInt32_output_0"]
    assert list(mask_equal.input) == [
        "/model/encoder/AttentionMaskInt32_output_0",
        "/model/encoder/AttentionMaskZeroInt32",
    ]
    assert list(mask_equal.output) == ["mask_bool_2"]
    assert "/model/encoder/Cast_2" not in nodes_by_name
    assert "/model/encoder/Cast_3" not in nodes_by_name
    assert [output.name for output in rewritten.graph.output] == ["output"]
