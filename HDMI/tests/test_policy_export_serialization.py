import io

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

import active_adaptation.utils.export as export_utils
from active_adaptation.learning.ppo.ppo_amp import DummyRefJointPos as AmpDummyRefJointPos
from active_adaptation.learning.ppo.ppo_amp import RefJointPos as AmpRefJointPos
from active_adaptation.learning.ppo.ppo_roa import DummyRefJointPos as RoaDummyRefJointPos
from active_adaptation.learning.ppo.ppo_roa import RefJointPos as RoaRefJointPos


def _roundtrip_tensordict_module(module):
    td_module = TensorDictModule(module, ["ref_joint_pos_", "loc"], ["loc"])
    buffer = io.BytesIO()
    torch.save(td_module, buffer)
    buffer.seek(0)
    return torch.load(buffer, weights_only=False)


def test_ppo_roa_residual_action_modules_are_export_serializable():
    td = TensorDict(
        {
            "ref_joint_pos_": torch.tensor([[0.1, 0.2]]),
            "loc": torch.tensor([[1.0, 2.0]]),
        },
        batch_size=[1],
    )

    ref_module = _roundtrip_tensordict_module(RoaRefJointPos())
    dummy_module = _roundtrip_tensordict_module(RoaDummyRefJointPos())

    assert torch.allclose(ref_module(td.clone())["loc"], torch.tensor([[1.1, 2.2]]))
    assert torch.allclose(dummy_module(td.clone())["loc"], torch.tensor([[1.0, 2.0]]))


def test_optional_onnx_export_reports_failure_without_raising(monkeypatch, tmp_path):
    def fail_export(*args, **kwargs):
        raise RuntimeError("onnx conversion failed")

    monkeypatch.setattr(export_utils, "export_onnx", fail_export)

    ok = export_utils.export_onnx_optional(object(), TensorDict({}, []), str(tmp_path / "policy.onnx"))

    assert ok is False


def test_optional_onnx_export_can_require_success(monkeypatch, tmp_path):
    def fail_export(*args, **kwargs):
        raise RuntimeError("onnx conversion failed")

    monkeypatch.setattr(export_utils, "export_onnx", fail_export)

    with pytest.raises(RuntimeError, match="onnx conversion failed"):
        export_utils.export_onnx_optional(
            object(),
            TensorDict({}, []),
            str(tmp_path / "policy.onnx"),
            required=True,
        )


def test_ppo_amp_residual_action_modules_are_export_serializable():
    td = TensorDict(
        {
            "ref_joint_pos_": torch.tensor([[0.1, 0.2]]),
            "loc": torch.tensor([[1.0, 2.0]]),
        },
        batch_size=[1],
    )

    ref_module = _roundtrip_tensordict_module(AmpRefJointPos())
    dummy_module = _roundtrip_tensordict_module(AmpDummyRefJointPos())

    assert torch.allclose(ref_module(td.clone())["loc"], torch.tensor([[1.1, 2.2]]))
    assert torch.allclose(dummy_module(td.clone())["loc"], torch.tensor([[1.0, 2.0]]))
