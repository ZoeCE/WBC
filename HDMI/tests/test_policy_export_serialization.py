import io
import json

import onnxruntime as ort
import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential

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

def test_export_onnx_writes_sim2real_compatible_tensordict_model(tmp_path):
    class TinyActor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(3, 4),
                torch.nn.LayerNorm(4),
                torch.nn.Mish(),
                torch.nn.Linear(4, 2),
            )

        def forward(self, policy):
            return self.net(policy)

    module = TensorDictSequential(
        TensorDictModule(TinyActor(), in_keys=["policy"], out_keys=["action"])
    ).eval()
    td = TensorDict(
        {"policy": torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)},
        batch_size=[1],
    )
    path = tmp_path / "policy.onnx"

    export_utils.export_onnx(module, td, str(path))

    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    output = session.run(None, {session.get_inputs()[0].name: td["policy"].numpy()})
    assert path.is_file()
    assert metadata["in_keys"] == ["policy"]
    assert metadata["out_keys"] == ["action"]
    assert output[0].shape == (1, 2)


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
