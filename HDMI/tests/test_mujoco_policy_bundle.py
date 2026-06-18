import yaml
import torch
from tensordict.nn import TensorDictModule

from active_adaptation.mujoco.observation_builder import MujocoPolicyState
from active_adaptation.mujoco.policy import MujocoPolicyBundle


def _state():
    return MujocoPolicyState(
        root_ang_vel_b=torch.zeros(1, 3),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
        joint_pos=torch.zeros(1, 2),
        applied_action=torch.tensor([[2.0, -4.0]]),
    )


def test_exported_policy_bundle_builds_tensordict_and_scaled_joint_targets(tmp_path):
    module = TensorDictModule(
        torch.nn.Linear(2, 2, bias=False),
        in_keys=["policy"],
        out_keys=["action"],
    )
    module.module.weight.data.copy_(torch.tensor([[1.0, 0.0], [0.0, 2.0]]))
    policy_path = tmp_path / "policy-test-final.pt"
    torch.save(module, policy_path)
    (tmp_path / "policy-test-final.yaml").write_text(
        yaml.safe_dump(
            {
                "observation": {
                    "policy": {
                        "applied_action": {},
                    },
                },
                "action_scale": {
                    "left_.*": 0.5,
                    "right_joint": 0.25,
                },
                "policy_joint_names": ["left_joint", "right_joint"],
                "isaac_joint_names": ["left_joint", "right_joint"],
                "isaac_body_names": ["pelvis", "left_foot"],
                "default_joint_pos": {
                    "left_joint": 1.0,
                    "right_joint": -1.0,
                },
            }
        )
    )

    bundle = MujocoPolicyBundle.load(policy_path)
    td = bundle.build_tensordict(_state())

    assert torch.allclose(td["policy"], torch.tensor([[2.0, -4.0]]))
    assert td["is_init"].shape == (1, 1)
    assert td["is_init"].dtype is torch.bool
    assert bundle.policy_joint_names == ["left_joint", "right_joint"]
    assert bundle.isaac_joint_names == ["left_joint", "right_joint"]
    assert bundle.isaac_body_names == ["pelvis", "left_foot"]
    assert torch.allclose(bundle.action_scale, torch.tensor([0.5, 0.25]))
    assert torch.allclose(bundle.default_joint_pos, torch.tensor([1.0, -1.0]))

    action = bundle.act(_state())

    assert torch.allclose(action.raw_action, torch.tensor([[2.0, -8.0]]))
    assert torch.allclose(action.scaled_action, torch.tensor([[1.0, -2.0]]))
    assert torch.allclose(action.joint_position_target, torch.tensor([[2.0, -3.0]]))
