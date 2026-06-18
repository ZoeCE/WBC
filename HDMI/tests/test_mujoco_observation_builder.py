import pytest
import torch

from active_adaptation.mujoco.observation_builder import (
    MujocoObservationBuilder,
    MujocoPolicyState,
)


def _as_float_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.float32)
    return torch.tensor(value, dtype=torch.float32)


def _state(root, gravity, joint, offset=None, action=None, history=None, **extra):
    root_t = _as_float_tensor(root)
    batch = root_t.shape[0]
    joint_t = _as_float_tensor(joint)
    if offset is None:
        offset = torch.zeros_like(joint_t)
    if action is None:
        action = torch.zeros(batch, joint_t.shape[1])
    state = MujocoPolicyState(
        root_ang_vel_b=root_t,
        projected_gravity_b=_as_float_tensor(gravity),
        joint_pos=joint_t,
        joint_pos_offset=_as_float_tensor(offset),
        applied_action=_as_float_tensor(action),
        action_history=history,
    )
    for name, value in extra.items():
        setattr(state, name, _as_float_tensor(value))
    return state


def test_policy_builder_matches_exported_policy_group_order_and_history():
    cfg = {
        "policy": {
            "root_ang_vel_history": {"history_steps": [0, 2], "noise_std": 0.0},
            "projected_gravity_history": {"history_steps": [1], "noise_std": 0.0},
            "joint_pos_history": {"history_steps": [0, 1], "noise_std": 0.0},
            "prev_actions": {"steps": 2},
        }
    }
    builder = MujocoObservationBuilder(cfg, policy_joint_names=["j0", "j1", "j2"])

    s0 = _state(
        root=[[0.0, 0.1, 0.2]],
        gravity=[[0.0, 0.0, -1.0]],
        joint=[[1.0, 2.0, 3.0]],
        offset=[[0.1, 0.2, 0.3]],
    )
    s1 = _state(
        root=[[1.0, 1.1, 1.2]],
        gravity=[[0.0, 1.0, 0.0]],
        joint=[[2.0, 3.0, 4.0]],
        offset=[[0.1, 0.2, 0.3]],
    )
    action_history = torch.tensor([[[0.5, 0.4, 0.3], [0.6, 0.5, 0.4], [0.7, 0.6, 0.5]]])
    s2 = _state(
        root=[[2.0, 2.1, 2.2]],
        gravity=[[1.0, 0.0, 0.0]],
        joint=[[3.0, 4.0, 5.0]],
        offset=[[0.1, 0.2, 0.3]],
        history=action_history,
    )

    builder.reset(s0)
    builder.update(s1)
    builder.update(s2)

    obs = builder.build_group("policy", s2)
    expected = torch.cat(
        [
            torch.tensor([[2.0, 2.1, 2.2, 0.0, 0.1, 0.2]]),
            torch.tensor([[0.0, 1.0, 0.0]]),
            torch.tensor([[2.9, 3.8, 4.7, 1.9, 2.8, 3.7]]),
            action_history[:, :, :2].reshape(1, -1),
        ],
        dim=-1,
    )

    assert torch.allclose(obs, expected)
    assert builder.group_dim("policy") == expected.shape[-1]


def test_policy_builder_can_return_named_components_for_debugging():
    cfg = {"policy": {"applied_action": {}}}
    builder = MujocoObservationBuilder(cfg, policy_joint_names=["j0", "j1"])
    state = _state(
        root=[[0.0, 0.0, 0.0]],
        gravity=[[0.0, 0.0, -1.0]],
        joint=[[0.0, 0.0]],
        action=[[0.2, -0.3]],
    )

    pieces = builder.build_group("policy", state, return_components=True)

    assert list(pieces.keys()) == ["applied_action"]
    assert torch.allclose(pieces["applied_action"], torch.tensor([[0.2, -0.3]]))


def test_history_observation_shared_by_groups_updates_once_per_state():
    cfg = {
        "policy": {"root_ang_vel_history": {"history_steps": [0, 1]}},
        "critic": {"root_ang_vel_history": {"history_steps": [0, 1]}},
    }
    builder = MujocoObservationBuilder(cfg, policy_joint_names=["j0"])

    s0 = _state(
        root=[[0.0, 0.1, 0.2]],
        gravity=[[0.0, 0.0, -1.0]],
        joint=[[0.0]],
    )
    s1 = _state(
        root=[[1.0, 1.1, 1.2]],
        gravity=[[0.0, 0.0, -1.0]],
        joint=[[0.0]],
    )
    s2 = _state(
        root=[[2.0, 2.1, 2.2]],
        gravity=[[0.0, 0.0, -1.0]],
        joint=[[0.0]],
    )

    builder.reset(s0)
    builder.update(s1)
    builder.update(s2)

    expected = torch.tensor([[2.0, 2.1, 2.2, 1.0, 1.1, 1.2]])
    assert torch.allclose(builder.build_group("policy", s2), expected)
    assert torch.allclose(builder.build_group("critic", s2), expected)


def test_command_group_builds_motion_reference_observations_in_export_order():
    cfg = {
        "command": {
            "ref_body_pos_future_local": {},
            "ref_joint_pos_future": {},
            "ref_motion_phase": {},
        }
    }
    builder = MujocoObservationBuilder(cfg, policy_joint_names=["j0", "j1"])
    sqrt_half = 2 ** -0.5
    state = _state(
        root=[[0.0, 0.0, 0.0]],
        gravity=[[0.0, 0.0, -1.0]],
        joint=[[0.0, 0.0]],
        ref_body_pos_future_w=[
            [
                [[1.0, 1.0, 2.0], [0.0, 2.0, 3.0]],
                [[2.0, 1.0, 4.0], [0.0, 3.0, 5.0]],
            ]
        ],
        ref_root_pos_w=[[1.0, 1.0, 0.5]],
        ref_root_quat_w=[[sqrt_half, 0.0, 0.0, sqrt_half]],
        ref_joint_pos_future=[[[0.1, 0.2], [0.3, 0.4]]],
        motion_t=[2],
        motion_len=[10],
    )

    obs = builder.build_group("command", state)

    expected = torch.tensor(
        [[
            0.0, 0.0, 2.0,
            1.0, 1.0, 3.0,
            0.0, -1.0, 4.0,
            2.0, 1.0, 5.0,
            0.1, 0.2, 0.3, 0.4,
            0.2,
        ]]
    )
    assert torch.allclose(obs, expected, atol=1e-6)
    assert builder.group_dim("command") == expected.shape[-1]


def test_policy_builder_rejects_unsupported_observation_key():
    cfg = {"policy": {"unknown_mujoco_observation": {}}}
    builder = MujocoObservationBuilder(cfg, policy_joint_names=["j0"])
    state = _state(
        root=[[0.0, 0.0, 0.0]],
        gravity=[[0.0, 0.0, -1.0]],
        joint=[[0.0]],
    )

    with pytest.raises(NotImplementedError, match="unknown_mujoco_observation"):
        builder.build_group("policy", state)


def test_command_group_builds_contact_observations_in_robot_root_frame():
    cfg = {
        "command": {
            "ref_contact_pos_b": {"yaw_only": True},
            "diff_contact_pos_b": {},
        }
    }
    builder = MujocoObservationBuilder(cfg, policy_joint_names=["j0"])
    sqrt_half = 2 ** -0.5
    state = _state(
        root=[[0.0, 0.0, 0.0]],
        gravity=[[0.0, 0.0, -1.0]],
        joint=[[0.0]],
        robot_root_pos_w=[[1.0, 1.0, 0.0]],
        robot_root_quat_w=[[sqrt_half, 0.0, 0.0, sqrt_half]],
        contact_target_pos_w=[
            [
                [1.0, 2.0, 0.0],
                [2.0, 1.0, 1.0],
            ]
        ],
        contact_eef_pos_w=[
            [
                [1.0, 1.5, 0.0],
                [1.0, 1.0, 1.0],
            ]
        ],
    )

    obs = builder.build_group("command", state)

    expected = torch.tensor(
        [[
            1.0, 0.0, 0.0,
            0.0, -1.0, 1.0,
            0.5, 0.0, 0.0,
            0.0, -1.0, 0.0,
        ]]
    )
    assert torch.allclose(obs, expected, atol=1e-6)
