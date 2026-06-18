import pytest
import torch

from active_adaptation.mujoco import playback_parity
from active_adaptation.mujoco.playback_parity import (
    compute_playback_parity,
)


def test_playback_parity_computes_joint_and_body_l2_per_env():
    metrics = compute_playback_parity(
        q_mujoco=torch.tensor([[1.0, 2.0, 4.0]]),
        q_ref=torch.tensor([[1.0, 0.0, 0.0]]),
        body_pos_mujoco_w=torch.tensor([[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]]]),
        body_pos_ref_w=torch.tensor([[[1.0, 1.0, 0.0], [1.0, 3.0, 3.0]]]),
        reward=torch.tensor([0.75]),
    )

    assert torch.allclose(metrics.q_l2, torch.tensor([20.0**0.5]))
    assert torch.allclose(metrics.body_pos_l2, torch.tensor([5.0**0.5]))
    assert torch.equal(metrics.reward, torch.tensor([0.75]))


def test_playback_parity_accepts_motion_reference_future_slice():
    ref_joint_pos_future = torch.tensor([[[0.2, 0.4], [9.0, 9.0]]])
    ref_body_pos_future_w = torch.tensor([[[[0.0, 0.0, 1.0]], [[9.0, 9.0, 9.0]]]])

    metrics = compute_playback_parity(
        q_mujoco=torch.tensor([[0.1, 0.1]]),
        q_ref=ref_joint_pos_future[:, 0],
        body_pos_mujoco_w=torch.tensor([[[0.0, 0.0, 0.0]]]),
        body_pos_ref_w=ref_body_pos_future_w[:, 0],
    )

    assert torch.allclose(metrics.q_l2, torch.tensor([(0.1**2 + 0.3**2) ** 0.5]))
    assert torch.allclose(metrics.body_pos_l2, torch.tensor([1.0]))
    assert metrics.reward is None


def test_playback_parity_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="q_mujoco shape.*q_ref shape"):
        compute_playback_parity(
            q_mujoco=torch.zeros(1, 2),
            q_ref=torch.zeros(1, 3),
            body_pos_mujoco_w=torch.zeros(1, 1, 3),
            body_pos_ref_w=torch.zeros(1, 1, 3),
        )


def test_reward_from_spec_matches_hdmi_group_sum_and_aliases():
    state = playback_parity.MujocoRewardState(
        actual_body_pos_w=torch.tensor([[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]]),
        ref_body_pos_w=torch.tensor([[[0.3, 0.4, 0.0]], [[1.0, 0.0, 0.0]]]),
        joint_pos=torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
        ref_joint_pos=torch.tensor([[0.2, 1.0], [1.5, 3.5]]),
        object_pos_w=torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
        ref_object_pos_w=torch.tensor([[0.5, 0.0, 0.0], [1.0, 1.5, 0.0]]),
    )
    reward_cfg = {
        "tracking": {
            "tracking_root_pos(keypoint_pos_tracking_product)": {"weight": 0.5, "sigma": 0.5},
            "joint_pos_tracking_product": {"weight": 2.0, "sigma": 0.25},
            "disabled_object": {"enabled": False, "weight": 100.0},
        },
        "object_tracking": {
            "object_pos_tracking": {"weight": 3.0, "sigma": 0.5},
        },
    }

    reward = playback_parity.compute_reward_from_spec(reward_cfg, state)

    keypoint = torch.exp(
        -((state.ref_body_pos_w - state.actual_body_pos_w).norm(dim=-1).mean(dim=1)) / 0.5
    ).unsqueeze(1)
    joint = torch.exp(-((state.ref_joint_pos - state.joint_pos).abs().mean(dim=1)) / 0.25).unsqueeze(1)
    object_pos = torch.exp(-((state.ref_object_pos_w - state.object_pos_w).norm(dim=-1)) / 0.5).unsqueeze(1)
    expected = torch.cat([0.5 * keypoint + 2.0 * joint, 3.0 * object_pos], dim=1)
    assert torch.allclose(reward, expected)

    metrics = compute_playback_parity(
        q_mujoco=state.joint_pos,
        q_ref=state.ref_joint_pos,
        body_pos_mujoco_w=state.actual_body_pos_w,
        body_pos_ref_w=state.ref_body_pos_w,
        reward=reward,
    )
    assert torch.allclose(metrics.reward, expected)


def test_reward_from_spec_matches_hdmi_multiplicative_group():
    state = playback_parity.MujocoRewardState(
        object_pos_w=torch.tensor([[0.0, 0.0, 0.0]]),
        ref_object_pos_w=torch.tensor([[0.2, 0.0, 0.0]]),
        object_joint_pos=torch.tensor([0.4]),
        ref_object_joint_pos=torch.tensor([0.1]),
    )
    reward_cfg = {
        "object_tracking": {
            "_multiplicative": True,
            "object_pos_tracking": {"weight": 2.0, "sigma": 0.5},
            "object_joint_pos_tracking": {"weight": 0.5, "sigma": 0.25},
        }
    }

    reward = playback_parity.compute_reward_from_spec(reward_cfg, state)

    object_pos = torch.exp(-torch.tensor([[0.2]]) / 0.5)
    object_joint = torch.exp(-torch.tensor([[0.3]]) / 0.25)
    expected = (2.0 * object_pos) * (0.5 * object_joint)
    assert torch.allclose(reward, expected)
