import pytest
import torch

from active_adaptation.mujoco.playback_parity import compute_playback_parity


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
