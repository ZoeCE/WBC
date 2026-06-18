import torch

from active_adaptation.mujoco.reward_parity import (
    eef_contact_exp,
    joint_position_tracking_product,
    keypoint_position_tracking_product,
)


def test_keypoint_position_tracking_product_matches_hdmi_world_formula():
    actual_body_pos_w = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ]
    )
    ref_body_pos_w = torch.tensor(
        [
            [[0.3, 0.4, 0.0], [1.0, 1.2, 0.0]],
            [[1.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
        ]
    )
    tolerance = torch.tensor([0.1, 0.2])

    reward = keypoint_position_tracking_product(
        actual_body_pos_w=actual_body_pos_w,
        ref_body_pos_w=ref_body_pos_w,
        sigma=0.5,
        tolerance=tolerance,
    )

    error = ((ref_body_pos_w - actual_body_pos_w).norm(dim=-1) - tolerance).clamp_min(0.0)
    expected = torch.exp(-error.mean(dim=1) / 0.5).unsqueeze(1)
    assert torch.allclose(reward, expected)


def test_keypoint_position_tracking_product_can_match_in_yaw_local_frame():
    sqrt_half = 2 ** -0.5
    actual_body_pos_w = torch.tensor([[[0.0, 1.0, 0.0]]])
    ref_body_pos_w = torch.tensor([[[1.0, 0.0, 0.0]]])

    reward = keypoint_position_tracking_product(
        actual_body_pos_w=actual_body_pos_w,
        ref_body_pos_w=ref_body_pos_w,
        sigma=0.1,
        local=True,
        root_pos_w=torch.tensor([[0.0, 0.0, 0.0]]),
        root_quat_w=torch.tensor([[sqrt_half, 0.0, 0.0, sqrt_half]]),
        ref_root_pos_w=torch.tensor([[0.0, 0.0, 0.0]]),
        ref_root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    )

    assert torch.allclose(reward, torch.ones(1, 1), atol=1e-6)


def test_joint_position_tracking_product_matches_hdmi_formula():
    joint_pos = torch.tensor([[0.0, 2.0, 4.0], [1.0, 1.0, 1.0]])
    ref_joint_pos = torch.tensor([[0.2, 1.5, 4.0], [1.6, 1.0, 0.7]])
    tolerance = torch.tensor([0.1, 0.2, 0.0])

    reward = joint_position_tracking_product(
        joint_pos=joint_pos,
        ref_joint_pos=ref_joint_pos,
        sigma=0.25,
        tolerance=tolerance,
    )

    error = ((ref_joint_pos - joint_pos).abs() - tolerance).clamp_min(0.0)
    expected = torch.exp(-error.mean(dim=1) / 0.25).unsqueeze(1)
    assert torch.allclose(reward, expected)


def test_eef_contact_exp_matches_hdmi_contact_reward_formula():
    contact_eef_pos_w = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        ]
    )
    contact_target_pos_w = torch.tensor(
        [
            [[0.2, 0.0, 0.0], [1.5, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 2.4, 0.0]],
        ]
    )
    eef_contact_forces_b = torch.tensor(
        [
            [[0.0, 0.0, 3.0], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
        ]
    )
    ref_object_contact = torch.tensor([[True, False], [False, True]])

    reward = eef_contact_exp(
        contact_eef_pos_w=contact_eef_pos_w,
        contact_target_pos_w=contact_target_pos_w,
        eef_contact_forces_b=eef_contact_forces_b,
        ref_object_contact=ref_object_contact,
        pos_sigma=0.5,
        pos_tolerance=0.1,
        frc_sigma=2.0,
        frc_thres=2.0,
        gain=0.8,
    )

    pos_error = ((contact_eef_pos_w - contact_target_pos_w).norm(dim=-1) - 0.1).clamp_min(0.0)
    contact_frc = (eef_contact_forces_b.norm(dim=-1) - 2.0).clamp_max(0.0)
    active_reward = torch.exp(-pos_error / 0.5) * torch.exp(contact_frc / 2.0)
    expected = (active_reward * ref_object_contact.float() * 0.8 + 1 - ref_object_contact.float()).mean(dim=-1)
    assert torch.allclose(reward, expected.unsqueeze(-1))
