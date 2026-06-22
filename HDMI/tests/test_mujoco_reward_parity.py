import torch

from active_adaptation.mujoco import reward_parity
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


def test_loco_common_rewards_match_hdmi_formulas():
    joint_vel = torch.tensor([[1.0, -3.0, 10.0], [0.5, -0.5, 0.0]])

    assert torch.allclose(reward_parity.survival(joint_vel), torch.ones(2, 1))
    assert torch.allclose(
        reward_parity.joint_velocity_l2(joint_vel),
        -joint_vel.square().clamp_max(5.0).sum(dim=1, keepdim=True),
    )

    joint_pos = torch.tensor([[-0.9, 0.0, 1.8], [0.6, 1.6, 0.2]])
    joint_pos_limits = torch.tensor(
        [
            [[-1.0, 1.0], [-2.0, 2.0], [0.0, 2.0]],
            [[-1.0, 1.0], [-2.0, 2.0], [0.0, 2.0]],
        ]
    )

    reward = reward_parity.joint_position_limits(
        joint_pos=joint_pos,
        joint_pos_limits=joint_pos_limits,
        soft_factor=0.5,
    )

    jpos_mean = joint_pos_limits.mean(dim=-1)
    jpos_range = joint_pos_limits[..., 1] - joint_pos_limits[..., 0]
    soft_lower = jpos_mean - 0.5 * jpos_range * 0.5
    soft_upper = jpos_mean + 0.5 * jpos_range * 0.5
    violation_min = (soft_lower - joint_pos).clamp_min(0.0)
    violation_max = (joint_pos - soft_upper).clamp_min(0.0)
    expected = -(violation_min + violation_max).sum(dim=1, keepdim=True)
    assert torch.allclose(reward, expected)


def test_feet_contact_rewards_match_hdmi_formulas():
    body_lin_vel_w = torch.tensor(
        [
            [[0.5, 0.0, 1.0], [0.0, 0.3, 0.0]],
            [[1.5, 0.0, 0.0], [0.2, 0.2, 0.0]],
        ]
    )
    current_contact_time = torch.tensor([[0.03, 0.01], [0.10, 0.04]])

    slip = reward_parity.feet_slip(
        body_lin_vel_w=body_lin_vel_w,
        current_contact_time=current_contact_time,
        tolerance=0.1,
    )

    in_contact = current_contact_time > 0.02
    feet_vel = (body_lin_vel_w[..., :2].norm(dim=-1) - 0.1).clamp(min=0.0, max=1.0)
    assert torch.allclose(slip, -(in_contact * feet_vel).sum(dim=1, keepdim=True))

    net_forces_w_history = torch.tensor(
        [
            [
                [[30.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[10.0, 0.0, 0.0], [0.0, 20.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]],
                [[0.0, 40.0, 0.0], [10.0, 0.0, 0.0]],
            ],
        ]
    )
    first_contact = torch.tensor([[True, False], [False, True]])
    default_mass_total = torch.tensor([20.0, 10.0])

    impact = reward_parity.impact_force_l2(
        net_forces_w_history=net_forces_w_history,
        first_contact=first_contact,
        default_mass_total=default_mass_total,
    )

    contact_forces = net_forces_w_history.norm(dim=-1).mean(dim=1)
    force = contact_forces / default_mass_total[:, None]
    expected_impact = -(force.square() * first_contact).clamp_max(10.0).sum(dim=1, keepdim=True)
    assert torch.allclose(impact, expected_impact)

    last_air_time = torch.tensor([[0.20, 0.50], [0.10, 0.40]])
    air_time = reward_parity.feet_air_time(
        last_air_time=last_air_time,
        first_contact=first_contact,
        thres=0.30,
        is_standing_env=torch.tensor([False, True]),
    )

    expected_air_time = ((last_air_time - 0.30).clamp_max(0.0) * first_contact).sum(dim=1, keepdim=True)
    expected_air_time[1] = 0.0
    assert torch.allclose(air_time, expected_air_time)

    net_forces_w = torch.tensor(
        [
            [[0.6, 0.0, 10.0], [0.3, 0.2, 0.0]],
            [[0.0, 0.0, 20.0], [0.0, 0.7, 0.0]],
        ]
    )

    stumble = reward_parity.feet_stumble(net_forces_w)

    in_xy_contact = net_forces_w[..., :2].norm(dim=-1) > 0.5
    expected_stumble = -in_xy_contact.float().mean(dim=1, keepdim=True)
    assert torch.allclose(stumble, expected_stumble)


def test_impact_force_l2_sanitizes_nonfinite_contact_sensor_values():
    net_forces_w_history = torch.tensor(
        [
            [
                [[float("nan"), float("inf"), -float("inf")], [30.0, 0.0, 0.0]],
                [[10.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]],
            ],
        ]
    )
    first_contact = torch.tensor([[True, True]])
    default_mass_total = torch.tensor([10.0])

    impact = reward_parity.impact_force_l2(
        net_forces_w_history=net_forces_w_history,
        first_contact=first_contact,
        default_mass_total=default_mass_total,
    )

    sanitized = torch.nan_to_num(net_forces_w_history, nan=0.0, posinf=0.0, neginf=0.0)
    contact_forces = sanitized.norm(dim=-1).mean(dim=1)
    force = contact_forces / default_mass_total[:, None]
    expected = -(force.square() * first_contact).clamp_max(10.0).sum(dim=1, keepdim=True)
    assert torch.isfinite(impact).all()
    assert torch.allclose(impact, expected)


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


def test_eef_contact_exp_sanitizes_nonfinite_contact_sensor_values():
    contact_eef_pos_w = torch.zeros(1, 2, 3)
    contact_target_pos_w = torch.zeros(1, 2, 3)
    eef_contact_forces_b = torch.tensor(
        [
            [
                [float("nan"), float("inf"), -float("inf")],
                [0.0, 0.0, 4.0],
            ]
        ]
    )
    ref_object_contact = torch.tensor([[True, True]])

    reward = eef_contact_exp(
        contact_eef_pos_w=contact_eef_pos_w,
        contact_target_pos_w=contact_target_pos_w,
        eef_contact_forces_b=eef_contact_forces_b,
        ref_object_contact=ref_object_contact,
        frc_sigma=2.0,
        frc_thres=2.0,
    )

    sanitized = torch.nan_to_num(eef_contact_forces_b, nan=0.0, posinf=0.0, neginf=0.0)
    expected = torch.exp((sanitized.norm(dim=-1) - 2.0).clamp_max(0.0) / 2.0).mean(dim=-1, keepdim=True)
    assert torch.isfinite(reward).all()
    assert torch.allclose(reward, expected)


def test_object_position_tracking_matches_hdmi_formula():
    object_pos_w = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    ref_object_pos_w = torch.tensor([[0.3, 0.4, 0.0], [1.0, 2.2, 0.0]])

    reward = reward_parity.object_position_tracking(
        object_pos_w=object_pos_w,
        ref_object_pos_w=ref_object_pos_w,
        sigma=0.5,
    )

    object_pos_error = (ref_object_pos_w - object_pos_w).norm(dim=-1)
    expected = torch.exp(-object_pos_error / 0.5).unsqueeze(1)
    assert torch.allclose(reward, expected)


def test_object_orientation_tracking_matches_hdmi_formula():
    sqrt_half = 2 ** -0.5
    object_quat_w = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [sqrt_half, 0.0, 0.0, sqrt_half],
        ]
    )
    ref_object_quat_w = torch.tensor(
        [
            [sqrt_half, 0.0, 0.0, sqrt_half],
            [sqrt_half, 0.0, 0.0, sqrt_half],
        ]
    )

    reward = reward_parity.object_orientation_tracking(
        object_quat_w=object_quat_w,
        ref_object_quat_w=ref_object_quat_w,
        sigma=0.25,
    )

    expected = torch.exp(-torch.tensor([torch.pi / 2, 0.0]) / 0.25).unsqueeze(1)
    assert torch.allclose(reward, expected, atol=1e-6)


def test_object_joint_position_tracking_matches_hdmi_formula():
    object_joint_pos = torch.tensor([0.0, 1.5, -0.2])
    ref_object_joint_pos = torch.tensor([0.4, 1.0, -0.2])

    reward = reward_parity.object_joint_position_tracking(
        object_joint_pos=object_joint_pos,
        ref_object_joint_pos=ref_object_joint_pos,
        sigma=0.2,
    )

    joint_pos_error = (ref_object_joint_pos - object_joint_pos).abs()
    expected = torch.exp(-joint_pos_error / 0.2).unsqueeze(1)
    assert torch.allclose(reward, expected)


def test_eef_contact_exp_max_matches_hdmi_contact_reward_formula():
    contact_eef_pos_w = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        ]
    )
    contact_target_pos_w = torch.tensor(
        [
            [[0.2, 0.0, 0.0], [1.8, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 2.4, 0.0]],
        ]
    )
    eef_contact_forces_b = torch.tensor(
        [
            [[0.0, 0.0, 3.0], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
        ]
    )
    ref_object_contact = torch.tensor([[True, False], [False, False]])

    reward = reward_parity.eef_contact_exp_max(
        contact_eef_pos_w=contact_eef_pos_w,
        contact_target_pos_w=contact_target_pos_w,
        eef_contact_forces_b=eef_contact_forces_b,
        ref_object_contact=ref_object_contact,
        pos_sigma=0.5,
        pos_tolerance=0.1,
        frc_sigma=2.0,
        frc_thres=2.0,
    )

    pos_error = ((contact_eef_pos_w - contact_target_pos_w).norm(dim=-1) - 0.1).clamp_min(0.0)
    contact_frc = (eef_contact_forces_b.norm(dim=-1) - 2.0).clamp_max(0.0)
    active_reward = torch.exp(-pos_error / 0.5) * torch.exp(contact_frc / 2.0)
    expected = active_reward.max(dim=-1).values * ref_object_contact.any(dim=-1).float()
    assert torch.allclose(reward, expected.unsqueeze(-1))


def test_eef_contact_all_matches_hdmi_contact_reward_formula():
    contact_eef_pos_w = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        ]
    )
    contact_target_pos_w = torch.tensor(
        [
            [[0.05, 0.0, 0.0], [1.5, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 2.08, 0.0]],
        ]
    )
    eef_contact_forces_b = torch.tensor(
        [
            [[0.0, 0.0, 3.0], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
        ]
    )
    ref_object_contact = torch.tensor([[True, True], [False, True]])

    reward = reward_parity.eef_contact_all(
        contact_eef_pos_w=contact_eef_pos_w,
        contact_target_pos_w=contact_target_pos_w,
        eef_contact_forces_b=eef_contact_forces_b,
        ref_object_contact=ref_object_contact,
        pos_thres=0.1,
        frc_thres=2.0,
        gain=0.5,
    )

    contact_pos = (contact_eef_pos_w - contact_target_pos_w).norm(dim=-1) < 0.1
    contact_frc = eef_contact_forces_b.norm(dim=-1) >= 2.0
    active_reward = (contact_pos & contact_frc).float()
    expected = (active_reward * ref_object_contact.float() * 0.5 + 1 - ref_object_contact.float()).mean(dim=-1)
    assert torch.allclose(reward, expected.unsqueeze(-1))


def test_action_rate_and_joint_torque_limit_rewards_match_hdmi_formulas():
    action_buf = torch.tensor(
        [
            [[1.0, 0.5], [-1.0, -0.25]],
            [[0.0, 0.5], [2.0, 1.5]],
        ]
    )

    action_rate = reward_parity.action_rate_l2(action_buf)

    action_diff = action_buf[:, :, 0] - action_buf[:, :, 1]
    expected_action_rate = -action_diff.square().sum(dim=-1, keepdim=True)
    assert torch.allclose(action_rate, expected_action_rate)

    applied_torque = torch.tensor(
        [
            [8.0, -12.0, 2.0],
            [-15.0, 3.0, 20.0],
        ]
    )
    joint_effort_limits = torch.tensor(
        [
            [10.0, 10.0, 5.0],
            [10.0, 10.0, 10.0],
        ]
    )

    torque_limit = reward_parity.joint_torque_limits(
        applied_torque=applied_torque,
        joint_effort_limits=joint_effort_limits,
        soft_factor=0.6,
    )

    soft_limits = joint_effort_limits * 0.6
    violation_high = (applied_torque / soft_limits - 1.0).clamp_min(0.0)
    violation_low = (-applied_torque / soft_limits - 1.0).clamp_min(0.0)
    expected_torque_limit = -(violation_high + violation_low).sum(dim=1, keepdim=True)
    assert torch.allclose(torque_limit, expected_torque_limit)
