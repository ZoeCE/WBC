from typing import Sequence

import torch


def keypoint_position_tracking_product(
    actual_body_pos_w: torch.Tensor,
    ref_body_pos_w: torch.Tensor,
    sigma: float = 0.03,
    tolerance: float | Sequence[float] | torch.Tensor = 0.0,
    *,
    local: bool = False,
    root_pos_w: torch.Tensor | None = None,
    root_quat_w: torch.Tensor | None = None,
    ref_root_pos_w: torch.Tensor | None = None,
    ref_root_quat_w: torch.Tensor | None = None,
) -> torch.Tensor:
    """MuJoCo tensor parity for HDMI keypoint position product rewards."""
    _require_same_shape("actual_body_pos_w", actual_body_pos_w, "ref_body_pos_w", ref_body_pos_w)
    _require_last_dim("actual_body_pos_w", actual_body_pos_w, 3)
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")

    if local:
        actual_body_pos_w = _body_pos_in_yaw_local_frame(
            body_pos_w=actual_body_pos_w,
            root_pos_w=_required_tensor("root_pos_w", root_pos_w),
            root_quat_w=_required_tensor("root_quat_w", root_quat_w),
        )
        ref_body_pos_w = _body_pos_in_yaw_local_frame(
            body_pos_w=ref_body_pos_w,
            root_pos_w=_required_tensor("ref_root_pos_w", ref_root_pos_w),
            root_quat_w=_required_tensor("ref_root_quat_w", ref_root_quat_w),
        )

    tolerance_t = _tolerance_tensor(tolerance, actual_body_pos_w)
    error = ((ref_body_pos_w - actual_body_pos_w).norm(dim=-1) - tolerance_t).clamp_min(0.0)
    return torch.exp(-error.mean(dim=1) / sigma).unsqueeze(1)


def joint_position_tracking_product(
    joint_pos: torch.Tensor,
    ref_joint_pos: torch.Tensor,
    sigma: float = 0.03,
    tolerance: float | Sequence[float] | torch.Tensor = 0.0,
) -> torch.Tensor:
    """MuJoCo tensor parity for HDMI joint position product rewards."""
    _require_same_shape("joint_pos", joint_pos, "ref_joint_pos", ref_joint_pos)
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")

    tolerance_t = _tolerance_tensor(tolerance, joint_pos)
    error = ((ref_joint_pos - joint_pos).abs() - tolerance_t).clamp_min(0.0)
    return torch.exp(-error.mean(dim=1) / sigma).unsqueeze(1)


def joint_velocity_tracking_product(
    joint_vel: torch.Tensor,
    ref_joint_vel: torch.Tensor,
    sigma: float = 0.03,
    tolerance: float | Sequence[float] | torch.Tensor = 0.0,
) -> torch.Tensor:
    """MuJoCo tensor parity for HDMI joint velocity product rewards."""
    _require_same_shape("joint_vel", joint_vel, "ref_joint_vel", ref_joint_vel)
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")

    tolerance_t = _tolerance_tensor(tolerance, joint_vel)
    error = ((ref_joint_vel - joint_vel).abs() - tolerance_t).clamp_min(0.0)
    return torch.exp(-error.mean(dim=1) / sigma).unsqueeze(1)


def keypoint_orientation_tracking_product(
    actual_body_quat_w: torch.Tensor,
    ref_body_quat_w: torch.Tensor,
    sigma: float = 0.03,
    tolerance: float | Sequence[float] | torch.Tensor = 0.0,
    *,
    local: bool = False,
    root_quat_w: torch.Tensor | None = None,
    ref_root_quat_w: torch.Tensor | None = None,
) -> torch.Tensor:
    """MuJoCo tensor parity for HDMI keypoint orientation product rewards."""
    _require_same_shape("actual_body_quat_w", actual_body_quat_w, "ref_body_quat_w", ref_body_quat_w)
    _require_last_dim("actual_body_quat_w", actual_body_quat_w, 4)
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")

    if local:
        root_quat = _yaw_quat(_required_tensor("root_quat_w", root_quat_w))[:, None, :].expand_as(actual_body_quat_w)
        ref_root_quat = _yaw_quat(_required_tensor("ref_root_quat_w", ref_root_quat_w))[:, None, :].expand_as(ref_body_quat_w)
        actual_body_quat_w = _quat_mul(_quat_conjugate(root_quat), actual_body_quat_w)
        ref_body_quat_w = _quat_mul(_quat_conjugate(ref_root_quat), ref_body_quat_w)

    diff = _quat_mul(_quat_conjugate(ref_body_quat_w), actual_body_quat_w)
    tolerance_t = _tolerance_tensor(tolerance, actual_body_quat_w)
    error = (_axis_angle_from_quat(diff).norm(dim=-1) - tolerance_t).clamp_min(0.0)
    return torch.exp(-error.mean(dim=1) / sigma).unsqueeze(1)


def keypoint_velocity_tracking_product(
    actual_body_vel_w: torch.Tensor,
    ref_body_vel_w: torch.Tensor,
    sigma: float = 0.03,
    tolerance: float | Sequence[float] | torch.Tensor = 0.0,
) -> torch.Tensor:
    """MuJoCo tensor parity for HDMI keypoint linear/angular velocity product rewards."""
    _require_same_shape("actual_body_vel_w", actual_body_vel_w, "ref_body_vel_w", ref_body_vel_w)
    _require_last_dim("actual_body_vel_w", actual_body_vel_w, 3)
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")

    tolerance_t = _tolerance_tensor(tolerance, actual_body_vel_w)
    error = ((ref_body_vel_w - actual_body_vel_w).norm(dim=-1) - tolerance_t).clamp_min(0.0)
    return torch.exp(-error.mean(dim=1) / sigma).unsqueeze(1)


def object_position_tracking(
    object_pos_w: torch.Tensor,
    ref_object_pos_w: torch.Tensor,
    sigma: float = 0.25,
) -> torch.Tensor:
    """MuJoCo tensor parity for HDMI object_pos_tracking rewards."""
    _require_same_shape("object_pos_w", object_pos_w, "ref_object_pos_w", ref_object_pos_w)
    _require_last_dim("object_pos_w", object_pos_w, 3)
    if object_pos_w.ndim != 2:
        raise ValueError(f"object_pos_w must have shape (num_envs, 3), got {tuple(object_pos_w.shape)}.")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")

    object_pos_error = (ref_object_pos_w - object_pos_w).norm(dim=-1)
    return torch.exp(-object_pos_error / sigma).unsqueeze(1)


def object_orientation_tracking(
    object_quat_w: torch.Tensor,
    ref_object_quat_w: torch.Tensor,
    sigma: float = 0.25,
) -> torch.Tensor:
    """MuJoCo tensor parity for HDMI object_ori_tracking rewards."""
    _require_same_shape("object_quat_w", object_quat_w, "ref_object_quat_w", ref_object_quat_w)
    _require_last_dim("object_quat_w", object_quat_w, 4)
    if object_quat_w.ndim != 2:
        raise ValueError(f"object_quat_w must have shape (num_envs, 4), got {tuple(object_quat_w.shape)}.")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")

    object_diff_quat = _quat_mul(_quat_conjugate(ref_object_quat_w), object_quat_w)
    object_ori_error = torch.norm(_axis_angle_from_quat(object_diff_quat), dim=-1)
    return torch.exp(-object_ori_error / sigma).unsqueeze(1)


def object_joint_position_tracking(
    object_joint_pos: torch.Tensor,
    ref_object_joint_pos: torch.Tensor,
    sigma: float = 0.25,
) -> torch.Tensor:
    """MuJoCo tensor parity for HDMI object_joint_pos_tracking rewards."""
    _require_same_shape("object_joint_pos", object_joint_pos, "ref_object_joint_pos", ref_object_joint_pos)
    if object_joint_pos.ndim != 1:
        raise ValueError(
            f"object_joint_pos must have shape (num_envs,) for HDMI single-object-joint rewards, "
            f"got {tuple(object_joint_pos.shape)}."
        )
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")

    joint_pos_error = (ref_object_joint_pos - object_joint_pos).abs()
    return torch.exp(-joint_pos_error / sigma).unsqueeze(1)


def eef_contact_exp(
    contact_eef_pos_w: torch.Tensor,
    contact_target_pos_w: torch.Tensor,
    eef_contact_forces_b: torch.Tensor,
    ref_object_contact: torch.Tensor,
    pos_sigma: float = 0.1,
    pos_tolerance: float = 0.0,
    frc_sigma: float = 10.0,
    frc_thres: float | Sequence[float] | torch.Tensor = 2.0,
    gain: float = 1.0,
) -> torch.Tensor:
    """MuJoCo tensor parity for HDMI eef_contact_exp reward."""
    _require_same_shape("contact_eef_pos_w", contact_eef_pos_w, "contact_target_pos_w", contact_target_pos_w)
    _require_same_shape("contact_eef_pos_w", contact_eef_pos_w, "eef_contact_forces_b", eef_contact_forces_b)
    _require_last_dim("contact_eef_pos_w", contact_eef_pos_w, 3)
    if ref_object_contact.shape != contact_eef_pos_w.shape[:-1]:
        raise ValueError(
            "ref_object_contact shape "
            f"{tuple(ref_object_contact.shape)} != contact tensors env/eef shape {tuple(contact_eef_pos_w.shape[:-1])}."
        )
    if pos_sigma <= 0:
        raise ValueError(f"pos_sigma must be positive, got {pos_sigma}.")
    if frc_sigma <= 0:
        raise ValueError(f"frc_sigma must be positive, got {frc_sigma}.")

    eef_pos_error = ((contact_eef_pos_w - contact_target_pos_w).norm(dim=-1) - pos_tolerance).clamp_min(0.0)
    contact_frc = _contact_force_penalty(eef_contact_forces_b, frc_thres)
    active_reward = torch.exp(-eef_pos_error / pos_sigma) * torch.exp(contact_frc / frc_sigma)
    contact_mask = ref_object_contact.to(dtype=active_reward.dtype)
    reward = active_reward * contact_mask * gain + 1 - contact_mask
    return reward.mean(dim=-1).unsqueeze(-1)


def eef_contact_exp_max(
    contact_eef_pos_w: torch.Tensor,
    contact_target_pos_w: torch.Tensor,
    eef_contact_forces_b: torch.Tensor,
    ref_object_contact: torch.Tensor,
    pos_sigma: float = 0.1,
    pos_tolerance: float = 0.0,
    frc_sigma: float = 10.0,
    frc_thres: float | Sequence[float] | torch.Tensor = 2.0,
) -> torch.Tensor:
    """MuJoCo tensor parity for HDMI eef_contact_exp_max rewards."""
    _require_contact_reward_shapes(
        contact_eef_pos_w=contact_eef_pos_w,
        contact_target_pos_w=contact_target_pos_w,
        eef_contact_forces_b=eef_contact_forces_b,
        ref_object_contact=ref_object_contact,
    )
    if pos_sigma <= 0:
        raise ValueError(f"pos_sigma must be positive, got {pos_sigma}.")
    if frc_sigma <= 0:
        raise ValueError(f"frc_sigma must be positive, got {frc_sigma}.")

    eef_pos_error = ((contact_eef_pos_w - contact_target_pos_w).norm(dim=-1) - pos_tolerance).clamp_min(0.0)
    contact_frc = _contact_force_penalty(eef_contact_forces_b, frc_thres)
    active_reward = torch.exp(-eef_pos_error / pos_sigma) * torch.exp(contact_frc / frc_sigma)
    reward = active_reward.max(dim=-1).values * ref_object_contact.any(dim=-1).to(dtype=active_reward.dtype)
    return reward.unsqueeze(-1)


def eef_contact_all(
    contact_eef_pos_w: torch.Tensor,
    contact_target_pos_w: torch.Tensor,
    eef_contact_forces_b: torch.Tensor,
    ref_object_contact: torch.Tensor,
    pos_thres: float = 0.1,
    frc_thres: float | Sequence[float] | torch.Tensor = 2.0,
    gain: float = 1.0,
) -> torch.Tensor:
    """MuJoCo tensor parity for HDMI eef_contact_all rewards."""
    _require_contact_reward_shapes(
        contact_eef_pos_w=contact_eef_pos_w,
        contact_target_pos_w=contact_target_pos_w,
        eef_contact_forces_b=eef_contact_forces_b,
        ref_object_contact=ref_object_contact,
    )

    contact_pos = (contact_eef_pos_w - contact_target_pos_w).norm(dim=-1) < pos_thres
    contact_frc = _contact_force_mask(eef_contact_forces_b, frc_thres)
    active_reward = (contact_pos & contact_frc).to(dtype=contact_eef_pos_w.dtype)
    contact_mask = ref_object_contact.to(dtype=active_reward.dtype)
    reward = active_reward * contact_mask * gain + 1 - contact_mask
    return reward.mean(dim=-1).unsqueeze(-1)


def _body_pos_in_yaw_local_frame(
    body_pos_w: torch.Tensor,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
) -> torch.Tensor:
    _require_last_dim("root_pos_w", root_pos_w, 3)
    _require_last_dim("root_quat_w", root_quat_w, 4)
    if root_pos_w.shape[0] != body_pos_w.shape[0]:
        raise ValueError(f"root_pos_w batch {root_pos_w.shape[0]} != body_pos_w batch {body_pos_w.shape[0]}.")
    if root_quat_w.shape[0] != body_pos_w.shape[0]:
        raise ValueError(f"root_quat_w batch {root_quat_w.shape[0]} != body_pos_w batch {body_pos_w.shape[0]}.")

    root_pos_w = root_pos_w[:, None, :].clone()
    root_pos_w[..., 2] = 0.0
    root_quat_w = _yaw_quat(root_quat_w)[:, None, :]
    return _quat_rotate_inverse(root_quat_w, body_pos_w - root_pos_w)


def _contact_force_penalty(
    eef_contact_forces_b: torch.Tensor,
    frc_thres: float | Sequence[float] | torch.Tensor,
) -> torch.Tensor:
    if isinstance(frc_thres, (float, int)):
        return (eef_contact_forces_b.norm(dim=-1) - float(frc_thres)).clamp_max(0.0)

    threshold = torch.as_tensor(frc_thres, dtype=eef_contact_forces_b.dtype, device=eef_contact_forces_b.device)
    if threshold.shape != (3,):
        raise ValueError(f"vector frc_thres must have shape (3,), got {tuple(threshold.shape)}.")
    return (eef_contact_forces_b.abs() - threshold).clamp_max(0.0).mean(dim=-1)


def _contact_force_mask(
    eef_contact_forces_b: torch.Tensor,
    frc_thres: float | Sequence[float] | torch.Tensor,
) -> torch.Tensor:
    if isinstance(frc_thres, (float, int)):
        return eef_contact_forces_b.norm(dim=-1) >= float(frc_thres)

    threshold = torch.as_tensor(frc_thres, dtype=eef_contact_forces_b.dtype, device=eef_contact_forces_b.device)
    if threshold.shape != (3,):
        raise ValueError(f"vector frc_thres must have shape (3,), got {tuple(threshold.shape)}.")
    return (eef_contact_forces_b.abs() >= threshold).all(dim=-1)


def _require_contact_reward_shapes(
    contact_eef_pos_w: torch.Tensor,
    contact_target_pos_w: torch.Tensor,
    eef_contact_forces_b: torch.Tensor,
    ref_object_contact: torch.Tensor,
) -> None:
    _require_same_shape("contact_eef_pos_w", contact_eef_pos_w, "contact_target_pos_w", contact_target_pos_w)
    _require_same_shape("contact_eef_pos_w", contact_eef_pos_w, "eef_contact_forces_b", eef_contact_forces_b)
    _require_last_dim("contact_eef_pos_w", contact_eef_pos_w, 3)
    if ref_object_contact.shape != contact_eef_pos_w.shape[:-1]:
        raise ValueError(
            "ref_object_contact shape "
            f"{tuple(ref_object_contact.shape)} != contact tensors env/eef shape {tuple(contact_eef_pos_w.shape[:-1])}."
        )


def _tolerance_tensor(
    tolerance: float | Sequence[float] | torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    tolerance_t = torch.as_tensor(tolerance, dtype=reference.dtype, device=reference.device)
    if tolerance_t.ndim == 0:
        return tolerance_t
    if tolerance_t.shape != reference.shape[1:-1] and tolerance_t.shape != reference.shape[1:]:
        raise ValueError(
            f"tolerance shape {tuple(tolerance_t.shape)} does not match term shape "
            f"{tuple(reference.shape[1:-1])} or {tuple(reference.shape[1:])}."
        )
    return tolerance_t


def _required_tensor(name: str, value: torch.Tensor | None) -> torch.Tensor:
    if value is None:
        raise ValueError(f"{name} is required for local reward parity.")
    return value


def _require_same_shape(lhs_name: str, lhs: torch.Tensor, rhs_name: str, rhs: torch.Tensor) -> None:
    if lhs.shape != rhs.shape:
        raise ValueError(f"{lhs_name} shape {tuple(lhs.shape)} != {rhs_name} shape {tuple(rhs.shape)}.")


def _require_last_dim(name: str, tensor: torch.Tensor, dim: int) -> None:
    if tensor.shape[-1] != dim:
        raise ValueError(f"{name} last dim {tensor.shape[-1]} != {dim}.")


def _yaw_quat(quat: torch.Tensor) -> torch.Tensor:
    qw, qx, qy, qz = torch.unbind(quat, dim=-1)
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    zeros = torch.zeros_like(yaw)
    return torch.stack((torch.cos(yaw / 2), zeros, zeros, torch.sin(yaw / 2)), dim=-1)


def _quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat((quat[..., 0:1], -quat[..., 1:]), dim=-1)


def _quat_mul(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    if lhs.shape != rhs.shape:
        raise ValueError(f"Quaternion shape mismatch: {tuple(lhs.shape)} != {tuple(rhs.shape)}.")

    w1, x1, y1, z1 = torch.unbind(lhs, dim=-1)
    w2, x2, y2, z2 = torch.unbind(rhs, dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _axis_angle_from_quat(quat: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    quat = torch.where(quat[..., 0:1] < 0.0, -quat, quat)
    mag = torch.linalg.norm(quat[..., 1:], dim=-1)
    half_angle = torch.atan2(mag, quat[..., 0])
    angle = 2.0 * half_angle
    sin_half_angles_over_angles = torch.where(
        angle.abs() > eps,
        torch.sin(half_angle) / angle,
        0.5 - angle * angle / 48,
    )
    return quat[..., 1:4] / sin_half_angles_over_angles.unsqueeze(-1)


def _quat_rotate_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    quat = quat.expand(*vec.shape[:-1], 4)
    xyz = quat[..., 1:]
    t = torch.linalg.cross(xyz, vec, dim=-1) * 2
    return vec - quat[..., 0:1] * t + torch.linalg.cross(xyz, t, dim=-1)
