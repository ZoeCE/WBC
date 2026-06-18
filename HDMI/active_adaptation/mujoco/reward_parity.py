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


def _quat_rotate_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    quat = quat.expand(*vec.shape[:-1], 4)
    xyz = quat[..., 1:]
    t = torch.linalg.cross(xyz, vec, dim=-1) * 2
    return vec - quat[..., 0:1] * t + torch.linalg.cross(xyz, t, dim=-1)
