from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PlaybackParityMetrics:
    q_l2: torch.Tensor
    body_pos_l2: torch.Tensor
    reward: torch.Tensor | None = None


def compute_playback_parity(
    q_mujoco: torch.Tensor,
    q_ref: torch.Tensor,
    body_pos_mujoco_w: torch.Tensor,
    body_pos_ref_w: torch.Tensor,
    reward: torch.Tensor | None = None,
) -> PlaybackParityMetrics:
    _require_same_shape("q_mujoco", q_mujoco, "q_ref", q_ref)
    _require_same_shape("body_pos_mujoco_w", body_pos_mujoco_w, "body_pos_ref_w", body_pos_ref_w)
    if body_pos_mujoco_w.shape[-1] != 3:
        raise ValueError(f"body position tensors must end in xyz dim 3, got {body_pos_mujoco_w.shape}.")

    q_l2 = torch.linalg.vector_norm(q_mujoco - q_ref, dim=-1)
    body_pos_delta = (body_pos_mujoco_w - body_pos_ref_w).reshape(body_pos_mujoco_w.shape[0], -1)
    body_pos_l2 = torch.linalg.vector_norm(body_pos_delta, dim=-1)
    if reward is not None and reward.ndim == 0:
        reward = reward.unsqueeze(0)
    return PlaybackParityMetrics(q_l2=q_l2, body_pos_l2=body_pos_l2, reward=reward)


def _require_same_shape(lhs_name: str, lhs: torch.Tensor, rhs_name: str, rhs: torch.Tensor) -> None:
    if lhs.shape != rhs.shape:
        raise ValueError(f"{lhs_name} shape {tuple(lhs.shape)} != {rhs_name} shape {tuple(rhs.shape)}.")
