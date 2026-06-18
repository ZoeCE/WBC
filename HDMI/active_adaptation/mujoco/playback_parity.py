from dataclasses import dataclass
from typing import Any, Mapping

import torch

from . import reward_parity


@dataclass(frozen=True)
class PlaybackParityMetrics:
    q_l2: torch.Tensor
    body_pos_l2: torch.Tensor
    reward: torch.Tensor | None = None


@dataclass(frozen=True)
class MujocoRewardState:
    actual_body_pos_w: torch.Tensor | None = None
    ref_body_pos_w: torch.Tensor | None = None
    root_pos_w: torch.Tensor | None = None
    root_quat_w: torch.Tensor | None = None
    ref_root_pos_w: torch.Tensor | None = None
    ref_root_quat_w: torch.Tensor | None = None
    joint_pos: torch.Tensor | None = None
    ref_joint_pos: torch.Tensor | None = None
    object_pos_w: torch.Tensor | None = None
    ref_object_pos_w: torch.Tensor | None = None
    object_quat_w: torch.Tensor | None = None
    ref_object_quat_w: torch.Tensor | None = None
    object_joint_pos: torch.Tensor | None = None
    ref_object_joint_pos: torch.Tensor | None = None
    contact_eef_pos_w: torch.Tensor | None = None
    contact_target_pos_w: torch.Tensor | None = None
    eef_contact_forces_b: torch.Tensor | None = None
    ref_object_contact: torch.Tensor | None = None


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


def compute_reward_from_spec(reward_cfg: Mapping[str, Any], state: MujocoRewardState) -> torch.Tensor:
    """Compute HDMI-style reward groups from MuJoCo playback tensors."""
    group_rewards: list[torch.Tensor] = []
    for group_name, group_cfg in reward_cfg.items():
        if group_name == "_mult_dt_":
            continue
        if not isinstance(group_cfg, Mapping):
            raise ValueError(f"Reward group {group_name!r} must be a mapping, got {type(group_cfg).__name__}.")

        multiplicative = bool(group_cfg.get("_multiplicative", False))
        term_rewards: list[torch.Tensor] = []
        for raw_term_name, raw_params in group_cfg.items():
            if raw_term_name == "_multiplicative" or raw_params is None:
                continue
            params = dict(raw_params)
            if not bool(params.pop("enabled", True)):
                continue

            weight = float(params.pop("weight", 1.0))
            term_name = _formula_name(raw_term_name)
            term_reward = _compute_reward_term(term_name, params, state)
            term_rewards.append(term_reward * weight)

        if not term_rewards:
            continue
        terms = torch.cat(term_rewards, dim=1)
        if multiplicative:
            group_rewards.append(terms.prod(dim=1, keepdim=True))
        else:
            group_rewards.append(terms.sum(dim=1, keepdim=True))

    if not group_rewards:
        raise ValueError("reward_cfg did not contain any enabled MuJoCo-supported reward terms.")
    return torch.cat(group_rewards, dim=1)


def _formula_name(term_name: str) -> str:
    if "(" not in term_name:
        return term_name
    if not term_name.endswith(")"):
        raise ValueError(f"Invalid reward term alias syntax: {term_name!r}.")
    return term_name.rsplit("(", 1)[1][:-1]


def _compute_reward_term(term_name: str, params: dict[str, Any], state: MujocoRewardState) -> torch.Tensor:
    if term_name in ("keypoint_pos_tracking_product", "keypoint_position_tracking_product"):
        return reward_parity.keypoint_position_tracking_product(
            actual_body_pos_w=_required_state_tensor(state, "actual_body_pos_w"),
            ref_body_pos_w=_required_state_tensor(state, "ref_body_pos_w"),
            sigma=float(params.pop("sigma", 0.03)),
            tolerance=params.pop("tolerance", 0.0),
        )
    if term_name in ("keypoint_pos_tracking_local_product", "keypoint_position_tracking_local_product"):
        return reward_parity.keypoint_position_tracking_product(
            actual_body_pos_w=_required_state_tensor(state, "actual_body_pos_w"),
            ref_body_pos_w=_required_state_tensor(state, "ref_body_pos_w"),
            sigma=float(params.pop("sigma", 0.03)),
            tolerance=params.pop("tolerance", 0.0),
            local=True,
            root_pos_w=_required_state_tensor(state, "root_pos_w"),
            root_quat_w=_required_state_tensor(state, "root_quat_w"),
            ref_root_pos_w=_required_state_tensor(state, "ref_root_pos_w"),
            ref_root_quat_w=_required_state_tensor(state, "ref_root_quat_w"),
        )
    if term_name in ("joint_pos_tracking_product", "joint_position_tracking_product"):
        return reward_parity.joint_position_tracking_product(
            joint_pos=_required_state_tensor(state, "joint_pos"),
            ref_joint_pos=_required_state_tensor(state, "ref_joint_pos"),
            sigma=float(params.pop("sigma", 0.03)),
            tolerance=params.pop("tolerance", 0.0),
        )
    if term_name == "object_pos_tracking":
        return reward_parity.object_position_tracking(
            object_pos_w=_required_state_tensor(state, "object_pos_w"),
            ref_object_pos_w=_required_state_tensor(state, "ref_object_pos_w"),
            sigma=float(params.pop("sigma", 0.25)),
        )
    if term_name == "object_ori_tracking":
        return reward_parity.object_orientation_tracking(
            object_quat_w=_required_state_tensor(state, "object_quat_w"),
            ref_object_quat_w=_required_state_tensor(state, "ref_object_quat_w"),
            sigma=float(params.pop("sigma", 0.25)),
        )
    if term_name == "object_joint_pos_tracking":
        return reward_parity.object_joint_position_tracking(
            object_joint_pos=_required_state_tensor(state, "object_joint_pos"),
            ref_object_joint_pos=_required_state_tensor(state, "ref_object_joint_pos"),
            sigma=float(params.pop("sigma", 0.25)),
        )
    if term_name == "eef_contact_exp":
        return reward_parity.eef_contact_exp(
            contact_eef_pos_w=_required_state_tensor(state, "contact_eef_pos_w"),
            contact_target_pos_w=_required_state_tensor(state, "contact_target_pos_w"),
            eef_contact_forces_b=_required_state_tensor(state, "eef_contact_forces_b"),
            ref_object_contact=_required_state_tensor(state, "ref_object_contact"),
            pos_sigma=float(params.pop("pos_sigma", 0.1)),
            pos_tolerance=float(params.pop("pos_tolerance", 0.0)),
            frc_sigma=float(params.pop("frc_sigma", 10.0)),
            frc_thres=params.pop("frc_thres", 2.0),
            gain=float(params.pop("gain", 1.0)),
        )
    if term_name == "eef_contact_exp_max":
        return reward_parity.eef_contact_exp_max(
            contact_eef_pos_w=_required_state_tensor(state, "contact_eef_pos_w"),
            contact_target_pos_w=_required_state_tensor(state, "contact_target_pos_w"),
            eef_contact_forces_b=_required_state_tensor(state, "eef_contact_forces_b"),
            ref_object_contact=_required_state_tensor(state, "ref_object_contact"),
            pos_sigma=float(params.pop("pos_sigma", 0.1)),
            pos_tolerance=float(params.pop("pos_tolerance", 0.0)),
            frc_sigma=float(params.pop("frc_sigma", 10.0)),
            frc_thres=params.pop("frc_thres", 2.0),
        )
    if term_name == "eef_contact_all":
        return reward_parity.eef_contact_all(
            contact_eef_pos_w=_required_state_tensor(state, "contact_eef_pos_w"),
            contact_target_pos_w=_required_state_tensor(state, "contact_target_pos_w"),
            eef_contact_forces_b=_required_state_tensor(state, "eef_contact_forces_b"),
            ref_object_contact=_required_state_tensor(state, "ref_object_contact"),
            pos_thres=float(params.pop("pos_thres", 0.1)),
            frc_thres=params.pop("frc_thres", 2.0),
            gain=float(params.pop("gain", 1.0)),
        )
    raise NotImplementedError(f"Unsupported MuJoCo reward parity term {term_name!r}.")


def _required_state_tensor(state: MujocoRewardState, name: str) -> torch.Tensor:
    value = getattr(state, name)
    if value is None:
        raise ValueError(f"MujocoRewardState.{name} is required for this reward term.")
    return value


def _require_same_shape(lhs_name: str, lhs: torch.Tensor, rhs_name: str, rhs: torch.Tensor) -> None:
    if lhs.shape != rhs.shape:
        raise ValueError(f"{lhs_name} shape {tuple(lhs.shape)} != {rhs_name} shape {tuple(rhs.shape)}.")
