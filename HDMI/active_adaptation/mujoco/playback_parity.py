import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

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
    actual_body_quat_w: torch.Tensor | None = None
    ref_body_quat_w: torch.Tensor | None = None
    actual_body_lin_vel_w: torch.Tensor | None = None
    ref_body_lin_vel_w: torch.Tensor | None = None
    actual_body_ang_vel_w: torch.Tensor | None = None
    ref_body_ang_vel_w: torch.Tensor | None = None
    body_names: Sequence[str] | None = None
    root_pos_w: torch.Tensor | None = None
    root_quat_w: torch.Tensor | None = None
    ref_root_pos_w: torch.Tensor | None = None
    ref_root_quat_w: torch.Tensor | None = None
    joint_pos: torch.Tensor | None = None
    ref_joint_pos: torch.Tensor | None = None
    joint_vel: torch.Tensor | None = None
    ref_joint_vel: torch.Tensor | None = None
    joint_pos_limits: torch.Tensor | None = None
    joint_names: Sequence[str] | None = None
    applied_torque: torch.Tensor | None = None
    joint_effort_limits: torch.Tensor | None = None
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
    contact_body_names: Sequence[str] | None = None
    contact_current_contact_time: torch.Tensor | None = None
    contact_last_air_time: torch.Tensor | None = None
    contact_first_contact: torch.Tensor | None = None
    contact_net_forces_w_history: torch.Tensor | None = None
    default_mass_total: torch.Tensor | None = None
    is_standing_env: torch.Tensor | None = None
    action_buf: torch.Tensor | None = None


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
    body_pos_delta = (body_pos_mujoco_w - body_pos_ref_w).flatten(start_dim=-2)
    body_pos_l2 = torch.linalg.vector_norm(body_pos_delta, dim=-1)
    if reward is not None and reward.ndim == 0:
        reward = reward.unsqueeze(0)
    return PlaybackParityMetrics(q_l2=q_l2, body_pos_l2=body_pos_l2, reward=reward)


def build_reward_state_from_scene(
    scene: Any,
    *,
    body_names: str | Sequence[str] | None = None,
    joint_names: str | Sequence[str] | None = None,
    ref_body_pos_w: torch.Tensor | None = None,
    ref_body_quat_w: torch.Tensor | None = None,
    ref_body_lin_vel_w: torch.Tensor | None = None,
    ref_body_ang_vel_w: torch.Tensor | None = None,
    ref_joint_pos: torch.Tensor | None = None,
    ref_joint_vel: torch.Tensor | None = None,
    ref_root_pos_w: torch.Tensor | None = None,
    ref_root_quat_w: torch.Tensor | None = None,
    object_name: str | None = None,
    object_body_name: str | None = None,
    object_joint_name: str | None = None,
    ref_object_pos_w: torch.Tensor | None = None,
    ref_object_quat_w: torch.Tensor | None = None,
    ref_object_joint_pos: torch.Tensor | None = None,
    contact_eef_pos_w: torch.Tensor | None = None,
    contact_target_pos_w: torch.Tensor | None = None,
    eef_contact_forces_b: torch.Tensor | None = None,
    ref_object_contact: torch.Tensor | None = None,
) -> MujocoRewardState:
    """Build reward tensors from the current MuJoCo scene state and a reference frame."""
    robot = scene["robot"]
    if robot is None:
        raise KeyError("MuJoCo scene does not contain a 'robot' articulation.")

    ref_body_pos_w = _current_reference_frame(ref_body_pos_w, state_rank=3)
    ref_body_quat_w = _current_reference_frame(ref_body_quat_w, state_rank=3)
    ref_body_lin_vel_w = _current_reference_frame(ref_body_lin_vel_w, state_rank=3)
    ref_body_ang_vel_w = _current_reference_frame(ref_body_ang_vel_w, state_rank=3)
    ref_joint_pos = _current_reference_frame(ref_joint_pos, state_rank=2)
    ref_joint_vel = _current_reference_frame(ref_joint_vel, state_rank=2)
    ref_root_pos_w = _current_reference_frame(ref_root_pos_w, state_rank=2)
    ref_root_quat_w = _current_reference_frame(ref_root_quat_w, state_rank=2)
    ref_object_pos_w = _current_reference_frame(ref_object_pos_w, state_rank=2)
    ref_object_quat_w = _current_reference_frame(ref_object_quat_w, state_rank=2)
    ref_object_joint_pos = _current_reference_frame(ref_object_joint_pos, state_rank=1)
    contact_eef_pos_w = _current_reference_frame(contact_eef_pos_w, state_rank=3)
    contact_target_pos_w = _current_reference_frame(contact_target_pos_w, state_rank=3)
    eef_contact_forces_b = _current_reference_frame(eef_contact_forces_b, state_rank=3)
    ref_object_contact = _current_reference_frame(ref_object_contact, state_rank=2)

    actual_body_pos_w = None
    actual_body_quat_w = None
    actual_body_lin_vel_w = None
    actual_body_ang_vel_w = None
    resolved_body_names = None
    if (
        body_names is not None
        or ref_body_pos_w is not None
        or ref_body_quat_w is not None
        or ref_body_lin_vel_w is not None
        or ref_body_ang_vel_w is not None
    ):
        body_indices = None
        if body_names is not None:
            body_indices, resolved_body_names = _resolve_indices_and_names(robot, "find_bodies", body_names)
        if body_indices is None:
            actual_body_pos_w = robot.data.body_link_pos_w
            actual_body_quat_w = robot.data.body_link_quat_w
            actual_body_lin_vel_w = robot.data.body_com_lin_vel_w
            actual_body_ang_vel_w = robot.data.body_com_ang_vel_w
            resolved_body_names = list(robot.body_names)
        else:
            actual_body_pos_w = robot.data.body_link_pos_w[:, body_indices]
            actual_body_quat_w = robot.data.body_link_quat_w[:, body_indices]
            actual_body_lin_vel_w = robot.data.body_com_lin_vel_w[:, body_indices]
            actual_body_ang_vel_w = robot.data.body_com_ang_vel_w[:, body_indices]

    joint_pos = None
    joint_vel = None
    joint_pos_limits = None
    resolved_joint_names = None
    applied_torque = None
    joint_effort_limits = None
    if joint_names is not None or ref_joint_pos is not None or ref_joint_vel is not None:
        joint_indices = None
        if joint_names is not None:
            joint_indices, resolved_joint_names = _resolve_indices_and_names(robot, "find_joints", joint_names)
        if joint_indices is None:
            joint_pos = robot.data.joint_pos
            joint_vel = robot.data.joint_vel
            joint_pos_limits = robot.data.soft_joint_pos_limits
            resolved_joint_names = list(robot.joint_names)
            applied_torque = robot.data.applied_torque
            joint_effort_limits = robot.data.joint_effort_limits
        else:
            joint_pos = robot.data.joint_pos[:, joint_indices]
            joint_vel = robot.data.joint_vel[:, joint_indices]
            joint_pos_limits = robot.data.soft_joint_pos_limits[:, joint_indices]
            applied_torque = robot.data.applied_torque[:, joint_indices]
            joint_effort_limits = robot.data.joint_effort_limits[:, joint_indices]

    object_pos_w = None
    object_quat_w = None
    object_joint_pos = None
    if object_name is not None:
        object_view = scene[object_name]
        if object_view is None:
            raise KeyError(f"MuJoCo scene does not contain object {object_name!r}.")

        body_index = 0 if object_body_name is None else _resolve_single_index(
            object_view, "find_bodies", object_body_name, "object body"
        )
        object_pos_w = object_view.data.body_link_pos_w[:, body_index]
        object_quat_w = object_view.data.body_link_quat_w[:, body_index]

        if object_view.num_joints:
            joint_index = 0 if object_joint_name is None else _resolve_single_index(
                object_view, "find_joints", object_joint_name, "object joint"
            )
            object_joint_pos = object_view.data.joint_pos[:, joint_index]
    elif ref_object_pos_w is not None or ref_object_quat_w is not None or ref_object_joint_pos is not None:
        raise ValueError("object_name is required when object reference tensors are provided.")

    return MujocoRewardState(
        actual_body_pos_w=actual_body_pos_w,
        ref_body_pos_w=ref_body_pos_w,
        actual_body_quat_w=actual_body_quat_w,
        ref_body_quat_w=ref_body_quat_w,
        actual_body_lin_vel_w=actual_body_lin_vel_w,
        ref_body_lin_vel_w=ref_body_lin_vel_w,
        actual_body_ang_vel_w=actual_body_ang_vel_w,
        ref_body_ang_vel_w=ref_body_ang_vel_w,
        body_names=resolved_body_names,
        root_pos_w=robot.data.root_link_pos_w,
        root_quat_w=robot.data.root_link_quat_w,
        ref_root_pos_w=ref_root_pos_w,
        ref_root_quat_w=ref_root_quat_w,
        joint_pos=joint_pos,
        ref_joint_pos=ref_joint_pos,
        joint_vel=joint_vel,
        ref_joint_vel=ref_joint_vel,
        joint_pos_limits=joint_pos_limits,
        joint_names=resolved_joint_names,
        applied_torque=applied_torque,
        joint_effort_limits=joint_effort_limits,
        object_pos_w=object_pos_w,
        ref_object_pos_w=ref_object_pos_w,
        object_quat_w=object_quat_w,
        ref_object_quat_w=ref_object_quat_w,
        object_joint_pos=object_joint_pos,
        ref_object_joint_pos=ref_object_joint_pos,
        contact_eef_pos_w=contact_eef_pos_w,
        contact_target_pos_w=contact_target_pos_w,
        eef_contact_forces_b=eef_contact_forces_b,
        ref_object_contact=ref_object_contact,
        **_contact_reward_state(scene, robot),
    )


def compute_kinematic_motion_playback_parity(
    scene: Any,
    reference: Any,
    *,
    steps: Sequence[int] | torch.Tensor | None = None,
    reward_cfg: Mapping[str, Any] | None = None,
    object_name: str | None = None,
    object_body_name: str | None = None,
    object_joint_name: str | None = None,
    contact_eef_body_names: Sequence[str] | None = None,
    contact_target_pos_offset: Sequence[Sequence[float]] | torch.Tensor | None = None,
    contact_eef_pos_offset: Sequence[Sequence[float]] | torch.Tensor | None = None,
) -> PlaybackParityMetrics:
    """Replay reference poses through MuJoCo kinematics and report per-step parity metrics."""
    robot = scene["robot"]
    if robot is None:
        raise KeyError("MuJoCo scene does not contain a 'robot' articulation.")
    object_view = _scene_optional_asset(scene, object_name)
    steps_t = _normalize_playback_steps(steps, reference.num_steps)

    q_l2: list[torch.Tensor] = []
    body_pos_l2: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    for step in steps_t.tolist():
        _write_reference_frame_to_scene(
            scene=scene,
            reference=reference,
            step=step,
            object_view=object_view,
            object_body_name=object_body_name or object_name,
        )

        actual_joint_pos = _gather_scene_joint_pos(
            robot=robot,
            joint_names=reference.requested_joint_names,
            object_view=object_view,
        )
        ref_joint_pos = _expand_reference_frame(_reference_joint_pos(reference, step), scene.num_envs)
        actual_body_pos_w = _gather_scene_body_pos_w(
            robot=robot,
            body_names=reference.requested_body_names,
            object_view=object_view,
        )
        actual_body_quat_w = _gather_scene_body_quat_w(
            robot=robot,
            body_names=reference.requested_body_names,
            object_view=object_view,
        )
        actual_body_lin_vel_w = _gather_scene_body_lin_vel_w(
            robot=robot,
            body_names=reference.requested_body_names,
            object_view=object_view,
        )
        actual_body_ang_vel_w = _gather_scene_body_ang_vel_w(
            robot=robot,
            body_names=reference.requested_body_names,
            object_view=object_view,
        )
        ref_body_pos_w = _expand_reference_frame(_reference_body_pos_w(reference, step), scene.num_envs)
        ref_body_quat_w = _expand_reference_frame(_reference_body_quat_w(reference, step), scene.num_envs)
        ref_body_lin_vel_w = _expand_reference_frame(_reference_body_lin_vel_w(reference, step), scene.num_envs)
        ref_body_ang_vel_w = _expand_reference_frame(_reference_body_ang_vel_w(reference, step), scene.num_envs)
        actual_joint_vel = _gather_scene_joint_vel(
            robot=robot,
            joint_names=reference.requested_joint_names,
            object_view=object_view,
        )
        actual_joint_pos_limits = _gather_scene_joint_pos_limits(
            robot=robot,
            joint_names=reference.requested_joint_names,
            object_view=object_view,
        )
        ref_joint_vel = _expand_reference_frame(_reference_joint_vel(reference, step), scene.num_envs)

        reward = None
        if reward_cfg is not None:
            reward_state = _build_kinematic_reward_state(
                scene=scene,
                robot=robot,
                object_view=object_view,
                reference=reference,
                step=step,
                actual_body_pos_w=actual_body_pos_w,
                ref_body_pos_w=ref_body_pos_w,
                actual_body_quat_w=actual_body_quat_w,
                ref_body_quat_w=ref_body_quat_w,
                actual_body_lin_vel_w=actual_body_lin_vel_w,
                ref_body_lin_vel_w=ref_body_lin_vel_w,
                actual_body_ang_vel_w=actual_body_ang_vel_w,
                ref_body_ang_vel_w=ref_body_ang_vel_w,
                actual_joint_pos=actual_joint_pos,
                ref_joint_pos=ref_joint_pos,
                actual_joint_vel=actual_joint_vel,
                ref_joint_vel=ref_joint_vel,
                actual_joint_pos_limits=actual_joint_pos_limits,
                object_name=object_name,
                object_body_name=object_body_name or object_name,
                object_joint_name=object_joint_name,
                contact_eef_body_names=contact_eef_body_names,
                contact_target_pos_offset=contact_target_pos_offset,
                contact_eef_pos_offset=contact_eef_pos_offset,
            )
            reward = compute_reward_from_spec(reward_cfg, reward_state)
            rewards.append(reward)

        metrics = compute_playback_parity(
            q_mujoco=actual_joint_pos,
            q_ref=ref_joint_pos,
            body_pos_mujoco_w=actual_body_pos_w,
            body_pos_ref_w=ref_body_pos_w,
            reward=reward,
        )
        q_l2.append(metrics.q_l2)
        body_pos_l2.append(metrics.body_pos_l2)

    return PlaybackParityMetrics(
        q_l2=torch.stack(q_l2),
        body_pos_l2=torch.stack(body_pos_l2),
        reward=torch.stack(rewards) if rewards else None,
    )


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
    if term_name in ("keypoint_ori_tracking_product", "keypoint_orientation_tracking_product"):
        return reward_parity.keypoint_orientation_tracking_product(
            actual_body_quat_w=_required_state_tensor(state, "actual_body_quat_w"),
            ref_body_quat_w=_required_state_tensor(state, "ref_body_quat_w"),
            sigma=float(params.pop("sigma", 0.03)),
            tolerance=params.pop("tolerance", 0.0),
        )
    if term_name in ("keypoint_ori_tracking_local_product", "keypoint_orientation_tracking_local_product"):
        return reward_parity.keypoint_orientation_tracking_product(
            actual_body_quat_w=_required_state_tensor(state, "actual_body_quat_w"),
            ref_body_quat_w=_required_state_tensor(state, "ref_body_quat_w"),
            sigma=float(params.pop("sigma", 0.03)),
            tolerance=params.pop("tolerance", 0.0),
            local=True,
            root_quat_w=_required_state_tensor(state, "root_quat_w"),
            ref_root_quat_w=_required_state_tensor(state, "ref_root_quat_w"),
        )
    if term_name == "keypoint_lin_vel_tracking_product":
        return reward_parity.keypoint_velocity_tracking_product(
            actual_body_vel_w=_required_state_tensor(state, "actual_body_lin_vel_w"),
            ref_body_vel_w=_required_state_tensor(state, "ref_body_lin_vel_w"),
            sigma=float(params.pop("sigma", 0.03)),
            tolerance=params.pop("tolerance", 0.0),
        )
    if term_name == "keypoint_ang_vel_tracking_product":
        return reward_parity.keypoint_velocity_tracking_product(
            actual_body_vel_w=_required_state_tensor(state, "actual_body_ang_vel_w"),
            ref_body_vel_w=_required_state_tensor(state, "ref_body_ang_vel_w"),
            sigma=float(params.pop("sigma", 0.03)),
            tolerance=params.pop("tolerance", 0.0),
        )
    if term_name in ("joint_pos_tracking_product", "joint_position_tracking_product"):
        return reward_parity.joint_position_tracking_product(
            joint_pos=_required_state_tensor(state, "joint_pos"),
            ref_joint_pos=_required_state_tensor(state, "ref_joint_pos"),
            sigma=float(params.pop("sigma", 0.03)),
            tolerance=params.pop("tolerance", 0.0),
        )
    if term_name in ("joint_vel_tracking_product", "joint_velocity_tracking_product"):
        return reward_parity.joint_velocity_tracking_product(
            joint_vel=_required_state_tensor(state, "joint_vel"),
            ref_joint_vel=_required_state_tensor(state, "ref_joint_vel"),
            sigma=float(params.pop("sigma", 0.03)),
            tolerance=params.pop("tolerance", 0.0),
        )
    if term_name == "survival":
        return reward_parity.survival(_state_batch_reference(state))
    if term_name == "joint_vel_l2":
        joint_indices = _state_joint_indices(state, params.pop("joint_names", None))
        return reward_parity.joint_velocity_l2(
            joint_vel=_select_state_joint_tensor(_required_state_tensor(state, "joint_vel"), joint_indices),
        )
    if term_name == "joint_pos_limits":
        joint_indices = _state_joint_indices(state, params.pop("joint_names", None))
        return reward_parity.joint_position_limits(
            joint_pos=_select_state_joint_tensor(_required_state_tensor(state, "joint_pos"), joint_indices),
            joint_pos_limits=_select_state_joint_tensor(_required_state_tensor(state, "joint_pos_limits"), joint_indices),
            soft_factor=float(params.pop("soft_factor", 0.9)),
        )
    if term_name == "action_rate_l2":
        return reward_parity.action_rate_l2(_required_state_tensor(state, "action_buf"))
    if term_name == "joint_torque_limits":
        joint_indices = _state_joint_indices(state, params.pop("joint_names", None))
        return reward_parity.joint_torque_limits(
            applied_torque=_select_state_joint_tensor(
                _required_state_tensor(state, "applied_torque"),
                joint_indices,
            ),
            joint_effort_limits=_select_state_joint_tensor(_required_state_tensor(state, "joint_effort_limits"), joint_indices),
            soft_factor=float(params.pop("soft_factor", 0.9)),
        )
    if term_name == "feet_slip":
        body_names = _required_param(params, "body_names", term_name)
        body_indices = _state_body_indices(state, body_names)
        contact_indices = _state_contact_body_indices(state, body_names)
        return reward_parity.feet_slip(
            body_lin_vel_w=_select_state_body_tensor(
                _required_state_tensor(state, "actual_body_lin_vel_w"),
                body_indices,
            ),
            current_contact_time=_select_state_body_tensor(
                _required_state_tensor(state, "contact_current_contact_time"),
                contact_indices,
            ),
            tolerance=float(params.pop("tolerance", 0.0)),
        )
    if term_name == "impact_force_l2":
        body_names = _required_param(params, "body_names", term_name)
        contact_indices = _state_contact_body_indices(state, body_names)
        return reward_parity.impact_force_l2(
            net_forces_w_history=_select_state_contact_history(
                _required_state_tensor(state, "contact_net_forces_w_history"),
                contact_indices,
            ),
            first_contact=_select_state_body_tensor(_required_state_tensor(state, "contact_first_contact"), contact_indices),
            default_mass_total=_required_state_tensor(state, "default_mass_total"),
        )
    if term_name == "feet_air_time":
        body_names = _required_param(params, "body_names", term_name)
        contact_indices = _state_contact_body_indices(state, body_names)
        params.pop("soft_discount", None)
        params.pop("condition_on_linvel", None)
        return reward_parity.feet_air_time(
            last_air_time=_select_state_body_tensor(_required_state_tensor(state, "contact_last_air_time"), contact_indices),
            first_contact=_select_state_body_tensor(_required_state_tensor(state, "contact_first_contact"), contact_indices),
            thres=float(params.pop("thres")),
            is_standing_env=state.is_standing_env,
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


def _state_batch_reference(state: MujocoRewardState) -> torch.Tensor:
    for name in (
        "joint_pos",
        "joint_vel",
        "actual_body_pos_w",
        "root_pos_w",
        "object_pos_w",
        "contact_eef_pos_w",
        "action_buf",
        "applied_torque",
    ):
        value = getattr(state, name)
        if value is not None:
            return value
    raise ValueError("MujocoRewardState needs at least one tensor to infer reward batch size.")


def _state_joint_indices(state: MujocoRewardState, joint_names: Any) -> list[int] | None:
    if joint_names is None:
        return None
    if state.joint_names is None:
        raise ValueError("MujocoRewardState.joint_names is required when reward term joint_names is set.")
    return _matching_name_indices(joint_names, state.joint_names, "joint")


def _state_body_indices(state: MujocoRewardState, body_names: Any) -> list[int]:
    if state.body_names is None:
        raise ValueError("MujocoRewardState.body_names is required when reward term body_names is set.")
    return _matching_name_indices(body_names, state.body_names, "body")


def _state_contact_body_indices(state: MujocoRewardState, body_names: Any) -> list[int]:
    if state.contact_body_names is None:
        raise ValueError("MujocoRewardState.contact_body_names is required when reward term body_names is set.")
    return _matching_name_indices(body_names, state.contact_body_names, "contact body")


def _select_state_joint_tensor(tensor: torch.Tensor, indices: list[int] | None) -> torch.Tensor:
    if indices is None:
        return tensor
    return tensor[:, indices]


def _select_state_body_tensor(tensor: torch.Tensor, indices: list[int]) -> torch.Tensor:
    return tensor[:, indices]


def _select_state_contact_history(tensor: torch.Tensor, indices: list[int]) -> torch.Tensor:
    return tensor[:, :, indices]


def _required_param(params: dict[str, Any], name: str, term_name: str) -> Any:
    if name not in params:
        raise ValueError(f"Reward term {term_name!r} requires parameter {name!r}.")
    return params.pop(name)


def _contact_reward_state(scene: Any, robot: Any) -> dict[str, Any]:
    default_mass_total = robot.data.default_mass.sum(dim=1) * 9.81
    sensor = getattr(scene, "sensors", {}).get("contact_forces")
    if sensor is None:
        return {"default_mass_total": default_mass_total}

    data = sensor.data
    return {
        "contact_body_names": list(sensor.body_names),
        "contact_current_contact_time": data.current_contact_time,
        "contact_last_air_time": data.last_air_time,
        "contact_first_contact": sensor.compute_first_contact(_scene_step_dt(scene)),
        "contact_net_forces_w_history": data.net_forces_w_history,
        "default_mass_total": default_mass_total,
    }


def _scene_step_dt(scene: Any) -> float:
    model = getattr(scene, "mj_model", None)
    opt = getattr(model, "opt", None)
    timestep = getattr(opt, "timestep", 0.0)
    return float(timestep)


def _matching_name_indices(name_keys: Any, available_names: Sequence[str], label: str) -> list[int]:
    if isinstance(name_keys, (str, bytes)):
        patterns = [str(name_keys)]
    elif isinstance(name_keys, Sequence):
        patterns = [str(name_key) for name_key in name_keys]
    else:
        raise ValueError(f"{label}_names must be a string or sequence, got {type(name_keys).__name__}.")

    indices: list[int] = []
    for pattern in patterns:
        matches = [index for index, name in enumerate(available_names) if re.fullmatch(pattern, name)]
        if not matches:
            raise ValueError(f"No {label} names matched {pattern!r}.")
        for index in matches:
            if index not in indices:
                indices.append(index)
    return indices


def _current_reference_frame(value: torch.Tensor | None, *, state_rank: int) -> torch.Tensor | None:
    if value is None:
        return None
    if value.ndim == state_rank + 1:
        return value[:, 0]
    return value


def _resolve_indices(asset: Any, resolver_name: str, name_keys: str | Sequence[str]) -> list[int]:
    indices, _ = getattr(asset, resolver_name)(name_keys, preserve_order=True)
    if not indices:
        raise ValueError(f"No names matched {name_keys!r}.")
    return indices


def _resolve_indices_and_names(
    asset: Any,
    resolver_name: str,
    name_keys: str | Sequence[str],
) -> tuple[list[int], list[str]]:
    indices, names = getattr(asset, resolver_name)(name_keys, preserve_order=True)
    if not indices:
        raise ValueError(f"No names matched {name_keys!r}.")
    return indices, list(names)


def _resolve_single_index(asset: Any, resolver_name: str, name_key: str, label: str) -> int:
    indices = _resolve_indices(asset, resolver_name, [name_key])
    if len(indices) != 1:
        raise ValueError(f"Expected one {label} match for {name_key!r}, got {len(indices)}.")
    return indices[0]


def _normalize_playback_steps(steps: Sequence[int] | torch.Tensor | None, num_steps: int) -> torch.Tensor:
    if steps is None:
        steps_t = torch.arange(num_steps, dtype=torch.long)
    else:
        steps_t = torch.as_tensor(steps, dtype=torch.long)
        if steps_t.ndim == 0:
            steps_t = steps_t.unsqueeze(0)
    if torch.any(steps_t < 0) or torch.any(steps_t >= num_steps):
        raise ValueError(f"playback steps must be in [0, {num_steps}), got {steps_t.tolist()}.")
    return steps_t


def _scene_optional_asset(scene: Any, asset_name: str | None) -> Any | None:
    if asset_name is None:
        return None
    asset = scene[asset_name]
    if asset is None:
        raise KeyError(f"MuJoCo scene does not contain asset {asset_name!r}.")
    return asset


def _write_reference_frame_to_scene(
    *,
    scene: Any,
    reference: Any,
    step: int,
    object_view: Any | None,
    object_body_name: str | None,
) -> None:
    robot = scene["robot"]
    root_pose = torch.cat(
        [
            reference.body_pos_w[step, reference.root_body_index],
            reference.body_quat_w[step, reference.root_body_index],
        ]
    )
    robot.write_root_link_pose_to_sim(root_pose)

    ref_joint_pos = _reference_joint_pos(reference, step)
    ref_joint_vel = _reference_joint_vel(reference, step)
    robot_pairs = _reference_to_asset_joint_pairs(reference.requested_joint_names, robot.joint_names)
    _write_asset_joint_pairs(robot, ref_joint_pos, ref_joint_vel, robot_pairs)

    if object_view is not None:
        if object_body_name is not None and object_body_name in reference.body_names:
            body_index = _name_index(reference.body_names, object_body_name, "body")
            object_pose = torch.cat(
                [
                    reference.body_pos_w[step, body_index],
                    reference.body_quat_w[step, body_index],
                ]
            )
            object_view.write_root_link_pose_to_sim(object_pose)

        object_pairs = _reference_to_asset_joint_pairs(reference.requested_joint_names, object_view.joint_names)
        _write_asset_joint_pairs(object_view, ref_joint_pos, ref_joint_vel, object_pairs)
    scene.update(0.0)


def _write_asset_joint_pairs(
    asset: Any,
    ref_joint_pos: torch.Tensor,
    ref_joint_vel: torch.Tensor,
    pairs: list[tuple[int, int]],
) -> None:
    if not pairs:
        return
    ref_columns, asset_joint_ids = zip(*pairs, strict=True)
    joint_pos = ref_joint_pos[list(ref_columns)].unsqueeze(0).expand(asset.num_instances, -1)
    joint_vel = ref_joint_vel[list(ref_columns)].unsqueeze(0).expand(asset.num_instances, -1)
    asset.write_joint_state_to_sim(
        joint_pos,
        joint_vel,
        joint_ids=list(asset_joint_ids),
    )


def _reference_joint_pos(reference: Any, step: int) -> torch.Tensor:
    return reference.joint_pos[step, reference.joint_indices]


def _reference_body_pos_w(reference: Any, step: int) -> torch.Tensor:
    return reference.body_pos_w[step, reference.body_indices]


def _reference_body_quat_w(reference: Any, step: int) -> torch.Tensor:
    return reference.body_quat_w[step, reference.body_indices]


def _reference_body_lin_vel_w(reference: Any, step: int) -> torch.Tensor:
    body_lin_vel_w = getattr(reference, "body_lin_vel_w", None)
    if body_lin_vel_w is None:
        return torch.zeros_like(reference.body_pos_w[step, reference.body_indices])
    return body_lin_vel_w[step, reference.body_indices]


def _reference_body_ang_vel_w(reference: Any, step: int) -> torch.Tensor:
    body_ang_vel_w = getattr(reference, "body_ang_vel_w", None)
    if body_ang_vel_w is None:
        return torch.zeros_like(reference.body_pos_w[step, reference.body_indices])
    return body_ang_vel_w[step, reference.body_indices]


def _reference_joint_vel(reference: Any, step: int) -> torch.Tensor:
    joint_vel = getattr(reference, "joint_vel", None)
    if joint_vel is None:
        return torch.zeros_like(reference.joint_pos[step, reference.joint_indices])
    return joint_vel[step, reference.joint_indices]


def _expand_reference_frame(frame: torch.Tensor, num_envs: int) -> torch.Tensor:
    return frame.unsqueeze(0).expand(num_envs, *frame.shape).clone()


def _gather_scene_joint_pos(robot: Any, joint_names: Sequence[str], object_view: Any | None) -> torch.Tensor:
    robot_index = _name_to_index(robot.joint_names)
    object_index = _name_to_index(object_view.joint_names) if object_view is not None else {}
    columns = []
    for joint_name in joint_names:
        if joint_name in robot_index:
            columns.append(robot.data.joint_pos[:, robot_index[joint_name]])
        elif object_view is not None and joint_name in object_index:
            columns.append(object_view.data.joint_pos[:, object_index[joint_name]])
        else:
            raise ValueError(f"missing playback joint name in MuJoCo scene: {joint_name!r}.")
    return torch.stack(columns, dim=1)


def _gather_scene_joint_vel(robot: Any, joint_names: Sequence[str], object_view: Any | None) -> torch.Tensor:
    robot_index = _name_to_index(robot.joint_names)
    object_index = _name_to_index(object_view.joint_names) if object_view is not None else {}
    columns = []
    for joint_name in joint_names:
        if joint_name in robot_index:
            columns.append(robot.data.joint_vel[:, robot_index[joint_name]])
        elif object_view is not None and joint_name in object_index:
            columns.append(object_view.data.joint_vel[:, object_index[joint_name]])
        else:
            raise ValueError(f"missing playback joint name in MuJoCo scene: {joint_name!r}.")
    return torch.stack(columns, dim=1)


def _gather_scene_joint_pos_limits(robot: Any, joint_names: Sequence[str], object_view: Any | None) -> torch.Tensor:
    robot_index = _name_to_index(robot.joint_names)
    object_index = _name_to_index(object_view.joint_names) if object_view is not None else {}
    robot_limits = robot.data.soft_joint_pos_limits
    object_limits = object_view.data.soft_joint_pos_limits if object_view is not None else None
    columns = []
    for joint_name in joint_names:
        if joint_name in robot_index:
            columns.append(robot_limits[:, robot_index[joint_name]])
        elif object_view is not None and joint_name in object_index:
            columns.append(object_limits[:, object_index[joint_name]])
        else:
            raise ValueError(f"missing playback joint name in MuJoCo scene: {joint_name!r}.")
    return torch.stack(columns, dim=1)


def _gather_scene_applied_torque(robot: Any, joint_names: Sequence[str], object_view: Any | None) -> torch.Tensor:
    robot_index = _name_to_index(robot.joint_names)
    object_index = _name_to_index(object_view.joint_names) if object_view is not None else {}
    object_zeros = None
    columns = []
    for joint_name in joint_names:
        if joint_name in robot_index:
            columns.append(robot.data.applied_torque[:, robot_index[joint_name]])
        elif object_view is not None and joint_name in object_index:
            if object_zeros is None:
                object_zeros = torch.zeros(object_view.num_instances, dtype=robot.data.applied_torque.dtype)
            columns.append(object_zeros)
        else:
            raise ValueError(f"missing playback joint name in MuJoCo scene: {joint_name!r}.")
    return torch.stack(columns, dim=1)


def _gather_scene_joint_effort_limits(robot: Any, joint_names: Sequence[str], object_view: Any | None) -> torch.Tensor:
    robot_index = _name_to_index(robot.joint_names)
    object_index = _name_to_index(object_view.joint_names) if object_view is not None else {}
    object_limits = None
    columns = []
    for joint_name in joint_names:
        if joint_name in robot_index:
            columns.append(robot.data.joint_effort_limits[:, robot_index[joint_name]])
        elif object_view is not None and joint_name in object_index:
            if object_limits is None:
                object_limits = torch.full((object_view.num_instances,), float("inf"), dtype=robot.data.applied_torque.dtype)
            columns.append(object_limits)
        else:
            raise ValueError(f"missing playback joint name in MuJoCo scene: {joint_name!r}.")
    return torch.stack(columns, dim=1)


def _gather_scene_body_pos_w(robot: Any, body_names: Sequence[str], object_view: Any | None) -> torch.Tensor:
    return _gather_scene_body_tensor(robot, body_names, object_view, "body_link_pos_w")


def _gather_scene_body_quat_w(robot: Any, body_names: Sequence[str], object_view: Any | None) -> torch.Tensor:
    return _gather_scene_body_tensor(robot, body_names, object_view, "body_link_quat_w")


def _gather_scene_body_lin_vel_w(robot: Any, body_names: Sequence[str], object_view: Any | None) -> torch.Tensor:
    return _gather_scene_body_tensor(robot, body_names, object_view, "body_com_lin_vel_w")


def _gather_scene_body_ang_vel_w(robot: Any, body_names: Sequence[str], object_view: Any | None) -> torch.Tensor:
    return _gather_scene_body_tensor(robot, body_names, object_view, "body_com_ang_vel_w")


def _gather_scene_body_tensor(
    robot: Any,
    body_names: Sequence[str],
    object_view: Any | None,
    tensor_name: str,
) -> torch.Tensor:
    robot_index = _name_to_index(robot.body_names)
    object_index = _name_to_index(object_view.body_names) if object_view is not None else {}
    robot_tensor = getattr(robot.data, tensor_name)
    object_tensor = getattr(object_view.data, tensor_name) if object_view is not None else None
    columns = []
    for body_name in body_names:
        if body_name in robot_index:
            columns.append(robot_tensor[:, robot_index[body_name]])
        elif object_view is not None and body_name in object_index:
            columns.append(object_tensor[:, object_index[body_name]])
        else:
            raise ValueError(f"missing playback body name in MuJoCo scene: {body_name!r}.")
    return torch.stack(columns, dim=1)


def _build_kinematic_reward_state(
    *,
    scene: Any,
    robot: Any,
    object_view: Any | None,
    reference: Any,
    step: int,
    actual_body_pos_w: torch.Tensor,
    ref_body_pos_w: torch.Tensor,
    actual_body_quat_w: torch.Tensor,
    ref_body_quat_w: torch.Tensor,
    actual_body_lin_vel_w: torch.Tensor,
    ref_body_lin_vel_w: torch.Tensor,
    actual_body_ang_vel_w: torch.Tensor,
    ref_body_ang_vel_w: torch.Tensor,
    actual_joint_pos: torch.Tensor,
    ref_joint_pos: torch.Tensor,
    actual_joint_vel: torch.Tensor,
    ref_joint_vel: torch.Tensor,
    actual_joint_pos_limits: torch.Tensor,
    object_name: str | None,
    object_body_name: str | None,
    object_joint_name: str | None,
    contact_eef_body_names: Sequence[str] | None,
    contact_target_pos_offset: Sequence[Sequence[float]] | torch.Tensor | None,
    contact_eef_pos_offset: Sequence[Sequence[float]] | torch.Tensor | None,
) -> MujocoRewardState:
    ref_root_pos_w = _expand_reference_frame(reference.body_pos_w[step, reference.root_body_index], robot.num_instances)
    ref_root_quat_w = _expand_reference_frame(
        reference.body_quat_w[step, reference.root_body_index],
        robot.num_instances,
    )

    object_pos_w = None
    object_quat_w = None
    ref_object_pos_w = None
    ref_object_quat_w = None
    object_joint_pos = None
    ref_object_joint_pos = None
    contact_eef_pos_w = None
    contact_target_pos_w = None
    eef_contact_forces_b = None
    ref_object_contact = None
    applied_torque = _gather_scene_applied_torque(
        robot=robot,
        joint_names=reference.requested_joint_names,
        object_view=object_view,
    )
    joint_effort_limits = _gather_scene_joint_effort_limits(robot, reference.requested_joint_names, object_view)
    if object_view is not None:
        object_body_index = 0
        if object_body_name is not None and object_body_name in object_view.body_names:
            object_body_index = _name_index(object_view.body_names, object_body_name, "object body")
        object_pos_w = object_view.data.body_link_pos_w[:, object_body_index]
        object_quat_w = object_view.data.body_link_quat_w[:, object_body_index]

        if object_body_name is not None and object_body_name in reference.body_names:
            reference_body_index = _name_index(reference.body_names, object_body_name, "body")
            ref_object_pos_w = _expand_reference_frame(
                reference.body_pos_w[step, reference_body_index],
                robot.num_instances,
            )
            ref_object_quat_w = _expand_reference_frame(
                reference.body_quat_w[step, reference_body_index],
                robot.num_instances,
            )

        resolved_joint_name = object_joint_name or (object_view.joint_names[0] if object_view.joint_names else None)
        if resolved_joint_name is not None and resolved_joint_name in object_view.joint_names:
            object_joint_index = _name_index(object_view.joint_names, resolved_joint_name, "object joint")
            object_joint_pos = object_view.data.joint_pos[:, object_joint_index]
        if resolved_joint_name is not None and resolved_joint_name in reference.requested_joint_names:
            reference_joint_index = reference.requested_joint_names.index(resolved_joint_name)
            ref_object_joint_pos = ref_joint_pos[:, reference_joint_index]

        if contact_eef_body_names:
            (
                contact_eef_pos_w,
                contact_target_pos_w,
                eef_contact_forces_b,
                ref_object_contact,
            ) = _build_contact_reward_tensors(
                scene=scene,
                robot=robot,
                object_view=object_view,
                object_name=object_name,
                object_body_name=object_body_name,
                object_body_index=object_body_index,
                reference=reference,
                step=step,
                contact_eef_body_names=contact_eef_body_names,
                contact_target_pos_offset=contact_target_pos_offset,
                contact_eef_pos_offset=contact_eef_pos_offset,
            )
    elif contact_eef_body_names:
        raise ValueError("object_name is required when contact EEF reward inputs are requested.")

    return MujocoRewardState(
        actual_body_pos_w=actual_body_pos_w,
        ref_body_pos_w=ref_body_pos_w,
        actual_body_quat_w=actual_body_quat_w,
        ref_body_quat_w=ref_body_quat_w,
        actual_body_lin_vel_w=actual_body_lin_vel_w,
        ref_body_lin_vel_w=ref_body_lin_vel_w,
        actual_body_ang_vel_w=actual_body_ang_vel_w,
        ref_body_ang_vel_w=ref_body_ang_vel_w,
        body_names=list(reference.requested_body_names),
        root_pos_w=robot.data.root_link_pos_w,
        root_quat_w=robot.data.root_link_quat_w,
        ref_root_pos_w=ref_root_pos_w,
        ref_root_quat_w=ref_root_quat_w,
        joint_pos=actual_joint_pos,
        ref_joint_pos=ref_joint_pos,
        joint_vel=actual_joint_vel,
        ref_joint_vel=ref_joint_vel,
        joint_pos_limits=actual_joint_pos_limits,
        joint_names=list(reference.requested_joint_names),
        applied_torque=applied_torque,
        joint_effort_limits=joint_effort_limits,
        object_pos_w=object_pos_w,
        ref_object_pos_w=ref_object_pos_w,
        object_quat_w=object_quat_w,
        ref_object_quat_w=ref_object_quat_w,
        object_joint_pos=object_joint_pos,
        ref_object_joint_pos=ref_object_joint_pos,
        contact_eef_pos_w=contact_eef_pos_w,
        contact_target_pos_w=contact_target_pos_w,
        eef_contact_forces_b=eef_contact_forces_b,
        ref_object_contact=ref_object_contact,
        **_contact_reward_state(scene, robot),
    )


def _build_contact_reward_tensors(
    *,
    scene: Any,
    robot: Any,
    object_view: Any,
    object_name: str | None,
    object_body_name: str | None,
    object_body_index: int,
    reference: Any,
    step: int,
    contact_eef_body_names: Sequence[str],
    contact_target_pos_offset: Sequence[Sequence[float]] | torch.Tensor | None,
    contact_eef_pos_offset: Sequence[Sequence[float]] | torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    eef_body_names = _as_string_list(contact_eef_body_names, "contact_eef_body_names")
    if not eef_body_names:
        raise ValueError("contact_eef_body_names must contain at least one body name.")

    eef_body_indices = [
        _resolve_single_index(robot, "find_bodies", eef_body_name, "contact EEF body")
        for eef_body_name in eef_body_names
    ]
    dtype = object_view.data.body_link_pos_w.dtype
    device = object_view.data.body_link_pos_w.device
    num_envs = robot.num_instances
    num_eefs = len(eef_body_names)

    target_offsets = _contact_offsets(
        contact_target_pos_offset,
        num_eefs=num_eefs,
        dtype=dtype,
        device=device,
        label="contact_target_pos_offset",
    )
    eef_offsets = _contact_offsets(
        contact_eef_pos_offset,
        num_eefs=num_eefs,
        dtype=dtype,
        device=device,
        label="contact_eef_pos_offset",
    )

    object_pos_w = object_view.data.body_link_pos_w[:, object_body_index]
    object_quat_w = object_view.data.body_link_quat_w[:, object_body_index]
    contact_target_pos_w = object_pos_w[:, None, :] + _quat_rotate(
        object_quat_w[:, None, :],
        target_offsets.unsqueeze(0).expand(num_envs, -1, -1),
    )

    eef_pos_w = robot.data.body_link_pos_w[:, eef_body_indices]
    eef_quat_w = robot.data.body_link_quat_w[:, eef_body_indices]
    contact_eef_pos_w = eef_pos_w + _quat_rotate(
        eef_quat_w,
        eef_offsets.unsqueeze(0).expand(num_envs, -1, -1),
    )

    eef_contact_forces_w = _contact_forces_w(
        scene=scene,
        eef_body_names=eef_body_names,
        object_name=object_name,
        object_view=object_view,
        dtype=dtype,
        device=device,
        num_envs=num_envs,
    )
    eef_contact_forces_b = _quat_rotate_inverse(object_quat_w[:, None, :], eef_contact_forces_w)
    ref_object_contact = _reference_object_contact(
        reference=reference,
        step=step,
        num_envs=num_envs,
        num_eefs=num_eefs,
        device=device,
    )
    return contact_eef_pos_w, contact_target_pos_w, eef_contact_forces_b, ref_object_contact


def _as_string_list(value: Sequence[str], label: str) -> list[str]:
    if isinstance(value, (str, bytes)):
        return [str(value)]
    try:
        return [str(item) for item in value]
    except TypeError as exc:
        raise ValueError(f"{label} must be a sequence of names.") from exc


def _contact_offsets(
    value: Sequence[Sequence[float]] | torch.Tensor | None,
    *,
    num_eefs: int,
    dtype: torch.dtype,
    device: torch.device,
    label: str,
) -> torch.Tensor:
    if value is None:
        return torch.zeros(num_eefs, 3, dtype=dtype, device=device)
    offsets = torch.as_tensor(value, dtype=dtype, device=device)
    if offsets.ndim == 1:
        offsets = offsets.unsqueeze(0)
    if offsets.ndim != 2 or offsets.shape[-1] != 3:
        raise ValueError(f"{label} must have shape (num_eefs, 3), got {tuple(offsets.shape)}.")
    if offsets.shape[0] == 1 and num_eefs > 1:
        offsets = offsets.expand(num_eefs, -1)
    if offsets.shape[0] != num_eefs:
        raise ValueError(f"{label} has {offsets.shape[0]} rows, but {num_eefs} contact EEFs were configured.")
    return offsets


def _contact_forces_w(
    *,
    scene: Any,
    eef_body_names: Sequence[str],
    object_name: str | None,
    object_view: Any,
    dtype: torch.dtype,
    device: torch.device,
    num_envs: int,
) -> torch.Tensor:
    forces_w = torch.zeros(num_envs, len(eef_body_names), 3, dtype=dtype, device=device)
    sensors = getattr(scene, "sensors", {})
    if not sensors:
        return forces_w

    for eef_index, eef_body_name in enumerate(eef_body_names):
        sensor = _contact_sensor_for_eef(
            sensors=sensors,
            eef_body_name=eef_body_name,
            object_name=object_name,
            object_view=object_view,
        )
        if sensor is None:
            continue
        sensor_body_index = _sensor_body_index(sensor, eef_body_name)
        force_matrix_w = sensor.data.force_matrix_w
        if force_matrix_w.shape[2] == 0:
            continue
        forces_w[:, eef_index] = force_matrix_w[:, sensor_body_index].sum(dim=1).to(dtype=dtype, device=device)
    return forces_w


def _contact_sensor_for_eef(
    *,
    sensors: Mapping[str, Any],
    eef_body_name: str,
    object_name: str | None,
    object_view: Any,
) -> Any | None:
    for sensor_name in _contact_sensor_names(eef_body_name, object_name, object_view):
        sensor = sensors.get(sensor_name)
        if sensor is not None:
            return sensor
    return None


def _contact_sensor_names(eef_body_name: str, object_name: str | None, object_view: Any) -> list[str]:
    object_names = []
    if object_name is not None:
        object_names.append(object_name)
    spec = getattr(object_view, "spec", None)
    spec_asset_name = getattr(spec, "asset_name", None)
    if spec_asset_name is not None:
        object_names.append(str(spec_asset_name))
    return [f"{eef_body_name}_{name}_contact_forces" for name in dict.fromkeys(object_names)]


def _sensor_body_index(sensor: Any, eef_body_name: str) -> int:
    indices, _ = sensor.find_bodies([eef_body_name], preserve_order=True)
    if len(indices) != 1:
        raise ValueError(f"Expected one contact sensor body for {eef_body_name!r}, got {len(indices)}.")
    return indices[0]


def _reference_object_contact(
    *,
    reference: Any,
    step: int,
    num_envs: int,
    num_eefs: int,
    device: torch.device,
) -> torch.Tensor:
    object_contact = getattr(reference, "object_contact", None)
    if object_contact is None:
        raise ValueError("reference.object_contact is required for contact reward parity.")
    contact = object_contact[step].to(dtype=torch.bool, device=device).reshape(-1)
    if contact.numel() == 1 and num_eefs > 1:
        contact = contact.expand(num_eefs)
    if contact.numel() != num_eefs:
        raise ValueError(
            f"reference.object_contact step {step} has {contact.numel()} values, "
            f"but {num_eefs} contact EEFs were configured."
        )
    return contact.unsqueeze(0).expand(num_envs, -1)


def _quat_rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    quat = quat.expand(*vec.shape[:-1], 4)
    xyz = quat[..., 1:]
    t = torch.linalg.cross(xyz, vec, dim=-1) * 2.0
    return vec + quat[..., 0:1] * t + torch.linalg.cross(xyz, t, dim=-1)


def _quat_rotate_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    quat_inv = torch.cat([quat[..., 0:1], -quat[..., 1:]], dim=-1)
    return _quat_rotate(quat_inv, vec)


def _reference_to_asset_joint_pairs(
    reference_joint_names: Sequence[str],
    asset_joint_names: Sequence[str],
) -> list[tuple[int, int]]:
    asset_index = _name_to_index(asset_joint_names)
    return [
        (reference_index, asset_index[name])
        for reference_index, name in enumerate(reference_joint_names)
        if name in asset_index
    ]


def _name_to_index(names: Sequence[str]) -> dict[str, int]:
    return {name: index for index, name in enumerate(names)}


def _name_index(names: Sequence[str], name: str, label: str) -> int:
    try:
        return names.index(name)
    except ValueError as exc:
        raise ValueError(f"missing {label} name: {name!r}.") from exc


def _require_same_shape(lhs_name: str, lhs: torch.Tensor, rhs_name: str, rhs: torch.Tensor) -> None:
    if lhs.shape != rhs.shape:
        raise ValueError(f"{lhs_name} shape {tuple(lhs.shape)} != {rhs_name} shape {tuple(rhs.shape)}.")
