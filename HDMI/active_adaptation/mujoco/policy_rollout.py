from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .motion_reference import MujocoMotionReference
from .observation_builder import MujocoPolicyState
from .policy import MujocoPolicyAction, MujocoPolicyBundle
from . import reward_parity
from .playback_parity import (
    _build_kinematic_reward_state,
    _expand_reference_frame,
    _gather_scene_body_ang_vel_w,
    _gather_scene_body_lin_vel_w,
    _gather_scene_body_quat_w,
    _gather_scene_joint_pos,
    _gather_scene_joint_pos_limits,
    _gather_scene_joint_vel,
    _reference_body_ang_vel_w as _reference_frame_body_ang_vel_w,
    _reference_body_lin_vel_w as _reference_frame_body_lin_vel_w,
    _reference_body_quat_w as _reference_frame_body_quat_w,
    _reference_joint_pos as _reference_frame_joint_pos,
    _reference_joint_vel as _reference_frame_joint_vel,
    _scene_object_views,
    _scene_optional_asset,
    compute_playback_parity,
    compute_reward_from_spec,
)

_OBJECT_POSE_OBS_KEYS = ("object_xy_b", "object_heading_b", "object_pos_b", "object_ori_b")


@dataclass(frozen=True)
class MujocoPolicyRolloutMetrics:
    q_l2: torch.Tensor
    body_pos_l2: torch.Tensor
    actions: torch.Tensor
    joint_position_targets: torch.Tensor
    action_rate_l2: torch.Tensor
    reward: torch.Tensor | None = None


def run_mujoco_policy_rollout(
    *,
    scene: Any,
    policy_bundle: MujocoPolicyBundle,
    reference: MujocoMotionReference,
    steps: Sequence[int] | torch.Tensor | None = None,
    decimation: int = 1,
    reward_cfg: Mapping[str, Any] | None = None,
    object_name: str | None = None,
    object_body_name: str | None = None,
    object_joint_name: str | None = None,
    contact_eef_body_names: Sequence[str] | None = None,
    contact_target_pos_offset: Sequence[Sequence[float]] | torch.Tensor | None = None,
    contact_eef_pos_offset: Sequence[Sequence[float]] | torch.Tensor | None = None,
) -> MujocoPolicyRolloutMetrics:
    if decimation < 1:
        raise ValueError(f"decimation must be >= 1, got {decimation}.")

    from active_adaptation.envs import mujoco as mujoco_env

    sim = mujoco_env.MJSim(scene, realtime=False)
    robot = scene["robot"]
    if robot is None:
        raise KeyError("MuJoCo scene does not contain a 'robot' articulation.")
    primary_object_view = _scene_optional_asset(scene, object_name)
    object_views = _policy_scene_object_views(scene, robot, primary_object_view)

    policy_joint_ids = _ordered_joint_ids(robot, policy_bundle.policy_joint_names, label="Policy joint")
    observation_joint_ids = _ordered_joint_ids(
        robot,
        policy_bundle.observation_joint_names,
        label="Policy observation joint",
    )
    steps_t = _normalize_steps(steps, reference.num_steps)

    _write_reference_frame_to_scene(scene, reference, int(steps_t[0].item()))
    scene.update(0.0)

    action_history = torch.zeros(scene.num_envs, policy_bundle.action_dim, _policy_action_history_steps(policy_bundle))
    applied_action = torch.zeros(scene.num_envs, policy_bundle.action_dim)
    policy_bundle.reset(
        _policy_state_from_scene(
            scene=scene,
            policy_bundle=policy_bundle,
            reference=reference,
            step=int(steps_t[0].item()),
            policy_joint_ids=policy_joint_ids,
            observation_joint_ids=observation_joint_ids,
            applied_action=applied_action,
            action_history=action_history,
        )
    )

    actions: list[torch.Tensor] = []
    joint_targets: list[torch.Tensor] = []
    q_l2: list[torch.Tensor] = []
    body_pos_l2: list[torch.Tensor] = []
    action_rate_l2: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []

    for rollout_index, step_t in enumerate(steps_t):
        step = int(step_t.item())
        state = _policy_state_from_scene(
            scene=scene,
            policy_bundle=policy_bundle,
            reference=reference,
            step=step,
            policy_joint_ids=policy_joint_ids,
            observation_joint_ids=observation_joint_ids,
            applied_action=applied_action,
            action_history=action_history,
        )
        if rollout_index:
            policy_bundle.update(state)
        action = policy_bundle.act(state, is_init=(rollout_index == 0))
        _apply_policy_action(robot, action, policy_joint_ids)
        scene.write_data_to_sim()
        for _ in range(decimation):
            sim.step()
        scene.update(sim.get_physics_dt() * decimation)

        current_joint_pos = robot.data.joint_pos[:, policy_joint_ids]
        ref_joint_pos = _reference_joint_pos(reference, policy_bundle.policy_joint_names, step, scene.num_envs)
        current_body_pos_w = _scene_body_pos_w(scene, reference.requested_body_names)
        ref_body_pos_w = _reference_body_pos_w(reference, step, scene.num_envs)
        metrics = compute_playback_parity(
            q_mujoco=current_joint_pos,
            q_ref=ref_joint_pos,
            body_pos_mujoco_w=current_body_pos_w,
            body_pos_ref_w=ref_body_pos_w,
        )
        q_l2.append(metrics.q_l2)
        body_pos_l2.append(metrics.body_pos_l2)
        actions.append(action.raw_action.detach().cpu())
        joint_targets.append(action.joint_position_target.detach().cpu())
        action_buf = torch.stack((action.raw_action.detach(), action_history[:, :, 0]), dim=2)
        action_rate_l2.append(
            reward_parity.action_rate_l2(action_buf).detach().cpu()
        )
        if reward_cfg is not None:
            reward = _policy_rollout_reward_from_scene(
                scene=scene,
                robot=robot,
                reference=reference,
                step=step,
                reward_cfg=reward_cfg,
                actual_body_pos_w=current_body_pos_w,
                ref_body_pos_w=ref_body_pos_w,
                action_buf=action_buf,
                primary_object_view=primary_object_view,
                object_views=object_views,
                object_name=object_name,
                object_body_name=object_body_name,
                object_joint_name=object_joint_name,
                contact_eef_body_names=contact_eef_body_names,
                contact_target_pos_offset=contact_target_pos_offset,
                contact_eef_pos_offset=contact_eef_pos_offset,
            )
            rewards.append(reward.detach().cpu())

        applied_action = action.raw_action.detach()
        action_history = action_history.roll(1, dims=2)
        action_history[:, :, 0] = applied_action

    return MujocoPolicyRolloutMetrics(
        q_l2=torch.stack(q_l2),
        body_pos_l2=torch.stack(body_pos_l2),
        actions=torch.stack(actions),
        joint_position_targets=torch.stack(joint_targets),
        action_rate_l2=torch.stack(action_rate_l2),
        reward=torch.stack(rewards) if rewards else None,
    )


def _policy_scene_object_views(scene: Any, robot: Any, primary_object_view: Any | None) -> tuple[Any, ...]:
    object_views = list(_scene_object_views(scene, primary_object_view))
    for candidate in (*getattr(scene, "articulations", {}).values(), *getattr(scene, "rigid_objects", {}).values()):
        if candidate is None or candidate is robot:
            continue
        if any(candidate is existing for existing in object_views):
            continue
        object_views.append(candidate)
    return tuple(object_views)


def _policy_rollout_reward_from_scene(
    *,
    scene: Any,
    robot: Any,
    reference: MujocoMotionReference,
    step: int,
    reward_cfg: Mapping[str, Any],
    actual_body_pos_w: torch.Tensor,
    ref_body_pos_w: torch.Tensor,
    action_buf: torch.Tensor,
    primary_object_view: Any | None,
    object_views: Any | None,
    object_name: str | None,
    object_body_name: str | None,
    object_joint_name: str | None,
    contact_eef_body_names: Sequence[str] | None,
    contact_target_pos_offset: Sequence[Sequence[float]] | torch.Tensor | None,
    contact_eef_pos_offset: Sequence[Sequence[float]] | torch.Tensor | None,
) -> torch.Tensor:
    actual_body_quat_w = _gather_scene_body_quat_w(
        robot=robot,
        body_names=reference.requested_body_names,
        object_view=object_views,
    )
    actual_body_lin_vel_w = _gather_scene_body_lin_vel_w(
        robot=robot,
        body_names=reference.requested_body_names,
        object_view=object_views,
    )
    actual_body_ang_vel_w = _gather_scene_body_ang_vel_w(
        robot=robot,
        body_names=reference.requested_body_names,
        object_view=object_views,
    )
    actual_joint_pos = _gather_scene_joint_pos(
        robot=robot,
        joint_names=reference.requested_joint_names,
        object_view=object_views,
    )
    actual_joint_vel = _gather_scene_joint_vel(
        robot=robot,
        joint_names=reference.requested_joint_names,
        object_view=object_views,
    )
    actual_joint_pos_limits = _gather_scene_joint_pos_limits(
        robot=robot,
        joint_names=reference.requested_joint_names,
        object_view=object_views,
    )
    ref_body_quat_w = _expand_reference_frame(_reference_frame_body_quat_w(reference, step), scene.num_envs)
    ref_body_lin_vel_w = _expand_reference_frame(_reference_frame_body_lin_vel_w(reference, step), scene.num_envs)
    ref_body_ang_vel_w = _expand_reference_frame(_reference_frame_body_ang_vel_w(reference, step), scene.num_envs)
    ref_joint_pos = _expand_reference_frame(_reference_frame_joint_pos(reference, step), scene.num_envs)
    ref_joint_vel = _expand_reference_frame(_reference_frame_joint_vel(reference, step), scene.num_envs)
    reward_state = _build_kinematic_reward_state(
        scene=scene,
        robot=robot,
        object_view=primary_object_view,
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
        action_buf=action_buf,
    )
    return compute_reward_from_spec(reward_cfg, reward_state)


def _ordered_joint_ids(robot: Any, joint_names_expected: Sequence[str], *, label: str) -> list[int]:
    joint_names_expected = list(joint_names_expected)
    joint_ids, joint_names = robot.find_joints(joint_names_expected, preserve_order=True)
    if joint_names != joint_names_expected:
        raise ValueError(f"{label} order mismatch: expected {joint_names_expected}, got {joint_names}.")
    return joint_ids


def _policy_state_from_scene(
    *,
    scene: Any,
    policy_bundle: MujocoPolicyBundle,
    reference: MujocoMotionReference,
    step: int,
    policy_joint_ids: Sequence[int],
    observation_joint_ids: Sequence[int],
    applied_action: torch.Tensor,
    action_history: torch.Tensor,
) -> MujocoPolicyState:
    robot = scene["robot"]
    step_ids = torch.full((scene.num_envs,), step, dtype=torch.long)
    fields = reference.observation_fields_at(step_ids)
    object_state = _policy_object_state_from_scene(scene=scene, policy_bundle=policy_bundle)
    return MujocoPolicyState(
        root_ang_vel_b=robot.data.root_ang_vel_b,
        projected_gravity_b=robot.data.projected_gravity_b,
        joint_pos=robot.data.joint_pos[:, observation_joint_ids],
        joint_pos_offset=torch.zeros_like(robot.data.joint_pos[:, observation_joint_ids]),
        applied_action=applied_action,
        action_history=action_history,
        ref_body_pos_future_w=fields.ref_body_pos_future_w,
        ref_root_pos_w=fields.ref_root_pos_w,
        ref_root_quat_w=fields.ref_root_quat_w,
        ref_joint_pos_future=fields.ref_joint_pos_future,
        motion_t=fields.motion_t,
        motion_len=fields.motion_len,
        robot_root_pos_w=robot.data.root_link_pos_w,
        robot_root_quat_w=robot.data.root_link_quat_w,
        **object_state,
    )


def _policy_object_state_from_scene(scene: Any, policy_bundle: MujocoPolicyBundle) -> dict[str, torch.Tensor]:
    object_name = _first_policy_observation_object_name(policy_bundle, _OBJECT_POSE_OBS_KEYS)
    contact_cfg = _first_policy_observation_cfg(policy_bundle, "ref_contact_pos_b")
    if contact_cfg is not None and object_name is None and contact_cfg.get("object_name") is not None:
        object_name = str(contact_cfg["object_name"])

    if object_name is None:
        if _uses_policy_observation(policy_bundle, (*_OBJECT_POSE_OBS_KEYS, "ref_contact_pos_b")):
            object_name = _default_scene_object_body_name(scene)
        else:
            return {}

    object_pos_w, object_quat_w = _scene_body_pose_w(scene, object_name)
    object_state = {
        "object_pos_w": object_pos_w,
        "object_quat_w": object_quat_w,
    }
    if contact_cfg is not None:
        object_state["contact_target_pos_w"] = _contact_target_pos_w(
            object_pos_w=object_pos_w,
            object_quat_w=object_quat_w,
            contact_cfg=contact_cfg,
        )
    return object_state


def _first_policy_observation_object_name(
    policy_bundle: MujocoPolicyBundle,
    obs_keys: Sequence[str],
) -> str | None:
    for obs_key in obs_keys:
        obs_cfg = _first_policy_observation_cfg(policy_bundle, obs_key)
        if obs_cfg is not None and obs_cfg.get("object_name") is not None:
            return str(obs_cfg["object_name"])
    return None


def _first_policy_observation_cfg(
    policy_bundle: MujocoPolicyBundle,
    obs_key: str,
) -> Mapping[str, Any] | None:
    for group_cfg in policy_bundle.observation_builder.observation_cfg.values():
        if obs_key not in group_cfg:
            continue
        obs_cfg = group_cfg[obs_key] or {}
        if not isinstance(obs_cfg, Mapping):
            raise ValueError(f"Policy observation {obs_key!r} config must be a mapping.")
        return obs_cfg
    return None


def _uses_policy_observation(policy_bundle: MujocoPolicyBundle, obs_keys: Sequence[str]) -> bool:
    return any(_first_policy_observation_cfg(policy_bundle, obs_key) is not None for obs_key in obs_keys)


def _default_scene_object_body_name(scene: Any) -> str:
    robot = scene["robot"]
    object_body_names: list[str] = []
    for object_view in (*scene.articulations.values(), *scene.rigid_objects.values()):
        if object_view is robot:
            continue
        object_body_names.extend(getattr(object_view, "body_names", ()) or ())
    if not object_body_names:
        raise ValueError("Policy uses object observations, but MuJoCo scene has no non-robot object body.")
    return object_body_names[0]


def _contact_target_pos_w(
    *,
    object_pos_w: torch.Tensor,
    object_quat_w: torch.Tensor,
    contact_cfg: Mapping[str, Any],
) -> torch.Tensor:
    offsets = torch.as_tensor(
        contact_cfg.get("contact_target_pos_offset", [[0.0, 0.0, 0.0]]),
        dtype=object_pos_w.dtype,
        device=object_pos_w.device,
    )
    if offsets.ndim == 1:
        offsets = offsets.unsqueeze(0)
    if offsets.shape[-1] != 3:
        raise ValueError(f"contact_target_pos_offset must end in xyz dim 3, got {tuple(offsets.shape)}.")
    offsets = offsets.unsqueeze(0).expand(object_pos_w.shape[0], -1, -1)
    return object_pos_w[:, None, :] + _quat_rotate(object_quat_w[:, None, :], offsets)


def _apply_policy_action(robot: Any, action: MujocoPolicyAction, policy_joint_ids: Sequence[int]) -> None:
    robot.set_joint_position_target(action.joint_position_target, joint_ids=list(policy_joint_ids))


def _write_reference_frame_to_scene(scene: Any, reference: MujocoMotionReference, step: int) -> None:
    robot = scene["robot"]
    root_pos = reference.body_pos_w[step, reference.root_body_index].unsqueeze(0).expand(scene.num_envs, -1)
    root_quat = reference.body_quat_w[step, reference.root_body_index].unsqueeze(0).expand(scene.num_envs, -1)
    root_state = torch.cat([root_pos, root_quat, torch.zeros(scene.num_envs, 6)], dim=-1)
    robot.write_root_state_to_sim(root_state)

    joint_names = _reference_robot_joint_names(scene, robot, reference.requested_joint_names)
    if joint_names:
        joint_ids, found_names = robot.find_joints(joint_names, preserve_order=True)
        if found_names != joint_names:
            raise ValueError(f"Reference joint order mismatch: expected {joint_names}, got {found_names}.")
        joint_pos = _reference_joint_pos(reference, joint_names, step, scene.num_envs)
        robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), joint_ids=joint_ids)

    _write_reference_objects_to_scene(scene, robot, reference, step)


def _write_reference_objects_to_scene(
    scene: Any,
    robot: Any,
    reference: MujocoMotionReference,
    step: int,
) -> None:
    seen_ids: set[int] = set()
    for object_view in (*scene.articulations.values(), *scene.rigid_objects.values()):
        if object_view is robot:
            continue
        object_id = id(object_view)
        if object_id in seen_ids:
            continue
        seen_ids.add(object_id)

        reference_object_body_name = _reference_object_body_name(reference, object_view)
        if reference_object_body_name is not None:
            body_index = reference.body_names.index(reference_object_body_name)
            object_pos = reference.body_pos_w[step, body_index].unsqueeze(0).expand(scene.num_envs, -1)
            object_quat = reference.body_quat_w[step, body_index].unsqueeze(0).expand(scene.num_envs, -1)
            object_view.write_root_link_pose_to_sim(torch.cat([object_pos, object_quat], dim=-1))

        object_joint_names = _reference_object_joint_names(reference.requested_joint_names, object_view)
        if not object_joint_names:
            continue
        object_joint_ids, found_names = object_view.find_joints(object_joint_names, preserve_order=True)
        if found_names != object_joint_names:
            raise ValueError(
                f"Reference object joint order mismatch: expected {object_joint_names}, got {found_names}."
            )
        object_joint_pos = _reference_joint_pos(reference, object_joint_names, step, scene.num_envs)
        object_view.write_joint_state_to_sim(
            object_joint_pos,
            torch.zeros_like(object_joint_pos),
            joint_ids=object_joint_ids,
        )


def _reference_object_body_name(reference: MujocoMotionReference, object_view: Any) -> str | None:
    for body_name in getattr(object_view, "body_names", ()) or ():
        if body_name in reference.body_names:
            return body_name
    return None


def _reference_object_joint_names(joint_names: Sequence[str], object_view: Any) -> list[str]:
    object_joint_names = set(getattr(object_view, "joint_names", ()) or ())
    return [joint_name for joint_name in joint_names if joint_name in object_joint_names]


def _reference_robot_joint_names(scene: Any, robot: Any, joint_names: Sequence[str]) -> list[str]:
    robot_joint_names = set(robot.joint_names)
    object_joint_names: set[str] = set()
    for object_view in (*scene.articulations.values(), *scene.rigid_objects.values()):
        if object_view is robot:
            continue
        object_joint_names.update(getattr(object_view, "joint_names", ()) or ())

    robot_reference_names: list[str] = []
    missing_names: list[str] = []
    for joint_name in joint_names:
        if joint_name in robot_joint_names:
            robot_reference_names.append(joint_name)
        elif joint_name not in object_joint_names:
            missing_names.append(joint_name)

    if missing_names:
        raise ValueError(f"Reference joints are absent from MuJoCo scene: {missing_names}.")
    return robot_reference_names


def _scene_body_pos_w(scene: Any, body_names: Sequence[str]) -> torch.Tensor:
    robot = scene["robot"]
    body_pos_w: list[torch.Tensor] = []
    for body_name in body_names:
        if body_name in robot.body_names:
            body_index = robot.body_names.index(body_name)
            body_pos_w.append(robot.data.body_link_pos_w[:, body_index])
            continue
        object_view, object_body_index = _find_object_body(scene, body_name)
        body_pos_w.append(object_view.data.body_link_pos_w[:, object_body_index])
    return torch.stack(body_pos_w, dim=1)


def _scene_body_pose_w(scene: Any, body_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    robot = scene["robot"]
    if body_name in robot.body_names:
        body_index = robot.body_names.index(body_name)
        return robot.data.body_link_pos_w[:, body_index], robot.data.body_link_quat_w[:, body_index]
    object_view, object_body_index = _find_object_body(scene, body_name)
    return (
        object_view.data.body_link_pos_w[:, object_body_index],
        object_view.data.body_link_quat_w[:, object_body_index],
    )


def _find_object_body(scene: Any, body_name: str) -> tuple[Any, int]:
    for object_view in (*scene.articulations.values(), *scene.rigid_objects.values()):
        if object_view is scene["robot"]:
            continue
        if body_name in getattr(object_view, "body_names", ()):
            return object_view, object_view.body_names.index(body_name)
    raise ValueError(f"Body {body_name!r} is not present in MuJoCo scene.")


def _reference_body_pos_w(reference: MujocoMotionReference, step: int, num_envs: int) -> torch.Tensor:
    return reference.body_pos_w[step, reference.body_indices].unsqueeze(0).expand(num_envs, -1, -1)


def _reference_joint_pos(
    reference: MujocoMotionReference,
    joint_names: Sequence[str],
    step: int,
    num_envs: int,
) -> torch.Tensor:
    missing = [name for name in joint_names if name not in reference.joint_names]
    if missing:
        raise ValueError(f"Policy rollout reference is missing joints: {missing}.")
    indices = [reference.joint_names.index(name) for name in joint_names]
    return reference.joint_pos[step, indices].unsqueeze(0).expand(num_envs, -1)


def _normalize_steps(steps: Sequence[int] | torch.Tensor | None, num_steps: int) -> torch.Tensor:
    if steps is None:
        return torch.arange(num_steps, dtype=torch.long)
    steps_t = torch.as_tensor(list(steps), dtype=torch.long)
    if steps_t.numel() == 0:
        raise ValueError("steps must contain at least one frame.")
    if torch.any(steps_t < 0) or torch.any(steps_t >= num_steps):
        raise ValueError(f"steps must be in [0, {num_steps}), got {steps_t.tolist()}.")
    return steps_t


def _policy_action_history_steps(policy_bundle: MujocoPolicyBundle) -> int:
    max_steps = 1
    for group_cfg in policy_bundle.observation_builder.observation_cfg.values():
        for obs_key, params in group_cfg.items():
            if obs_key == "prev_actions":
                max_steps = max(max_steps, int((params or {}).get("steps", 1)))
    return max_steps


def _quat_rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    quat = quat.expand(*vec.shape[:-1], 4)
    xyz = quat[..., 1:]
    t = torch.linalg.cross(xyz, vec, dim=-1) * 2
    return vec + quat[..., 0:1] * t + torch.linalg.cross(xyz, t, dim=-1)
