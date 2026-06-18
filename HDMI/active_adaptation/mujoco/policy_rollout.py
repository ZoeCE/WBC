from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .motion_reference import MujocoMotionReference
from .observation_builder import MujocoPolicyState
from .policy import MujocoPolicyAction, MujocoPolicyBundle
from .playback_parity import compute_playback_parity


@dataclass(frozen=True)
class MujocoPolicyRolloutMetrics:
    q_l2: torch.Tensor
    body_pos_l2: torch.Tensor
    actions: torch.Tensor
    joint_position_targets: torch.Tensor


def run_mujoco_policy_rollout(
    *,
    scene: Any,
    policy_bundle: MujocoPolicyBundle,
    reference: MujocoMotionReference,
    steps: Sequence[int] | torch.Tensor | None = None,
    decimation: int = 1,
) -> MujocoPolicyRolloutMetrics:
    if decimation < 1:
        raise ValueError(f"decimation must be >= 1, got {decimation}.")

    from active_adaptation.envs import mujoco as mujoco_env

    sim = mujoco_env.MJSim(scene, realtime=False)
    robot = scene["robot"]
    if robot is None:
        raise KeyError("MuJoCo scene does not contain a 'robot' articulation.")

    policy_joint_ids = _policy_joint_ids(robot, policy_bundle.policy_joint_names)
    steps_t = _normalize_steps(steps, reference.num_steps)

    _write_reference_frame_to_scene(scene, reference, int(steps_t[0].item()))
    scene.update(0.0)

    action_history = torch.zeros(scene.num_envs, policy_bundle.action_dim, _policy_action_history_steps(policy_bundle))
    applied_action = torch.zeros(scene.num_envs, policy_bundle.action_dim)
    policy_bundle.reset(
        _policy_state_from_scene(
            scene=scene,
            reference=reference,
            step=int(steps_t[0].item()),
            policy_joint_ids=policy_joint_ids,
            applied_action=applied_action,
            action_history=action_history,
        )
    )

    actions: list[torch.Tensor] = []
    joint_targets: list[torch.Tensor] = []
    q_l2: list[torch.Tensor] = []
    body_pos_l2: list[torch.Tensor] = []

    for rollout_index, step_t in enumerate(steps_t):
        step = int(step_t.item())
        state = _policy_state_from_scene(
            scene=scene,
            reference=reference,
            step=step,
            policy_joint_ids=policy_joint_ids,
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

        applied_action = action.raw_action.detach()
        action_history = action_history.roll(1, dims=2)
        action_history[:, :, 0] = applied_action

    return MujocoPolicyRolloutMetrics(
        q_l2=torch.stack(q_l2),
        body_pos_l2=torch.stack(body_pos_l2),
        actions=torch.stack(actions),
        joint_position_targets=torch.stack(joint_targets),
    )


def _policy_joint_ids(robot: Any, policy_joint_names: Sequence[str]) -> list[int]:
    joint_ids, joint_names = robot.find_joints(list(policy_joint_names), preserve_order=True)
    if joint_names != list(policy_joint_names):
        raise ValueError(f"Policy joint order mismatch: expected {list(policy_joint_names)}, got {joint_names}.")
    return joint_ids


def _policy_state_from_scene(
    *,
    scene: Any,
    reference: MujocoMotionReference,
    step: int,
    policy_joint_ids: Sequence[int],
    applied_action: torch.Tensor,
    action_history: torch.Tensor,
) -> MujocoPolicyState:
    robot = scene["robot"]
    step_ids = torch.full((scene.num_envs,), step, dtype=torch.long)
    fields = reference.observation_fields_at(step_ids)
    return MujocoPolicyState(
        root_ang_vel_b=robot.data.root_ang_vel_b,
        projected_gravity_b=robot.data.projected_gravity_b,
        joint_pos=robot.data.joint_pos[:, policy_joint_ids],
        joint_pos_offset=torch.zeros_like(robot.data.joint_pos[:, policy_joint_ids]),
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
    )


def _apply_policy_action(robot: Any, action: MujocoPolicyAction, policy_joint_ids: Sequence[int]) -> None:
    robot.set_joint_position_target(action.joint_position_target, joint_ids=list(policy_joint_ids))


def _write_reference_frame_to_scene(scene: Any, reference: MujocoMotionReference, step: int) -> None:
    robot = scene["robot"]
    root_pos = reference.body_pos_w[step, reference.root_body_index].unsqueeze(0).expand(scene.num_envs, -1)
    root_quat = reference.body_quat_w[step, reference.root_body_index].unsqueeze(0).expand(scene.num_envs, -1)
    root_state = torch.cat([root_pos, root_quat, torch.zeros(scene.num_envs, 6)], dim=-1)
    robot.write_root_state_to_sim(root_state)

    joint_names = _reference_robot_joint_names(scene, robot, reference.requested_joint_names)
    if not joint_names:
        return
    joint_ids, found_names = robot.find_joints(joint_names, preserve_order=True)
    if found_names != joint_names:
        raise ValueError(f"Reference joint order mismatch: expected {joint_names}, got {found_names}.")
    joint_pos = _reference_joint_pos(reference, joint_names, step, scene.num_envs)
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), joint_ids=joint_ids)


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
