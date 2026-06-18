import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml


HDMI_ROOT = Path(__file__).resolve().parents[1]
if str(HDMI_ROOT) not in sys.path:
    sys.path.insert(0, str(HDMI_ROOT))

from active_adaptation.assets_mjcf import ROBOTS
from active_adaptation.mujoco import (
    MujocoMotionReference,
    MujocoPolicyBundle,
    MujocoPolicyRolloutMetrics,
    MujocoPolicyState,
    PlaybackParityMetrics,
    compute_kinematic_motion_playback_parity,
    run_mujoco_policy_rollout,
)
from active_adaptation.mujoco.task_mapping import validate_task_motion_mapping


KINEMATIC_REWARD_TERMS = {
    "keypoint_pos_tracking_product",
    "keypoint_position_tracking_product",
    "keypoint_pos_tracking_local_product",
    "keypoint_position_tracking_local_product",
    "keypoint_ori_tracking_product",
    "keypoint_orientation_tracking_product",
    "keypoint_ori_tracking_local_product",
    "keypoint_orientation_tracking_local_product",
    "keypoint_lin_vel_tracking_product",
    "keypoint_ang_vel_tracking_product",
    "joint_pos_tracking_product",
    "joint_position_tracking_product",
    "joint_vel_tracking_product",
    "joint_velocity_tracking_product",
    "survival",
    "joint_vel_l2",
    "joint_pos_limits",
    "action_rate_l2",
    "joint_torque_limits",
    "feet_slip",
    "impact_force_l2",
    "feet_air_time",
    "object_pos_tracking",
    "object_ori_tracking",
    "object_joint_pos_tracking",
    "eef_contact_exp",
    "eef_contact_exp_max",
    "eef_contact_all",
}

_OBJECT_POSE_OBS_KEYS = ("object_xy_b", "object_heading_b", "object_pos_b", "object_ori_b")


def run_parity(
    *,
    motion_dir: str | Path,
    robot_name: str = "g1_29dof",
    object_name: str | None = None,
    object_type: str | None = None,
    object_body_name: str | None = None,
    object_joint_name: str | None = None,
    root_body_name: str | None = None,
    contact_eef_body_names: Sequence[str] | None = None,
    contact_target_pos_offset: Sequence[Sequence[float]] | torch.Tensor | None = None,
    contact_eef_pos_offset: Sequence[Sequence[float]] | torch.Tensor | None = None,
    steps: Sequence[int] | None = None,
    num_envs: int = 1,
    reward_config: Mapping[str, Any] | None = None,
    policy_path: str | Path | None = None,
    policy_rollout: bool = False,
) -> dict[str, Any]:
    if policy_rollout and policy_path is None:
        raise ValueError("--policy-rollout requires --policy-path.")

    motion_dir = Path(motion_dir)
    meta = _load_motion_meta(motion_dir)
    body_names = list(meta["body_names"])
    joint_names = list(meta["joint_names"])
    root_body_name = root_body_name or body_names[0]

    scene = _build_scene(
        robot_name=robot_name,
        object_name=object_name,
        object_type=object_type,
        object_body_name=object_body_name,
        contact_eef_body_names=contact_eef_body_names,
        num_envs=num_envs,
    )
    reference = MujocoMotionReference.from_motion_dir(
        motion_dir=motion_dir,
        body_names=body_names,
        joint_names=joint_names,
        root_body_name=root_body_name,
        future_steps=[0],
    )
    reward_config, reward_terms_used, reward_terms_skipped = filter_kinematic_reward_config(reward_config)
    metrics = compute_kinematic_motion_playback_parity(
        scene=scene,
        reference=reference,
        steps=steps,
        reward_cfg=reward_config,
        object_name=object_name,
        object_body_name=object_body_name,
        object_joint_name=object_joint_name,
        contact_eef_body_names=contact_eef_body_names,
        contact_target_pos_offset=contact_target_pos_offset,
        contact_eef_pos_offset=contact_eef_pos_offset,
    )
    summary = summarize_metrics(
        metrics,
        reward_terms_used=reward_terms_used,
        reward_terms_skipped=reward_terms_skipped,
    )
    summary.update(
        {
            "motion_dir": str(motion_dir),
            "object_name": object_name,
            "object_body_name": object_body_name,
            "object_joint_name": object_joint_name,
            "root_body_name": root_body_name,
        }
    )
    if policy_path is not None:
        policy_reference = _policy_reference_from_export(
            motion_dir=motion_dir,
            policy_path=policy_path,
            fallback_body_names=body_names,
            fallback_joint_names=joint_names,
            fallback_root_body_name=root_body_name,
        )
        summary.update(
            run_policy_playback_smoke(
                policy_path=policy_path,
                reference=policy_reference,
                steps=steps,
                num_envs=num_envs,
            )
        )
    if policy_rollout:
        policy_reference = _policy_reference_from_export(
            motion_dir=motion_dir,
            policy_path=policy_path,
            fallback_body_names=body_names,
            fallback_joint_names=joint_names,
            fallback_root_body_name=root_body_name,
        )
        summary.update(
            summarize_policy_rollout_metrics(
                run_mujoco_policy_rollout(
                    scene=scene,
                    policy_bundle=MujocoPolicyBundle.load(policy_path),
                    reference=policy_reference,
                    steps=steps,
                )
            )
        )
    return summary


def summarize_metrics(
    metrics: PlaybackParityMetrics,
    *,
    reward_terms_used: Sequence[str] = (),
    reward_terms_skipped: Sequence[str] = (),
) -> dict[str, Any]:
    reward = metrics.reward
    summary = {
        "steps": int(metrics.q_l2.shape[0]),
        "envs": int(metrics.q_l2.shape[1]) if metrics.q_l2.ndim > 1 else 1,
        "q_l2_max": float(metrics.q_l2.max().item()),
        "q_l2_mean": float(metrics.q_l2.mean().item()),
        "body_pos_l2_max": float(metrics.body_pos_l2.max().item()),
        "body_pos_l2_mean": float(metrics.body_pos_l2.mean().item()),
        "reward_shape": None,
        "reward_mean": None,
        "reward_terms_used": list(reward_terms_used),
        "reward_terms_skipped": list(reward_terms_skipped),
    }
    if reward is not None:
        summary["reward_shape"] = list(reward.shape)
        summary["reward_mean"] = float(reward.mean().item())
    return summary


def summarize_policy_rollout_metrics(metrics: MujocoPolicyRolloutMetrics) -> dict[str, Any]:
    return {
        "policy_rollout_q_l2_shape": list(metrics.q_l2.shape),
        "policy_rollout_q_l2_max": float(metrics.q_l2.max().item()),
        "policy_rollout_body_pos_l2_shape": list(metrics.body_pos_l2.shape),
        "policy_rollout_body_pos_l2_max": float(metrics.body_pos_l2.max().item()),
        "policy_rollout_action_shape": list(metrics.actions.shape),
        "policy_rollout_joint_target_shape": list(metrics.joint_position_targets.shape),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    with contextlib.redirect_stdout(sys.stderr):
        task_cfg = _load_task_config_from_args(args)
        playback_inputs = _resolve_playback_inputs(args, task_cfg)
        contact_inputs = _resolve_contact_inputs(args, task_cfg)
        mapping_report = _task_mapping_report_from_args(args, task_cfg)
        reward_config = _load_reward_config_from_args(args, task_cfg)
        summary = run_parity(
            motion_dir=playback_inputs["motion_dir"],
            robot_name=args.robot_name,
            object_name=playback_inputs["object_name"],
            object_type=args.object_type,
            object_body_name=playback_inputs["object_body_name"],
            object_joint_name=playback_inputs["object_joint_name"],
            root_body_name=playback_inputs["root_body_name"],
            contact_eef_body_names=contact_inputs["contact_eef_body_names"],
            contact_target_pos_offset=contact_inputs["contact_target_pos_offset"],
            contact_eef_pos_offset=contact_inputs["contact_eef_pos_offset"],
            steps=_parse_steps(args.steps),
            num_envs=args.num_envs,
            reward_config=reward_config,
            policy_path=args.policy_path,
            policy_rollout=args.policy_rollout,
        )
        if mapping_report is not None:
            summary.update(_task_mapping_summary(mapping_report))
    print(json.dumps(summary, sort_keys=True))
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MuJoCo kinematic playback parity on a motion.npz + meta.json directory."
    )
    parser.add_argument(
        "--motion-dir",
        default=None,
        help="Directory containing motion.npz and meta.json. Defaults to task YAML command.data_path.",
    )
    parser.add_argument("--robot-name", default="g1_29dof")
    parser.add_argument("--object-name", default=None)
    parser.add_argument("--object-type", default=None)
    parser.add_argument("--object-body-name", default=None)
    parser.add_argument("--object-joint-name", default=None)
    parser.add_argument("--root-body-name", default=None)
    parser.add_argument("--steps", default=None, help="Comma-separated playback frame indices. Defaults to all frames.")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--reward-config-json", default=None)
    parser.add_argument("--task-yaml", default=None, help="HDMI task YAML. Its reward section is used for playback.")
    parser.add_argument("--policy-path", default=None, help="Exported HDMI policy .pt to smoke-run on playback states.")
    parser.add_argument(
        "--policy-rollout",
        action="store_true",
        help="Run the exported policy in a closed-loop MuJoCo rollout and report rollout parity metrics.",
    )
    return parser.parse_args(argv)


def _parse_steps(raw_steps: str | None) -> list[int] | None:
    if raw_steps is None or raw_steps == "":
        return None
    steps = []
    for raw_step in raw_steps.split(","):
        raw_step = raw_step.strip()
        if not raw_step:
            continue
        steps.append(int(raw_step))
    return steps


def _load_motion_meta(motion_dir: Path) -> dict[str, Any]:
    meta_path = motion_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing motion metadata: {meta_path}")
    return json.loads(meta_path.read_text())


def _policy_reference_from_export(
    *,
    motion_dir: Path,
    policy_path: str | Path,
    fallback_body_names: Sequence[str],
    fallback_joint_names: Sequence[str],
    fallback_root_body_name: str,
) -> MujocoMotionReference:
    policy_cfg = _load_yaml_mapping(_policy_config_path(policy_path))
    command_cfg = _policy_observation_group(policy_cfg, ("command", "command_"))
    if command_cfg is None:
        return MujocoMotionReference.from_motion_dir(
            motion_dir=motion_dir,
            body_names=list(fallback_body_names),
            joint_names=list(fallback_joint_names),
            root_body_name=fallback_root_body_name,
            future_steps=[0],
        )

    return MujocoMotionReference.from_motion_dir(
        motion_dir=motion_dir,
        body_names=_policy_observation_list(command_cfg, "body_names", fallback_body_names, str),
        joint_names=_policy_observation_list(command_cfg, "joint_names", fallback_joint_names, str),
        root_body_name=str(_policy_observation_value(command_cfg, "root_body_name", fallback_root_body_name)),
        future_steps=_policy_observation_list(command_cfg, "future_steps", [0], int),
    )


def _policy_config_path(policy_path: str | Path) -> Path:
    policy_path = Path(policy_path)
    for suffix in (".yaml", ".yml"):
        candidate = policy_path.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    return policy_path.with_suffix(".yaml")


def _policy_observation_group(
    policy_cfg: Mapping[str, Any],
    group_names: Sequence[str],
) -> Mapping[str, Any] | None:
    observation_cfg = policy_cfg.get("observation", {})
    if not isinstance(observation_cfg, Mapping):
        raise ValueError("Exported policy observation config must be a mapping.")
    for group_name in group_names:
        group_cfg = observation_cfg.get(group_name)
        if group_cfg is None:
            continue
        if not isinstance(group_cfg, Mapping):
            raise ValueError(f"Exported policy observation group {group_name!r} must be a mapping.")
        return group_cfg
    return None


def _policy_observation_value(
    group_cfg: Mapping[str, Any],
    key: str,
    fallback: Any,
) -> Any:
    for obs_cfg in group_cfg.values():
        if isinstance(obs_cfg, Mapping) and key in obs_cfg:
            return obs_cfg[key]
    return fallback


def _policy_observation_list(
    group_cfg: Mapping[str, Any],
    key: str,
    fallback: Sequence[Any],
    item_type: type,
) -> list[Any]:
    value = _policy_observation_value(group_cfg, key, fallback)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"Exported policy observation {key!r} must be a sequence, got {type(value).__name__}.")
    return [item_type(item) for item in value]


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _load_task_config_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.task_yaml is None:
        return None
    return load_task_config(args.task_yaml)


def _resolve_playback_inputs(
    args: argparse.Namespace,
    task_cfg: Mapping[str, Any] | None,
) -> dict[str, Any]:
    command_cfg = _task_command_config(task_cfg, args.task_yaml) if task_cfg is not None else {}
    motion_dir = args.motion_dir
    if motion_dir is None:
        data_path = command_cfg.get("data_path")
        if data_path is None:
            raise ValueError("--motion-dir is required unless --task-yaml command.data_path is set.")
        motion_dir = _resolve_task_data_path(Path(args.task_yaml), data_path)

    return {
        "motion_dir": motion_dir,
        "object_name": args.object_name or command_cfg.get("object_asset_name"),
        "object_body_name": args.object_body_name or command_cfg.get("object_body_name"),
        "object_joint_name": args.object_joint_name or command_cfg.get("object_joint_name"),
        "root_body_name": args.root_body_name or command_cfg.get("root_body_name"),
    }


def _task_mapping_report_from_args(
    args: argparse.Namespace,
    task_cfg: Mapping[str, Any] | None,
):
    if task_cfg is None:
        return None
    command_cfg = _task_command_config(task_cfg, args.task_yaml)
    if "object_asset_name" not in command_cfg:
        return None
    if any(
        value is not None
        for value in (
            args.motion_dir,
            args.object_name,
            args.object_type,
            args.object_body_name,
            args.object_joint_name,
            args.root_body_name,
        )
    ):
        return None
    return validate_task_motion_mapping(args.task_yaml, robot_name=args.robot_name)


def _task_mapping_summary(mapping_report) -> dict[str, Any]:
    return {
        "task_object_body_name": mapping_report.task_object_body_name,
        "reference_object_body_name": mapping_report.reference_object_body_name,
        "asset_object_body_names": list(mapping_report.asset_object_body_names),
        "asset_object_joint_names": list(mapping_report.asset_object_joint_names),
        "extra_object_names": list(mapping_report.extra_object_names),
    }


def _resolve_contact_inputs(
    args: argparse.Namespace,
    task_cfg: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if task_cfg is None:
        return {
            "contact_eef_body_names": None,
            "contact_target_pos_offset": None,
            "contact_eef_pos_offset": None,
        }
    command_cfg = _task_command_config(task_cfg, args.task_yaml)
    return {
        "contact_eef_body_names": command_cfg.get("contact_eef_body_name"),
        "contact_target_pos_offset": command_cfg.get("contact_target_pos_offset"),
        "contact_eef_pos_offset": command_cfg.get("contact_eef_pos_offset"),
    }


def _task_command_config(
    task_cfg: Mapping[str, Any],
    task_yaml: str | Path | None,
) -> Mapping[str, Any]:
    command_cfg = task_cfg.get("command", {})
    if not isinstance(command_cfg, Mapping):
        raise ValueError(f"Task YAML command section must be a mapping: {task_yaml}")
    return command_cfg


def _resolve_task_data_path(task_yaml: Path, data_path: str | Path) -> Path:
    data_path = Path(data_path)
    if data_path.is_absolute():
        return data_path

    roots = (_task_project_root(task_yaml), HDMI_ROOT, Path.cwd())
    for root in roots:
        candidate = root / data_path
        if candidate.exists():
            return candidate
    return roots[0] / data_path


def _task_project_root(task_yaml: Path) -> Path:
    resolved = task_yaml.resolve()
    parts = resolved.parts
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "cfg":
            return Path(*parts[:index])
    return HDMI_ROOT


def _load_reward_config_from_args(
    args: argparse.Namespace,
    task_cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    if args.reward_config_json:
        return _load_json(args.reward_config_json)
    if task_cfg is not None:
        return _task_reward_config(task_cfg, args.task_yaml)
    if args.task_yaml:
        return load_task_reward_config(args.task_yaml)
    return None


def load_task_config(task_yaml: str | Path) -> dict[str, Any]:
    return _load_task_yaml_with_defaults(Path(task_yaml))


def run_policy_playback_smoke(
    *,
    policy_path: str | Path,
    reference: MujocoMotionReference,
    steps: Sequence[int] | None,
    num_envs: int,
) -> dict[str, Any]:
    bundle = MujocoPolicyBundle.load(policy_path)
    steps_t = _normalize_policy_steps(steps, reference.num_steps)
    action_history = torch.zeros(num_envs, bundle.action_dim, _policy_action_history_steps(bundle))
    applied_action = torch.zeros(num_envs, bundle.action_dim)
    default_joint_pos = bundle.observation_default_joint_pos.unsqueeze(0).expand(num_envs, -1)

    raw_actions: list[torch.Tensor] = []
    joint_targets: list[torch.Tensor] = []
    for index, step in enumerate(steps_t.tolist()):
        state = _policy_state_from_reference(
            bundle=bundle,
            reference=reference,
            step=step,
            num_envs=num_envs,
            joint_pos=default_joint_pos,
            applied_action=applied_action,
            action_history=action_history,
        )
        if index == 0:
            bundle.reset(state)
        else:
            bundle.update(state)
        action = bundle.act(state, is_init=(index == 0))
        raw_action = action.raw_action.detach().cpu()
        raw_actions.append(raw_action)
        joint_targets.append(action.joint_position_target.detach().cpu())

        applied_action = raw_action
        action_history = action_history.roll(1, dims=2)
        action_history[:, :, 0] = raw_action

    actions_t = torch.stack(raw_actions)
    targets_t = torch.stack(joint_targets)
    return {
        "policy_path": str(policy_path),
        "policy_action_shape": list(actions_t.shape),
        "policy_joint_target_shape": list(targets_t.shape),
        "policy_action_mean": float(actions_t.mean().item()),
        "policy_action_max_abs": float(actions_t.abs().max().item()),
        "policy_joint_target_mean": float(targets_t.mean().item()),
    }


def _policy_state_from_reference(
    *,
    bundle: MujocoPolicyBundle,
    reference: MujocoMotionReference,
    step: int,
    num_envs: int,
    joint_pos: torch.Tensor,
    applied_action: torch.Tensor,
    action_history: torch.Tensor,
) -> MujocoPolicyState:
    fields = reference.observation_fields_at(torch.full((num_envs,), step, dtype=torch.long))
    zeros_ang_vel = torch.zeros(num_envs, 3, dtype=joint_pos.dtype)
    projected_gravity = torch.zeros(num_envs, 3, dtype=joint_pos.dtype)
    projected_gravity[:, 2] = -1.0
    object_state = _policy_object_state_from_reference(
        bundle=bundle,
        reference=reference,
        step=step,
        num_envs=num_envs,
        dtype=joint_pos.dtype,
    )
    return MujocoPolicyState(
        root_ang_vel_b=zeros_ang_vel,
        projected_gravity_b=projected_gravity,
        joint_pos=joint_pos,
        joint_pos_offset=torch.zeros_like(joint_pos),
        applied_action=applied_action,
        action_history=action_history,
        ref_body_pos_future_w=fields.ref_body_pos_future_w,
        ref_root_pos_w=fields.ref_root_pos_w,
        ref_root_quat_w=fields.ref_root_quat_w,
        ref_joint_pos_future=fields.ref_joint_pos_future,
        motion_t=fields.motion_t,
        motion_len=fields.motion_len,
        robot_root_pos_w=fields.ref_root_pos_w,
        robot_root_quat_w=fields.ref_root_quat_w,
        **object_state,
    )


def _normalize_policy_steps(steps: Sequence[int] | None, num_steps: int) -> torch.Tensor:
    if steps is None:
        return torch.arange(num_steps, dtype=torch.long)
    return torch.as_tensor(list(steps), dtype=torch.long)


def _policy_action_history_steps(bundle: MujocoPolicyBundle) -> int:
    max_steps = 1
    for group_cfg in bundle.observation_builder.observation_cfg.values():
        for obs_key, params in group_cfg.items():
            if obs_key == "prev_actions":
                max_steps = max(max_steps, int((params or {}).get("steps", 1)))
    return max_steps


def _policy_object_state_from_reference(
    *,
    bundle: MujocoPolicyBundle,
    reference: MujocoMotionReference,
    step: int,
    num_envs: int,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    object_name = _first_policy_observation_object_name(bundle, _OBJECT_POSE_OBS_KEYS)
    contact_cfg = _first_policy_observation_cfg(bundle, "ref_contact_pos_b")
    if object_name is None and contact_cfg is not None:
        object_name = contact_cfg.get("object_name")

    state: dict[str, torch.Tensor] = {}
    if object_name is not None:
        object_pos_w, object_quat_w = _reference_body_pose(
            reference=reference,
            body_name=str(object_name),
            step=step,
            num_envs=num_envs,
            dtype=dtype,
        )
        state["object_pos_w"] = object_pos_w
        state["object_quat_w"] = object_quat_w

    if contact_cfg is not None:
        contact_object_name = contact_cfg.get("object_name", object_name)
        if contact_object_name is None:
            raise ValueError("ref_contact_pos_b policy observation requires object_name.")
        contact_object_pos_w, contact_object_quat_w = _reference_body_pose(
            reference=reference,
            body_name=str(contact_object_name),
            step=step,
            num_envs=num_envs,
            dtype=dtype,
        )
        offsets = torch.as_tensor(
            contact_cfg.get("contact_target_pos_offset", [[0.0, 0.0, 0.0]]),
            dtype=dtype,
        )
        if offsets.ndim == 1:
            offsets = offsets.unsqueeze(0)
        if offsets.shape[-1] != 3:
            raise ValueError(f"contact_target_pos_offset must end in xyz dim 3, got {tuple(offsets.shape)}.")
        offsets = offsets.unsqueeze(0).expand(num_envs, -1, -1)
        state["contact_target_pos_w"] = contact_object_pos_w[:, None, :] + _quat_rotate(
            contact_object_quat_w[:, None, :],
            offsets,
        )
    return state


def _first_policy_observation_object_name(
    bundle: MujocoPolicyBundle,
    obs_keys: Sequence[str],
) -> str | None:
    for obs_key in obs_keys:
        obs_cfg = _first_policy_observation_cfg(bundle, obs_key)
        if obs_cfg is not None and obs_cfg.get("object_name") is not None:
            return str(obs_cfg["object_name"])
    return None


def _first_policy_observation_cfg(bundle: MujocoPolicyBundle, obs_key: str) -> Mapping[str, Any] | None:
    for group_cfg in bundle.observation_builder.observation_cfg.values():
        if obs_key not in group_cfg:
            continue
        obs_cfg = group_cfg[obs_key] or {}
        if not isinstance(obs_cfg, Mapping):
            raise ValueError(f"Policy observation {obs_key!r} config must be a mapping.")
        return obs_cfg
    return None


def _reference_body_pose(
    *,
    reference: MujocoMotionReference,
    body_name: str,
    step: int,
    num_envs: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if body_name not in reference.body_names:
        raise ValueError(f"Policy observation body {body_name!r} is missing from motion metadata.")
    body_index = reference.body_names.index(body_name)
    body_pos_w = reference.body_pos_w[step, body_index].to(dtype=dtype).unsqueeze(0).expand(num_envs, -1)
    body_quat_w = reference.body_quat_w[step, body_index].to(dtype=dtype).unsqueeze(0).expand(num_envs, -1)
    return body_pos_w, body_quat_w


def _quat_rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    quat = quat.expand(*vec.shape[:-1], 4)
    xyz = quat[..., 1:]
    t = torch.linalg.cross(xyz, vec, dim=-1) * 2
    return vec + quat[..., 0:1] * t + torch.linalg.cross(xyz, t, dim=-1)


def load_task_reward_config(task_yaml: str | Path) -> dict[str, Any]:
    cfg = load_task_config(task_yaml)
    return _task_reward_config(cfg, task_yaml)


def _task_reward_config(task_cfg: Mapping[str, Any], task_yaml: str | Path | None) -> dict[str, Any]:
    cfg = dict(task_cfg)
    reward_cfg = cfg.get("reward", {})
    if not isinstance(reward_cfg, dict):
        raise ValueError(
            f"Task YAML reward section must be a mapping in {task_yaml}, got {type(reward_cfg).__name__}."
        )
    return reward_cfg


def filter_kinematic_reward_config(
    reward_config: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, list[str], list[str]]:
    if reward_config is None:
        return None, [], []

    filtered: dict[str, Any] = {}
    used: list[str] = []
    skipped: list[str] = []
    for group_name, group_cfg in reward_config.items():
        if group_name == "_mult_dt_":
            continue
        if not isinstance(group_cfg, Mapping):
            continue

        group_out: dict[str, Any] = {}
        if bool(group_cfg.get("_multiplicative", False)):
            group_out["_multiplicative"] = True

        for term_name, term_cfg in group_cfg.items():
            if term_name == "_multiplicative" or term_cfg is None:
                continue
            if not isinstance(term_cfg, Mapping):
                term_cfg = {}
            if not bool(term_cfg.get("enabled", True)):
                continue

            formula_name = _reward_formula_name(term_name)
            qualified_name = f"{group_name}.{term_name}"
            if formula_name not in KINEMATIC_REWARD_TERMS:
                skipped.append(qualified_name)
                continue
            group_out[term_name] = dict(term_cfg)
            used.append(qualified_name)

        if any(key != "_multiplicative" for key in group_out):
            filtered[group_name] = group_out

    return (filtered or None), used, skipped


def _load_task_yaml_with_defaults(path: Path) -> dict[str, Any]:
    raw_cfg = _load_yaml_mapping(path)
    defaults = raw_cfg.pop("defaults", None)
    self_cfg = raw_cfg
    if defaults is None:
        return self_cfg

    merged: dict[str, Any] = {}
    inserted_self = False
    for entry in defaults:
        if entry == "_self_":
            merged = _deep_merge_dicts(merged, self_cfg)
            inserted_self = True
            continue
        default_path = _resolve_default_yaml(path, entry)
        merged = _deep_merge_dicts(merged, _load_task_yaml_with_defaults(default_path))
    if not inserted_self:
        merged = _deep_merge_dicts(merged, self_cfg)
    return merged


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text()) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return cfg


def _resolve_default_yaml(path: Path, entry: Any) -> Path:
    if isinstance(entry, Mapping):
        if len(entry) != 1:
            raise ValueError(f"Unsupported Hydra default entry in {path}: {entry!r}")
        entry = next(iter(entry.values()))
    if not isinstance(entry, str):
        raise ValueError(f"Unsupported Hydra default entry in {path}: {entry!r}")
    if entry == "_self_":
        return path

    rel = Path(entry.lstrip("/")).with_suffix(".yaml")
    for root in (path.parent, *path.parents):
        candidate = root / rel
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not resolve Hydra default {entry!r} from {path}.")


def _deep_merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _reward_formula_name(term_name: str) -> str:
    if "(" not in term_name:
        return term_name
    if not term_name.endswith(")"):
        raise ValueError(f"Invalid reward term alias syntax: {term_name!r}.")
    return term_name.rsplit("(", 1)[1][:-1]


def _build_scene(
    *,
    robot_name: str,
    object_name: str | None,
    object_type: str | None,
    object_body_name: str | None,
    contact_eef_body_names: Sequence[str] | None,
    num_envs: int,
):
    from active_adaptation.envs import mujoco as mujoco_env

    class SceneCfg:
        pass

    if object_name is None:
        SceneCfg.robot = ROBOTS[robot_name]
    else:
        SceneCfg.robot = ROBOTS.with_object(
            robot_name,
            object_asset_name=object_name,
            object_type=object_type,
        )
        if object_body_name is not None:
            for eef_body_name in _contact_eef_body_name_list(contact_eef_body_names):
                contact_sensor_name = f"{eef_body_name}_{object_name}_contact_forces"
                setattr(
                    SceneCfg,
                    contact_sensor_name,
                    mujoco_env.MJContactSensorCfg(
                        target="robot",
                        body_names=[eef_body_name],
                        filter_body_names=[object_body_name],
                    ),
                )
    SceneCfg.contact_forces = "robot"
    return mujoco_env.MJScene(SceneCfg(), num_envs=num_envs, launch_viewer=False)


def _contact_eef_body_name_list(value: Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [str(value)]
    return [str(item) for item in value]


if __name__ == "__main__":
    raise SystemExit(main())
