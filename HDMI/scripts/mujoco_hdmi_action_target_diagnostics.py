from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
HDMI_ROOT = SCRIPT_DIR.parent
if str(HDMI_ROOT) not in sys.path:
    sys.path.insert(0, str(HDMI_ROOT))

from active_adaptation.mujoco import MujocoPolicyBundle, run_mujoco_policy_rollout
from scripts import mujoco_playback_parity as playback


DEFAULT_OUTPUT_DIR = Path("/tmp/wbc_hdmi_goal_full_parity_v2/diagnostics/open_loop/action_target")
DEFAULT_CONTACT_EEF_BODY_NAMES = ("left_wrist_yaw_link", "right_wrist_yaw_link")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    jobs = _parse_jobs(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_reports: list[dict[str, Any]] = []
    for job in jobs:
        try:
            report = run_action_target_diagnostic(
                task_yaml=job.task_yaml,
                policy_path=job.policy_path,
                robot_name=args.robot_name,
                object_type=args.object_type,
                steps=playback._parse_steps(args.steps),
                max_steps=args.max_steps,
                num_envs=args.num_envs,
                decimation=args.decimation,
                initial_state=args.initial_state,
                near_zero_action_threshold=args.near_zero_action_threshold,
                target_ref_l2_threshold=args.target_ref_l2_threshold,
            )
            report_path = output_dir / f"{report['task_stem']}_action_target_diagnostic.json"
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
            report["report_path"] = str(report_path)
        except Exception as exc:  # pragma: no cover - exercised by real fleet runs.
            report = {
                **_task_identity(job.task_yaml),
                "policy_path": str(job.policy_path),
                "error": f"{type(exc).__name__}: {exc}",
                "diagnosis_flags": [],
                "gate_passed": False,
            }
        task_reports.append(report)

    aggregate = aggregate_action_target_reports(task_reports)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(aggregate, sort_keys=True))
    return 0 if aggregate["error_count"] == 0 else 1


def run_action_target_diagnostic(
    *,
    task_yaml: str | Path,
    policy_path: str | Path,
    robot_name: str | None = None,
    object_type: str | None = None,
    steps: Sequence[int] | None = None,
    max_steps: int | None = 30,
    num_envs: int = 1,
    decimation: int | None = None,
    initial_state: str = "reference_frame",
    near_zero_action_threshold: float = 0.05,
    target_ref_l2_threshold: float = 0.5,
) -> dict[str, Any]:
    task_yaml = Path(task_yaml)
    policy_path = Path(policy_path)
    task_cfg = playback.load_task_config(task_yaml)
    command_cfg = _task_command_config(task_cfg, task_yaml)
    motion_dir = playback._resolve_task_data_path(task_yaml, command_cfg["data_path"])
    motion_meta = playback._load_motion_meta(motion_dir)
    body_names = list(motion_meta["body_names"])
    joint_names = list(motion_meta["joint_names"])
    root_body_name = str(command_cfg.get("root_body_name") or body_names[0])
    object_name = command_cfg.get("object_asset_name")
    object_body_name = command_cfg.get("object_body_name")
    contact_eef_body_names = _contact_eef_body_names_from_command(command_cfg)
    diagnostic_inputs = resolve_diagnostic_scene_inputs(
        robot_name=robot_name,
        object_type=object_type,
        task_cfg=task_cfg,
        command_cfg=command_cfg,
    )

    scene = playback._build_scene(
        robot_name=diagnostic_inputs["robot_name"],
        object_name=object_name,
        object_type=diagnostic_inputs["object_type"],
        object_body_name=object_body_name,
        contact_eef_body_names=contact_eef_body_names,
        num_envs=num_envs,
    )
    policy_reference = playback._policy_reference_from_export(
        motion_dir=motion_dir,
        policy_path=policy_path,
        fallback_body_names=body_names,
        fallback_joint_names=joint_names,
        fallback_root_body_name=root_body_name,
    )
    policy_bundle = MujocoPolicyBundle.load(
        policy_path,
        policy_config_overrides=diagnostic_inputs["policy_config_overrides"],
    )
    steps_t = _diagnostic_steps(
        requested_steps=steps,
        num_reference_steps=policy_reference.num_steps,
        max_steps=max_steps,
    )
    rollout_decimation = decimation if decimation is not None else playback._policy_decimation_from_task_cfg(task_cfg)
    action_adapter_config = playback._policy_action_adapter_config_from_task_cfg(task_cfg)
    metrics = run_mujoco_policy_rollout(
        scene=scene,
        policy_bundle=policy_bundle,
        reference=policy_reference,
        steps=steps_t,
        decimation=rollout_decimation,
        action_adapter_config=action_adapter_config,
        object_name=str(object_name) if object_name is not None else None,
        object_body_name=str(object_body_name) if object_body_name is not None else None,
        object_joint_name=str(command_cfg.get("object_joint_name")) if command_cfg.get("object_joint_name") is not None else None,
        contact_eef_body_names=contact_eef_body_names,
        contact_target_pos_offset=command_cfg.get("contact_target_pos_offset"),
        contact_eef_pos_offset=command_cfg.get("contact_eef_pos_offset"),
        initial_state=initial_state,
    )
    reference_joint_pos = reference_joint_pos_for_policy(
        reference=policy_reference,
        policy_joint_names=policy_bundle.policy_joint_names,
        steps=steps_t,
        num_envs=num_envs,
    )
    return summarize_action_target_metrics(
        task_identity=_task_identity(task_yaml),
        policy_path=policy_path,
        policy_config_path=policy_bundle.config_path,
        steps=steps_t,
        decimation=rollout_decimation,
        initial_state=initial_state,
        actions=metrics.actions,
        joint_position_targets=metrics.joint_position_targets,
        reference_joint_pos=reference_joint_pos,
        policy_joint_names=policy_bundle.policy_joint_names,
        robot_name=diagnostic_inputs["robot_name"],
        object_type=diagnostic_inputs["object_type"],
        q_l2=metrics.q_l2,
        body_pos_l2=metrics.body_pos_l2,
        near_zero_action_threshold=near_zero_action_threshold,
        target_ref_l2_threshold=target_ref_l2_threshold,
    )


def summarize_action_target_metrics(
    *,
    task_identity: Mapping[str, str],
    policy_path: str | Path,
    policy_config_path: str | Path,
    steps: Sequence[int] | torch.Tensor,
    decimation: int,
    actions: torch.Tensor,
    joint_position_targets: torch.Tensor,
    reference_joint_pos: torch.Tensor,
    q_l2: torch.Tensor,
    body_pos_l2: torch.Tensor,
    policy_joint_names: Sequence[str] | None = None,
    robot_name: str | None = None,
    object_type: str | None = None,
    initial_state: str = "reference_frame",
    near_zero_action_threshold: float = 0.05,
    target_ref_l2_threshold: float = 0.5,
) -> dict[str, Any]:
    actions = _require_series_tensor(actions, "actions")
    joint_position_targets = _require_series_tensor(joint_position_targets, "joint_position_targets")
    reference_joint_pos = _require_series_tensor(reference_joint_pos, "reference_joint_pos")
    if joint_position_targets.shape != reference_joint_pos.shape:
        raise ValueError(
            "joint_position_targets and reference_joint_pos must have matching shapes, "
            f"got {tuple(joint_position_targets.shape)} and {tuple(reference_joint_pos.shape)}."
        )
    if actions.shape[:2] != joint_position_targets.shape[:2]:
        raise ValueError(
            "actions and joint_position_targets must share time/env dimensions, "
            f"got {tuple(actions.shape)} and {tuple(joint_position_targets.shape)}."
        )
    if policy_joint_names is not None and len(policy_joint_names) != joint_position_targets.shape[-1]:
        raise ValueError(
            f"policy_joint_names length {len(policy_joint_names)} does not match target dimension "
            f"{joint_position_targets.shape[-1]}."
        )

    steps_t = torch.as_tensor(list(steps) if not isinstance(steps, torch.Tensor) else steps, dtype=torch.long)
    if steps_t.numel() != actions.shape[0]:
        raise ValueError(f"steps length {steps_t.numel()} does not match action time dimension {actions.shape[0]}.")
    q_l2 = _require_metric_tensor(q_l2, actions.shape[:2], "q_l2")
    body_pos_l2 = _require_metric_tensor(body_pos_l2, actions.shape[:2], "body_pos_l2")

    target_ref_l2 = torch.linalg.vector_norm(joint_position_targets - reference_joint_pos, dim=-1)
    target_ref_abs_error = (joint_position_targets - reference_joint_pos).abs()
    raw_action_abs_max_by_step = actions.abs().amax(dim=(1, 2))
    raw_action_l2_by_step = torch.linalg.vector_norm(actions, dim=-1).amax(dim=1)
    target_ref_l2_max_by_step = target_ref_l2.amax(dim=1)
    target_abs_max_by_step = joint_position_targets.abs().amax(dim=(1, 2))
    reference_abs_max_by_step = reference_joint_pos.abs().amax(dim=(1, 2))
    q_l2_max_by_step = q_l2.amax(dim=1)
    body_pos_l2_max_by_step = body_pos_l2.amax(dim=1)

    rows = [
        {
            "index": int(index),
            "step": int(steps_t[index].item()),
            "raw_action_abs_max": _json_float(raw_action_abs_max_by_step[index]),
            "raw_action_l2_max": _json_float(raw_action_l2_by_step[index]),
            "joint_target_ref_l2_max": _json_float(target_ref_l2_max_by_step[index]),
            "joint_target_abs_max": _json_float(target_abs_max_by_step[index]),
            "reference_joint_abs_max": _json_float(reference_abs_max_by_step[index]),
            "q_l2_max": _json_float(q_l2_max_by_step[index]),
            "body_pos_l2_max": _json_float(body_pos_l2_max_by_step[index]),
        }
        for index in range(actions.shape[0])
    ]
    first = rows[0]
    flags = []
    if first["raw_action_abs_max"] <= near_zero_action_threshold:
        flags.append("near_zero_initial_action")
    if first["joint_target_ref_l2_max"] >= target_ref_l2_threshold:
        flags.append("target_far_from_reference_at_initial_state")

    return {
        **dict(task_identity),
        "policy_path": str(policy_path),
        "policy_config_path": str(policy_config_path),
        "decimation": int(decimation),
        "initial_state": str(initial_state),
        "robot_name": robot_name,
        "object_type": object_type,
        "step_count": int(actions.shape[0]),
        "env_count": int(actions.shape[1]),
        "action_dim": int(actions.shape[2]),
        "near_zero_action_threshold": float(near_zero_action_threshold),
        "target_ref_l2_threshold": float(target_ref_l2_threshold),
        "diagnosis_flags": flags,
        "gate_passed": not flags,
        "max_raw_action_abs": _json_float(actions.abs().max()),
        "mean_raw_action_abs": _json_float(actions.abs().mean()),
        "max_joint_target_ref_l2": _json_float(target_ref_l2.max()),
        "mean_joint_target_ref_l2": _json_float(target_ref_l2.mean()),
        "top_joint_target_ref_errors": _top_joint_target_ref_errors(
            target_ref_abs_error=target_ref_abs_error,
            steps=steps_t,
            policy_joint_names=policy_joint_names,
        ),
        "max_q_l2": _json_float(q_l2.max()),
        "mean_q_l2": _json_float(q_l2.mean()),
        "max_body_pos_l2": _json_float(body_pos_l2.max()),
        "mean_body_pos_l2": _json_float(body_pos_l2.mean()),
        "first_step": first,
        "last_step": rows[-1],
        "rows": rows,
    }


def resolve_diagnostic_scene_inputs(
    *,
    robot_name: str | None,
    object_type: str | None,
    task_cfg: Mapping[str, Any],
    command_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    resolved_robot_name = playback._resolve_robot_name_from_task_cfg(robot_name, task_cfg)
    resolved_object_type = playback._resolve_object_type_from_task_cfg(
        object_type,
        command_cfg,
        robot_name=resolved_robot_name,
    )
    return {
        "robot_name": resolved_robot_name,
        "object_type": resolved_object_type,
        "policy_config_overrides": playback._policy_config_overrides_from_task_cfg(task_cfg),
    }


def reference_joint_pos_for_policy(
    *,
    reference: Any,
    policy_joint_names: Sequence[str],
    steps: Sequence[int] | torch.Tensor,
    num_envs: int,
) -> torch.Tensor:
    missing = [name for name in policy_joint_names if name not in reference.joint_names]
    if missing:
        raise ValueError(f"Policy action-target diagnostic reference is missing joints: {missing}.")
    indices = [reference.joint_names.index(name) for name in policy_joint_names]
    steps_t = torch.as_tensor(list(steps) if not isinstance(steps, torch.Tensor) else steps, dtype=torch.long)
    values = reference.joint_pos[steps_t][:, indices]
    return values.unsqueeze(1).expand(-1, num_envs, -1).contiguous()


def aggregate_action_target_reports(task_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tasks = [dict(report) for report in task_reports]
    return {
        "task_count": len(tasks),
        "error_count": sum(1 for task in tasks if task.get("error")),
        "gate_passed_count": sum(1 for task in tasks if task.get("gate_passed") is True),
        "near_zero_initial_action_count": sum(
            1 for task in tasks if "near_zero_initial_action" in task.get("diagnosis_flags", [])
        ),
        "target_far_initial_count": sum(
            1
            for task in tasks
            if "target_far_from_reference_at_initial_state" in task.get("diagnosis_flags", [])
        ),
        "tasks": tasks,
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose HDMI MuJoCo policy rollout action targets against motion-reference joint poses."
    )
    parser.add_argument("--task-policy", action="append", default=[], metavar="TASK_YAML=POLICY_PATH")
    parser.add_argument("--task-yaml", default=None)
    parser.add_argument("--policy-path", default=None)
    parser.add_argument("--robot-name", default=None)
    parser.add_argument("--object-type", default=None)
    parser.add_argument("--steps", default=None)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--decimation", type=int, default=None)
    parser.add_argument("--initial-state", choices=("reference_frame", "scene_default"), default="reference_frame")
    parser.add_argument("--near-zero-action-threshold", type=float, default=0.05)
    parser.add_argument("--target-ref-l2-threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


class _Job:
    def __init__(self, task_yaml: Path, policy_path: Path) -> None:
        self.task_yaml = task_yaml
        self.policy_path = policy_path


def _parse_jobs(args: argparse.Namespace) -> list[_Job]:
    jobs = []
    for raw in args.task_policy:
        if "=" not in raw:
            raise ValueError(f"--task-policy must be TASK_YAML=POLICY_PATH, got {raw!r}.")
        task_yaml, policy_path = raw.split("=", 1)
        jobs.append(_Job(Path(task_yaml), Path(policy_path)))
    if args.task_yaml or args.policy_path:
        if not args.task_yaml or not args.policy_path:
            raise ValueError("--task-yaml and --policy-path must be provided together.")
        jobs.append(_Job(Path(args.task_yaml), Path(args.policy_path)))
    if not jobs:
        raise ValueError("Provide at least one --task-policy or a --task-yaml/--policy-path pair.")
    return jobs



def _contact_eef_body_names_from_command(command_cfg: Mapping[str, Any]) -> list[str] | None:
    raw_names = command_cfg.get("contact_eef_body_name")
    if raw_names is not None:
        return _as_string_list(raw_names)
    target_offsets = command_cfg.get("contact_target_pos_offset") or []
    if len(target_offsets) == len(DEFAULT_CONTACT_EEF_BODY_NAMES):
        return list(DEFAULT_CONTACT_EEF_BODY_NAMES)
    return None


def _as_string_list(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)):
        return [str(value)]
    return [str(item) for item in value]

def _diagnostic_steps(
    *,
    requested_steps: Sequence[int] | None,
    num_reference_steps: int,
    max_steps: int | None,
) -> torch.Tensor:
    if requested_steps is not None:
        steps = torch.as_tensor(list(requested_steps), dtype=torch.long)
    else:
        count = num_reference_steps if max_steps is None or max_steps <= 0 else min(num_reference_steps, max_steps)
        steps = torch.arange(count, dtype=torch.long)
    if steps.numel() == 0:
        raise ValueError("Diagnostic steps must not be empty.")
    if torch.any(steps < 0) or torch.any(steps >= num_reference_steps):
        raise ValueError(f"Diagnostic steps must be in [0, {num_reference_steps}), got {steps.tolist()}.")
    return steps


def _task_identity(task_yaml: Path) -> dict[str, str]:
    cfg = _load_yaml_mapping(task_yaml)
    task_name = str(cfg.get("name") or task_yaml.stem)
    return {
        "task_name": task_name,
        "task_stem": task_yaml.stem,
        "task_path": str(task_yaml),
    }


def _task_command_config(task_cfg: Mapping[str, Any], task_yaml: str | Path | None) -> Mapping[str, Any]:
    command_cfg = task_cfg.get("command", {})
    if not isinstance(command_cfg, Mapping):
        raise ValueError(f"Task YAML command section must be a mapping: {task_yaml}")
    if "data_path" not in command_cfg:
        raise ValueError(f"Task YAML command.data_path is required: {task_yaml}")
    return command_cfg


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text()) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return cfg


def _require_series_tensor(value: torch.Tensor, name: str) -> torch.Tensor:
    value = torch.as_tensor(value, dtype=torch.float32)
    if value.ndim != 3:
        raise ValueError(f"{name} must have shape [time, env, dim], got {tuple(value.shape)}.")
    if value.shape[0] < 1 or value.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one time step and one env, got {tuple(value.shape)}.")
    return value


def _require_metric_tensor(value: torch.Tensor, expected_time_env: torch.Size, name: str) -> torch.Tensor:
    value = torch.as_tensor(value, dtype=torch.float32)
    if value.ndim == 1:
        value = value.unsqueeze(1)
    if tuple(value.shape[:2]) != tuple(expected_time_env):
        raise ValueError(f"{name} must have time/env shape {tuple(expected_time_env)}, got {tuple(value.shape)}.")
    return value


def _top_joint_target_ref_errors(
    *,
    target_ref_abs_error: torch.Tensor,
    steps: torch.Tensor,
    policy_joint_names: Sequence[str] | None,
    top_n: int = 8,
) -> list[dict[str, Any]]:
    errors = torch.as_tensor(target_ref_abs_error, dtype=torch.float32)
    if errors.ndim != 3 or errors.shape[-1] == 0:
        return []
    joint_names = list(policy_joint_names) if policy_joint_names is not None else [f"joint_{index}" for index in range(errors.shape[-1])]
    mean_by_joint = errors.mean(dim=(0, 1))
    max_by_joint, flat_indices = errors.reshape(-1, errors.shape[-1]).max(dim=0)
    env_count = errors.shape[1]
    top_indices = torch.argsort(max_by_joint, descending=True)[:top_n]
    rows = []
    for joint_index_t in top_indices:
        joint_index = int(joint_index_t.item())
        flat_index = int(flat_indices[joint_index].item())
        time_index = flat_index // env_count
        rows.append(
            {
                "name": str(joint_names[joint_index]),
                "mean_abs_error": _json_float(mean_by_joint[joint_index]),
                "max_abs_error": _json_float(max_by_joint[joint_index]),
                "max_step": int(steps[time_index].item()),
            }
        )
    return rows


def _json_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        value = float(value.detach().cpu().item())
    return round(float(value), 6)


if __name__ == "__main__":
    raise SystemExit(main())
