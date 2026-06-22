from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


HDMI_ROOT = Path(__file__).resolve().parents[1]
if str(HDMI_ROOT) not in sys.path:
    sys.path.insert(0, str(HDMI_ROOT))

from active_adaptation.mujoco.observation_builder import MujocoPolicyState
from active_adaptation.mujoco.policy import MujocoPolicyBundle


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.env_trace_json is not None or args.builder_trace_json is not None:
        if args.env_trace_json is None or args.builder_trace_json is None:
            raise SystemExit("--env-trace-json and --builder-trace-json must be provided together.")
        report = compare_observation_component_traces(
            _load_json(args.env_trace_json),
            _load_json(args.builder_trace_json),
            max_abs=args.max_abs,
        )
        report["mode"] = "offline_trace"
    else:
        if args.task_yaml is None or args.policy_path is None:
            raise SystemExit("live mode requires --task-yaml and --policy-path.")
        log_buffer = io.StringIO()
        redirect = contextlib.nullcontext() if args.verbose else contextlib.redirect_stdout(log_buffer)
        redirect_err = contextlib.nullcontext() if args.verbose else contextlib.redirect_stderr(log_buffer)
        with redirect, redirect_err:
            report = build_live_obs_builder_parity(
                task_yaml=args.task_yaml,
                policy_path=args.policy_path,
                steps=args.steps,
                action_source=args.action_source,
                max_abs=args.max_abs,
            )
        report["mode"] = "live_mujoco_env"
        if not args.verbose:
            report["captured_log_tail"] = _tail_lines(log_buffer.getvalue(), max_lines=80)

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if args.require_pass and not report["gate_passed"]:
        return 1
    return 0


def compare_observation_component_traces(
    env_trace: Any,
    builder_trace: Any,
    *,
    max_abs: float = 1e-6,
) -> dict[str, Any]:
    env_steps = _coerce_trace_steps(env_trace, label="env_trace")
    builder_steps = _coerce_trace_steps(builder_trace, label="builder_trace")
    env_by_step = {entry["step"]: entry for entry in env_steps}
    builder_by_step = {entry["step"]: entry for entry in builder_steps}
    step_ids = sorted(set(env_by_step) | set(builder_by_step))

    failures: list[dict[str, Any]] = []
    groups: dict[str, Any] = {}
    compared_components = 0
    abs_sum = 0.0
    element_count = 0
    global_max = 0.0

    for step_id in step_ids:
        env_entry = env_by_step.get(step_id)
        builder_entry = builder_by_step.get(step_id)
        if env_entry is None:
            failures.append({"step": step_id, "reason": "missing_env_step"})
            continue
        if builder_entry is None:
            failures.append({"step": step_id, "reason": "missing_builder_step"})
            continue

        env_groups = env_entry["groups"]
        builder_groups = builder_entry["groups"]
        group_names = sorted(set(env_groups) | set(builder_groups))
        for group_name in group_names:
            if group_name not in env_groups:
                failures.append({"step": step_id, "group": group_name, "reason": "missing_env_group"})
                continue
            if group_name not in builder_groups:
                failures.append({"step": step_id, "group": group_name, "reason": "missing_builder_group"})
                continue

            env_components = env_groups[group_name]
            builder_components = builder_groups[group_name]
            component_names = list(dict.fromkeys([*env_components.keys(), *builder_components.keys()]))
            for component_name in component_names:
                if component_name not in env_components:
                    failures.append(
                        {
                            "step": step_id,
                            "group": group_name,
                            "component": component_name,
                            "reason": "missing_env_component",
                        }
                    )
                    continue
                if component_name not in builder_components:
                    failures.append(
                        {
                            "step": step_id,
                            "group": group_name,
                            "component": component_name,
                            "reason": "missing_builder_component",
                        }
                    )
                    continue

                env_tensor = _as_float_tensor(env_components[component_name])
                builder_tensor = _as_float_tensor(builder_components[component_name])
                if tuple(env_tensor.shape) != tuple(builder_tensor.shape):
                    failures.append(
                        {
                            "step": step_id,
                            "group": group_name,
                            "component": component_name,
                            "reason": "shape_mismatch",
                            "env_shape": list(env_tensor.shape),
                            "builder_shape": list(builder_tensor.shape),
                        }
                    )
                    continue

                diff = (env_tensor - builder_tensor).abs()
                comp_max = float(diff.max().item()) if diff.numel() else 0.0
                comp_abs_sum = float(diff.sum().item()) if diff.numel() else 0.0
                comp_count = int(diff.numel())
                compared_components += 1
                abs_sum += comp_abs_sum
                element_count += comp_count
                global_max = max(global_max, comp_max)
                _accumulate_component(
                    groups,
                    group_name=group_name,
                    component_name=component_name,
                    shape=list(env_tensor.shape),
                    max_abs=comp_max,
                    abs_sum=comp_abs_sum,
                    element_count=comp_count,
                )
                if comp_max > max_abs:
                    failures.append(
                        {
                            "step": step_id,
                            "group": group_name,
                            "component": component_name,
                            "reason": "max_abs_exceeded",
                            "max_abs": comp_max,
                            "threshold": float(max_abs),
                            "mean_abs": comp_abs_sum / comp_count if comp_count else 0.0,
                            "shape": list(env_tensor.shape),
                        }
                    )

    _finalize_group_stats(groups)
    return {
        "step_count": len(step_ids),
        "compared_component_count": compared_components,
        "max_abs": global_max,
        "mean_abs": abs_sum / element_count if element_count else 0.0,
        "element_count": element_count,
        "threshold": float(max_abs),
        "gate_passed": len(failures) == 0,
        "failure_count": len(failures),
        "failures": failures,
        "groups": groups,
    }


def build_live_obs_builder_parity(
    *,
    task_yaml: str | Path,
    policy_path: str | Path,
    steps: int,
    action_source: str,
    max_abs: float,
) -> dict[str, Any]:
    if steps < 1:
        raise ValueError("--steps must be >= 1 for live mode.")

    bundle = MujocoPolicyBundle.load(policy_path)
    base_env = _make_mujoco_base_env(task_yaml)
    groups = _policy_groups_present_in_bundle(bundle)
    policy_joint_ids = _ordered_joint_ids(
        base_env.scene["robot"],
        bundle.policy_joint_names,
        label="policy joints",
    )
    observation_joint_ids = _ordered_joint_ids(
        base_env.scene["robot"],
        bundle.observation_joint_names,
        label="observation joints",
    )

    base_env.eval()
    base_env.reset()
    state = _policy_state_from_env(
        base_env=base_env,
        bundle=bundle,
        observation_joint_ids=observation_joint_ids,
    )
    builder = bundle.observation_builder
    builder.reset(state)

    env_trace = []
    builder_trace = []
    from tensordict import TensorDict

    for step_id in range(steps):
        action = _action_for_step(
            step_id=step_id,
            source=action_source,
            num_envs=base_env.num_envs,
            action_dim=base_env.action_dim,
            device=base_env.device,
        )
        base_env.step(TensorDict({"action": action}, batch_size=[base_env.num_envs], device=base_env.device))
        state = _policy_state_from_env(
            base_env=base_env,
            bundle=bundle,
            observation_joint_ids=observation_joint_ids,
        )
        builder.update(state)
        env_trace.append(
            {
                "step": step_id,
                "groups": _env_component_groups(base_env=base_env, groups=groups, bundle=bundle),
            }
        )
        builder_trace.append(
            {
                "step": step_id,
                "groups": _builder_component_groups(builder=builder, groups=groups, state=state),
            }
        )

    report = compare_observation_component_traces(env_trace, builder_trace, max_abs=max_abs)
    report.update(
        {
            "task_yaml": str(task_yaml),
            "task_override": _task_override_from_path(task_yaml),
            "policy_path": str(policy_path),
            "policy_config_path": str(bundle.config_path),
            "policy_joint_count": len(policy_joint_ids),
            "observation_joint_count": len(observation_joint_ids),
            "action_source": action_source,
            "groups_compared": groups,
            "env_action_dim": int(base_env.action_dim),
            "policy_action_dim": int(bundle.action_dim),
            "command_t": _tensor_to_json(getattr(base_env.command_manager, "t", None)),
            "motion_len": _tensor_to_json(getattr(base_env.command_manager, "motion_len", None)),
        }
    )
    return report


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare HDMI MuJoCo env observation components against the exported "
            "MujocoObservationBuilder components used by deployed policies."
        )
    )
    parser.add_argument("--env-trace-json", default=None, help="Offline env component trace JSON.")
    parser.add_argument("--builder-trace-json", default=None, help="Offline builder component trace JSON.")
    parser.add_argument("--task-yaml", default=None, help="HDMI task YAML for live MuJoCo env mode.")
    parser.add_argument("--policy-path", default=None, help="Exported policy .pt for live MuJoCo env mode.")
    parser.add_argument("--steps", type=int, default=4, help="Live env rollout steps to compare.")
    parser.add_argument(
        "--action-source",
        choices=("zero", "ramp"),
        default="zero",
        help="Deterministic raw env action source for live mode.",
    )
    parser.add_argument("--max-abs", type=float, default=1e-6, help="Per-component max abs gate.")
    parser.add_argument("--require-pass", action="store_true", help="Exit nonzero if any component fails.")
    parser.add_argument("--output", default=None, help="Optional JSON report output path.")
    parser.add_argument("--verbose", action="store_true", help="Do not suppress MuJoCo/Hydra setup logs.")
    return parser.parse_args(argv)


def _make_mujoco_base_env(task_yaml: str | Path):
    import active_adaptation as aa
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    aa.set_backend("mujoco")
    with initialize_config_dir(config_dir=str(HDMI_ROOT / "cfg"), version_base=None):
        cfg = compose(
            config_name="play",
            overrides=[
                "backend=mujoco",
                "headless=true",
                f"task={_task_override_from_path(task_yaml)}",
                "algo=ppo_roa_adapt",
                "task.num_envs=1",
                "checkpoint_path=null",
                "wandb.mode=disabled",
            ],
        )
    OmegaConf.set_struct(cfg, False)
    _disable_observation_noise(cfg.task.observation)
    OmegaConf.resolve(cfg)
    from scripts.helpers import make_env_policy

    env, _, _ = make_env_policy(cfg)
    base_env = env.base_env
    base_env.eval()
    env.eval()
    return base_env


def _disable_observation_noise(observation_cfg: Any) -> None:
    for group_cfg in observation_cfg.values():
        if not isinstance(group_cfg, Mapping):
            continue
        for params in group_cfg.values():
            if params is None or not isinstance(params, Mapping):
                continue
            if "noise_std" in params:
                params["noise_std"] = 0.0
            if "episodic_noise_std" in params:
                params["episodic_noise_std"] = 0.0


def _policy_state_from_env(
    *,
    base_env: Any,
    bundle: MujocoPolicyBundle,
    observation_joint_ids: Sequence[int],
) -> MujocoPolicyState:
    robot = base_env.scene["robot"]
    command = base_env.command_manager
    joint_pos = robot.data.joint_pos[:, observation_joint_ids]
    joint_offset = bundle.observation_default_joint_pos.to(device=joint_pos.device, dtype=joint_pos.dtype).unsqueeze(0)
    env_joint_history = _env_joint_pos_history_func(base_env)
    if env_joint_history is not None:
        reorder_indices = _joint_history_reorder_indices(
            env_joint_history,
            bundle.observation_builder.joint_pos_names,
        )
    else:
        reorder_indices = None
    if env_joint_history is not None and reorder_indices is not None:
        if hasattr(env_joint_history, "buffer"):
            joint_pos = env_joint_history.buffer.index_select(2, reorder_indices)[:, 0].detach().clone()
        if hasattr(env_joint_history, "joint_pos_offset") and hasattr(env_joint_history, "joint_ids"):
            env_offset = env_joint_history.joint_pos_offset[:, env_joint_history.joint_ids]
            joint_offset = env_offset.index_select(1, reorder_indices).detach().clone()
    object_state = _object_state_from_env(base_env)
    return MujocoPolicyState(
        root_ang_vel_b=robot.data.root_ang_vel_b,
        root_lin_vel_b=getattr(robot.data, "root_lin_vel_b", None),
        projected_gravity_b=robot.data.projected_gravity_b,
        joint_pos=joint_pos,
        joint_names=list(bundle.observation_builder.joint_pos_names),
        joint_pos_offset=joint_offset.expand_as(joint_pos),
        applied_action=getattr(base_env.action_manager, "applied_action", None),
        applied_torque=getattr(robot.data, "applied_torque", None),
        action_history=getattr(base_env.action_manager, "action_buf", None),
        body_names=list(getattr(robot, "body_names", ())),
        body_pos_w=getattr(robot.data, "body_pos_w", None),
        body_quat_w=getattr(robot.data, "body_quat_w", None),
        body_lin_vel_w=getattr(robot.data, "body_lin_vel_w", None),
        body_ang_vel_w=getattr(robot.data, "body_ang_vel_w", None),
        tracking_body_pos_w=getattr(command, "robot_body_pos_w", None),
        tracking_body_quat_w=getattr(command, "robot_body_quat_w", None),
        tracking_body_lin_vel_w=getattr(command, "robot_body_lin_vel_w", None),
        tracking_body_ang_vel_w=getattr(command, "robot_body_ang_vel_w", None),
        ref_body_pos_future_w=getattr(command, "ref_body_pos_future_w", None),
        ref_body_quat_future_w=getattr(command, "ref_body_quat_future_w", None),
        ref_body_lin_vel_future_w=getattr(command, "ref_body_lin_vel_future_w", None),
        ref_body_ang_vel_future_w=getattr(command, "ref_body_ang_vel_future_w", None),
        ref_root_pos_w=getattr(command, "ref_root_pos_w", None),
        ref_root_quat_w=getattr(command, "ref_root_quat_w", None),
        ref_root_pos_future_w=getattr(command, "ref_root_pos_future_w", None),
        ref_root_quat_future_w=getattr(command, "ref_root_quat_future_w", None),
        ref_joint_pos_future=getattr(command, "ref_joint_pos_future_", None),
        ref_joint_pos_action=_ref_joint_pos_action_from_env(command, bundle),
        ref_object_pos_future_w=getattr(command, "ref_object_pos_future_w", None),
        ref_object_quat_future_w=getattr(command, "ref_object_quat_future_w", None),
        ref_object_contact_future=getattr(command, "ref_object_contact_future", None),
        motion_t=getattr(command, "t", None),
        motion_len=getattr(command, "motion_len", None),
        robot_root_pos_w=getattr(command, "robot_root_pos_w", robot.data.root_link_pos_w),
        robot_root_quat_w=getattr(command, "robot_root_quat_w", robot.data.root_link_quat_w),
        **object_state,
    )


def _ref_joint_pos_action_from_env(command: Any, bundle: MujocoPolicyBundle) -> torch.Tensor | None:
    current_ref_motion = getattr(command, "current_ref_motion", None)
    ref_joint_pos_all = getattr(current_ref_motion, "joint_pos", None)
    dataset = getattr(command, "dataset", None)
    dataset_joint_names = [str(name) for name in (getattr(dataset, "joint_names", None) or [])]
    policy_joint_names = [str(name) for name in getattr(bundle, "policy_joint_names", [])]
    if ref_joint_pos_all is None or not dataset_joint_names or not policy_joint_names:
        return None

    index_by_name = {name: index for index, name in enumerate(dataset_joint_names)}
    missing = [name for name in policy_joint_names if name not in index_by_name]
    if missing:
        raise ValueError(f"Policy joints missing from motion dataset joint_names: {missing}.")

    if not isinstance(ref_joint_pos_all, torch.Tensor):
        ref_joint_pos_all = torch.as_tensor(ref_joint_pos_all, dtype=torch.float32)
    joint_indices = torch.as_tensor(
        [index_by_name[name] for name in policy_joint_names],
        device=ref_joint_pos_all.device,
        dtype=torch.long,
    )
    ref_joint_pos = ref_joint_pos_all.index_select(1, joint_indices)
    default_joint_pos = bundle.default_joint_pos.to(device=ref_joint_pos.device, dtype=ref_joint_pos.dtype).unsqueeze(0)
    action_scale = bundle.action_scale.to(device=ref_joint_pos.device, dtype=ref_joint_pos.dtype).clamp_min(1.0e-6).unsqueeze(0)
    return (ref_joint_pos - default_joint_pos) / action_scale


def _env_joint_pos_history_func(base_env: Any) -> Any | None:
    policy_group = base_env.observation_funcs.get("policy")
    if policy_group is None:
        return None
    joint_history = policy_group.funcs.get("joint_pos_history")
    if joint_history is None:
        return None
    if not hasattr(joint_history, "joint_names"):
        return None
    return joint_history


def _object_state_from_env(base_env: Any) -> dict[str, torch.Tensor]:
    command = base_env.command_manager
    result: dict[str, torch.Tensor] = {}
    object_view = getattr(command, "object", None)
    if object_view is not None:
        result["object_pos_w"] = object_view.data.root_link_pos_w
        result["object_quat_w"] = object_view.data.root_link_quat_w
        for state_name, data_name in (
            ("object_joint_pos", "joint_pos"),
            ("object_joint_vel", "joint_vel"),
            ("object_joint_torque", "applied_torque"),
        ):
            value = getattr(object_view.data, data_name, None)
            if value is not None:
                result[state_name] = _as_joint_column(value)
    for attr_name in ("object_joint_pos", "object_joint_vel"):
        value = getattr(command, attr_name, None)
        if value is not None:
            result[attr_name] = _as_joint_column(value)
    for attr_name in ("contact_target_pos_w", "contact_eef_pos_w"):
        value = getattr(command, attr_name, None)
        if value is not None:
            result[attr_name] = value
    return result


def _as_joint_column(value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value, dtype=torch.float32)
    if value.ndim == 1:
        return value.unsqueeze(1)
    return value


def _env_component_groups(
    *,
    base_env: Any,
    groups: Sequence[str],
    bundle: MujocoPolicyBundle | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    result: dict[str, dict[str, torch.Tensor]] = {}
    for group_name in groups:
        group = base_env.observation_funcs.get(group_name)
        if group is None:
            continue
        components: dict[str, torch.Tensor] = {}
        for component_name, func in group.funcs.items():
            value = func.compute().detach().clone()
            if bundle is not None:
                value = _align_env_component_to_bundle(
                    group_name=group_name,
                    component_name=component_name,
                    func=func,
                    value=value,
                    bundle=bundle,
                )
            components[component_name] = value
        result[group_name] = components
    return result


def _align_env_component_to_bundle(
    *,
    group_name: str,
    component_name: str,
    func: Any,
    value: torch.Tensor,
    bundle: MujocoPolicyBundle,
) -> torch.Tensor:
    if group_name != "policy" or component_name != "joint_pos_history":
        return value
    reorder_indices = _joint_history_reorder_indices(
        func,
        bundle.observation_builder.joint_pos_names,
        device=value.device,
    )
    if reorder_indices is None:
        return value
    joint_count = len(list(getattr(func, "joint_names", [])))
    if joint_count == 0 or value.shape[-1] % joint_count != 0:
        return value
    history_count = value.shape[-1] // joint_count
    aligned = value.reshape(*value.shape[:-1], history_count, joint_count)
    aligned = aligned.index_select(-1, reorder_indices)
    return aligned.reshape(*value.shape[:-1], history_count * joint_count)


def _joint_history_reorder_indices(
    joint_history: Any,
    target_joint_names: Sequence[str],
    *,
    device: torch.device | None = None,
) -> torch.Tensor | None:
    env_names = list(getattr(joint_history, "joint_names", []))
    target_names = list(target_joint_names)
    if not env_names or len(env_names) != len(target_names):
        return None
    if set(env_names) != set(target_names):
        return None
    if device is None:
        if hasattr(joint_history, "buffer"):
            device = joint_history.buffer.device
        elif hasattr(joint_history, "joint_ids"):
            device = joint_history.joint_ids.device
    indices = [env_names.index(name) for name in target_names]
    return torch.as_tensor(indices, dtype=torch.long, device=device)


def _builder_component_groups(
    *,
    builder: Any,
    groups: Sequence[str],
    state: MujocoPolicyState,
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        group_name: dict(builder.build_group(group_name, state, return_components=True))
        for group_name in groups
    }


def _policy_groups_present_in_bundle(bundle: MujocoPolicyBundle) -> list[str]:
    return list(bundle.observation_builder.observation_cfg.keys())


def _ordered_joint_ids(robot: Any, joint_names_expected: Sequence[str], *, label: str) -> list[int]:
    expected = list(joint_names_expected)
    joint_ids, joint_names = robot.find_joints(expected, preserve_order=True)
    if joint_names != expected:
        raise ValueError(f"{label} order mismatch: expected {expected}, got {joint_names}.")
    return joint_ids


def _action_for_step(
    *,
    step_id: int,
    source: str,
    num_envs: int,
    action_dim: int,
    device: torch.device,
) -> torch.Tensor:
    if source == "zero":
        return torch.zeros(num_envs, action_dim, device=device)
    if source == "ramp":
        base = torch.linspace(-0.25, 0.25, action_dim, device=device)
        return base.mul(float(step_id + 1) / 10.0).unsqueeze(0).expand(num_envs, -1)
    raise ValueError(f"Unsupported action source {source!r}.")


def _coerce_trace_steps(trace: Any, *, label: str) -> list[dict[str, Any]]:
    if isinstance(trace, Mapping):
        trace = trace.get("steps", trace)
    if not isinstance(trace, Sequence) or isinstance(trace, (str, bytes)):
        raise TypeError(f"{label} must be a list of step entries or a mapping with a 'steps' list.")
    result = []
    for default_step, entry in enumerate(trace):
        if not isinstance(entry, Mapping):
            raise TypeError(f"{label}[{default_step}] must be a mapping.")
        groups = entry.get("groups")
        if not isinstance(groups, Mapping):
            raise TypeError(f"{label}[{default_step}].groups must be a mapping.")
        result.append({"step": int(entry.get("step", default_step)), "groups": groups})
    return result


def _accumulate_component(
    groups: dict[str, Any],
    *,
    group_name: str,
    component_name: str,
    shape: list[int],
    max_abs: float,
    abs_sum: float,
    element_count: int,
) -> None:
    group_stats = groups.setdefault(
        group_name,
        {"max_abs": 0.0, "_abs_sum": 0.0, "_element_count": 0, "components": {}},
    )
    comp_stats = group_stats["components"].setdefault(
        component_name,
        {
            "shape": shape,
            "max_abs": 0.0,
            "_abs_sum": 0.0,
            "_element_count": 0,
            "steps_compared": 0,
        },
    )
    comp_stats["max_abs"] = max(comp_stats["max_abs"], max_abs)
    comp_stats["_abs_sum"] += abs_sum
    comp_stats["_element_count"] += element_count
    comp_stats["steps_compared"] += 1
    group_stats["max_abs"] = max(group_stats["max_abs"], max_abs)
    group_stats["_abs_sum"] += abs_sum
    group_stats["_element_count"] += element_count


def _finalize_group_stats(groups: dict[str, Any]) -> None:
    for group_stats in groups.values():
        count = group_stats.pop("_element_count")
        abs_sum = group_stats.pop("_abs_sum")
        group_stats["mean_abs"] = abs_sum / count if count else 0.0
        for comp_stats in group_stats["components"].values():
            comp_count = comp_stats.pop("_element_count")
            comp_abs_sum = comp_stats.pop("_abs_sum")
            comp_stats["mean_abs"] = comp_abs_sum / comp_count if comp_count else 0.0


def _as_float_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32)


def _tensor_to_json(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def _task_override_from_path(task_yaml: str | Path) -> str:
    path = Path(task_yaml)
    try:
        rel_path = path.resolve().relative_to((HDMI_ROOT / "cfg" / "task").resolve())
        return str(rel_path.with_suffix("")).replace("\\", "/")
    except ValueError:
        return path.with_suffix("").name


def _tail_lines(text: str, *, max_lines: int) -> list[str]:
    if not text:
        return []
    return text.splitlines()[-max_lines:]


if __name__ == "__main__":
    raise SystemExit(main())
