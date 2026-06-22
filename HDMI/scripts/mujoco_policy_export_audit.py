from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml
from omegaconf import OmegaConf
import onnxruntime as ort


HDMI_ROOT = Path(__file__).resolve().parents[1]
if str(HDMI_ROOT) not in sys.path:
    sys.path.insert(0, str(HDMI_ROOT))

from active_adaptation.mujoco.observation_builder import MujocoPolicyState
from active_adaptation.mujoco.policy import MujocoPolicyBundle
from active_adaptation.mujoco.task_mapping import (
    validate_policy_task_motion_mapping,
    validate_task_motion_mapping,
)

_REFERENCE_OBSERVATION_KEYS = frozenset(
    {
        "ref_body_pos_future_local",
        "ref_joint_pos_future",
        "ref_motion_phase",
        "ref_root_pos_future_b",
        "ref_root_ori_future_b",
        "diff_body_pos_future_local",
        "diff_body_ori_future_local",
        "diff_body_lin_vel_future_local",
        "diff_body_ang_vel_future_local",
        "ref_contact_pos_b",
        "diff_contact_pos_b",
        "diff_object_pos_future",
        "diff_object_ori_future",
        "ref_object_contact_future",
        "ref_joint_pos_action_policy",
    }
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_policy_export_audit(
        task_yaml=args.task_yaml,
        policy_path=args.policy_path,
        exports_dir=args.exports_dir,
        checkpoint_path=args.checkpoint_path,
        robot_name=args.robot_name,
        require_reference_observation=args.require_reference_observation,
        require_obs_action_smoke=args.require_obs_action_smoke,
        require_onnx_policy=args.require_onnx_policy,
        require_checkpoint_algo=args.require_checkpoint_algo,
        min_checkpoint_total_frames=args.min_checkpoint_total_frames,
    )
    print(json.dumps(report, sort_keys=True))
    gate_requested = (
        args.require_policy
        or args.require_reference_observation
        or args.require_obs_action_smoke
        or args.require_onnx_policy
        or args.require_checkpoint_algo is not None
        or args.min_checkpoint_total_frames is not None
    )
    if gate_requested and not report["gate_passed"]:
        return 1
    return 0


def build_policy_export_audit(
    *,
    task_yaml: str | Path,
    policy_path: str | Path | None = None,
    exports_dir: str | Path | None = None,
    checkpoint_path: str | None = None,
    robot_name: str = "g1_29dof",
    require_reference_observation: bool = False,
    require_obs_action_smoke: bool = False,
    require_onnx_policy: bool = False,
    require_checkpoint_algo: str | None = None,
    min_checkpoint_total_frames: int | None = None,
) -> dict[str, Any]:
    task_path = Path(task_yaml)
    task_cfg = _load_yaml_mapping(task_path)
    task_name = str(task_cfg.get("name") or task_path.stem)
    task_override = _task_override_from_path(task_path)
    exports_root = Path(exports_dir) if exports_dir is not None else HDMI_ROOT / "scripts" / "exports"
    expected_export_dir = exports_root / task_name

    resolved_policy_path = Path(policy_path) if policy_path is not None else _find_latest_policy(expected_export_dir)
    policy_exists = resolved_policy_path is not None and resolved_policy_path.is_file()
    policy_config_path = _default_policy_config_path(resolved_policy_path) if resolved_policy_path is not None else None
    policy_config_exists = policy_config_path is not None and policy_config_path.is_file()
    policy_onnx_path = _default_policy_onnx_path(resolved_policy_path) if resolved_policy_path is not None else None
    policy_onnx_metadata_path = _default_policy_onnx_metadata_path(resolved_policy_path) if resolved_policy_path is not None else None
    policy_onnx_exists = policy_onnx_path is not None and policy_onnx_path.is_file()
    policy_onnx_metadata_exists = policy_onnx_metadata_path is not None and policy_onnx_metadata_path.is_file()

    task_report = validate_task_motion_mapping(task_path, robot_name=robot_name)
    report: dict[str, Any] = {
        "task_path": str(task_path),
        "task_name": task_name,
        "task_override": task_override,
        "motion_dir": str(task_report.motion_dir),
        "object_asset_name": task_report.object_asset_name,
        "expected_export_dir": str(expected_export_dir),
        "available_policy_paths": [str(path) for path in _list_policy_exports(expected_export_dir)],
        "policy_path": str(resolved_policy_path) if resolved_policy_path is not None else None,
        "policy_config_path": str(policy_config_path) if policy_config_path is not None else None,
        "policy_onnx_path": str(policy_onnx_path) if policy_onnx_path is not None else None,
        "policy_onnx_metadata_path": str(policy_onnx_metadata_path) if policy_onnx_metadata_path is not None else None,
        "policy_exists": bool(policy_exists),
        "policy_config_exists": bool(policy_config_exists),
        "policy_onnx_exists": bool(policy_onnx_exists),
        "policy_onnx_metadata_exists": bool(policy_onnx_metadata_exists),
        "checkpoint_path": checkpoint_path,
        **_checkpoint_summary(checkpoint_path),
        "export_command": _export_command(task_override, checkpoint_path),
        "task_mapping": {
            "num_motion_bodies": len(task_report.motion_body_names),
            "num_motion_joints": len(task_report.motion_joint_names),
            "num_body_mappings": len(task_report.body_name_mapping),
            "num_joint_mappings": len(task_report.joint_name_mapping),
            "reference_object_body_name": task_report.reference_object_body_name,
            "object_joint_name": task_report.object_joint_name,
        },
        "policy_loadable": None,
        "policy_load_error": None,
        "policy_mapping_ok": None,
        "policy_mapping_error": None,
        "policy_action_dim": None,
        "policy_joint_names": [],
        "policy_observation_joint_names": [],
        "policy_isaac_body_names": [],
        "policy_observation_groups": [],
        "policy_observation_keys": {},
        "policy_has_reference_observation": None,
        "policy_reference_observation_keys": [],
        "require_reference_observation": bool(require_reference_observation),
        "require_obs_action_smoke": bool(require_obs_action_smoke),
        "require_onnx_policy": bool(require_onnx_policy),
        "policy_obs_action_smoke_ok": None,
        "policy_obs_action_smoke_error": None,
        "policy_onnx_loadable": None,
        "policy_onnx_load_error": None,
        "policy_onnx_input_count": None,
        "policy_onnx_output_count": None,
        "policy_onnx_metadata_in_keys": [],
        "policy_onnx_metadata_out_keys": [],
        "policy_smoke_observation_shapes": {},
        "policy_smoke_action_shape": None,
        "policy_smoke_joint_target_shape": None,
        "require_checkpoint_algo": require_checkpoint_algo,
        "min_checkpoint_total_frames": min_checkpoint_total_frames,
        "num_policy_body_mappings": None,
        "num_policy_joint_mappings": None,
    }

    if policy_exists and policy_config_exists:
        _annotate_policy_load(report, resolved_policy_path)
        _annotate_policy_mapping(report, policy_config_path, task_path, robot_name=robot_name)
    if policy_onnx_exists and policy_onnx_metadata_exists:
        _annotate_onnx_load(report, policy_onnx_path, policy_onnx_metadata_path)

    report["missing_requirements"] = _missing_requirements(
        report,
        require_reference_observation=require_reference_observation,
        require_obs_action_smoke=require_obs_action_smoke,
        require_onnx_policy=require_onnx_policy,
        require_checkpoint_algo=require_checkpoint_algo,
        min_checkpoint_total_frames=min_checkpoint_total_frames,
    )
    report["gate_passed"] = not report["missing_requirements"]
    return report


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether an HDMI policy export is ready for MuJoCo playback/rollout parity. "
            "The gate checks the exported .pt, its YAML metadata, and task/motion/MJCF name mapping."
        )
    )
    parser.add_argument("--task-yaml", required=True, help="HDMI task YAML used for the original policy.")
    parser.add_argument(
        "--policy-path",
        default=None,
        help="Exported policy .pt. Defaults to latest policy-*.pt under scripts/exports/<task.name>.",
    )
    parser.add_argument(
        "--exports-dir",
        default=None,
        help="Policy exports root. Defaults to HDMI/scripts/exports.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Original play.py checkpoint_path value, used only for readiness reporting and command rendering.",
    )
    parser.add_argument(
        "--robot-name",
        default="g1_29dof",
        help="MuJoCo robot registry name used to resolve assets_mjcf scenes.",
    )
    parser.add_argument(
        "--require-policy",
        action="store_true",
        help="Return exit code 1 unless the policy export exists, loads, and maps to the task/motion/MJCF names.",
    )
    parser.add_argument(
        "--require-reference-observation",
        action="store_true",
        help=(
            "Return exit code 1 unless the exported actor observation config contains a reference/command "
            "observation needed for closed-loop motion tracking parity."
        ),
    )
    parser.add_argument(
        "--require-obs-action-smoke",
        action="store_true",
        help=(
            "Return exit code 1 unless a synthetic MuJoCo policy state can build exported observations "
            "and run one policy action with the expected action dimension."
        ),
    )
    parser.add_argument(
        "--require-onnx-policy",
        action="store_true",
        help="Return exit code 1 unless the matching .onnx and .json metadata exist and load in onnxruntime.",
    )
    parser.add_argument(
        "--require-checkpoint-algo",
        default=None,
        help="Return exit code 1 unless a local checkpoint cfg has this algo.name.",
    )
    parser.add_argument(
        "--min-checkpoint-total-frames",
        type=int,
        default=None,
        help="Return exit code 1 unless a local checkpoint cfg.total_frames is at least this value.",
    )
    return parser.parse_args(argv)


def _annotate_policy_load(report: dict[str, Any], policy_path: Path | None) -> None:
    if policy_path is None:
        return
    try:
        bundle = MujocoPolicyBundle.load(policy_path)
    except Exception as exc:  # pragma: no cover - exact third-party loader errors vary.
        report["policy_loadable"] = False
        report["policy_load_error"] = f"{type(exc).__name__}: {exc}"
        return

    report["policy_loadable"] = True
    report["policy_action_dim"] = bundle.action_dim
    report["policy_joint_names"] = list(bundle.policy_joint_names)
    report["policy_observation_joint_names"] = list(bundle.observation_joint_names)
    report["policy_isaac_body_names"] = list(bundle.isaac_body_names)
    observation_keys = _policy_observation_keys(bundle.observation_builder.observation_cfg)
    reference_keys = _policy_reference_observation_keys(observation_keys)
    report["policy_observation_groups"] = list(observation_keys)
    report["policy_observation_keys"] = observation_keys
    report["policy_reference_observation_keys"] = reference_keys
    report["policy_has_reference_observation"] = bool(reference_keys)
    _annotate_obs_action_smoke(report, bundle)


def _annotate_policy_mapping(
    report: dict[str, Any],
    policy_config_path: Path | None,
    task_path: Path,
    *,
    robot_name: str,
) -> None:
    if policy_config_path is None:
        return
    try:
        policy_report = validate_policy_task_motion_mapping(
            policy_config_path,
            task_path,
            robot_name=robot_name,
        )
    except Exception as exc:
        report["policy_mapping_ok"] = False
        report["policy_mapping_error"] = f"{type(exc).__name__}: {exc}"
        return

    report["policy_mapping_ok"] = True
    report["num_policy_body_mappings"] = len(policy_report.policy_body_name_mapping)
    report["num_policy_joint_mappings"] = len(policy_report.policy_joint_name_mapping)


def _annotate_onnx_load(report: dict[str, Any], onnx_path: Path | None, metadata_path: Path | None) -> None:
    if onnx_path is None or metadata_path is None:
        return
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    except Exception as exc:  # pragma: no cover - exact loader errors vary by runtime.
        report["policy_onnx_loadable"] = False
        report["policy_onnx_load_error"] = f"{type(exc).__name__}: {exc}"
        return

    report["policy_onnx_loadable"] = True
    report["policy_onnx_load_error"] = None
    report["policy_onnx_input_count"] = len(session.get_inputs())
    report["policy_onnx_output_count"] = len(session.get_outputs())
    report["policy_onnx_metadata_in_keys"] = metadata.get("in_keys", [])
    report["policy_onnx_metadata_out_keys"] = metadata.get("out_keys", [])


def _missing_requirements(
    report: dict[str, Any],
    *,
    require_reference_observation: bool,
    require_obs_action_smoke: bool,
    require_onnx_policy: bool,
    require_checkpoint_algo: str | None,
    min_checkpoint_total_frames: int | None,
) -> list[str]:
    missing: list[str] = []
    if not report["policy_exists"]:
        missing.append("exported_policy_pt")
    if not report["policy_config_exists"]:
        missing.append("exported_policy_yaml")
    if report["policy_loadable"] is not True:
        missing.append("policy_loadable")
    if report["policy_mapping_ok"] is not True:
        missing.append("policy_task_motion_mjcf_mapping")
    if require_reference_observation and report["policy_has_reference_observation"] is not True:
        missing.append("policy_reference_observation")
    if require_obs_action_smoke and report["policy_obs_action_smoke_ok"] is not True:
        missing.append("policy_obs_action_smoke")
    if require_onnx_policy:
        if not report["policy_onnx_exists"]:
            missing.append("exported_policy_onnx")
        if not report["policy_onnx_metadata_exists"]:
            missing.append("exported_policy_onnx_json")
        if report["policy_onnx_loadable"] is not True:
            missing.append("policy_onnx_loadable")
    if require_checkpoint_algo is not None and report.get("checkpoint_algo_name") != require_checkpoint_algo:
        missing.append("checkpoint_algo")
    if min_checkpoint_total_frames is not None and not _at_least(
        report.get("checkpoint_total_frames"),
        float(min_checkpoint_total_frames),
    ):
        missing.append("checkpoint_total_frames")
    return missing


def _policy_observation_keys(observation_cfg: Mapping[str, Any]) -> dict[str, list[str]]:
    return {
        str(group_name): [str(obs_key) for obs_key in (group_cfg or {}).keys()]
        for group_name, group_cfg in observation_cfg.items()
    }


def _policy_reference_observation_keys(observation_keys: Mapping[str, Sequence[str]]) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for group_keys in observation_keys.values():
        for obs_key in group_keys:
            if obs_key not in _REFERENCE_OBSERVATION_KEYS or obs_key in seen:
                continue
            seen.add(obs_key)
            found.append(obs_key)
    return found


def _annotate_obs_action_smoke(report: dict[str, Any], bundle: MujocoPolicyBundle) -> None:
    try:
        state = _synthetic_policy_state(bundle)
        bundle.reset(state)
        td = bundle.build_tensordict(state, is_init=True)
        action = bundle.act(state, is_init=True)
        if action.raw_action.shape[-1] != bundle.action_dim:
            raise ValueError(
                f"policy returned action dim {action.raw_action.shape[-1]} != exported action dim {bundle.action_dim}"
            )
    except Exception as exc:
        report["policy_obs_action_smoke_ok"] = False
        report["policy_obs_action_smoke_error"] = f"{type(exc).__name__}: {exc}"
        return

    report["policy_obs_action_smoke_ok"] = True
    report["policy_obs_action_smoke_error"] = None
    report["policy_smoke_observation_shapes"] = {
        str(group_name): list(td[group_name].shape)
        for group_name in bundle.observation_builder.observation_cfg
    }
    report["policy_smoke_action_shape"] = list(action.raw_action.shape)
    report["policy_smoke_joint_target_shape"] = list(action.joint_position_target.shape)


def _synthetic_policy_state(bundle: MujocoPolicyBundle) -> MujocoPolicyState:
    batch = 1
    dtype = torch.float32
    device = torch.device("cpu")
    action_dim = bundle.action_dim
    joint_dim = max(len(bundle.observation_joint_names), 1)
    torque_joint_names = list(bundle.isaac_joint_names or bundle.observation_joint_names or bundle.policy_joint_names)
    if not torque_joint_names:
        torque_joint_names = ["joint"]
    body_names = list(bundle.isaac_body_names or ["pelvis"])
    observation_cfg = bundle.observation_builder.observation_cfg
    future_count = _max_param_list_len(observation_cfg, "future_steps", default=1)
    reference_body_count = _max_obs_param_list_len(
        observation_cfg,
        "ref_body_pos_future_local",
        "body_names",
        default=1,
    )
    body_count = max(len(body_names), 1)
    while len(body_names) < body_count:
        body_names.append(f"body_{len(body_names)}")
    ref_joint_dim = _max_obs_param_list_len(
        observation_cfg,
        "ref_joint_pos_future",
        "joint_names",
        default=joint_dim,
    )
    contact_count = _max_obs_nested_list_len(
        observation_cfg,
        "ref_contact_pos_b",
        "contact_target_pos_offset",
        default=1,
    )
    action_history_steps = _max_prev_action_steps(observation_cfg)
    identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype, device=device)
    body_identity_quat = identity_quat[:, None, :].expand(batch, body_count, 4).clone()
    tracking_body_identity_quat = identity_quat[:, None, :].expand(batch, reference_body_count, 4).clone()
    future_body_identity_quat = (
        identity_quat[:, None, None, :]
        .expand(batch, future_count, reference_body_count, 4)
        .clone()
    )
    future_root_identity_quat = identity_quat[:, None, :].expand(batch, future_count, 4).clone()
    future_object_identity_quat = identity_quat[:, None, :].expand(batch, future_count, 4).clone()

    return MujocoPolicyState(
        root_ang_vel_b=torch.zeros(batch, 3, dtype=dtype, device=device),
        root_lin_vel_b=torch.zeros(batch, 3, dtype=dtype, device=device),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]], dtype=dtype, device=device),
        joint_pos=torch.zeros(batch, joint_dim, dtype=dtype, device=device),
        joint_names=torque_joint_names,
        joint_pos_offset=torch.zeros(batch, joint_dim, dtype=dtype, device=device),
        applied_action=torch.zeros(batch, action_dim, dtype=dtype, device=device),
        applied_torque=torch.zeros(batch, len(torque_joint_names), dtype=dtype, device=device),
        action_history=torch.zeros(batch, action_dim, action_history_steps, dtype=dtype, device=device),
        body_names=body_names,
        body_pos_w=torch.zeros(batch, body_count, 3, dtype=dtype, device=device),
        body_quat_w=body_identity_quat.clone(),
        body_lin_vel_w=torch.zeros(batch, body_count, 3, dtype=dtype, device=device),
        body_ang_vel_w=torch.zeros(batch, body_count, 3, dtype=dtype, device=device),
        tracking_body_pos_w=torch.zeros(batch, reference_body_count, 3, dtype=dtype, device=device),
        tracking_body_quat_w=tracking_body_identity_quat.clone(),
        tracking_body_lin_vel_w=torch.zeros(batch, reference_body_count, 3, dtype=dtype, device=device),
        tracking_body_ang_vel_w=torch.zeros(batch, reference_body_count, 3, dtype=dtype, device=device),
        ref_body_pos_future_w=torch.zeros(batch, future_count, reference_body_count, 3, dtype=dtype, device=device),
        ref_body_quat_future_w=future_body_identity_quat.clone(),
        ref_body_lin_vel_future_w=torch.zeros(
            batch,
            future_count,
            reference_body_count,
            3,
            dtype=dtype,
            device=device,
        ),
        ref_body_ang_vel_future_w=torch.zeros(
            batch,
            future_count,
            reference_body_count,
            3,
            dtype=dtype,
            device=device,
        ),
        ref_root_pos_w=torch.zeros(batch, 3, dtype=dtype, device=device),
        ref_root_quat_w=identity_quat.clone(),
        ref_root_pos_future_w=torch.zeros(batch, future_count, 3, dtype=dtype, device=device),
        ref_root_quat_future_w=future_root_identity_quat.clone(),
        ref_joint_pos_future=torch.zeros(batch, future_count, ref_joint_dim, dtype=dtype, device=device),
        ref_joint_pos_action=torch.zeros(batch, action_dim, dtype=dtype, device=device),
        motion_t=torch.zeros(batch, 1, dtype=dtype, device=device),
        motion_len=torch.ones(batch, 1, dtype=dtype, device=device),
        robot_root_pos_w=torch.zeros(batch, 3, dtype=dtype, device=device),
        robot_root_quat_w=identity_quat.clone(),
        contact_target_pos_w=torch.zeros(batch, contact_count, 3, dtype=dtype, device=device),
        contact_eef_pos_w=torch.zeros(batch, contact_count, 3, dtype=dtype, device=device),
        object_pos_w=torch.zeros(batch, 3, dtype=dtype, device=device),
        object_quat_w=identity_quat.clone(),
        object_joint_pos=torch.zeros(batch, 1, dtype=dtype, device=device),
        object_joint_vel=torch.zeros(batch, 1, dtype=dtype, device=device),
        object_joint_torque=torch.zeros(batch, 1, dtype=dtype, device=device),
        ref_object_pos_future_w=torch.zeros(batch, future_count, 3, dtype=dtype, device=device),
        ref_object_quat_future_w=future_object_identity_quat.clone(),
        ref_object_contact_future=torch.zeros(batch, future_count, dtype=dtype, device=device),
    )


def _max_param_list_len(observation_cfg: Mapping[str, Any], key: str, *, default: int) -> int:
    best = int(default)
    for params in _iter_observation_params(observation_cfg):
        value = params.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            best = max(best, len(value))
        elif value is not None:
            best = max(best, 1)
    return max(best, 1)


def _max_obs_param_list_len(
    observation_cfg: Mapping[str, Any],
    obs_key: str,
    param_key: str,
    *,
    default: int,
) -> int:
    best: int | None = None
    for params in _iter_observation_params_for_key(observation_cfg, obs_key):
        value = params.get(param_key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            best = max(best or 0, len(value))
        elif value is not None:
            best = max(best or 0, 1)
    return max(best if best is not None else int(default), 1)


def _max_nested_list_len(observation_cfg: Mapping[str, Any], key: str, *, default: int) -> int:
    best = int(default)
    for params in _iter_observation_params(observation_cfg):
        value = params.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            best = max(best, len(value))
    return max(best, 1)


def _max_obs_nested_list_len(
    observation_cfg: Mapping[str, Any],
    obs_key: str,
    param_key: str,
    *,
    default: int,
) -> int:
    best: int | None = None
    for params in _iter_observation_params_for_key(observation_cfg, obs_key):
        value = params.get(param_key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            best = max(best or 0, len(value))
    return max(best if best is not None else int(default), 1)


def _max_prev_action_steps(observation_cfg: Mapping[str, Any]) -> int:
    best = 1
    for group_cfg in observation_cfg.values():
        if not isinstance(group_cfg, Mapping):
            continue
        params = group_cfg.get("prev_actions")
        if isinstance(params, Mapping):
            best = max(best, int(params.get("steps", 1)))
    return best


def _iter_observation_params(observation_cfg: Mapping[str, Any]):
    for group_cfg in observation_cfg.values():
        if not isinstance(group_cfg, Mapping):
            continue
        for params in group_cfg.values():
            yield params or {}


def _iter_observation_params_for_key(observation_cfg: Mapping[str, Any], obs_key: str):
    for group_cfg in observation_cfg.values():
        if not isinstance(group_cfg, Mapping):
            continue
        params = group_cfg.get(obs_key)
        if params is not None:
            yield params or {}


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected YAML mapping, got {type(data).__name__}.")
    return data


def _task_override_from_path(task_path: Path) -> str:
    parts = task_path.with_suffix("").parts
    if "task" in parts:
        index = parts.index("task")
        return "/".join(parts[index + 1 :])
    return str(task_path)


def _find_latest_policy(expected_export_dir: Path) -> Path | None:
    policies = _list_policy_exports(expected_export_dir)
    return policies[-1] if policies else None


def _list_policy_exports(expected_export_dir: Path) -> list[Path]:
    if not expected_export_dir.is_dir():
        return []
    return sorted(
        expected_export_dir.glob("policy-*.pt"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )


def _default_policy_config_path(policy_path: Path | None) -> Path | None:
    if policy_path is None:
        return None
    return policy_path.with_suffix(".yaml")


def _default_policy_onnx_path(policy_path: Path | None) -> Path | None:
    if policy_path is None:
        return None
    return policy_path.with_suffix(".onnx")


def _default_policy_onnx_metadata_path(policy_path: Path | None) -> Path | None:
    if policy_path is None:
        return None
    return policy_path.with_suffix(".json")


def _checkpoint_summary(checkpoint_path: str | None) -> dict[str, Any]:
    summary = _checkpoint_provenance_defaults()
    if checkpoint_path is None:
        summary.update({"checkpoint_kind": "none", "checkpoint_exists": None})
        return summary
    if checkpoint_path.startswith("run:"):
        summary.update({"checkpoint_kind": "wandb_run", "checkpoint_exists": None})
        return summary

    path = Path(checkpoint_path)
    summary.update({"checkpoint_kind": "local", "checkpoint_exists": path.is_file()})
    if path.is_file():
        summary.update(_local_checkpoint_provenance(path))
    return summary


def _checkpoint_provenance_defaults() -> dict[str, Any]:
    return {
        "checkpoint_cfg_loadable": None,
        "checkpoint_cfg_load_error": None,
        "checkpoint_algo_name": None,
        "checkpoint_algo_target": None,
        "checkpoint_backend": None,
        "checkpoint_total_frames": None,
        "checkpoint_task_name": None,
        "checkpoint_num_envs": None,
        "checkpoint_source_checkpoint_path": None,
        "checkpoint_wandb_id": None,
        "checkpoint_wandb_name": None,
    }


def _local_checkpoint_provenance(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # pragma: no cover - loader errors depend on checkpoint contents.
        return {
            "checkpoint_cfg_loadable": False,
            "checkpoint_cfg_load_error": f"{type(exc).__name__}: {exc}",
        }

    checkpoint_mapping = _as_mapping(checkpoint)
    cfg = _as_mapping(checkpoint_mapping.get("cfg"))
    if not cfg:
        return {
            "checkpoint_cfg_loadable": False,
            "checkpoint_cfg_load_error": "checkpoint cfg missing or not a mapping",
        }

    algo = _as_mapping(cfg.get("algo"))
    task = _as_mapping(cfg.get("task"))
    wandb_info = _as_mapping(checkpoint_mapping.get("wandb"))
    return {
        "checkpoint_cfg_loadable": True,
        "checkpoint_cfg_load_error": None,
        "checkpoint_algo_name": _optional_str(algo.get("name")),
        "checkpoint_algo_target": _optional_str(algo.get("_target_")),
        "checkpoint_backend": _optional_str(cfg.get("backend")),
        "checkpoint_total_frames": _optional_int(cfg.get("total_frames")),
        "checkpoint_task_name": _optional_str(task.get("name")),
        "checkpoint_num_envs": _optional_int(task.get("num_envs")),
        "checkpoint_source_checkpoint_path": _optional_str(cfg.get("checkpoint_path")),
        "checkpoint_wandb_id": _optional_str(wandb_info.get("id")),
        "checkpoint_wandb_name": _optional_str(wandb_info.get("name")),
    }


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    return value if isinstance(value, Mapping) else {}


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _at_least(actual: Any, limit: float) -> bool:
    try:
        actual_value = float(actual)
    except (TypeError, ValueError):
        return False
    return actual_value >= limit


def _export_command(task_override: str, checkpoint_path: str | None) -> list[str]:
    checkpoint = checkpoint_path if checkpoint_path is not None else "<checkpoint_path>"
    return [
        "python",
        "scripts/play.py",
        f"task={task_override}",
        f"checkpoint_path={checkpoint}",
        "export_policy=true",
        "export_policy_exit=true",
        "export_policy_benchmark_iters=0",
        "export_onnx_policy=true",
        "export_onnx_required=true",
        "headless=true",
        "backend=mujoco",
    ]


if __name__ == "__main__":
    raise SystemExit(main())
