#!/usr/bin/env python3
import argparse
import contextlib
import io
import json
import math
import os
import sys
import time
from dataclasses import dataclass
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
HDMI_ROOT = SCRIPT_DIR.parent
DEFAULT_BUNDLED_SIM2REAL_ROOT = HDMI_ROOT / "third_party" / "sim2real_hdmi"
DEFAULT_SIM2REAL_ROOT = Path(os.environ.get("HDMI_SIM2REAL_ROOT", str(DEFAULT_BUNDLED_SIM2REAL_ROOT)))
DEFAULT_WBC_ROOT = Path(os.environ.get("HDMI_WBC_ROOT", str(HDMI_ROOT)))
DEFAULT_OUTPUT_DIR = Path("/tmp/wbc_hdmi_goal_full_parity_v2/sim2real_runner")
DEFAULT_ROBOT_CONFIG = "config/robot/g1.yaml"
DEFAULT_WBC_CONTACT_EEF_BODY_NAMES = ("left_wrist_yaw_link", "right_wrist_yaw_link")


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    aliases: tuple[str, ...]
    scene_config: str
    policy_config: str
    publish_object_names: tuple[str, ...]
    success_metric: str
    success_threshold: float


def official_scenario_specs() -> dict[str, ScenarioSpec]:
    return {
        "G1Dance1Subject2": ScenarioSpec(
            name="G1Dance1Subject2",
            aliases=("dance", "lafan", "G1Dance1Subject2"),
            scene_config="config/scene/g1_29dof_rubberhand.yaml",
            policy_config="checkpoints/G1Dance1Subject2/policy-1781wsjf-final.yaml",
            publish_object_names=("pelvis",),
            success_metric="not_fallen",
            success_threshold=1.0,
        ),
        "G1TrackSuitcase": ScenarioSpec(
            name="G1TrackSuitcase",
            aliases=("suitcase", "track_suitcase", "move_suitcase", "G1TrackSuitcase"),
            scene_config="config/scene/g1_29dof_rubberhand-suitcase.yaml",
            policy_config="checkpoints/G1TrackSuitcase/policy-v55m8a23-final.yaml",
            publish_object_names=("suitcase", "pelvis"),
            success_metric="suitcase_xy_displacement",
            success_threshold=0.5,
        ),
        "G1PushDoorHand": ScenarioSpec(
            name="G1PushDoorHand",
            aliases=("door", "push_door", "push_door-hand", "G1PushDoorHand"),
            scene_config="config/scene/g1_29dof_rubberhand-door.yaml",
            policy_config="checkpoints/G1PushDoorHand/policy-xg6644nr-final.yaml",
            publish_object_names=("door", "door_panel", "pelvis"),
            success_metric="door_joint_abs",
            success_threshold=0.2,
        ),
        "G1RollBall": ScenarioSpec(
            name="G1RollBall",
            aliases=("ball", "roll_ball", "roll_ball-hand", "G1RollBall"),
            scene_config="config/scene/g1_29dof_rubberhand-ball.yaml",
            policy_config="checkpoints/G1RollBall/policy-yte3rr8b-final.yaml",
            publish_object_names=("ball", "pelvis"),
            success_metric="ball_xy_displacement",
            success_threshold=0.3,
        ),
    }


def select_scenarios(names: Sequence[str] | None) -> list[ScenarioSpec]:
    scenarios = official_scenario_specs()
    if not names:
        return list(scenarios.values())
    lowered = [name.lower() for name in names]
    if "all" in lowered:
        return list(scenarios.values())
    alias_map: dict[str, ScenarioSpec] = {}
    for scenario in scenarios.values():
        alias_map[scenario.name.lower()] = scenario
        for alias in scenario.aliases:
            alias_map[alias.lower()] = scenario
    selected: list[ScenarioSpec] = []
    seen: set[str] = set()
    for raw_name in names:
        key = raw_name.lower()
        if key not in alias_map:
            available = sorted(alias_map)
            raise ValueError(f"Unknown scenario {raw_name!r}. Available aliases: {available}")
        scenario = alias_map[key]
        if scenario.name not in seen:
            selected.append(scenario)
            seen.add(scenario.name)
    return selected


def resolve_sim2real_scenario_paths(root: str | Path, scenario: ScenarioSpec) -> dict[str, Path]:
    root = Path(root)
    policy_config = root / scenario.policy_config
    paths = {
        "root": root,
        "robot_config": root / DEFAULT_ROBOT_CONFIG,
        "scene_config": root / scenario.scene_config,
        "policy_config": policy_config,
        "policy_model": policy_config.with_suffix(".onnx"),
        "policy_metadata": policy_config.with_suffix(".json"),
    }
    missing = [str(path) for key, path in paths.items() if key != "root" and not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required sim2real files for {scenario.name}: {missing}")
    return paths


def resolve_wbc_export_inputs(
    *,
    wbc_root: str | Path = DEFAULT_WBC_ROOT,
    sim2real_root: str | Path = DEFAULT_SIM2REAL_ROOT,
    task_yaml: str | Path,
    policy_model: str | Path,
) -> dict[str, Any]:
    wbc_root = Path(wbc_root).resolve()
    sim2real_root = Path(sim2real_root).resolve()
    task_path = _resolve_under_root(wbc_root, task_yaml)
    policy_model_path = _resolve_under_root(wbc_root, policy_model)
    if policy_model_path.suffix == ".pt":
        policy_model_path = policy_model_path.with_suffix(".onnx")
    policy_config_path = policy_model_path.with_suffix(".yaml")
    policy_metadata_path = policy_model_path.with_suffix(".json")
    task_cfg = _load_wbc_task_config(task_path, wbc_root=wbc_root)
    command_cfg = _mapping(task_cfg.get("command", {}), f"{task_path}: command")
    robot_cfg = _mapping(task_cfg.get("robot", {}), f"{task_path}: robot")
    policy_config = _load_yaml(policy_config_path)
    _rewrite_policy_motion_paths(policy_config, wbc_root=wbc_root)
    _apply_wbc_robot_overrides_to_policy_config(policy_config, robot_cfg)
    _annotate_wbc_contact_eef_metadata(policy_config, command_cfg)

    object_asset_name = str(command_cfg["object_asset_name"])
    object_body_name = str(command_cfg.get("object_body_name") or object_asset_name)
    extra_object_names = tuple(str(name) for name in command_cfg.get("extra_object_names", ()))
    object_type = _infer_wbc_object_type(
        object_asset_name=object_asset_name,
        explicit_object_type=command_cfg.get("object_type"),
        extra_object_names=extra_object_names,
    )
    publish_object_names = _dedupe_names(
        (
            object_asset_name,
            object_body_name,
            *extra_object_names,
            str(command_cfg.get("root_body_name", "pelvis")),
        )
    )
    scene_config = _wbc_scene_config(
        sim2real_root=sim2real_root,
        robot_type=str(robot_cfg.get("robot_type", "g1_29dof")),
        object_asset_name=object_asset_name,
        object_type=object_type,
        object_joint_name=command_cfg.get("object_joint_name"),
        command_cfg=command_cfg,
    )
    _apply_wbc_task_sim_dt(scene_config, task_cfg.get("sim", {}), path=task_path)
    scenario = ScenarioSpec(
        name=str(task_cfg.get("name") or task_path.stem),
        aliases=(str(task_cfg.get("name") or task_path.stem), task_path.stem),
        scene_config=str(scene_config["ROBOT_SCENE"]),
        policy_config=str(policy_config_path),
        publish_object_names=publish_object_names,
        success_metric="not_fallen",
        success_threshold=1.0,
    )
    missing = [path for path in (task_path, policy_model_path, policy_config_path, policy_metadata_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required WBC export files: {[str(path) for path in missing]}")
    return {
        "root": wbc_root,
        "task_yaml": task_path,
        "scenario": scenario,
        "scene_config": scene_config,
        "policy_config": policy_config,
        "policy_config_path": policy_config_path,
        "policy_model": policy_model_path,
        "policy_metadata": policy_metadata_path,
        "publish_object_names": publish_object_names,
        "object_asset_name": object_asset_name,
        "object_body_name": object_body_name,
        "object_type": object_type,
        "scene_asset_source": scene_config.pop("_asset_source", "unknown"),
    }


def _infer_wbc_object_type(
    *,
    object_asset_name: str,
    explicit_object_type: Any,
    extra_object_names: Sequence[str],
) -> str:
    if explicit_object_type:
        return str(explicit_object_type)
    extras = {str(name) for name in extra_object_names}
    if object_asset_name == "foam" and "stool_support" in extras:
        return "foam_with_support"
    if object_asset_name == "stool" and "stool_support" in extras:
        return "stool_with_support"
    return str(object_asset_name)


def load_wbc_reference_initial_state(
    *,
    policy_config: Mapping[str, Any],
    root_body_name: str,
    object_body_names: Sequence[str] = (),
    initial_step: int = 0,
    target_fps: int = 50,
) -> dict[str, Any]:
    motion_path = _policy_motion_path(policy_config)
    motion_npz_path = _select_single_motion_npz(motion_path)
    meta_path = motion_npz_path.parent / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing motion meta.json next to {motion_npz_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    source_fps = int(meta.get("fps", target_fps))
    if source_fps != int(target_fps):
        raise NotImplementedError(
            f"WBC reference initialization currently requires motion fps {target_fps}, got {source_fps} from {meta_path}."
        )
    if initial_step < 0:
        raise ValueError("initial_step must be non-negative")
    motion = np.load(motion_npz_path, allow_pickle=True)
    body_names = [str(name) for name in meta["body_names"]]
    joint_names = [str(name) for name in meta["joint_names"]]
    if root_body_name not in body_names:
        raise ValueError(f"Root body {root_body_name!r} is absent from {meta_path}")
    num_steps = int(motion["joint_pos"].shape[0])
    step = min(int(initial_step), num_steps - 1)
    root_idx = body_names.index(root_body_name)
    body_pose_by_name = {}
    for body_name in _dedupe_names(object_body_names):
        if body_name == root_body_name or body_name not in body_names:
            continue
        body_idx = body_names.index(body_name)
        body_pose_by_name[body_name] = {
            "pos": _array_to_float_list(motion["body_pos_w"][step, body_idx]),
            "quat": _normalize_quat_list(motion["body_quat_w"][step, body_idx]),
            "lin_vel": _array_to_float_list(motion["body_lin_vel_w"][step, body_idx]),
            "ang_vel": _array_to_float_list(motion["body_ang_vel_w"][step, body_idx]),
        }
    return {
        "motion_path": str(motion_npz_path),
        "meta_path": str(meta_path),
        "motion_step": step,
        "motion_num_steps": num_steps,
        "root_body_name": root_body_name,
        "root_pos": _array_to_float_list(motion["body_pos_w"][step, root_idx]),
        "root_quat": _normalize_quat_list(motion["body_quat_w"][step, root_idx]),
        "root_lin_vel": _array_to_float_list(motion["body_lin_vel_w"][step, root_idx]),
        "root_ang_vel": _array_to_float_list(motion["body_ang_vel_w"][step, root_idx]),
        "joint_pos_by_name": {
            joint_name: float(motion["joint_pos"][step, joint_idx])
            for joint_idx, joint_name in enumerate(joint_names)
        },
        "joint_vel_by_name": {
            joint_name: float(motion["joint_vel"][step, joint_idx])
            for joint_idx, joint_name in enumerate(joint_names)
        },
        "body_pose_by_name": body_pose_by_name,
    }


def build_wbc_reference_tracker(
    *,
    model: Any,
    policy_config: Mapping[str, Any],
    joint_names: Sequence[str],
    object_body_names: Sequence[str] = (),
    initial_step: int = 0,
    target_fps: int = 50,
) -> dict[str, Any]:
    import mujoco

    motion_path = _policy_motion_path(policy_config)
    motion_npz_path = _select_single_motion_npz(motion_path)
    meta_path = motion_npz_path.parent / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing motion meta.json next to {motion_npz_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    source_fps = int(meta.get("fps", target_fps))
    if source_fps != int(target_fps):
        raise NotImplementedError(
            f"WBC reference tracking currently requires motion fps {target_fps}, got {source_fps} from {meta_path}."
        )
    if initial_step < 0:
        raise ValueError("initial_step must be non-negative")
    motion = np.load(motion_npz_path, allow_pickle=True)
    meta_joint_names = [str(name) for name in meta["joint_names"]]
    meta_body_names = [str(name) for name in meta["body_names"]]
    num_steps = int(motion["joint_pos"].shape[0])

    joint_qpos_adrs: list[int] = []
    joint_motion_indices: list[int] = []
    joint_target_indices: list[int] = []
    tracked_joint_names: list[str] = []
    missing_joint_names: list[str] = []
    for target_index, joint_name in enumerate(joint_names):
        joint_name = str(joint_name)
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0 or joint_name not in meta_joint_names:
            missing_joint_names.append(joint_name)
            continue
        joint_type = int(model.jnt_type[joint_id])
        if joint_type not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            missing_joint_names.append(joint_name)
            continue
        joint_qpos_adrs.append(int(model.jnt_qposadr[joint_id]))
        joint_target_indices.append(int(target_index))
        joint_motion_indices.append(meta_joint_names.index(joint_name))
        tracked_joint_names.append(joint_name)

    body_ids: list[int] = []
    body_motion_indices: list[int] = []
    tracked_body_names: list[str] = []
    for body_idx, body_name in enumerate(meta_body_names):
        body_id = _mujoco_body_id(model, body_name)
        if body_id < 0:
            continue
        body_ids.append(int(body_id))
        body_motion_indices.append(body_idx)
        tracked_body_names.append(body_name)

    object_body_ids: list[int] = []
    object_motion_indices: list[int] = []
    tracked_object_body_names: list[str] = []
    missing_object_body_names: list[str] = []
    for body_name in _dedupe_names(object_body_names):
        if body_name not in meta_body_names:
            missing_object_body_names.append(body_name)
            continue
        body_id = _mujoco_body_id(model, body_name)
        if body_id < 0:
            missing_object_body_names.append(body_name)
            continue
        object_body_ids.append(int(body_id))
        object_motion_indices.append(meta_body_names.index(body_name))
        tracked_object_body_names.append(body_name)

    return {
        "motion_path": str(motion_npz_path),
        "meta_path": str(meta_path),
        "initial_step": int(initial_step),
        "num_steps": num_steps,
        "motion": motion,
        "joint_qpos_adrs": np.asarray(joint_qpos_adrs, dtype=np.int64),
        "joint_motion_indices": np.asarray(joint_motion_indices, dtype=np.int64),
        "joint_target_indices": np.asarray(joint_target_indices, dtype=np.int64),
        "tracked_joint_names": tracked_joint_names,
        "missing_joint_names": missing_joint_names,
        "body_ids": np.asarray(body_ids, dtype=np.int64),
        "body_motion_indices": np.asarray(body_motion_indices, dtype=np.int64),
        "tracked_body_names": tracked_body_names,
        "object_body_ids": np.asarray(object_body_ids, dtype=np.int64),
        "object_motion_indices": np.asarray(object_motion_indices, dtype=np.int64),
        "tracked_object_body_names": tracked_object_body_names,
        "missing_object_body_names": missing_object_body_names,
    }


def sample_wbc_reference_tracking(
    model: Any,
    data: Any,
    reference_tracker: Mapping[str, Any],
    policy_step: int,
    *,
    reference_step: int | None = None,
) -> dict[str, Any]:
    del model
    num_steps = int(reference_tracker["num_steps"])
    if reference_step is None:
        reference_step = min(int(reference_tracker["initial_step"]) + int(policy_step) + 1, num_steps - 1)
    else:
        reference_step = int(np.clip(int(reference_step), 0, num_steps - 1))
    motion = reference_tracker["motion"]
    sample: dict[str, Any] = {
        "reference_step": reference_step,
        "reference_joint_count": int(len(reference_tracker["joint_qpos_adrs"])),
        "reference_body_count": int(len(reference_tracker["body_ids"])),
        "reference_object_body_count": int(len(reference_tracker["object_body_ids"])),
    }
    offset_errors = _sample_reference_offset_errors(data, reference_tracker, reference_step)
    if offset_errors:
        sample["reference_offset_errors"] = offset_errors

    joint_qpos_adrs = reference_tracker["joint_qpos_adrs"]
    joint_motion_indices = reference_tracker["joint_motion_indices"]
    if len(joint_qpos_adrs) > 0:
        q_current = np.asarray(data.qpos[joint_qpos_adrs], dtype=np.float64)
        q_reference = np.asarray(motion["joint_pos"][reference_step, joint_motion_indices], dtype=np.float64)
        q_error = q_current - q_reference
        sample["q_ref_l2"] = _json_float(np.linalg.norm(q_error))
        sample["q_ref_linf"] = _json_float(np.max(np.abs(q_error)))
        sample["q_ref_joint_error_top"] = _top_scalar_error_entries(
            reference_tracker["tracked_joint_names"],
            q_error,
            top_n=8,
        )

    body_ids = reference_tracker["body_ids"]
    body_motion_indices = reference_tracker["body_motion_indices"]
    if len(body_ids) > 0:
        body_current = np.asarray(data.xpos[body_ids], dtype=np.float64)
        body_reference = np.asarray(motion["body_pos_w"][reference_step, body_motion_indices], dtype=np.float64)
        body_error = body_current - body_reference
        sample["body_pos_ref_l2"] = _json_float(np.linalg.norm(body_error.reshape(-1)))
        sample["body_pos_ref_linf"] = _json_float(np.max(np.abs(body_error)))
        sample["body_pos_ref_error_top"] = _top_vector_error_entries(
            reference_tracker["tracked_body_names"],
            body_error,
            top_n=8,
        )

    object_body_ids = reference_tracker["object_body_ids"]
    object_motion_indices = reference_tracker["object_motion_indices"]
    if len(object_body_ids) > 0:
        object_current = np.asarray(data.xpos[object_body_ids], dtype=np.float64)
        object_reference = np.asarray(motion["body_pos_w"][reference_step, object_motion_indices], dtype=np.float64)
        object_error = object_current - object_reference
        sample["object_pos_ref_l2"] = _json_float(np.linalg.norm(object_error.reshape(-1)))
        sample["object_pos_ref_linf"] = _json_float(np.max(np.abs(object_error)))
        sample["object_pos_ref_error_top"] = _top_vector_error_entries(
            reference_tracker["tracked_object_body_names"],
            object_error,
            top_n=8,
        )
    return sample


def sample_wbc_q_target_reference_tracking(
    *,
    q_target: np.ndarray,
    reference_tracker: Mapping[str, Any],
    policy_step: int,
    reference_step: int | None = None,
) -> dict[str, Any]:
    num_steps = int(reference_tracker["num_steps"])
    if reference_step is None:
        reference_step = min(int(reference_tracker["initial_step"]) + int(policy_step) + 1, num_steps - 1)
    else:
        reference_step = int(np.clip(int(reference_step), 0, num_steps - 1))
    motion = reference_tracker["motion"]
    joint_motion_indices = np.asarray(reference_tracker["joint_motion_indices"], dtype=np.int64)
    joint_target_indices = np.asarray(
        reference_tracker.get("joint_target_indices", np.arange(len(joint_motion_indices))),
        dtype=np.int64,
    )
    sample: dict[str, Any] = {
        "reference_step": reference_step,
        "q_target_ref_joint_count": int(len(joint_motion_indices)),
    }
    if len(joint_motion_indices) == 0:
        return sample
    if len(joint_target_indices) != len(joint_motion_indices):
        sample["q_target_ref_error"] = (
            f"joint_target_indices length {len(joint_target_indices)} != "
            f"joint_motion_indices length {len(joint_motion_indices)}"
        )
        return sample
    q_target = np.asarray(q_target, dtype=np.float64).reshape(-1)
    max_target_index = int(np.max(joint_target_indices))
    if q_target.size <= max_target_index:
        sample["q_target_ref_error"] = f"q_target size {q_target.size} <= max joint target index {max_target_index}"
        return sample
    q_target_selected = q_target[joint_target_indices]
    q_reference = np.asarray(motion["joint_pos"][reference_step, joint_motion_indices], dtype=np.float64)
    q_error = q_target_selected - q_reference
    sample["q_target_ref_l2"] = _json_float(np.linalg.norm(q_error))
    sample["q_target_ref_linf"] = _json_float(np.max(np.abs(q_error)))
    sample["q_target_ref_joint_error_top"] = _top_scalar_error_entries(
        reference_tracker["tracked_joint_names"],
        q_error,
        top_n=8,
    )
    return sample


def _sample_reference_offset_errors(
    data: Any,
    reference_tracker: Mapping[str, Any],
    reference_step: int,
    *,
    offsets: Sequence[int] = (-10, -5, -2, -1, 0, 1, 2, 5, 10),
) -> list[dict[str, Any]]:
    motion = reference_tracker["motion"]
    num_steps = int(reference_tracker["num_steps"])
    joint_qpos_adrs = reference_tracker["joint_qpos_adrs"]
    joint_motion_indices = reference_tracker["joint_motion_indices"]
    body_ids = reference_tracker["body_ids"]
    body_motion_indices = reference_tracker["body_motion_indices"]
    object_body_ids = reference_tracker["object_body_ids"]
    object_motion_indices = reference_tracker["object_motion_indices"]
    q_current = np.asarray(data.qpos[joint_qpos_adrs], dtype=np.float64) if len(joint_qpos_adrs) > 0 else None
    body_current = np.asarray(data.xpos[body_ids], dtype=np.float64) if len(body_ids) > 0 else None
    object_current = np.asarray(data.xpos[object_body_ids], dtype=np.float64) if len(object_body_ids) > 0 else None
    rows = []
    for offset in offsets:
        candidate_step = int(np.clip(reference_step + int(offset), 0, num_steps - 1))
        row: dict[str, Any] = {
            "offset": int(offset),
            "reference_step": candidate_step,
        }
        if q_current is not None:
            q_reference = np.asarray(motion["joint_pos"][candidate_step, joint_motion_indices], dtype=np.float64)
            row["q_ref_l2"] = _json_float(np.linalg.norm(q_current - q_reference))
        if body_current is not None:
            body_reference = np.asarray(motion["body_pos_w"][candidate_step, body_motion_indices], dtype=np.float64)
            row["body_pos_ref_l2"] = _json_float(np.linalg.norm((body_current - body_reference).reshape(-1)))
        if object_current is not None:
            object_reference = np.asarray(motion["body_pos_w"][candidate_step, object_motion_indices], dtype=np.float64)
            row["object_pos_ref_l2"] = _json_float(np.linalg.norm((object_current - object_reference).reshape(-1)))
        rows.append(row)
    return rows


def summarize_rollout(
    *,
    scenario: ScenarioSpec,
    duration_sec: float,
    policy_steps: int,
    sim_steps: int,
    decimation: int,
    samples: Sequence[Mapping[str, Any]],
    actions_abs_max: Sequence[float],
    actions_abs_mean: Sequence[float],
    q_target_abs_max: Sequence[float],
    video_path: str | Path | None,
    fall_height: float = 0.4,
    control_latency_steps: int = 0,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("samples must not be empty")
    pelvis_z_values = np.asarray([sample.get("pelvis_z", np.nan) for sample in samples], dtype=np.float64)
    pelvis_up_values = np.asarray([sample.get("pelvis_up_z", np.nan) for sample in samples], dtype=np.float64)
    pelvis_up_available = bool(np.isfinite(pelvis_up_values).any())
    finite = bool(np.isfinite(pelvis_z_values).all())
    if pelvis_up_available:
        finite = finite and bool(np.isfinite(pelvis_up_values).all())
    min_pelvis_z = _json_float(np.nanmin(pelvis_z_values))
    final_pelvis_z = _json_float(pelvis_z_values[-1])
    min_pelvis_up_z = _json_float(np.nanmin(pelvis_up_values)) if pelvis_up_available else None
    final_pelvis_up_z = _json_float(pelvis_up_values[-1]) if pelvis_up_available else None
    low_pelvis_observed = bool(min_pelvis_z is not None and min_pelvis_z < fall_height)
    posture_up_threshold = 0.5
    final_height_ok = bool(final_pelvis_z is not None and final_pelvis_z >= fall_height)
    final_up_ok = bool(final_pelvis_up_z is None or final_pelvis_up_z >= posture_up_threshold)
    final_posture_ok = bool(finite and final_height_ok and final_up_ok)
    not_fallen = final_posture_ok
    object_metrics = _object_metrics(samples)
    metric_value = 1.0 if scenario.success_metric == "not_fallen" else object_metrics.get(scenario.success_metric, 0.0)
    if metric_value is None:
        metric_value = 0.0
    heuristic_success = bool(not_fallen and metric_value >= scenario.success_threshold)
    first_xy = np.asarray(samples[0].get("pelvis_xy", [np.nan, np.nan]), dtype=np.float64)
    last_xy = np.asarray(samples[-1].get("pelvis_xy", [np.nan, np.nan]), dtype=np.float64)
    pelvis_xy_displacement = _json_float(np.linalg.norm(last_xy - first_xy))
    reference_tracking = _reference_tracking_summary(samples)
    return {
        "scenario": scenario.name,
        "duration_sec": float(duration_sec),
        "policy_steps": int(policy_steps),
        "sim_steps": int(sim_steps),
        "decimation": int(decimation),
        "control_latency_steps": int(control_latency_steps),
        "finite": finite,
        "fall_height": float(fall_height),
        "posture_up_threshold": float(posture_up_threshold),
        "not_fallen": bool(not_fallen),
        "low_pelvis_observed": low_pelvis_observed,
        "final_posture_ok": final_posture_ok,
        "min_pelvis_z": min_pelvis_z,
        "final_pelvis_z": final_pelvis_z,
        "min_pelvis_up_z": min_pelvis_up_z,
        "final_pelvis_up_z": final_pelvis_up_z,
        "pelvis_xy_displacement": pelvis_xy_displacement,
        "success_metric": scenario.success_metric,
        "success_threshold": float(scenario.success_threshold),
        "success_metric_value": _json_float(metric_value),
        "heuristic_success": heuristic_success,
        "object_metrics": object_metrics,
        "reference_tracking": reference_tracking,
        "max_abs_action": _json_float(max(actions_abs_max) if actions_abs_max else np.nan),
        "mean_abs_action": _json_float(float(np.mean(actions_abs_mean)) if actions_abs_mean else np.nan),
        "max_abs_q_target": _json_float(max(q_target_abs_max) if q_target_abs_max else np.nan),
        "first_sample": _json_sample(samples[0]),
        "last_sample": _json_sample(samples[-1]),
        "video_path": str(video_path) if video_path is not None else None,
    }


def build_rollout_trace(
    *,
    scenario: ScenarioSpec,
    samples: Sequence[Mapping[str, Any]],
    actions_abs_max: Sequence[float],
    actions_abs_mean: Sequence[float],
    q_target_abs_max: Sequence[float],
    fall_height: float = 0.4,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("samples must not be empty")
    first_sample = samples[0]
    series = []
    for index, sample in enumerate(samples):
        success_metric_value = _sample_success_metric_value(
            scenario=scenario,
            first_sample=first_sample,
            sample=sample,
            fall_height=fall_height,
        )
        success_progress = _success_progress(success_metric_value, scenario.success_threshold)
        pelvis_z = _json_float(sample.get("pelvis_z"))
        pelvis_up_z = _json_float(sample.get("pelvis_up_z"))
        not_fallen_proxy = _sample_not_fallen(sample, fall_height=fall_height)
        row = {
            "policy_step": int(sample.get("policy_step", index)),
            "time": _json_float(sample.get("time")),
            "pelvis_z": pelvis_z,
            "pelvis_up_z": pelvis_up_z,
            "not_fallen_proxy": bool(not_fallen_proxy),
            "success_metric": scenario.success_metric,
            "success_metric_value": _json_float(success_metric_value),
            "success_progress": _json_float(success_progress),
            "reward_proxy": _json_float(success_progress if not_fallen_proxy else 0.0),
            "action_abs_max": _sequence_json_float(actions_abs_max, index),
            "action_abs_mean": _sequence_json_float(actions_abs_mean, index),
            "q_target_abs_max": _sequence_json_float(q_target_abs_max, index),
            "pelvis_xy_displacement": _sample_xy_displacement(first_sample, sample, key="pelvis_xy"),
            "suitcase_xy_displacement": _sample_xy_displacement(first_sample, sample, key="suitcase_xy"),
            "ball_xy_displacement": _sample_xy_displacement(first_sample, sample, key="ball_xy"),
            "door_joint_abs": _json_float(abs(float(sample["door_joint"]))) if "door_joint" in sample else None,
            "q_ref_l2": _json_float(sample.get("q_ref_l2")),
            "body_pos_ref_l2": _json_float(sample.get("body_pos_ref_l2")),
            "object_pos_ref_l2": _json_float(sample.get("object_pos_ref_l2")),
        }
        series.append(row)
    return {
        "scenario": scenario.name,
        "success_metric": scenario.success_metric,
        "success_threshold": float(scenario.success_threshold),
        "series": series,
    }


def write_rollout_artifacts(
    trace: Mapping[str, Any],
    *,
    trace_json_path: str | Path | None = None,
    plot_path: str | Path | None = None,
) -> dict[str, str | None]:
    paths: dict[str, str | None] = {"trace_json_path": None, "plot_path": None}
    if trace_json_path is not None:
        path = Path(trace_json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(trace, indent=2, sort_keys=True), encoding="utf-8")
        paths["trace_json_path"] = str(path)
    if plot_path is not None:
        path = Path(plot_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _plot_rollout_trace(trace, path)
        paths["plot_path"] = str(path)
    return paths


def _plot_rollout_trace(trace: Mapping[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    series = list(trace.get("series", []))
    if not series:
        raise ValueError("trace series must not be empty")
    t = np.asarray([_nan_if_none(row.get("time")) for row in series], dtype=np.float64)
    if not np.isfinite(t).any():
        t = np.arange(len(series), dtype=np.float64)
    panels = [
        (
            "policy / posture",
            (
                ("reward_proxy", "reward proxy"),
                ("success_progress", "success progress"),
                ("pelvis_z", "pelvis z"),
                ("pelvis_up_z", "pelvis up z"),
            ),
        ),
        (
            "object / reference",
            (
                ("success_metric_value", str(trace.get("success_metric", "success metric"))),
                ("pelvis_xy_displacement", "pelvis xy displacement"),
                ("q_ref_l2", "q ref L2"),
                ("body_pos_ref_l2", "body ref L2"),
                ("object_pos_ref_l2", "object ref L2"),
            ),
        ),
        (
            "action",
            (
                ("action_abs_max", "|action|max"),
                ("action_abs_mean", "|action|mean"),
                ("q_target_abs_max", "|q_target|max"),
            ),
        ),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 8), sharex=True)
    if len(panels) == 1:
        axes = [axes]
    for axis, (title, keys) in zip(axes, panels):
        plotted = False
        for key, label in keys:
            y = np.asarray([_nan_if_none(row.get(key)) for row in series], dtype=np.float64)
            if np.isfinite(y).any():
                axis.plot(t, y, label=label, linewidth=1.8)
                plotted = True
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
        if plotted:
            axis.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(f"{trace.get('scenario', 'rollout')} MuJoCo policy rollout curves")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _sample_success_metric_value(
    *,
    scenario: ScenarioSpec,
    first_sample: Mapping[str, Any],
    sample: Mapping[str, Any],
    fall_height: float,
) -> float:
    if scenario.success_metric == "not_fallen":
        return 1.0 if _sample_not_fallen(sample, fall_height=fall_height) else 0.0
    if scenario.success_metric == "suitcase_xy_displacement":
        return _sample_xy_displacement(first_sample, sample, key="suitcase_xy") or 0.0
    if scenario.success_metric == "ball_xy_displacement":
        return _sample_xy_displacement(first_sample, sample, key="ball_xy") or 0.0
    if scenario.success_metric == "door_joint_abs":
        return abs(float(sample.get("door_joint", 0.0)))
    return 0.0


def _sample_not_fallen(sample: Mapping[str, Any], *, fall_height: float, posture_up_threshold: float = 0.5) -> bool:
    pelvis_z = _json_float(sample.get("pelvis_z"))
    pelvis_up_z = _json_float(sample.get("pelvis_up_z"))
    return bool(
        pelvis_z is not None
        and pelvis_z >= fall_height
        and (pelvis_up_z is None or pelvis_up_z >= posture_up_threshold)
    )


def _success_progress(value: float | None, threshold: float) -> float:
    if value is None or threshold <= 0.0:
        return 0.0
    return float(np.clip(float(value) / float(threshold), 0.0, 1.0))


def _sample_xy_displacement(first_sample: Mapping[str, Any], sample: Mapping[str, Any], *, key: str) -> float | None:
    if key not in first_sample or key not in sample:
        return None
    first = np.asarray(first_sample.get(key), dtype=np.float64)
    current = np.asarray(sample.get(key), dtype=np.float64)
    if first.shape != current.shape or first.size < 2:
        return None
    return _json_float(np.linalg.norm(current[:2] - first[:2]))


def _sequence_json_float(values: Sequence[float], index: int) -> float | None:
    if index >= len(values):
        return None
    return _json_float(values[index])


def _nan_if_none(value: Any) -> float:
    number = _json_float(value)
    return float("nan") if number is None else float(number)


def run_scenario(
    *,
    root: str | Path,
    scenario: ScenarioSpec,
    output_dir: str | Path,
    duration_sec: float = 6.0,
    rl_rate: float = 50.0,
    width: int = 640,
    height: int = 480,
    render_video: bool = True,
    quiet_observations: bool = True,
    fall_height: float = 0.4,
    control_latency_steps: int = 0,
    disable_virtual_gantry_after_start: bool = True,
    video_stride: int = 1,
    viewer: bool = False,
    viewer_speed: float = 1.0,
    trace_json_path: str | Path | None = None,
    plot_path: str | Path | None = None,
) -> dict[str, Any]:
    video_stride = normalize_video_stride(video_stride)
    if viewer_speed <= 0.0:
        raise ValueError(f"viewer_speed must be > 0, got {viewer_speed}.")
    if render_video and not viewer:
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    paths = resolve_sim2real_scenario_paths(root, scenario)
    root = paths["root"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with _sim2real_context(root):
        import mujoco

        robot_config = _load_yaml(paths["robot_config"])
        raw_scene_config = _load_yaml(paths["scene_config"])
        virtual_gantry_configured = bool(raw_scene_config.get("ENABLE_ELASTIC_BAND", False))
        scene_config = official_sim2real_scene_config(
            raw_scene_config,
            disable_virtual_gantry_after_start=disable_virtual_gantry_after_start,
        )
        virtual_gantry_applied = bool(scene_config.get("ENABLE_ELASTIC_BAND", False))
        policy_config = _load_yaml(paths["policy_config"])
        robot_scene = root / str(scene_config["ROBOT_SCENE"])
        model = mujoco.MjModel.from_xml_path(str(robot_scene))
        data = mujoco.MjData(model)
        sim_dt = float(scene_config["SIMULATE_DT"])
        model.opt.timestep = sim_dt
        mujoco.mj_forward(model, data)
        elastic_band = _make_elastic_band(model, scene_config)
        policy = DirectSim2RealPolicy(
            model=model,
            robot_config=robot_config,
            scene_config=scene_config,
            policy_config=policy_config,
            model_path=paths["policy_model"],
            publish_object_names=scenario.publish_object_names,
            quiet_observations=quiet_observations,
        )
        decimation = max(1, int(round((1.0 / float(rl_rate)) / sim_dt)))
        policy_steps = int(round(duration_sec * float(rl_rate)))
        samples: list[dict[str, Any]] = []
        actions_abs_max: list[float] = []
        actions_abs_mean: list[float] = []
        q_target_abs_max: list[float] = []
        pending_q_targets: list[np.ndarray] = []
        video_path = output_dir / f"{scenario.name}_sim2real_headless_{duration_sec:g}s.mp4" if render_video else None
        video_fps = max(1, int(round(float(rl_rate) / video_stride)))
        writer = _make_video_writer(video_path, fps=video_fps) if render_video else contextlib.nullcontext(None)
        renderer = mujoco.Renderer(model, height=height, width=width) if render_video else None
        camera = _make_camera() if render_video else None
        with writer as video_writer, _policy_viewer_context(model, data, enabled=viewer) as policy_viewer:
            viewer_clock = _ViewerClock(data.time)
            for policy_step in range(policy_steps):
                if policy_viewer is not None and not policy_viewer.is_running():
                    break
                action, q_target = policy.step(model, data)
                active_q_target = select_control_q_target(
                    pending_q_targets,
                    q_target,
                    policy.state_processor.joint_pos,
                    control_latency_steps=control_latency_steps,
                )
                actions_abs_max.append(float(np.max(np.abs(action))))
                actions_abs_mean.append(float(np.mean(np.abs(action))))
                q_target_abs_max.append(float(np.max(np.abs(active_q_target))))
                for _ in range(decimation):
                    ctrl = policy.compute_robot_ctrl(model, data, active_q_target)
                    data.ctrl[:] = ctrl
                    _apply_object_joint_control(model, data, scene_config)
                    _apply_elastic_band_force(data, elastic_band)
                    mujoco.mj_step(model, data)
                    if policy_viewer is not None:
                        _sync_policy_viewer(policy_viewer, data, viewer_clock, speed=viewer_speed)
                sample = _sample_scene(model, data, scenario, scene_config)
                sample["policy_step"] = policy_step
                sample["time"] = float(data.time)
                samples.append(sample)
                if renderer is not None and video_writer is not None and policy_step % video_stride == 0:
                    _render_frame(renderer, model, data, camera, video_writer)
        if renderer is not None:
            renderer.close()
        summary = summarize_rollout(
            scenario=scenario,
            duration_sec=duration_sec,
            policy_steps=policy_steps,
            sim_steps=policy_steps * decimation,
            decimation=decimation,
            samples=samples,
            actions_abs_max=actions_abs_max,
            actions_abs_mean=actions_abs_mean,
            q_target_abs_max=q_target_abs_max,
            video_path=video_path,
            fall_height=fall_height,
            control_latency_steps=control_latency_steps,
        )
        summary.update(
            {
                "sim2real_root": str(root),
                "robot_config": str(paths["robot_config"]),
                "scene_config": str(paths["scene_config"]),
                "policy_config": str(paths["policy_config"]),
                "policy_model": str(paths["policy_model"]),
                "policy_metadata": str(paths["policy_metadata"]),
                "virtual_gantry_configured": virtual_gantry_configured,
                "virtual_gantry_applied": virtual_gantry_applied,
                "virtual_gantry_mode": (
                    "kept_enabled"
                    if virtual_gantry_applied
                    else "disabled_after_start"
                    if virtual_gantry_configured
                    else "not_configured"
                ),
                "video_stride": int(video_stride),
                "video_fps": int(video_fps) if render_video else None,
                "viewer_enabled": bool(viewer),
                "viewer_speed": float(viewer_speed) if viewer else None,
            }
        )
        trace = build_rollout_trace(
            scenario=scenario,
            samples=samples,
            actions_abs_max=actions_abs_max,
            actions_abs_mean=actions_abs_mean,
            q_target_abs_max=q_target_abs_max,
            fall_height=fall_height,
        )
        summary.update(write_rollout_artifacts(trace, trace_json_path=trace_json_path, plot_path=plot_path))
        summary_path = output_dir / f"{scenario.name}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        summary["summary_path"] = str(summary_path)
        return summary


def run_wbc_export_scenario(
    *,
    sim2real_root: str | Path = DEFAULT_SIM2REAL_ROOT,
    wbc_root: str | Path = DEFAULT_WBC_ROOT,
    task_yaml: str | Path,
    policy_model: str | Path,
    output_dir: str | Path,
    duration_sec: float = 2.0,
    rl_rate: float = 50.0,
    width: int = 640,
    height: int = 480,
    render_video: bool = True,
    quiet_observations: bool = True,
    fall_height: float = 0.4,
    control_latency_steps: int = 0,
    wbc_initial_state: str = "scene_default",
    wbc_initial_step: int = 0,
    wbc_initial_pause_sec: float = 0.0,
    disable_virtual_gantry_after_start: bool = True,
    video_stride: int = 1,
) -> dict[str, Any]:
    video_stride = normalize_video_stride(video_stride)
    if render_video:
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    resolved = resolve_wbc_export_inputs(
        wbc_root=wbc_root,
        sim2real_root=sim2real_root,
        task_yaml=task_yaml,
        policy_model=policy_model,
    )
    sim2real_root = Path(sim2real_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with _sim2real_context(sim2real_root):
        import mujoco

        robot_config = _load_yaml(sim2real_root / DEFAULT_ROBOT_CONFIG)
        raw_scene_config = dict(resolved["scene_config"])
        virtual_gantry_configured = bool(raw_scene_config.get("ENABLE_ELASTIC_BAND", False))
        scene_config = official_sim2real_scene_config(
            raw_scene_config,
            disable_virtual_gantry_after_start=disable_virtual_gantry_after_start,
        )
        virtual_gantry_applied = bool(scene_config.get("ENABLE_ELASTIC_BAND", False))
        policy_config = dict(resolved["policy_config"])
        reference_initial_state: dict[str, Any] | None = None
        static_body_pose_mjcf: dict[str, Any] | None = None
        if wbc_initial_state == "reference_frame":
            root_body_name = _wbc_root_body_name(policy_config)
            object_body_names = [
                name for name in resolved["publish_object_names"]
                if name != root_body_name
            ]
            reference_initial_state = load_wbc_reference_initial_state(
                policy_config=policy_config,
                root_body_name=root_body_name,
                object_body_names=object_body_names,
                initial_step=wbc_initial_step,
                target_fps=int(round(rl_rate)),
            )
            static_body_pose_mjcf = materialize_mjcf_static_body_poses(
                scene_path=scene_config["ROBOT_SCENE"],
                body_pose_by_name=reference_initial_state.get("body_pose_by_name", {}),
                cache_dir=output_dir / "mjcf_static_pose_cache",
            )
            scene_config["ROBOT_SCENE"] = str(static_body_pose_mjcf["scene_path"])
        elif wbc_initial_state != "scene_default":
            raise ValueError(f"Unknown WBC initial state mode: {wbc_initial_state!r}")
        model = mujoco.MjModel.from_xml_path(str(scene_config["ROBOT_SCENE"]))
        data = mujoco.MjData(model)
        sim_dt = float(scene_config["SIMULATE_DT"])
        model.opt.timestep = sim_dt
        initial_state_summary: dict[str, Any] | None = None
        if reference_initial_state is not None:
            initial_state_summary = apply_wbc_reference_initial_state(
                model=model,
                data=data,
                reference_initial_state=reference_initial_state,
            )
            initial_state_summary["static_body_pose_mjcf"] = static_body_pose_mjcf
        mujoco.mj_forward(model, data)
        elastic_band = _make_elastic_band(model, scene_config)
        wbc_action_adapter_config = _wbc_action_adapter_config_from_task_yaml(resolved["task_yaml"])
        policy = DirectSim2RealPolicy(
            model=model,
            robot_config=robot_config,
            scene_config=scene_config,
            policy_config=policy_config,
            model_path=resolved["policy_model"],
            publish_object_names=resolved["publish_object_names"],
            quiet_observations=quiet_observations,
            observation_semantics="wbc",
            wbc_action_delay=wbc_action_adapter_config["delay"],
            wbc_action_alpha=wbc_action_adapter_config["alpha"],
        )
        root_body_name = _wbc_root_body_name(policy_config)
        reference_tracker = build_wbc_reference_tracker(
            model=model,
            policy_config=policy_config,
            joint_names=policy.isaac_joint_names,
            object_body_names=[
                name for name in resolved["publish_object_names"]
                if name != root_body_name
            ],
            initial_step=wbc_initial_step,
            target_fps=int(round(rl_rate)),
        )
        decimation = max(1, int(round((1.0 / float(rl_rate)) / sim_dt)))
        active_policy_steps = int(round(duration_sec * float(rl_rate)))
        initial_pause_steps = int(round(float(wbc_initial_pause_sec) * float(rl_rate)))
        total_policy_steps = initial_pause_steps + active_policy_steps
        samples: list[dict[str, Any]] = []
        actions_abs_max: list[float] = []
        actions_abs_mean: list[float] = []
        q_target_abs_max: list[float] = []
        pending_q_targets: list[np.ndarray] = []
        scenario = resolved["scenario"]
        video_path = output_dir / f"{scenario.name}_wbc_onnx_headless_{duration_sec:g}s.mp4" if render_video else None
        video_fps = max(1, int(round(float(rl_rate) / video_stride)))
        writer = _make_video_writer(video_path, fps=video_fps) if render_video else contextlib.nullcontext(None)
        renderer = mujoco.Renderer(model, height=height, width=width) if render_video else None
        camera = _make_camera() if render_video else None
        with writer as video_writer:
            for policy_step in range(total_policy_steps):
                policy_reference_step = wbc_reference_step_for_policy_step(
                    policy_step,
                    initial_pause_steps=initial_pause_steps,
                    num_steps=int(reference_tracker["num_steps"]),
                )
                action, q_target = policy.step(model, data, reference_step=policy_reference_step)
                active_q_target = select_control_q_target(
                    pending_q_targets,
                    q_target,
                    policy.state_processor.joint_pos,
                    control_latency_steps=control_latency_steps,
                )
                for substep in range(decimation):
                    if getattr(policy, "observation_semantics", "sim2real") == "wbc":
                        active_q_target = policy.wbc_joint_position_target(substep=substep, decimation=decimation)
                    ctrl = policy.compute_robot_ctrl(model, data, active_q_target)
                    data.ctrl[:] = ctrl
                    _apply_object_joint_control(model, data, scene_config)
                    _apply_elastic_band_force(data, elastic_band)
                    mujoco.mj_step(model, data)
                if policy_step < initial_pause_steps:
                    continue
                active_policy_step = policy_step - initial_pause_steps
                tracking_reference_step = wbc_tracking_reference_step_for_policy_step(
                    policy_step,
                    initial_pause_steps=initial_pause_steps,
                    num_steps=int(reference_tracker["num_steps"]),
                )
                actions_abs_max.append(float(np.max(np.abs(action))))
                actions_abs_mean.append(float(np.mean(np.abs(action))))
                q_target_abs_max.append(float(np.max(np.abs(active_q_target))))
                sample = _sample_scene(model, data, scenario, scene_config)
                sample["policy_step"] = active_policy_step
                sample["total_policy_step"] = policy_step
                sample["time"] = float(data.time)
                sample.update(
                    sample_wbc_reference_tracking(
                        model,
                        data,
                        reference_tracker,
                        active_policy_step,
                        reference_step=tracking_reference_step,
                    )
                )
                sample.update(
                    sample_wbc_q_target_reference_tracking(
                        q_target=active_q_target,
                        reference_tracker=reference_tracker,
                        policy_step=active_policy_step,
                        reference_step=tracking_reference_step,
                    )
                )
                samples.append(sample)
                if renderer is not None and video_writer is not None and active_policy_step % video_stride == 0:
                    _render_frame(renderer, model, data, camera, video_writer)
        if renderer is not None:
            renderer.close()
        summary = summarize_rollout(
            scenario=scenario,
            duration_sec=duration_sec,
            policy_steps=active_policy_steps,
            sim_steps=total_policy_steps * decimation,
            decimation=decimation,
            samples=samples,
            actions_abs_max=actions_abs_max,
            actions_abs_mean=actions_abs_mean,
            q_target_abs_max=q_target_abs_max,
            video_path=video_path,
            fall_height=fall_height,
            control_latency_steps=control_latency_steps,
        )
        summary.update(
            {
                "runner_kind": "wbc_export_onnx",
                "wbc_root": str(resolved["root"]),
                "task_yaml": str(resolved["task_yaml"]),
                "scene_config": scene_config,
                "scene_asset_source": resolved["scene_asset_source"],
                "policy_config": str(resolved["policy_config_path"]),
                "policy_model": str(resolved["policy_model"]),
                "policy_metadata": str(resolved["policy_metadata"]),
                "publish_object_names": list(resolved["publish_object_names"]),
                "wbc_initial_state": wbc_initial_state,
                "wbc_initial_step": int(wbc_initial_step),
                "wbc_initial_pause_sec": float(wbc_initial_pause_sec),
                "wbc_initial_pause_steps": int(initial_pause_steps),
                "total_policy_steps": int(total_policy_steps),
                "applied_reference_initial_state": initial_state_summary,
                "virtual_gantry_configured": virtual_gantry_configured,
                "virtual_gantry_applied": virtual_gantry_applied,
                "virtual_gantry_mode": (
                    "kept_enabled"
                    if virtual_gantry_applied
                    else "disabled_after_start"
                    if virtual_gantry_configured
                    else "not_configured"
                ),
                "video_stride": int(video_stride),
                "video_fps": int(video_fps) if render_video else None,
            }
        )
        summary_path = output_dir / f"{scenario.name}_wbc_onnx_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        summary["summary_path"] = str(summary_path)
        return summary


def select_control_q_target(
    pending_q_targets: list[np.ndarray],
    policy_q_target: np.ndarray,
    hold_q_target: np.ndarray,
    *,
    control_latency_steps: int = 0,
) -> np.ndarray:
    if control_latency_steps < 0:
        raise ValueError("control_latency_steps must be non-negative")
    pending_q_targets.append(np.asarray(policy_q_target, dtype=np.float32).copy())
    if len(pending_q_targets) <= control_latency_steps:
        return np.asarray(hold_q_target, dtype=np.float32).copy()
    return pending_q_targets.pop(0)


def wbc_reference_step_for_policy_step(
    policy_step: int,
    *,
    initial_pause_steps: int,
    num_steps: int,
) -> int:
    if initial_pause_steps < 0:
        raise ValueError("initial_pause_steps must be non-negative")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    active_step = max(0, int(policy_step) - int(initial_pause_steps))
    return min(active_step, int(num_steps) - 1)


def wbc_tracking_reference_step_for_policy_step(
    policy_step: int,
    *,
    initial_pause_steps: int,
    num_steps: int,
) -> int:
    return min(
        wbc_reference_step_for_policy_step(
            policy_step,
            initial_pause_steps=initial_pause_steps,
            num_steps=num_steps,
        )
        + 1,
        int(num_steps) - 1,
    )


def normalize_video_stride(video_stride: int) -> int:
    stride = int(video_stride)
    if stride < 1:
        raise ValueError(f"video_stride must be >= 1, got {video_stride}.")
    return stride


def compute_elastic_band_force(
    *,
    point: Sequence[float],
    position: Sequence[float],
    linear_velocity: Sequence[float],
    stiffness: float = 200.0,
    damping: float = 100.0,
    length: float = 0.0,
) -> np.ndarray:
    delta = np.asarray(point, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-9:
        return np.zeros(3, dtype=np.float64)
    direction = delta / distance
    velocity_along_band = float(np.dot(np.asarray(linear_velocity, dtype=np.float64), direction))
    return (float(stiffness) * (distance - float(length)) - float(damping) * velocity_along_band) * direction


def official_sim2real_scene_config(
    scene_config: Mapping[str, Any],
    *,
    disable_virtual_gantry_after_start: bool = True,
) -> dict[str, Any]:
    """Return the official sim2real scene config for a headless rollout.

    The HDMI sim2sim README says to press 9 in the MuJoCo viewer immediately
    after starting policy control. That key toggles the elastic-band gantry off.
    """
    cfg = dict(scene_config)
    if disable_virtual_gantry_after_start and cfg.get("ENABLE_ELASTIC_BAND", False):
        cfg["ENABLE_ELASTIC_BAND"] = False
    return cfg


def _make_elastic_band(model: Any, scene_config: Mapping[str, Any]) -> dict[str, Any] | None:
    if not scene_config.get("ENABLE_ELASTIC_BAND", False):
        return None
    import mujoco

    for body_name in ("torso_link", "base_link"):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id >= 0:
            return {
                "body_id": int(body_id),
                "point": np.asarray([0.0, 0.0, 3.0], dtype=np.float64),
                "stiffness": 200.0,
                "damping": 100.0,
                "length": 0.0,
            }
    return None


def _apply_elastic_band_force(data: Any, elastic_band: Mapping[str, Any] | None) -> None:
    if elastic_band is None:
        return
    body_id = int(elastic_band["body_id"])
    force = compute_elastic_band_force(
        point=elastic_band["point"],
        position=data.xpos[body_id],
        linear_velocity=data.cvel[body_id, 3:6],
        stiffness=float(elastic_band["stiffness"]),
        damping=float(elastic_band["damping"]),
        length=float(elastic_band["length"]),
    )
    data.xfrc_applied[body_id, :3] = force


def materialize_mjcf_static_body_poses(
    *,
    scene_path: str | Path,
    body_pose_by_name: Mapping[str, Mapping[str, Any]],
    cache_dir: str | Path,
) -> dict[str, Any]:
    scene_path = Path(scene_path).resolve()
    cache_dir = Path(cache_dir).resolve()
    requested = {
        str(name): pose
        for name, pose in body_pose_by_name.items()
        if name and pose is not None
    }
    if not requested:
        return {
            "scene_path": str(scene_path),
            "applied_static_body_names": [],
            "skipped_freejoint_body_names": [],
            "missing_body_names": [],
        }

    cache_dir.mkdir(parents=True, exist_ok=True)
    materialized_by_source: dict[Path, Path] = {}
    seen_body_names: set[str] = set()
    applied_static_body_names: list[str] = []
    skipped_freejoint_body_names: list[str] = []

    def materialize_file(source_path: Path) -> Path:
        source_path = source_path.resolve()
        cached_path = materialized_by_source.get(source_path)
        if cached_path is not None:
            return cached_path

        tree = ET.parse(source_path)
        root = tree.getroot()
        _absolutize_mjcf_compiler_paths(root, source_path)
        for include in root.findall(".//include"):
            include_file = include.get("file")
            if not include_file:
                continue
            child_path = _resolve_mjcf_include_path(source_path, include_file)
            include.set("file", str(materialize_file(child_path)))

        for body in root.findall(".//body"):
            body_name = body.get("name")
            if body_name not in requested:
                continue
            seen_body_names.add(str(body_name))
            if _mjcf_body_has_freejoint(body):
                if body_name not in skipped_freejoint_body_names:
                    skipped_freejoint_body_names.append(str(body_name))
                continue
            pose = requested[str(body_name)]
            body.set("pos", _format_mjcf_float_list(pose["pos"], expected_len=3, field_name=f"{body_name}.pos"))
            body.set(
                "quat",
                _format_mjcf_float_list(
                    _normalize_quat_list(pose["quat"]),
                    expected_len=4,
                    field_name=f"{body_name}.quat",
                ),
            )
            if body_name not in applied_static_body_names:
                applied_static_body_names.append(str(body_name))

        destination = cache_dir / f"{len(materialized_by_source):04d}_{source_path.name}"
        tree.write(destination, encoding="utf-8", xml_declaration=False)
        materialized_by_source[source_path] = destination
        return destination

    materialized_scene_path = materialize_file(scene_path)
    missing_body_names = [name for name in requested if name not in seen_body_names]
    return {
        "scene_path": str(materialized_scene_path),
        "applied_static_body_names": applied_static_body_names,
        "skipped_freejoint_body_names": skipped_freejoint_body_names,
        "missing_body_names": missing_body_names,
    }


def _resolve_mjcf_include_path(source_path: Path, include_file: str) -> Path:
    include_path = Path(include_file)
    if include_path.is_absolute():
        return include_path
    return (source_path.parent / include_path).resolve()


def _absolutize_mjcf_compiler_paths(root: ET.Element, source_path: Path) -> None:
    compiler = root.find("compiler")
    if compiler is None:
        return
    for attr in ("meshdir", "texturedir"):
        raw_value = compiler.get(attr)
        if not raw_value:
            continue
        path = Path(raw_value)
        if not path.is_absolute():
            compiler.set(attr, str((source_path.parent / path).resolve()))


def _mjcf_body_has_freejoint(body: ET.Element) -> bool:
    for child in body:
        if child.tag == "freejoint":
            return True
        if child.tag == "joint" and child.get("type") == "free":
            return True
    return False


def _format_mjcf_float_list(values: Any, *, expected_len: int, field_name: str) -> str:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if int(array.shape[0]) != int(expected_len):
        raise ValueError(f"{field_name} must contain {expected_len} values, got {int(array.shape[0])}.")
    return " ".join(f"{float(value):.10g}" for value in array)


def apply_wbc_reference_initial_state(
    *,
    model: Any,
    data: Any,
    reference_initial_state: Mapping[str, Any],
) -> dict[str, Any]:
    import mujoco

    root_body_name = str(reference_initial_state["root_body_name"])
    root_binding = _find_freejoint_binding(model, root_body_name)
    if root_binding is None:
        raise ValueError(f"Could not find a freejoint for root body {root_body_name!r}")
    _write_freejoint_state(
        data,
        qpos_adr=root_binding["qpos_adr"],
        qvel_adr=root_binding["qvel_adr"],
        pos=reference_initial_state["root_pos"],
        quat=reference_initial_state["root_quat"],
        lin_vel=reference_initial_state["root_lin_vel"],
        ang_vel=reference_initial_state["root_ang_vel"],
    )
    applied_body_names = [root_body_name]
    missing_body_names: list[str] = []
    for body_name, body_pose in reference_initial_state.get("body_pose_by_name", {}).items():
        binding = _find_freejoint_binding(model, str(body_name))
        if binding is None:
            missing_body_names.append(str(body_name))
            continue
        _write_freejoint_state(
            data,
            qpos_adr=binding["qpos_adr"],
            qvel_adr=binding["qvel_adr"],
            pos=body_pose["pos"],
            quat=body_pose["quat"],
            lin_vel=body_pose["lin_vel"],
            ang_vel=body_pose["ang_vel"],
        )
        applied_body_names.append(str(body_name))

    applied_joint_names: list[str] = []
    missing_joint_names: list[str] = []
    skipped_joint_names: list[str] = []
    joint_vel_by_name = reference_initial_state.get("joint_vel_by_name", {})
    for joint_name, joint_pos in reference_initial_state.get("joint_pos_by_name", {}).items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(joint_name))
        if joint_id < 0:
            missing_joint_names.append(str(joint_name))
            continue
        joint_type = int(model.jnt_type[joint_id])
        if joint_type not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            skipped_joint_names.append(str(joint_name))
            continue
        qpos_adr = int(model.jnt_qposadr[joint_id])
        qvel_adr = int(model.jnt_dofadr[joint_id])
        data.qpos[qpos_adr] = float(joint_pos)
        data.qvel[qvel_adr] = float(joint_vel_by_name.get(joint_name, 0.0))
        applied_joint_names.append(str(joint_name))

    return {
        "motion_path": str(reference_initial_state.get("motion_path")),
        "motion_step": int(reference_initial_state["motion_step"]),
        "root_body_name": root_body_name,
        "applied_body_names": applied_body_names,
        "missing_body_names": missing_body_names,
        "applied_joint_count": len(applied_joint_names),
        "missing_joint_names": missing_joint_names,
        "skipped_joint_names": skipped_joint_names,
        "root_freejoint_name": root_binding["joint_name"],
    }


class DirectStateProcessor:
    def __init__(self, model: Any, robot_config: Mapping[str, Any], dest_joint_names: Sequence[str], object_names: Sequence[str]):
        import mujoco
        from utils.math import quat_conjugate, quat_mul, yaw_quat
        from utils.strings import unitree_joint_names

        self._mujoco = mujoco
        self._quat_conjugate = quat_conjugate
        self._quat_mul = quat_mul
        self._yaw_quat = yaw_quat
        self._unitree_joint_names = list(unitree_joint_names)
        self.mocap_ip = robot_config.get("MOCAP_IP", "localhost")
        self.num_dof = len(dest_joint_names)
        self.joint_names = list(dest_joint_names)
        self.joint_indices_in_source = [self._unitree_joint_names.index(name) for name in dest_joint_names]
        self.qpos = np.zeros(3 + 4 + self.num_dof, dtype=np.float32)
        self.qvel = np.zeros(3 + 3 + self.num_dof, dtype=np.float32)
        self.root_pos_w = self.qpos[0:3]
        self.root_lin_vel_w = self.qvel[0:3]
        self.root_quat_b = self.qpos[3:7]
        self.root_ang_vel_b = self.qvel[3:6]
        self.joint_pos = self.qpos[7:]
        self.joint_vel = self.qvel[6:]
        self.mocap_data: dict[str, np.ndarray] = {}
        self._joint_qpos_adrs = []
        self._joint_qvel_adrs = []
        for joint_name in self.joint_names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise ValueError(f"Joint {joint_name!r} is absent from MuJoCo model")
            self._joint_qpos_adrs.append(int(model.jnt_qposadr[joint_id]))
            self._joint_qvel_adrs.append(int(model.jnt_dofadr[joint_id]))
        joint_names_mujoco = [model.joint(i).name for i in range(model.njnt)]
        root_name = "pelvis_root" if "pelvis_root" in joint_names_mujoco else "floating_base_joint"
        root_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, root_name)
        if root_joint_id < 0:
            raise ValueError("No pelvis_root or floating_base_joint found in MuJoCo model")
        self.root_qpos_adr = int(model.jnt_qposadr[root_joint_id])
        self.root_qvel_adr = int(model.jnt_dofadr[root_joint_id])
        self._object_names = tuple(object_names)
        self._object_body_ids: dict[str, int] = {}
        self._object_sensor_adrs: dict[str, tuple[int | None, int | None]] = {}
        for object_name in self._object_names:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_name)
            if body_id >= 0:
                self._object_body_ids[object_name] = int(body_id)
            pos_sensor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"{object_name}_pos")
            quat_sensor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"{object_name}_quat")
            self._object_sensor_adrs[object_name] = (
                int(model.sensor_adr[pos_sensor]) if pos_sensor >= 0 else None,
                int(model.sensor_adr[quat_sensor]) if quat_sensor >= 0 else None,
            )

    def register_subscriber(self, object_name: str, port: int | None = None):
        return None

    def get_mocap_data(self, key: str):
        return self.mocap_data.get(key)

    def update_from_mujoco(self, model: Any, data: Any) -> None:
        root_qpos = self.root_qpos_adr
        root_qvel = self.root_qvel_adr
        root_pos = np.asarray(data.qpos[root_qpos:root_qpos + 3], dtype=np.float32)
        root_quat_w = np.asarray(data.qpos[root_qpos + 3:root_qpos + 7], dtype=np.float32)
        root_quat_yaw_w = self._yaw_quat(root_quat_w)
        root_quat_b = self._quat_mul(self._quat_conjugate(root_quat_yaw_w), root_quat_w)
        self.root_pos_w[:] = root_pos
        self.root_quat_b[:] = root_quat_b.astype(np.float32)
        self.root_lin_vel_w[:] = np.asarray(data.qvel[root_qvel:root_qvel + 3], dtype=np.float32)
        self.root_ang_vel_b[:] = np.asarray(data.qvel[root_qvel + 3:root_qvel + 6], dtype=np.float32)
        self.joint_pos[:] = np.asarray(data.qpos[self._joint_qpos_adrs], dtype=np.float32)
        self.joint_vel[:] = np.asarray(data.qvel[self._joint_qvel_adrs], dtype=np.float32)
        for object_name in self._object_names:
            pos_sensor_adr, quat_sensor_adr = self._object_sensor_adrs.get(object_name, (None, None))
            if pos_sensor_adr is not None and quat_sensor_adr is not None:
                pos = np.asarray(data.sensordata[pos_sensor_adr:pos_sensor_adr + 3], dtype=np.float32)
                quat = np.asarray(data.sensordata[quat_sensor_adr:quat_sensor_adr + 4], dtype=np.float32)
            elif object_name in self._object_body_ids:
                body_id = self._object_body_ids[object_name]
                pos = np.asarray(data.xpos[body_id], dtype=np.float32)
                quat = np.asarray(data.xquat[body_id], dtype=np.float32)
            else:
                continue
            self.mocap_data[f"{object_name}_pos"] = pos.copy()
            self.mocap_data[f"{object_name}_quat"] = quat.copy()


class DirectSim2RealPolicy:
    def __init__(
        self,
        *,
        model: Any,
        robot_config: Mapping[str, Any],
        scene_config: Mapping[str, Any],
        policy_config: Mapping[str, Any],
        model_path: str | Path,
        publish_object_names: Sequence[str],
        quiet_observations: bool = True,
        observation_semantics: str = "sim2real",
        wbc_action_delay: int = 0,
        wbc_action_alpha: float = 1.0,
    ) -> None:
        from observations import ObsGroup, Observation
        from rl_policy.utils.onnx_module import ONNXModule
        from utils.strings import resolve_matching_names_values, unitree_joint_names

        if observation_semantics not in {"sim2real", "wbc"}:
            raise ValueError(f"Unknown observation semantics: {observation_semantics!r}")
        self.robot_config = dict(robot_config)
        self.scene_config = dict(scene_config)
        self.policy_config = dict(policy_config)
        self.quiet_observations = bool(quiet_observations)
        self.observation_semantics = observation_semantics
        self.onnx_module = ONNXModule(str(model_path))
        self.isaac_joint_names = list(policy_config["isaac_joint_names"])
        self.num_dofs = len(self.isaac_joint_names)
        self.policy_joint_names = list(policy_config["policy_joint_names"])
        self.num_actions = len(self.policy_joint_names)
        self.controlled_joint_indices = [self.isaac_joint_names.index(name) for name in self.policy_joint_names]
        self.default_dof_angles = _resolve_named_array(
            policy_config["default_joint_pos"],
            self.isaac_joint_names,
            resolve_matching_names_values,
        )
        self.action_scale = np.ones(self.num_actions, dtype=np.float32)
        action_scale_cfg = policy_config["action_scale"]
        if isinstance(action_scale_cfg, (float, int)):
            self.action_scale[:] = float(action_scale_cfg)
        elif isinstance(action_scale_cfg, Mapping):
            indices, _, values = resolve_matching_names_values(
                action_scale_cfg,
                self.policy_joint_names,
                preserve_order=True,
                strict=True,
            )
            self.action_scale[np.asarray(indices, dtype=np.int64)] = np.asarray(values, dtype=np.float32)
        else:
            raise ValueError(f"Invalid action_scale type: {type(action_scale_cfg)!r}")
        self.unitree_joint_names = list(unitree_joint_names)
        self.joint_kp_unitree = _resolve_unitree_array(policy_config["joint_kp"], resolve_matching_names_values)
        self.joint_kd_unitree = _resolve_unitree_array(policy_config["joint_kd"], resolve_matching_names_values)
        self.default_joint_pos_unitree = _resolve_unitree_array(policy_config["default_joint_pos"], resolve_matching_names_values)
        self.state_processor = DirectStateProcessor(model, robot_config, self.isaac_joint_names, publish_object_names)
        self.env = SimpleNamespace(
            state_processor=self.state_processor,
            num_actions=self.num_actions,
            use_joystick=False,
            key_pressed=set(),
            wc_msg=None,
        )
        self.observations: dict[str, Any] = {}
        self.reset_callbacks = []
        self.update_callbacks = []
        self.wbc_observation_bridge = None
        self.wbc_action_adapter = None
        if self.observation_semantics == "wbc":
            self.wbc_observation_bridge = WBCDirectObservationBridge(
                model=model,
                policy_config=policy_config,
                policy_joint_names=self.policy_joint_names,
                action_dim=self.num_actions,
                object_names=publish_object_names,
            )
            self.wbc_action_adapter = WBCDirectActionAdapter(
                default_joint_pos=self.default_dof_angles[np.asarray(self.controlled_joint_indices, dtype=np.int64)],
                action_scale=self.action_scale,
                delay=wbc_action_delay,
                alpha=wbc_action_alpha,
            )
        else:
            for obs_group, obs_items in policy_config["observation"].items():
                obs_funcs = {}
                for obs_name, obs_config in obs_items.items():
                    obs_class = Observation.registry[obs_name]
                    obs_func = obs_class(env=self.env, **(obs_config or {}))
                    obs_funcs[obs_name] = obs_func
                    self.reset_callbacks.append(obs_func.reset)
                    self.update_callbacks.append(obs_func.update)
                self.observations[obs_group] = ObsGroup(obs_group, obs_funcs)
        self.state_dict: dict[str, Any] = {
            "action": np.zeros(self.num_actions, dtype=np.float32),
            "paused": False,
        }
        for reset_callback in self.reset_callbacks:
            reset_callback()
        self._bind_control_indices(model, resolve_matching_names_values)

    def _bind_control_indices(self, model: Any, resolve_matching_names_values: Any) -> None:
        import mujoco

        joint_names_mujoco = [model.joint(i).name for i in range(model.njnt)]
        actuator_names_mujoco = [model.actuator(i).name for i in range(model.nu)]
        self._control_entries = []
        for joint_name in joint_names_mujoco:
            if joint_name not in self.unitree_joint_names or joint_name not in actuator_names_mujoco:
                continue
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
            if joint_id < 0 or actuator_id < 0:
                continue
            unitree_idx = self.unitree_joint_names.index(joint_name)
            isaac_idx = self.isaac_joint_names.index(joint_name) if joint_name in self.isaac_joint_names else None
            self._control_entries.append(
                (joint_name, unitree_idx, isaac_idx, int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id]), int(actuator_id))
            )
        self._effort_limit_by_actuator: dict[int, float] = {}
        indices, matched_names, limits = resolve_matching_names_values(
            self.robot_config["joint_effort_limit"],
            joint_names_mujoco,
            preserve_order=True,
            strict=False,
        )
        for joint_name, effort_limit in zip(matched_names, limits):
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
            if actuator_id >= 0:
                self._effort_limit_by_actuator[int(actuator_id)] = float(effort_limit)

    def step(self, model: Any, data: Any, *, reference_step: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        if self.observation_semantics == "wbc":
            return self._step_wbc(model, data, reference_step=reference_step)
        self.state_processor.update_from_mujoco(model, data)
        def update_and_compute():
            for update_callback in self.update_callbacks:
                update_callback(self.state_dict)
            obs_dict = {}
            for obs_group in self.observations.values():
                obs = obs_group.compute()
                obs_dict[obs_group.name] = obs[None, :].astype(np.float32)
            return obs_dict
        if self.quiet_observations:
            with contextlib.redirect_stdout(io.StringIO()):
                obs_dict = update_and_compute()
        else:
            obs_dict = update_and_compute()
        self.state_dict.update(obs_dict)
        self.state_dict["is_init"] = np.zeros(1, dtype=bool)
        output_dict = self.onnx_module(self.state_dict)
        action = np.asarray(output_dict["action"].squeeze(0), dtype=np.float32)
        action = np.clip(action, -100.0, 100.0)
        next_state = {key[1]: value for key, value in output_dict.items() if isinstance(key, tuple) and key[0] == "next"}
        q_target = self.default_dof_angles.copy()
        q_target[np.asarray(self.controlled_joint_indices, dtype=np.int64)] += action * self.action_scale
        self.state_dict.update(next_state)
        self.state_dict["action"] = action
        self.state_dict["q_target"] = q_target
        return action, q_target

    def _step_wbc(self, model: Any, data: Any, *, reference_step: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        if self.wbc_observation_bridge is None:
            raise RuntimeError("WBC observation bridge is not initialized.")
        self.state_processor.update_from_mujoco(model, data)
        obs_dict = self.wbc_observation_bridge.build(model, data, reference_step=reference_step)
        self.state_dict.update(obs_dict)
        self.state_dict["is_init"] = np.asarray([not self.wbc_observation_bridge.has_acted], dtype=bool)
        output_dict = self.onnx_module(self.state_dict)
        action = np.asarray(output_dict["action"].squeeze(0), dtype=np.float32)
        action = np.clip(action, -100.0, 100.0)
        next_state = {key[1]: value for key, value in output_dict.items() if isinstance(key, tuple) and key[0] == "next"}
        q_target = self.default_dof_angles.copy()
        q_target[np.asarray(self.controlled_joint_indices, dtype=np.int64)] += action * self.action_scale
        self.state_dict.update(next_state)
        self.state_dict["action"] = action
        self.state_dict["q_target"] = q_target
        self.wbc_observation_bridge.record_action(action)
        if self.wbc_action_adapter is not None:
            self.wbc_action_adapter.record(action)
        return action, q_target

    def wbc_joint_position_target(self, *, substep: int, decimation: int) -> np.ndarray:
        if self.wbc_action_adapter is None:
            return np.asarray(self.state_dict["q_target"], dtype=np.float32)
        policy_target = self.wbc_action_adapter.joint_position_target(substep=substep, decimation=decimation)
        q_target = self.default_dof_angles.copy()
        q_target[np.asarray(self.controlled_joint_indices, dtype=np.int64)] = policy_target
        if self.wbc_observation_bridge is not None:
            self.wbc_observation_bridge.record_applied_action(self.wbc_action_adapter.applied_action)
        return q_target

    def compute_robot_ctrl(self, model: Any, data: Any, q_target: np.ndarray) -> np.ndarray:
        ctrl = np.zeros(model.nu, dtype=np.float64)
        for _, unitree_idx, isaac_idx, qpos_adr, qvel_adr, actuator_id in self._control_entries:
            q_des = self.default_joint_pos_unitree[unitree_idx] if isaac_idx is None else q_target[isaac_idx]
            torque = (
                self.joint_kp_unitree[unitree_idx] * (q_des - data.qpos[qpos_adr])
                + self.joint_kd_unitree[unitree_idx] * (0.0 - data.qvel[qvel_adr])
            )
            limit = self._effort_limit_by_actuator.get(actuator_id)
            if limit is not None:
                torque = float(np.clip(torque, -limit, limit))
            ctrl[actuator_id] = torque
        return ctrl


class WBCDirectActionAdapter:
    def __init__(
        self,
        *,
        default_joint_pos: Sequence[float],
        action_scale: Sequence[float],
        delay: int = 0,
        alpha: float = 1.0,
    ) -> None:
        if delay < 0:
            raise ValueError(f"delay must be non-negative, got {delay}.")
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}.")
        self.default_joint_pos = np.asarray(default_joint_pos, dtype=np.float32)
        self.action_scale = np.asarray(action_scale, dtype=np.float32)
        self.delay = int(delay)
        self.alpha = float(alpha)
        self.history = np.zeros((self.default_joint_pos.shape[0], 3), dtype=np.float32)
        self.applied_action = np.zeros(self.default_joint_pos.shape[0], dtype=np.float32)

    def record(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != self.applied_action.shape:
            raise ValueError(f"action shape {action.shape} != {self.applied_action.shape}")
        self.history = np.roll(self.history, shift=1, axis=1)
        self.history[:, 0] = action

    def joint_position_target(self, *, substep: int, decimation: int) -> np.ndarray:
        if not 0 <= int(substep) < int(decimation):
            raise ValueError(f"substep must be in [0, {decimation}), got {substep}.")
        history_index = (self.delay - int(substep) + int(decimation) - 1) // int(decimation)
        history_index = max(0, min(int(history_index), self.history.shape[1] - 1))
        selected_action = self.history[:, history_index]
        self.applied_action += self.alpha * (selected_action - self.applied_action)
        return self.default_joint_pos + self.applied_action * self.action_scale


class WBCDirectObservationBridge:
    def __init__(
        self,
        *,
        model: Any,
        policy_config: Mapping[str, Any],
        policy_joint_names: Sequence[str],
        action_dim: int,
        object_names: Sequence[str],
    ) -> None:
        from active_adaptation.mujoco.motion_reference import MujocoMotionReference
        from active_adaptation.mujoco.observation_builder import MujocoObservationBuilder

        self.policy_config = dict(policy_config)
        self.action_dim = int(action_dim)
        self.policy_joint_names = list(policy_joint_names)
        self.observation_builder = MujocoObservationBuilder(
            self.policy_config["observation"],
            policy_joint_names=self.policy_joint_names,
            observation_joint_names=self.policy_config.get("isaac_joint_names", self.policy_joint_names),
        )
        self.joint_names = list(self.observation_builder.joint_pos_names)
        self.joint_qpos_adrs = _mujoco_joint_qpos_addresses(model, self.joint_names)
        self.joint_names_for_torque = list(self.policy_config.get("isaac_joint_names", self.joint_names))
        self.joint_actuator_ids_for_torque = _mujoco_actuator_ids_for_joint_names(model, self.joint_names_for_torque)
        self.default_joint_pos = _resolve_named_float_array(
            self.policy_config.get("default_joint_pos", 0.0),
            self.joint_names,
            field_name="default_joint_pos",
        )
        self.policy_default_joint_pos = _resolve_named_float_array(
            self.policy_config.get("default_joint_pos", 0.0),
            self.policy_joint_names,
            field_name="default_joint_pos",
        )
        self.policy_action_scale = _resolve_action_scale_array(
            self.policy_config.get("action_scale", 1.0),
            self.policy_joint_names,
        )
        self.root_body_name = _wbc_root_body_name(self.policy_config)
        self.root_body_id = _mujoco_body_id(model, self.root_body_name)
        if self.root_body_id < 0:
            raise ValueError(f"WBC root body {self.root_body_name!r} is absent from MuJoCo model.")
        root_binding = _find_freejoint_binding(model, self.root_body_name)
        if root_binding is None:
            raise ValueError(f"WBC root body {self.root_body_name!r} does not have a free joint.")
        self.root_qpos_adr = int(root_binding["qpos_adr"])
        self.root_qvel_adr = int(root_binding["qvel_adr"])
        self.reference = None
        self.reference_body_ids: list[int] = []
        self.policy_joint_motion_indices: list[int] = []
        reference_cfg = _wbc_reference_observation_config(self.policy_config)
        if reference_cfg is not None:
            self.reference = MujocoMotionReference.from_motion_dir(
                _policy_motion_path(self.policy_config),
                body_names=[str(name) for name in reference_cfg.get("body_names", [])],
                joint_names=[str(name) for name in reference_cfg.get("joint_names", [])],
                root_body_name=str(reference_cfg.get("root_body_name", self.root_body_name)),
                future_steps=[int(step) for step in reference_cfg.get("future_steps", [1])],
            )
            self.reference_body_ids = _mujoco_body_ids_for_names(model, self.reference.requested_body_names)
            self.policy_joint_motion_indices = [
                self.reference.joint_names.index(joint_name)
                for joint_name in self.policy_joint_names
            ]
        self.model_body_names = [str(model.body(index).name) for index in range(model.nbody)]
        self.model_body_ids = list(range(model.nbody))
        self.object_names = tuple(str(name) for name in object_names)
        self.object_body_ids = {
            name: _mujoco_body_id(model, name)
            for name in self.object_names
            if _mujoco_body_id(model, name) >= 0
        }
        self.primary_object_name = _wbc_primary_object_name(self.policy_config)
        if self.primary_object_name is not None and self.primary_object_name not in self.object_body_ids:
            object_body_id = _mujoco_body_id(model, self.primary_object_name)
            if object_body_id >= 0:
                self.object_body_ids[self.primary_object_name] = object_body_id
        self.contact_cfg = _wbc_observation_config(self.policy_config, "ref_contact_pos_b")
        self.contact_eef_body_names = _as_string_list((self.contact_cfg or {}).get("contact_eef_body_name", []))
        self.contact_eef_body_ids = _mujoco_body_ids_for_names(model, self.contact_eef_body_names) if self.contact_eef_body_names else []
        self.contact_eef_offsets = np.asarray((self.contact_cfg or {}).get("contact_eef_pos_offset", []), dtype=np.float32)
        if self.contact_eef_offsets.ndim == 1 and self.contact_eef_offsets.size:
            self.contact_eef_offsets = self.contact_eef_offsets.reshape(1, 3)
        self.primary_object_joint_name = _infer_primary_object_joint_name(
            model,
            self.primary_object_name,
            excluded_joint_names=self.policy_config.get("isaac_joint_names", ()),
        )
        self.primary_object_joint_qpos_adr = None
        self.primary_object_joint_qvel_adr = None
        self.primary_object_joint_actuator_id = None
        if self.primary_object_joint_name is not None:
            self.primary_object_joint_qpos_adr, self.primary_object_joint_qvel_adr = _mujoco_joint_qpos_qvel_address(
                model,
                self.primary_object_joint_name,
            )
            self.primary_object_joint_actuator_id = _mujoco_actuator_id(model, self.primary_object_joint_name)
        self.action_history = np.zeros((1, self.action_dim, _wbc_prev_action_steps(self.policy_config)), dtype=np.float32)
        self.applied_action = np.zeros((1, self.action_dim), dtype=np.float32)
        self.policy_step = 0
        self.initialized = False
        self.has_acted = False

    def build(self, model: Any, data: Any, *, reference_step: int | None = None) -> dict[str, np.ndarray]:
        state = self._state(model, data, reference_step=reference_step)
        if not self.initialized:
            self.observation_builder.reset(state)
            self.initialized = True
        else:
            self.observation_builder.update(state)
        obs_dict = {
            group_name: self.observation_builder.build_group(group_name, state).detach().cpu().numpy().astype(np.float32)
            for group_name in self.observation_builder.observation_cfg
        }
        self.policy_step += 1
        return obs_dict

    def record_action(self, action: np.ndarray) -> None:
        self.action_history = np.roll(self.action_history, shift=1, axis=2)
        self.action_history[:, :, 0] = np.asarray(action, dtype=np.float32).reshape(1, self.action_dim)
        self.has_acted = True

    def record_applied_action(self, applied_action: np.ndarray) -> None:
        applied = np.asarray(applied_action, dtype=np.float32).reshape(1, self.action_dim)
        if applied.shape != self.applied_action.shape:
            raise ValueError(f"applied_action shape {applied.shape} != {self.applied_action.shape}")
        self.applied_action[:] = applied

    def _state(self, model: Any, data: Any, *, reference_step: int | None = None):
        import torch
        from active_adaptation.mujoco.observation_builder import MujocoPolicyState

        root_quat = torch.as_tensor(
            data.qpos[self.root_qpos_adr + 3:self.root_qpos_adr + 7],
            dtype=torch.float32,
        ).unsqueeze(0)
        root_pos = torch.as_tensor(
            data.qpos[self.root_qpos_adr:self.root_qpos_adr + 3],
            dtype=torch.float32,
        ).unsqueeze(0)
        root_ang_vel_w = torch.as_tensor(data.cvel[self.root_body_id, :3], dtype=torch.float32).unsqueeze(0)
        root_ang_vel_b = _torch_quat_rotate_inverse(root_quat, root_ang_vel_w)
        root_lin_vel_w = torch.as_tensor(data.qvel[self.root_qvel_adr:self.root_qvel_adr + 3], dtype=torch.float32).unsqueeze(0)
        root_lin_vel_b = _torch_quat_rotate_inverse(root_quat, root_lin_vel_w)
        projected_gravity_b = _torch_quat_rotate_inverse(
            root_quat,
            torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
        )
        joint_pos = torch.as_tensor(data.qpos[self.joint_qpos_adrs], dtype=torch.float32).unsqueeze(0)
        joint_offset = torch.as_tensor(self.default_joint_pos, dtype=torch.float32).unsqueeze(0)
        body_ids = np.asarray(self.model_body_ids, dtype=np.int64)
        body_pos_w = torch.as_tensor(data.xpos[body_ids], dtype=torch.float32).unsqueeze(0)
        body_quat_w = torch.as_tensor(data.xquat[body_ids], dtype=torch.float32).unsqueeze(0)
        body_lin_vel_w = torch.as_tensor(data.cvel[body_ids, 3:6], dtype=torch.float32).unsqueeze(0)
        body_ang_vel_w = torch.as_tensor(data.cvel[body_ids, :3], dtype=torch.float32).unsqueeze(0)
        if self.reference_body_ids:
            reference_body_ids = np.asarray(self.reference_body_ids, dtype=np.int64)
            tracking_body_pos_w = torch.as_tensor(data.xpos[reference_body_ids], dtype=torch.float32).unsqueeze(0)
            tracking_body_quat_w = torch.as_tensor(data.xquat[reference_body_ids], dtype=torch.float32).unsqueeze(0)
            tracking_body_lin_vel_w = torch.as_tensor(data.cvel[reference_body_ids, 3:6], dtype=torch.float32).unsqueeze(0)
            tracking_body_ang_vel_w = torch.as_tensor(data.cvel[reference_body_ids, :3], dtype=torch.float32).unsqueeze(0)
        else:
            tracking_body_pos_w = tracking_body_quat_w = tracking_body_lin_vel_w = tracking_body_ang_vel_w = None
        reference_fields = self._reference_fields(reference_step=reference_step)
        object_state = self._object_state(model, data)
        return MujocoPolicyState(
            root_ang_vel_b=root_ang_vel_b,
            root_lin_vel_b=root_lin_vel_b,
            projected_gravity_b=projected_gravity_b,
            joint_pos=joint_pos,
            joint_names=self.joint_names_for_torque,
            joint_pos_offset=joint_offset,
            applied_action=torch.as_tensor(self.applied_action, dtype=torch.float32),
            applied_torque=_torch_applied_torque_from_actuators(data, self.joint_actuator_ids_for_torque),
            action_history=torch.as_tensor(self.action_history, dtype=torch.float32),
            body_names=self.model_body_names,
            body_pos_w=body_pos_w,
            body_quat_w=body_quat_w,
            body_lin_vel_w=body_lin_vel_w,
            body_ang_vel_w=body_ang_vel_w,
            tracking_body_pos_w=tracking_body_pos_w,
            tracking_body_quat_w=tracking_body_quat_w,
            tracking_body_lin_vel_w=tracking_body_lin_vel_w,
            tracking_body_ang_vel_w=tracking_body_ang_vel_w,
            robot_root_pos_w=root_pos,
            robot_root_quat_w=root_quat,
            **reference_fields,
            **object_state,
        )

    def _reference_fields(self, *, reference_step: int | None = None) -> dict[str, Any]:
        if self.reference is None:
            return {}
        import torch

        step = self.policy_step if reference_step is None else int(reference_step)
        fields = self.reference.observation_fields_at(step)
        reference_fields: dict[str, Any] = {
            "ref_body_pos_future_w": fields.ref_body_pos_future_w,
            "ref_body_quat_future_w": fields.ref_body_quat_future_w,
            "ref_body_lin_vel_future_w": fields.ref_body_lin_vel_future_w,
            "ref_body_ang_vel_future_w": fields.ref_body_ang_vel_future_w,
            "ref_root_pos_w": fields.ref_root_pos_w,
            "ref_root_quat_w": fields.ref_root_quat_w,
            "ref_root_pos_future_w": fields.ref_root_pos_future_w,
            "ref_root_quat_future_w": fields.ref_root_quat_future_w,
            "ref_joint_pos_future": fields.ref_joint_pos_future,
            "motion_t": fields.motion_t,
            "motion_len": fields.motion_len,
        }
        current_step = min(int(step), self.reference.num_steps - 1)
        if self.policy_joint_motion_indices:
            ref_joint_pos = self.reference.joint_pos[current_step, self.policy_joint_motion_indices].unsqueeze(0)
            default = torch.as_tensor(self.policy_default_joint_pos, dtype=torch.float32).unsqueeze(0)
            scale = torch.as_tensor(self.policy_action_scale, dtype=torch.float32).unsqueeze(0).clamp_min(1e-6)
            reference_fields["ref_joint_pos_action"] = (ref_joint_pos - default) / scale
        future_indices = (
            torch.as_tensor([step], dtype=torch.long)[:, None]
            + self.reference.future_steps[None]
        ).clamp_max(self.reference.num_steps - 1)
        if self.primary_object_name in self.reference.body_names:
            object_index = self.reference.body_names.index(self.primary_object_name)
            reference_fields["ref_object_pos_future_w"] = self.reference.body_pos_w[future_indices, object_index]
            reference_fields["ref_object_quat_future_w"] = self.reference.body_quat_w[future_indices, object_index]
        if self.reference.object_contact is not None:
            reference_fields["ref_object_contact_future"] = self.reference.object_contact[future_indices]
        return reference_fields

    def _object_state(self, model: Any, data: Any) -> dict[str, Any]:
        if self.primary_object_name is None:
            return {}
        import torch

        object_body_id = self.object_body_ids.get(self.primary_object_name)
        if object_body_id is None:
            return {}
        object_pos_w = torch.as_tensor(data.xpos[object_body_id], dtype=torch.float32).unsqueeze(0)
        object_quat_w = torch.as_tensor(data.xquat[object_body_id], dtype=torch.float32).unsqueeze(0)
        object_state: dict[str, Any] = {
            "object_pos_w": object_pos_w,
            "object_quat_w": object_quat_w,
        }
        if self.primary_object_joint_qpos_adr is not None and self.primary_object_joint_qvel_adr is not None:
            object_state["object_joint_pos"] = torch.as_tensor([[data.qpos[self.primary_object_joint_qpos_adr]]], dtype=torch.float32)
            object_state["object_joint_vel"] = torch.as_tensor([[data.qvel[self.primary_object_joint_qvel_adr]]], dtype=torch.float32)
            torque = 0.0
            if self.primary_object_joint_actuator_id is not None:
                torque = float(data.actuator_force[self.primary_object_joint_actuator_id])
            object_state["object_joint_torque"] = torch.as_tensor([[torque]], dtype=torch.float32)
        if self.contact_cfg is not None:
            offsets = torch.as_tensor(
                self.contact_cfg.get("contact_target_pos_offset", [[0.0, 0.0, 0.0]]),
                dtype=torch.float32,
            )
            if offsets.ndim == 1:
                offsets = offsets.unsqueeze(0)
            object_state["contact_target_pos_w"] = object_pos_w[:, None, :] + _torch_quat_rotate(
                object_quat_w[:, None, :],
                offsets.unsqueeze(0),
            )
            if self.contact_eef_body_ids:
                body_ids = np.asarray(self.contact_eef_body_ids, dtype=np.int64)
                eef_pos = torch.as_tensor(data.xpos[body_ids], dtype=torch.float32).unsqueeze(0)
                if self.contact_eef_offsets.size:
                    offsets_t = torch.as_tensor(self.contact_eef_offsets, dtype=torch.float32).unsqueeze(0)
                    eef_quat = torch.as_tensor(data.xquat[body_ids], dtype=torch.float32).unsqueeze(0)
                    eef_pos = eef_pos + _torch_quat_rotate(eef_quat, offsets_t)
                object_state["contact_eef_pos_w"] = eef_pos
        return object_state


def _annotate_wbc_contact_eef_metadata(policy_config: dict[str, Any], command_cfg: Mapping[str, Any]) -> None:
    body_names = command_cfg.get("contact_eef_body_name")
    if body_names is None and command_cfg.get("contact_target_pos_offset") is not None:
        target_offsets = command_cfg.get("contact_target_pos_offset") or []
        if len(target_offsets) == len(DEFAULT_WBC_CONTACT_EEF_BODY_NAMES):
            body_names = list(DEFAULT_WBC_CONTACT_EEF_BODY_NAMES)
    if body_names is None:
        return
    body_name_list = _as_string_list(body_names)
    pos_offsets = command_cfg.get("contact_eef_pos_offset")
    if pos_offsets is None:
        pos_offsets = [[0.0, 0.0, 0.0] for _ in body_name_list]
    for obs_name in ("ref_contact_pos_b", "diff_contact_pos_b"):
        obs_cfg = _wbc_observation_config(policy_config, obs_name)
        if obs_cfg is None:
            continue
        obs_cfg.setdefault("contact_eef_body_name", body_name_list)
        obs_cfg.setdefault("contact_eef_pos_offset", pos_offsets)


def _resolve_action_scale_array(spec: Any, names: Sequence[str]) -> np.ndarray:
    if isinstance(spec, (int, float)):
        return np.full((len(names),), float(spec), dtype=np.float32)
    values = _resolve_named_value_map(spec, names, field_name="action_scale", require_full=True)
    return np.asarray([float(values[str(name)]) for name in names], dtype=np.float32)


def _mujoco_body_ids_for_names(model: Any, body_names: Sequence[str]) -> list[int]:
    ids: list[int] = []
    missing: list[str] = []
    for body_name in body_names:
        body_id = _mujoco_body_id(model, str(body_name))
        if body_id < 0:
            missing.append(str(body_name))
        else:
            ids.append(body_id)
    if missing:
        raise ValueError(f"MuJoCo model is missing body names: {missing}")
    return ids


def _mujoco_joint_qpos_qvel_address(model: Any, joint_name: str) -> tuple[int, int]:
    import mujoco

    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(joint_name))
    if joint_id < 0:
        raise ValueError(f"MuJoCo model is missing joint {joint_name!r}.")
    return int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])


def _mujoco_actuator_id(model: Any, actuator_name: str) -> int | None:
    import mujoco

    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, str(actuator_name))
    return int(actuator_id) if actuator_id >= 0 else None


def _mujoco_actuator_ids_for_joint_names(model: Any, joint_names: Sequence[str]) -> list[int | None]:
    return [_mujoco_actuator_id(model, str(joint_name)) for joint_name in joint_names]


def _infer_primary_object_joint_name(
    model: Any,
    primary_object_name: str | None,
    *,
    excluded_joint_names: Sequence[str],
) -> str | None:
    import mujoco

    excluded = {str(name) for name in excluded_joint_names}
    joint_names = [str(model.joint(index).name) for index in range(model.njnt)]
    candidates: list[str] = []
    if primary_object_name:
        candidates.extend([f"{primary_object_name}_joint", str(primary_object_name)])
        candidates.extend([name for name in joint_names if str(primary_object_name) in name])
    for name in joint_names:
        if name in excluded or not name:
            continue
        if name in candidates or primary_object_name is None:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id >= 0 and int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
                return name
    return None


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _torch_applied_torque_from_actuators(data: Any, actuator_ids: Sequence[int | None]):
    import torch

    values = [0.0 if actuator_id is None else float(data.actuator_force[int(actuator_id)]) for actuator_id in actuator_ids]
    return torch.as_tensor([values], dtype=torch.float32)





def aggregate_summaries(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    items = [dict(summary) for summary in summaries]
    return {
        "scenario_count": len(items),
        "heuristic_success_count": sum(1 for item in items if item.get("heuristic_success") is True),
        "not_fallen_count": sum(1 for item in items if item.get("not_fallen") is True),
        "finite_count": sum(1 for item in items if item.get("finite") is True),
        "items": items,
    }


def _scenario_artifact_path(
    *,
    scenario: ScenarioSpec,
    output_dir: str | Path,
    enabled: bool,
    explicit_path: str | Path | None,
    artifact_dir: str | Path | None,
    suffix: str,
    scenario_count: int,
    option_name: str,
) -> Path | None:
    if explicit_path is not None:
        if scenario_count != 1:
            raise ValueError(f"{option_name} can only be used with exactly one scenario; use a directory option instead.")
        return Path(explicit_path)
    if not enabled:
        return None
    base_dir = Path(artifact_dir) if artifact_dir is not None else Path(output_dir)
    return base_dir / f"{scenario.name}{suffix}"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.wbc_task_yaml or args.wbc_policy_model:
        if not args.wbc_task_yaml or not args.wbc_policy_model:
            raise ValueError("--wbc-task-yaml and --wbc-policy-model must be provided together.")
        try:
            summary = run_wbc_export_scenario(
                sim2real_root=args.sim2real_root,
                wbc_root=args.wbc_root,
                task_yaml=args.wbc_task_yaml,
                policy_model=args.wbc_policy_model,
                output_dir=args.output_dir,
                duration_sec=args.duration_sec,
                rl_rate=args.rl_rate,
                width=args.width,
                height=args.height,
                render_video=not args.no_video,
                quiet_observations=not args.verbose_observations,
                fall_height=args.fall_height,
                control_latency_steps=args.control_latency_steps,
                wbc_initial_state=args.wbc_initial_state,
                wbc_initial_step=args.wbc_initial_step,
                wbc_initial_pause_sec=args.wbc_initial_pause_sec,
                disable_virtual_gantry_after_start=not args.keep_virtual_gantry,
                video_stride=args.video_stride,
            )
        except Exception as exc:
            summary = {
                "scenario": str(args.wbc_task_yaml),
                "runner_kind": "wbc_export_onnx",
                "error": f"{type(exc).__name__}: {exc}",
                "heuristic_success": False,
                "finite": False,
                "not_fallen": False,
            }
            aggregate = aggregate_summaries([summary])
            _write_aggregate(Path(args.output_dir), aggregate, args.output)
            print(json.dumps(aggregate, sort_keys=True))
            return 1
        aggregate = aggregate_summaries([summary])
        _write_aggregate(Path(args.output_dir), aggregate, args.output)
        print(json.dumps(aggregate, sort_keys=True))
        return 0

    scenarios = select_scenarios(args.scenario)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for scenario in scenarios:
        try:
            trace_json_path = _scenario_artifact_path(
                scenario=scenario,
                output_dir=output_dir,
                enabled=args.trace,
                explicit_path=args.trace_json,
                artifact_dir=args.trace_dir,
                suffix="_trace.json",
                scenario_count=len(scenarios),
                option_name="--trace-json",
            )
            plot_path = _scenario_artifact_path(
                scenario=scenario,
                output_dir=output_dir,
                enabled=args.plot,
                explicit_path=args.plot_path,
                artifact_dir=args.plot_dir,
                suffix="_curves.png",
                scenario_count=len(scenarios),
                option_name="--plot-path",
            )
            summary = run_scenario(
                root=args.sim2real_root,
                scenario=scenario,
                output_dir=output_dir,
                duration_sec=args.duration_sec,
                rl_rate=args.rl_rate,
                width=args.width,
                height=args.height,
                render_video=(not args.no_video and not args.viewer),
                quiet_observations=not args.verbose_observations,
                fall_height=args.fall_height,
                control_latency_steps=args.control_latency_steps,
                disable_virtual_gantry_after_start=not args.keep_virtual_gantry,
                video_stride=args.video_stride,
                viewer=args.viewer,
                viewer_speed=args.viewer_speed,
                trace_json_path=trace_json_path,
                plot_path=plot_path,
            )
        except Exception as exc:
            summary = {
                "scenario": scenario.name,
                "error": f"{type(exc).__name__}: {exc}",
                "heuristic_success": False,
                "finite": False,
                "not_fallen": False,
            }
            if not args.keep_going:
                summaries.append(summary)
                aggregate = aggregate_summaries(summaries)
                _write_aggregate(output_dir, aggregate, args.output)
                print(json.dumps(aggregate, sort_keys=True))
                return 1
        summaries.append(summary)
    aggregate = aggregate_summaries(summaries)
    _write_aggregate(output_dir, aggregate, args.output)
    print(json.dumps(aggregate, sort_keys=True))
    return 0 if all("error" not in summary for summary in summaries) else 1


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official sim2real HDMI ONNX policies headlessly from WBC.")
    parser.add_argument("--sim2real-root", default=str(DEFAULT_SIM2REAL_ROOT))
    parser.add_argument("--wbc-root", default=str(DEFAULT_WBC_ROOT))
    parser.add_argument("--wbc-task-yaml", default=None)
    parser.add_argument("--wbc-policy-model", default=None)
    parser.add_argument("--wbc-initial-state", choices=("scene_default", "reference_frame"), default="scene_default")
    parser.add_argument("--wbc-initial-step", type=int, default=0)
    parser.add_argument("--wbc-initial-pause-sec", type=float, default=0.0)
    parser.add_argument("--scenario", action="append", default=None, help="Scenario alias, or all. Repeatable.")
    parser.add_argument("--duration-sec", type=float, default=6.0)
    parser.add_argument("--rl-rate", type=float, default=50.0)
    parser.add_argument("--fall-height", type=float, default=0.4)
    parser.add_argument("--control-latency-steps", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--video-stride", type=int, default=1)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--viewer", action="store_true", help="Open a live MuJoCo policy viewer instead of EGL video rendering.")
    parser.add_argument("--viewer-speed", type=float, default=1.0, help="Realtime viewer speed multiplier.")
    parser.add_argument("--trace", action="store_true", help="Write per-step rollout trace JSON files under --trace-dir or --output-dir.")
    parser.add_argument("--trace-dir", default=None, help="Directory for per-scenario trace JSON files.")
    parser.add_argument("--trace-json", default=None, help="Exact trace JSON path. Requires exactly one selected scenario.")
    parser.add_argument("--plot", action="store_true", help="Write per-step rollout curve PNG files under --plot-dir or --output-dir.")
    parser.add_argument("--plot-dir", default=None, help="Directory for per-scenario curve PNG files.")
    parser.add_argument("--plot-path", default=None, help="Exact curve PNG path. Requires exactly one selected scenario.")
    parser.add_argument("--verbose-observations", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument(
        "--keep-virtual-gantry",
        action="store_true",
        help="Keep the sim2real elastic-band gantry enabled instead of simulating the README's immediate 9-key disable.",
    )
    return parser.parse_args(argv)


def _write_aggregate(output_dir: Path, aggregate: Mapping[str, Any], output: str | None) -> None:
    path = Path(output) if output else output_dir / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")


def _sim2real_context(root: Path):
    @contextlib.contextmanager
    def _ctx():
        old_cwd = Path.cwd()
        root_str = str(root)
        rl_policy_str = str(root / "rl_policy")
        for path in (root_str, rl_policy_str):
            if path not in sys.path:
                sys.path.insert(0, path)
        os.chdir(root)
        try:
            yield
        finally:
            os.chdir(old_cwd)
    return _ctx()


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return data


def _load_wbc_task_config(task_path: Path, *, wbc_root: Path) -> dict[str, Any]:
    raw_task = _load_yaml(task_path)
    task_root = wbc_root / "cfg" / "task"
    merged: dict[str, Any] = {}
    for default_path in _wbc_task_default_paths(raw_task.get("defaults", ()), task_root=task_root):
        _deep_update(merged, _load_wbc_task_config(default_path, wbc_root=wbc_root))
    _deep_update(merged, raw_task)
    return merged


def _wbc_task_default_paths(defaults: Any, *, task_root: Path) -> list[Path]:
    if defaults is None:
        return []
    if not isinstance(defaults, Sequence) or isinstance(defaults, (str, bytes)):
        raise ValueError(f"WBC task defaults must be a list, got {type(defaults).__name__}.")
    paths: list[Path] = []
    for entry in defaults:
        if isinstance(entry, str):
            if entry == "_self_":
                continue
            default_name = entry
        elif isinstance(entry, Mapping):
            if "_self_" in entry:
                continue
            if len(entry) != 1:
                raise ValueError(f"Unsupported WBC task default entry: {entry!r}.")
            key, value = next(iter(entry.items()))
            if value in (None, "_self_"):
                continue
            default_name = f"{key}/{value}"
        else:
            raise ValueError(f"Unsupported WBC task default entry: {entry!r}.")
        default_path = task_root / f"{default_name}.yaml"
        if not default_path.exists():
            raise FileNotFoundError(f"Missing WBC task default config: {default_path}")
        paths.append(default_path)
    return paths


def _deep_update(target: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in overlay.items():
        current = target.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            _deep_update(current, value)
        else:
            target[key] = dict(value) if isinstance(value, Mapping) else value
    return target


def _apply_wbc_task_sim_dt(scene_config: dict[str, Any], sim_cfg: Any, *, path: Path) -> None:
    if sim_cfg is None:
        return
    if not isinstance(sim_cfg, Mapping):
        raise ValueError(f"{path}: sim must be a mapping, got {type(sim_cfg).__name__}.")
    physics_dt = sim_cfg.get("mujoco_physics_dt")
    if physics_dt is None:
        return
    physics_dt_f = float(physics_dt)
    if not math.isfinite(physics_dt_f) or physics_dt_f <= 0.0:
        raise ValueError(f"{path}: sim.mujoco_physics_dt must be finite and positive, got {physics_dt!r}.")
    scene_config["SIMULATE_DT"] = physics_dt_f


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping, got {type(value).__name__}.")
    return value


def _resolve_under_root(root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def _rewrite_policy_motion_paths(policy_config: Mapping[str, Any], *, wbc_root: Path) -> None:
    observation = policy_config.get("observation", {})
    if not isinstance(observation, Mapping):
        return
    for group_cfg in observation.values():
        if not isinstance(group_cfg, Mapping):
            continue
        for obs_cfg in group_cfg.values():
            if not isinstance(obs_cfg, Mapping) or "motion_path" not in obs_cfg:
                continue
            motion_path = Path(str(obs_cfg["motion_path"]))
            obs_cfg["motion_path"] = str(motion_path if motion_path.is_absolute() else wbc_root / motion_path)


def _apply_wbc_robot_overrides_to_policy_config(policy_config: dict[str, Any], robot_cfg: Mapping[str, Any]) -> None:
    override_params = robot_cfg.get("override_params")
    if not isinstance(override_params, Mapping):
        return
    joint_names = policy_config.get("isaac_joint_names")
    if not isinstance(joint_names, Sequence) or isinstance(joint_names, (str, bytes)):
        raise ValueError("Policy config must contain isaac_joint_names before applying robot overrides.")
    joint_names = [str(name) for name in joint_names]

    init_state = override_params.get("init_state")
    if isinstance(init_state, Mapping):
        joint_pos_override = init_state.get("joint_pos")
        if isinstance(joint_pos_override, Mapping):
            policy_config["default_joint_pos"] = _merge_named_value_overrides(
                policy_config.get("default_joint_pos", 0.0),
                joint_pos_override,
                joint_names,
                field_name="default_joint_pos",
            )

    stiffness_overrides: dict[str, Any] = {}
    damping_overrides: dict[str, Any] = {}
    actuators = override_params.get("actuators")
    if isinstance(actuators, Mapping):
        for actuator_name, actuator_cfg in actuators.items():
            if not isinstance(actuator_cfg, Mapping):
                raise TypeError(f"robot.override_params.actuators.{actuator_name} must be a mapping.")
            _collect_actuator_override_values(stiffness_overrides, actuator_cfg.get("stiffness"), field_name="stiffness")
            _collect_actuator_override_values(damping_overrides, actuator_cfg.get("damping"), field_name="damping")
    if stiffness_overrides:
        policy_config["joint_kp"] = _merge_named_value_overrides(
            policy_config.get("joint_kp", 0.0),
            stiffness_overrides,
            joint_names,
            field_name="joint_kp",
        )
    if damping_overrides:
        policy_config["joint_kd"] = _merge_named_value_overrides(
            policy_config.get("joint_kd", 0.0),
            damping_overrides,
            joint_names,
            field_name="joint_kd",
        )


def _collect_actuator_override_values(target: dict[str, Any], value: Any, *, field_name: str) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise TypeError(f"Only mapping {field_name} overrides are supported for WBC sim2real export playback.")
    target.update({str(key): item for key, item in value.items()})


def _merge_named_value_overrides(
    base: Any,
    overrides: Mapping[str, Any],
    joint_names: Sequence[str],
    *,
    field_name: str,
) -> dict[str, Any]:
    values = _resolve_named_value_map(base, joint_names, field_name=field_name, require_full=True)
    override_values = _resolve_named_value_map(
        dict(overrides),
        joint_names,
        field_name=f"{field_name}_override",
        require_full=False,
        require_keys_matched=True,
    )
    values.update(override_values)
    return {str(name): values[str(name)] for name in joint_names}


def _resolve_named_value_map(
    mapping_or_scalar: Any,
    joint_names: Sequence[str],
    *,
    field_name: str,
    require_full: bool,
    require_keys_matched: bool = False,
) -> dict[str, Any]:
    joint_names = [str(name) for name in joint_names]
    if isinstance(mapping_or_scalar, (int, float)):
        return {name: float(mapping_or_scalar) for name in joint_names}
    if not isinstance(mapping_or_scalar, Mapping):
        raise TypeError(f"{field_name} must be a scalar or mapping, got {type(mapping_or_scalar).__name__}.")

    import re

    resolved: dict[str, Any] = {}
    key_matches: dict[str, list[str]] = {str(key): [] for key in mapping_or_scalar.keys()}
    for joint_name in joint_names:
        matches = [
            (str(regex), value)
            for regex, value in mapping_or_scalar.items()
            if re.fullmatch(str(regex), joint_name)
        ]
        if len(matches) > 1:
            raise ValueError(f"Multiple {field_name} matches for {joint_name!r}: {[regex for regex, _ in matches]}")
        if matches:
            regex, value = matches[0]
            resolved[joint_name] = value
            key_matches[regex].append(joint_name)
        elif require_full:
            raise ValueError(f"{field_name} does not provide a value for joint {joint_name!r}.")

    if require_keys_matched:
        unmatched = [key for key, matches in key_matches.items() if not matches]
        if unmatched:
            raise ValueError(f"{field_name} override keys did not match any policy joints: {unmatched}")
    return resolved


def _policy_motion_path(policy_config: Mapping[str, Any]) -> Path:
    motion_paths: list[Path] = []
    observation = policy_config.get("observation", {})
    if isinstance(observation, Mapping):
        for group_cfg in observation.values():
            if not isinstance(group_cfg, Mapping):
                continue
            for obs_cfg in group_cfg.values():
                if isinstance(obs_cfg, Mapping) and "motion_path" in obs_cfg:
                    motion_paths.append(Path(str(obs_cfg["motion_path"])))
    unique_paths = []
    for motion_path in motion_paths:
        if motion_path not in unique_paths:
            unique_paths.append(motion_path)
    if not unique_paths:
        raise ValueError("Policy config does not contain any observation motion_path.")
    if len(unique_paths) > 1:
        raise ValueError(f"Policy config contains multiple motion_path values: {[str(path) for path in unique_paths]}")
    return unique_paths[0]


def _wbc_root_body_name(policy_config: Mapping[str, Any]) -> str:
    observation = policy_config.get("observation", {})
    if isinstance(observation, Mapping):
        for group_cfg in observation.values():
            if not isinstance(group_cfg, Mapping):
                continue
            for obs_cfg in group_cfg.values():
                if isinstance(obs_cfg, Mapping) and obs_cfg.get("root_body_name"):
                    return str(obs_cfg["root_body_name"])
    return "pelvis"


def _select_single_motion_npz(motion_path: Path) -> Path:
    if motion_path.is_file():
        if motion_path.name != "motion.npz":
            raise ValueError(f"Expected a motion.npz file, got {motion_path}")
        return motion_path
    if not motion_path.exists():
        raise FileNotFoundError(f"Motion path does not exist: {motion_path}")
    motion_paths = sorted(motion_path.rglob("motion.npz"))
    if not motion_paths:
        raise FileNotFoundError(f"No motion.npz found under {motion_path}")
    if len(motion_paths) > 1:
        raise ValueError(f"Expected one WBC motion under {motion_path}, found {len(motion_paths)}")
    return motion_paths[0]


def _wbc_action_adapter_config_from_task_yaml(task_yaml: str | Path) -> dict[str, float | int]:
    try:
        from scripts import mujoco_playback_parity as playback

        task_cfg = playback.load_task_config(task_yaml)
        adapter_cfg = playback._policy_action_adapter_config_from_task_cfg(task_cfg)
        return {"delay": int(adapter_cfg.delay), "alpha": float(adapter_cfg.alpha)}
    except Exception:
        return {"delay": 0, "alpha": 1.0}


def _mujoco_joint_qpos_addresses(model: Any, joint_names: Sequence[str]) -> np.ndarray:
    import mujoco

    qpos_adrs: list[int] = []
    missing: list[str] = []
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(joint_name))
        if joint_id < 0:
            missing.append(str(joint_name))
            continue
        qpos_adrs.append(int(model.jnt_qposadr[joint_id]))
    if missing:
        raise ValueError(f"MuJoCo model is missing WBC observation joints: {missing}")
    return np.asarray(qpos_adrs, dtype=np.int64)


def _resolve_named_float_array(spec: Any, names: Sequence[str], *, field_name: str) -> np.ndarray:
    if isinstance(spec, (int, float)):
        return np.full((len(names),), float(spec), dtype=np.float32)
    values = _resolve_named_value_map(spec, names, field_name=field_name, require_full=False)
    return np.asarray([float(values.get(str(name), 0.0)) for name in names], dtype=np.float32)


def _wbc_reference_observation_config(policy_config: Mapping[str, Any]) -> Mapping[str, Any] | None:
    observation = policy_config.get("observation", {})
    if not isinstance(observation, Mapping):
        return None
    for group_cfg in observation.values():
        if not isinstance(group_cfg, Mapping):
            continue
        for obs_cfg in group_cfg.values():
            if isinstance(obs_cfg, Mapping) and "motion_path" in obs_cfg:
                return obs_cfg
    return None


def _wbc_observation_config(policy_config: Mapping[str, Any], obs_name: str) -> Mapping[str, Any] | None:
    observation = policy_config.get("observation", {})
    if not isinstance(observation, Mapping):
        return None
    for group_cfg in observation.values():
        if isinstance(group_cfg, Mapping) and isinstance(group_cfg.get(obs_name), Mapping):
            return group_cfg[obs_name]
    return None


def _wbc_primary_object_name(policy_config: Mapping[str, Any]) -> str | None:
    for obs_name in ("object_xy_b", "object_heading_b", "object_pos_b", "object_ori_b", "ref_contact_pos_b"):
        obs_cfg = _wbc_observation_config(policy_config, obs_name)
        if obs_cfg is not None and obs_cfg.get("object_name") is not None:
            return str(obs_cfg["object_name"])
    return None


def _wbc_prev_action_steps(policy_config: Mapping[str, Any]) -> int:
    obs_cfg = _wbc_observation_config(policy_config, "prev_actions")
    if obs_cfg is None:
        return 1
    return max(1, int(obs_cfg.get("steps", 1)))


def _torch_quat_rotate(quat: Any, vec: Any):
    import torch

    quat = quat.expand(*vec.shape[:-1], 4)
    xyz = quat[..., 1:]
    t = torch.linalg.cross(xyz, vec, dim=-1) * 2
    return vec + quat[..., 0:1] * t + torch.linalg.cross(xyz, t, dim=-1)


def _torch_quat_rotate_inverse(quat: Any, vec: Any):
    import torch

    quat = quat.expand(*vec.shape[:-1], 4)
    xyz = quat[..., 1:]
    t = torch.linalg.cross(xyz, vec, dim=-1) * 2
    return vec - quat[..., 0:1] * t + torch.linalg.cross(xyz, t, dim=-1)


def _array_to_float_list(value: Any) -> list[float]:
    return [float(item) for item in np.asarray(value, dtype=np.float64).reshape(-1).tolist()]


def _normalize_quat_list(value: Any) -> list[float]:
    quat = np.asarray(value, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"Invalid quaternion: {quat.tolist()}")
    return [float(item) for item in (quat / norm).tolist()]


def _find_freejoint_binding(model: Any, body_name: str) -> dict[str, Any] | None:
    import mujoco

    body_id = _mujoco_body_id(model, body_name)
    if body_id >= 0:
        joint_start = int(model.body_jntadr[body_id])
        joint_count = int(model.body_jntnum[body_id])
        for joint_id in range(joint_start, joint_start + joint_count):
            if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
                return _freejoint_binding(model, joint_id)
    joint_names = [f"{body_name}_root", f"{body_name}_freejoint"]
    if body_name in ("pelvis", "base", "base_link", "floating_base"):
        joint_names.append("floating_base_joint")
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id >= 0 and int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
            return _freejoint_binding(model, joint_id)
    return None


def _mujoco_body_id(model: Any, body_name: str) -> int:
    import mujoco

    for candidate in (body_name, f"{body_name}_body"):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(candidate))
        if body_id >= 0:
            return int(body_id)
    return -1


def _freejoint_binding(model: Any, joint_id: int) -> dict[str, Any]:
    return {
        "joint_name": str(model.joint(joint_id).name),
        "qpos_adr": int(model.jnt_qposadr[joint_id]),
        "qvel_adr": int(model.jnt_dofadr[joint_id]),
    }


def _write_freejoint_state(
    data: Any,
    *,
    qpos_adr: int,
    qvel_adr: int,
    pos: Sequence[float],
    quat: Sequence[float],
    lin_vel: Sequence[float],
    ang_vel: Sequence[float],
) -> None:
    data.qpos[qpos_adr:qpos_adr + 3] = np.asarray(pos, dtype=np.float64)
    data.qpos[qpos_adr + 3:qpos_adr + 7] = np.asarray(_normalize_quat_list(quat), dtype=np.float64)
    data.qvel[qvel_adr:qvel_adr + 3] = np.asarray(lin_vel, dtype=np.float64)
    data.qvel[qvel_adr + 3:qvel_adr + 6] = np.asarray(ang_vel, dtype=np.float64)


def _wbc_scene_config(
    *,
    sim2real_root: Path,
    robot_type: str,
    object_asset_name: str,
    object_type: str,
    object_joint_name: Any,
    command_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    sim2real_scene = _sim2real_scene_config_for_wbc_task(
        sim2real_root=sim2real_root,
        robot_type=robot_type,
        object_type=object_type,
        object_joint_name=object_joint_name,
        command_cfg=command_cfg,
    )
    if sim2real_scene is not None:
        return sim2real_scene

    from active_adaptation.assets_mjcf import ROBOTS

    robot_cfg = ROBOTS.with_object("g1_29dof", object_asset_name, object_type)
    scene_config: dict[str, Any] = {
        "ROBOT_SCENE": str(robot_cfg.mjcf_path),
        "SIMULATE_DT": 0.002,
        "VIEWER_DT": 0.02,
        "ENABLE_ELASTIC_BAND": _robot_type_uses_elastic_band(robot_type),
        "publish_object_names": [],
        "_asset_source": "wbc_assets_mjcf",
    }
    if object_joint_name:
        scene_config.update(
            {
                "object_joint_name": str(object_joint_name),
                "joint_friction": float(command_cfg.get("joint_friction", 0.0)),
                "joint_damping": float(command_cfg.get("joint_damping", 0.0)),
                "joint_stiffness": float(command_cfg.get("joint_stiffness", 0.0)),
            }
        )
    return scene_config


def _sim2real_scene_config_for_wbc_task(
    *,
    sim2real_root: Path,
    robot_type: str,
    object_type: str,
    object_joint_name: Any,
    command_cfg: Mapping[str, Any],
) -> dict[str, Any] | None:
    robot_prefix = _sim2real_robot_prefix(robot_type)
    if robot_prefix is None:
        return None
    scene_path = sim2real_root / "data/robots/g1" / f"{robot_prefix}-{object_type}.xml"
    if not scene_path.exists() or not _mjcf_is_loadable(scene_path):
        return None
    scene_yaml = sim2real_root / "config/scene" / f"{robot_prefix}-{object_type}.yaml"
    if scene_yaml.exists():
        scene_config = _load_yaml(scene_yaml)
        scene_config["ROBOT_SCENE"] = str(_resolve_under_root(sim2real_root, scene_config["ROBOT_SCENE"]))
    else:
        scene_config = {
            "ROBOT_SCENE": str(scene_path),
            "SIMULATE_DT": 0.005,
            "VIEWER_DT": 0.02,
            "ENABLE_ELASTIC_BAND": False,
            "USE_JOYSTICK": 0,
            "publish_object_names": [],
        }
    scene_config["ROBOT_SCENE"] = str(scene_path)
    scene_config["_asset_source"] = "sim2real_hdmi"
    if object_joint_name:
        scene_config.update(
            {
                "object_joint_name": str(object_joint_name),
                "joint_friction": float(command_cfg.get("joint_friction", scene_config.get("joint_friction", 0.0))),
                "joint_damping": float(command_cfg.get("joint_damping", scene_config.get("joint_damping", 0.0))),
                "joint_stiffness": float(command_cfg.get("joint_stiffness", scene_config.get("joint_stiffness", 0.0))),
            }
        )
    return scene_config


def _sim2real_robot_prefix(robot_type: str) -> str | None:
    if "rubberhand" in robot_type:
        return "g1_29dof_rubberhand"
    if "nohand" in robot_type:
        return "g1_29dof_nohand"
    return None


def _robot_type_uses_elastic_band(robot_type: str) -> bool:
    return "rubberhand" in robot_type


def _mjcf_is_loadable(path: Path) -> bool:
    try:
        import mujoco

        mujoco.MjModel.from_xml_path(str(path))
        return True
    except Exception:
        return False


def _dedupe_names(names: Sequence[str]) -> tuple[str, ...]:
    deduped: list[str] = []
    for name in names:
        if name and name not in deduped:
            deduped.append(str(name))
    return tuple(deduped)


def _resolve_named_array(config: Any, names: Sequence[str], resolver: Any) -> np.ndarray:
    values = np.zeros(len(names), dtype=np.float32)
    if isinstance(config, (float, int)):
        values[:] = float(config)
        return values
    indices, _, resolved = resolver(config, names, preserve_order=True, strict=False)
    values[np.asarray(indices, dtype=np.int64)] = np.asarray(resolved, dtype=np.float32)
    return values


def _resolve_unitree_array(config: Any, resolver: Any) -> np.ndarray:
    from utils.strings import unitree_joint_names
    return _resolve_named_array(config, list(unitree_joint_names), resolver)


def _apply_object_joint_control(model: Any, data: Any, scene_config: Mapping[str, Any]) -> None:
    if not scene_config.get("object_joint_name"):
        return
    import mujoco
    joint_name = str(scene_config["object_joint_name"])
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
    if joint_id < 0 or actuator_id < 0:
        return
    qpos_adr = int(model.jnt_qposadr[joint_id])
    qvel_adr = int(model.jnt_dofadr[joint_id])
    qpos = data.qpos[qpos_adr]
    qvel = data.qvel[qvel_adr]
    friction = float(scene_config.get("joint_friction", 0.0))
    damping = float(scene_config.get("joint_damping", 0.0))
    stiffness = float(scene_config.get("joint_stiffness", 0.0))
    data.ctrl[actuator_id] = (
        -friction * np.sign(qvel) * (abs(qvel) > 0.01)
        + stiffness * (0.0 - qpos)
        + damping * (0.0 - qvel)
    )


def _sample_scene(model: Any, data: Any, scenario: ScenarioSpec, scene_config: Mapping[str, Any]) -> dict[str, Any]:
    import mujoco
    sample: dict[str, Any] = {}
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_id >= 0:
        pelvis_pos = np.asarray(data.xpos[pelvis_id], dtype=np.float64)
        pelvis_xmat = np.asarray(data.xmat[pelvis_id], dtype=np.float64).reshape(3, 3)
        sample["pelvis_z"] = _json_float(pelvis_pos[2])
        sample["pelvis_xy"] = [_json_float(pelvis_pos[0]), _json_float(pelvis_pos[1])]
        sample["pelvis_up_z"] = _json_float(pelvis_xmat[2, 2])
    for object_name in ("suitcase", "ball", "door", "door_panel"):
        pos = _named_position(model, data, object_name)
        if pos is not None:
            sample[f"{object_name}_xy"] = [_json_float(pos[0]), _json_float(pos[1])]
            sample[f"{object_name}_z"] = _json_float(pos[2])
    if scene_config.get("object_joint_name"):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(scene_config["object_joint_name"]))
        if joint_id >= 0:
            qpos_adr = int(model.jnt_qposadr[joint_id])
            sample["door_joint"] = _json_float(data.qpos[qpos_adr])
    return sample


def _named_position(model: Any, data: Any, object_name: str) -> np.ndarray | None:
    import mujoco
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"{object_name}_pos")
    if sensor_id >= 0:
        adr = int(model.sensor_adr[sensor_id])
        return np.asarray(data.sensordata[adr:adr + 3], dtype=np.float64)
    body_id = _mujoco_body_id(model, object_name)
    if body_id >= 0:
        return np.asarray(data.xpos[body_id], dtype=np.float64)
    return None


def _object_metrics(samples: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    metrics = {
        "suitcase_xy_displacement": _xy_displacement(samples, "suitcase_xy"),
        "ball_xy_displacement": _xy_displacement(samples, "ball_xy"),
        "door_joint_abs": None,
    }
    door_values = [sample.get("door_joint") for sample in samples if sample.get("door_joint") is not None]
    if door_values:
        metrics["door_joint_abs"] = _json_float(abs(float(door_values[-1])))
    return metrics


def _reference_tracking_summary(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reference_steps = [
        int(sample["reference_step"])
        for sample in samples
        if sample.get("reference_step") is not None
    ]
    metric_keys = (
        "q_ref_l2",
        "q_ref_linf",
        "q_target_ref_l2",
        "q_target_ref_linf",
        "body_pos_ref_l2",
        "body_pos_ref_linf",
        "object_pos_ref_l2",
        "object_pos_ref_linf",
    )
    summary: dict[str, Any] = {"available": bool(reference_steps)}
    if not reference_steps:
        return summary
    summary["reference_step_first"] = reference_steps[0]
    summary["reference_step_last"] = reference_steps[-1]
    for metric_key in metric_keys:
        values = [
            float(sample[metric_key])
            for sample in samples
            if sample.get(metric_key) is not None and math.isfinite(float(sample[metric_key]))
        ]
        if not values:
            continue
        summary[f"{metric_key}_mean"] = _json_float(np.mean(values))
        summary[f"{metric_key}_max"] = _json_float(np.max(values))
    for count_key in (
        "reference_joint_count",
        "reference_body_count",
        "reference_object_body_count",
        "q_target_ref_joint_count",
    ):
        counts = [
            int(sample[count_key])
            for sample in samples
            if sample.get(count_key) is not None
        ]
        if counts:
            summary[count_key] = int(max(counts))
    top_joint_errors = _aggregate_top_scalar_errors(samples, "q_ref_joint_error_top")
    if top_joint_errors:
        summary["top_q_ref_joint_errors"] = top_joint_errors
    top_q_target_errors = _aggregate_top_scalar_errors(samples, "q_target_ref_joint_error_top")
    if top_q_target_errors:
        summary["top_q_target_ref_joint_errors"] = top_q_target_errors
    top_body_errors = _aggregate_top_vector_errors(samples, "body_pos_ref_error_top")
    if top_body_errors:
        summary["top_body_pos_ref_errors"] = top_body_errors
    top_object_errors = _aggregate_top_vector_errors(samples, "object_pos_ref_error_top")
    if top_object_errors:
        summary["top_object_pos_ref_errors"] = top_object_errors
    phase_offset_diagnostic = _reference_phase_offset_diagnostic(samples)
    if phase_offset_diagnostic:
        summary["phase_offset_diagnostic"] = phase_offset_diagnostic
    return summary


def _top_scalar_error_entries(names: Sequence[str], errors: np.ndarray, *, top_n: int) -> list[dict[str, Any]]:
    errors = np.asarray(errors, dtype=np.float64).reshape(-1)
    if len(errors) == 0:
        return []
    order = np.argsort(np.abs(errors))[::-1][:top_n]
    return [
        {
            "name": str(names[int(index)]),
            "abs_error": _json_float(abs(errors[int(index)])),
            "signed_error": _json_float(errors[int(index)]),
        }
        for index in order
    ]


def _top_vector_error_entries(names: Sequence[str], errors: np.ndarray, *, top_n: int) -> list[dict[str, Any]]:
    errors = np.asarray(errors, dtype=np.float64).reshape((-1, 3))
    if len(errors) == 0:
        return []
    l2 = np.linalg.norm(errors, axis=1)
    order = np.argsort(l2)[::-1][:top_n]
    return [
        {
            "name": str(names[int(index)]),
            "l2": _json_float(l2[int(index)]),
            "linf": _json_float(np.max(np.abs(errors[int(index)]))),
            "error_xyz": [_json_float(value) for value in errors[int(index)].tolist()],
        }
        for index in order
    ]


def _aggregate_top_scalar_errors(samples: Sequence[Mapping[str, Any]], key: str, *, top_n: int = 8) -> list[dict[str, Any]]:
    by_name: dict[str, list[tuple[float, int | None]]] = {}
    for sample in samples:
        reference_step = int(sample["reference_step"]) if sample.get("reference_step") is not None else None
        for entry in sample.get(key, []) or []:
            name = str(entry.get("name"))
            value = entry.get("abs_error")
            if value is None:
                continue
            by_name.setdefault(name, []).append((float(value), reference_step))
    rows = []
    for name, values in by_name.items():
        error_values = [value for value, _ in values]
        max_index = int(np.argmax(error_values))
        rows.append(
            {
                "name": name,
                "mean_abs_error": _json_float(np.mean(error_values)),
                "max_abs_error": _json_float(error_values[max_index]),
                "max_reference_step": values[max_index][1],
            }
        )
    return sorted(rows, key=lambda row: (row["max_abs_error"] or -math.inf, row["mean_abs_error"] or -math.inf), reverse=True)[:top_n]


def _aggregate_top_vector_errors(samples: Sequence[Mapping[str, Any]], key: str, *, top_n: int = 8) -> list[dict[str, Any]]:
    by_name: dict[str, list[tuple[float, int | None]]] = {}
    for sample in samples:
        reference_step = int(sample["reference_step"]) if sample.get("reference_step") is not None else None
        for entry in sample.get(key, []) or []:
            name = str(entry.get("name"))
            value = entry.get("l2")
            if value is None:
                continue
            by_name.setdefault(name, []).append((float(value), reference_step))
    rows = []
    for name, values in by_name.items():
        l2_values = [value for value, _ in values]
        max_index = int(np.argmax(l2_values))
        rows.append(
            {
                "name": name,
                "mean_l2": _json_float(np.mean(l2_values)),
                "max_l2": _json_float(l2_values[max_index]),
                "max_reference_step": values[max_index][1],
            }
        )
    return sorted(rows, key=lambda row: (row["max_l2"] or -math.inf, row["mean_l2"] or -math.inf), reverse=True)[:top_n]


def _reference_phase_offset_diagnostic(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    by_offset: dict[int, dict[str, list[float]]] = {}
    for sample in samples:
        for row in sample.get("reference_offset_errors", []) or []:
            offset = int(row["offset"])
            metrics = by_offset.setdefault(offset, {})
            for metric_key in ("q_ref_l2", "body_pos_ref_l2", "object_pos_ref_l2"):
                value = row.get(metric_key)
                if value is not None:
                    metrics.setdefault(metric_key, []).append(float(value))
    if not by_offset:
        return None
    offset_rows = []
    for offset, metrics in sorted(by_offset.items()):
        offset_row: dict[str, Any] = {"offset": int(offset)}
        for metric_key, values in metrics.items():
            if values:
                offset_row[f"{metric_key}_mean"] = _json_float(np.mean(values))
        offset_rows.append(offset_row)
    rows_with_q = [row for row in offset_rows if row.get("q_ref_l2_mean") is not None]
    if not rows_with_q:
        return {"offsets": offset_rows}
    best_row = min(rows_with_q, key=lambda row: float(row["q_ref_l2_mean"]))
    zero_row = next((row for row in rows_with_q if row["offset"] == 0), None)
    result: dict[str, Any] = {
        "best_offset": int(best_row["offset"]),
        "best_q_ref_l2_mean": best_row.get("q_ref_l2_mean"),
        "best_body_pos_ref_l2_mean": best_row.get("body_pos_ref_l2_mean"),
        "best_object_pos_ref_l2_mean": best_row.get("object_pos_ref_l2_mean"),
        "offsets": offset_rows,
    }
    if zero_row is not None:
        result["zero_offset_q_ref_l2_mean"] = zero_row.get("q_ref_l2_mean")
        if best_row.get("q_ref_l2_mean") is not None and zero_row.get("q_ref_l2_mean") is not None:
            result["q_ref_l2_improvement_vs_zero"] = _json_float(
                float(zero_row["q_ref_l2_mean"]) - float(best_row["q_ref_l2_mean"])
            )
    return result


def _xy_displacement(samples: Sequence[Mapping[str, Any]], key: str) -> float | None:
    if key not in samples[0] or key not in samples[-1]:
        return None
    first = np.asarray(samples[0][key], dtype=np.float64)
    last = np.asarray(samples[-1][key], dtype=np.float64)
    return _json_float(np.linalg.norm(last - first))


def _json_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _json_value(value) for key, value in sample.items()}


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_value(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return _json_float(float(value))
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _json_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, 8)


def _make_video_writer(video_path: Path | None, fps: int):
    @contextlib.contextmanager
    def _writer():
        if video_path is None:
            yield None
            return
        import imageio.v2 as imageio
        video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(video_path), fps=fps)
        try:
            yield writer
        finally:
            writer.close()
    return _writer()


def _make_camera():
    import mujoco
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 4.0
    camera.azimuth = 135.0
    camera.elevation = -18.0
    return camera


@dataclass
class _ViewerClock:
    sim_start: float
    wall_start: float = 0.0

    def __post_init__(self) -> None:
        self.wall_start = time.monotonic()


def _policy_viewer_context(model: Any, data: Any, *, enabled: bool):
    if not enabled:
        return contextlib.nullcontext(None)
    import mujoco.viewer

    return mujoco.viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=True)


def _sync_policy_viewer(viewer: Any, data: Any, clock: _ViewerClock, *, speed: float) -> None:
    target_wall = clock.wall_start + max(0.0, float(data.time) - clock.sim_start) / float(speed)
    sleep_sec = target_wall - time.monotonic()
    if sleep_sec > 0.0:
        time.sleep(min(sleep_sec, 0.05))
    viewer.sync()


def _render_frame(renderer: Any, model: Any, data: Any, camera: Any, writer: Any) -> None:
    import mujoco
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_id >= 0:
        camera.lookat[:] = data.xpos[pelvis_id]
    renderer.update_scene(data, camera=camera)
    writer.append_data(renderer.render())


if __name__ == "__main__":
    raise SystemExit(main())
