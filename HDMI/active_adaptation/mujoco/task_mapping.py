from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from active_adaptation.assets_mjcf import ROBOTS
from active_adaptation.assets_mjcf.manifest import build_name_index, load_mujoco_asset_manifest


HDMI_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class NameIndexMapping:
    name: str
    motion_index: int
    mujoco_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "motion_index": self.motion_index,
            "mujoco_index": self.mujoco_index,
        }


@dataclass(frozen=True)
class PolicyNameIndexMapping:
    name: str
    policy_index: int
    motion_index: int
    mujoco_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "policy_index": self.policy_index,
            "motion_index": self.motion_index,
            "mujoco_index": self.mujoco_index,
        }


@dataclass(frozen=True)
class TaskMotionMappingReport:
    task_path: Path
    motion_dir: Path
    object_asset_name: str
    object_type: str
    task_object_body_name: str
    reference_object_body_name: str
    object_joint_name: str | None
    asset_object_body_names: tuple[str, ...]
    asset_object_joint_names: tuple[str, ...]
    extra_object_names: tuple[str, ...]
    motion_body_names: tuple[str, ...]
    motion_joint_names: tuple[str, ...]
    body_name_mapping: tuple[NameIndexMapping, ...]
    joint_name_mapping: tuple[NameIndexMapping, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_path": str(self.task_path),
            "motion_dir": str(self.motion_dir),
            "object_asset_name": self.object_asset_name,
            "object_type": self.object_type,
            "task_object_body_name": self.task_object_body_name,
            "reference_object_body_name": self.reference_object_body_name,
            "object_joint_name": self.object_joint_name,
            "asset_object_body_names": list(self.asset_object_body_names),
            "asset_object_joint_names": list(self.asset_object_joint_names),
            "extra_object_names": list(self.extra_object_names),
            "motion_body_names": list(self.motion_body_names),
            "motion_joint_names": list(self.motion_joint_names),
            "body_name_mapping": [entry.to_dict() for entry in self.body_name_mapping],
            "joint_name_mapping": [entry.to_dict() for entry in self.joint_name_mapping],
        }


@dataclass(frozen=True)
class PolicyTaskMotionMappingReport:
    task_report: TaskMotionMappingReport
    policy_config_path: Path
    policy_body_names: tuple[str, ...]
    policy_joint_names: tuple[str, ...]
    policy_body_name_mapping: tuple[PolicyNameIndexMapping, ...]
    policy_joint_name_mapping: tuple[PolicyNameIndexMapping, ...]

    def to_dict(self) -> dict[str, Any]:
        data = self.task_report.to_dict()
        data.update(
            {
                "policy_config_path": str(self.policy_config_path),
                "policy_body_names": list(self.policy_body_names),
                "policy_joint_names": list(self.policy_joint_names),
                "policy_body_name_mapping": [entry.to_dict() for entry in self.policy_body_name_mapping],
                "policy_joint_name_mapping": [entry.to_dict() for entry in self.policy_joint_name_mapping],
            }
        )
        return data


def validate_policy_task_motion_mapping(
    policy_config_path: str | Path,
    task_yaml: str | Path,
    *,
    robot_name: str = "g1_29dof",
) -> PolicyTaskMotionMappingReport:
    policy_config_path = Path(policy_config_path)
    policy_cfg = _load_yaml_mapping(policy_config_path)
    task_report = validate_task_motion_mapping(task_yaml, robot_name=robot_name)
    policy_body_names = tuple(
        _string_list(policy_cfg.get("isaac_body_names", ()), "isaac_body_names", policy_config_path)
    )
    policy_joint_names = tuple(
        _string_list(
            policy_cfg.get("isaac_joint_names") or policy_cfg.get("policy_joint_names", ()),
            "isaac_joint_names",
            policy_config_path,
        )
    )

    return PolicyTaskMotionMappingReport(
        task_report=task_report,
        policy_config_path=policy_config_path,
        policy_body_names=policy_body_names,
        policy_joint_names=policy_joint_names,
        policy_body_name_mapping=_build_policy_name_mapping(
            policy_body_names,
            task_report.motion_body_names,
            task_report.body_name_mapping,
            label="policy body",
            path=policy_config_path,
        ),
        policy_joint_name_mapping=_build_policy_name_mapping(
            policy_joint_names,
            task_report.motion_joint_names,
            task_report.joint_name_mapping,
            label="policy joint",
            path=policy_config_path,
        ),
    )


def validate_task_motion_mapping(
    task_yaml: str | Path,
    *,
    robot_name: str = "g1_29dof",
) -> TaskMotionMappingReport:
    task_path = Path(task_yaml)
    task_cfg = _load_yaml_mapping(task_path)
    command_cfg = _mapping(task_cfg.get("command", {}), f"{task_path}: command")
    if "object_asset_name" not in command_cfg:
        raise ValueError(f"{task_path}: command.object_asset_name is required for object mapping validation.")

    object_asset_name = str(command_cfg["object_asset_name"])
    object_type = str(command_cfg.get("object_type") or object_asset_name)
    task_object_body_name = _required_str(command_cfg, "object_body_name", task_path)
    object_joint_name = _optional_str(command_cfg.get("object_joint_name"))
    extra_object_names = tuple(_string_list(command_cfg.get("extra_object_names", ()), "extra_object_names", task_path))

    motion_dir = _resolve_motion_dir(task_path, _required_path(command_cfg, "data_path", task_path))
    _require_file(motion_dir / "motion.npz")
    motion_meta = _load_json_mapping(motion_dir / "meta.json")
    motion_body_names = tuple(_string_list(motion_meta.get("body_names"), "body_names", motion_dir / "meta.json"))
    motion_joint_names = tuple(_string_list(motion_meta.get("joint_names"), "joint_names", motion_dir / "meta.json"))

    robot_cfg = ROBOTS.with_object(robot_name, object_asset_name=object_asset_name, object_type=object_type)
    object_specs = robot_cfg.object_specs
    if object_asset_name not in object_specs:
        raise ValueError(
            f"{task_path}: object_asset_name={object_asset_name!r} is absent from MuJoCo object specs "
            f"{sorted(object_specs)}."
        )
    object_spec = object_specs[object_asset_name]
    asset_object_body_names = tuple(object_spec.body_names)
    asset_object_joint_names = tuple(object_spec.joint_names)

    if task_object_body_name not in asset_object_body_names:
        raise ValueError(
            f"{task_path}: command.object_body_name={task_object_body_name!r} is absent from "
            f"{object_asset_name!r} MuJoCo bodies {asset_object_body_names!r}."
        )
    if object_joint_name is not None and object_joint_name not in asset_object_joint_names:
        raise ValueError(
            f"{task_path}: command.object_joint_name={object_joint_name!r} is absent from "
            f"{object_asset_name!r} MuJoCo joints {asset_object_joint_names!r}."
        )

    missing_extra_specs = [name for name in extra_object_names if name not in object_specs]
    if missing_extra_specs:
        raise ValueError(
            f"{task_path}: extra_object_names are absent from MuJoCo object specs: {missing_extra_specs}."
        )
    missing_extra_motion = [name for name in extra_object_names if name not in motion_body_names]
    if missing_extra_motion:
        raise ValueError(f"{task_path}: extra_object_names are absent from motion bodies: {missing_extra_motion}.")

    manifest = load_mujoco_asset_manifest(robot_cfg.mjcf_path)
    body_name_indices = build_name_index(motion_body_names, manifest.body_names, label=f"{task_path.name} body")
    joint_name_indices = build_name_index(
        motion_joint_names,
        manifest.tracking_joint_names,
        label=f"{task_path.name} joint",
    )
    body_name_mapping = _build_name_mapping(motion_body_names, body_name_indices)
    joint_name_mapping = _build_name_mapping(motion_joint_names, joint_name_indices)

    root_body_name = _optional_str(command_cfg.get("root_body_name"))
    if root_body_name is not None and root_body_name not in motion_body_names:
        raise ValueError(f"{task_path}: root_body_name={root_body_name!r} is absent from motion bodies.")
    if object_joint_name is not None and object_joint_name not in motion_joint_names:
        raise ValueError(f"{task_path}: object_joint_name={object_joint_name!r} is absent from motion joints.")

    reference_object_body_name = _resolve_reference_object_body_name(
        motion_body_names=motion_body_names,
        object_asset_name=object_asset_name,
        task_object_body_name=task_object_body_name,
        asset_object_body_names=asset_object_body_names,
        task_path=task_path,
    )

    return TaskMotionMappingReport(
        task_path=task_path,
        motion_dir=motion_dir,
        object_asset_name=object_asset_name,
        object_type=object_type,
        task_object_body_name=task_object_body_name,
        reference_object_body_name=reference_object_body_name,
        object_joint_name=object_joint_name,
        asset_object_body_names=asset_object_body_names,
        asset_object_joint_names=asset_object_joint_names,
        extra_object_names=extra_object_names,
        motion_body_names=motion_body_names,
        motion_joint_names=motion_joint_names,
        body_name_mapping=body_name_mapping,
        joint_name_mapping=joint_name_mapping,
    )


def validate_all_task_motion_mappings(
    task_dir: str | Path,
    *,
    robot_name: str = "g1_29dof",
) -> tuple[TaskMotionMappingReport, ...]:
    reports: list[TaskMotionMappingReport] = []
    for task_path in sorted(Path(task_dir).glob("*.yaml")):
        task_cfg = _load_yaml_mapping(task_path)
        command_cfg = _mapping(task_cfg.get("command", {}), f"{task_path}: command")
        if "object_asset_name" not in command_cfg:
            continue
        reports.append(validate_task_motion_mapping(task_path, robot_name=robot_name))
    return tuple(reports)


def _build_policy_name_mapping(
    policy_names: Sequence[str],
    motion_names: Sequence[str],
    motion_mapping: Sequence[NameIndexMapping],
    *,
    label: str,
    path: Path,
) -> tuple[PolicyNameIndexMapping, ...]:
    motion_index = {name: index for index, name in enumerate(motion_names)}
    mujoco_index = {entry.name: entry.mujoco_index for entry in motion_mapping}
    missing_motion = [name for name in policy_names if name not in motion_index]
    if missing_motion:
        raise ValueError(f"{path}: {label} names are absent from motion metadata: {missing_motion}.")
    return tuple(
        PolicyNameIndexMapping(
            name=name,
            policy_index=policy_index,
            motion_index=motion_index[name],
            mujoco_index=mujoco_index[name],
        )
        for policy_index, name in enumerate(policy_names)
    )


def _build_name_mapping(
    motion_names: Sequence[str],
    mujoco_indices: Sequence[int],
) -> tuple[NameIndexMapping, ...]:
    if len(motion_names) != len(mujoco_indices):
        raise ValueError("motion names and MuJoCo indices must have the same length.")
    return tuple(
        NameIndexMapping(name=name, motion_index=motion_index, mujoco_index=int(mujoco_index))
        for motion_index, (name, mujoco_index) in enumerate(zip(motion_names, mujoco_indices))
    )


def _resolve_reference_object_body_name(
    *,
    motion_body_names: Sequence[str],
    object_asset_name: str,
    task_object_body_name: str,
    asset_object_body_names: Sequence[str],
    task_path: Path,
) -> str:
    candidates = _unique_preserve_order((task_object_body_name, object_asset_name, *asset_object_body_names))
    for candidate in candidates:
        if candidate in motion_body_names:
            return candidate
    raise ValueError(
        f"{task_path}: none of the MuJoCo object bodies {tuple(asset_object_body_names)!r} appear in motion bodies."
    )


def _resolve_motion_dir(task_path: Path, data_path: Path) -> Path:
    if data_path.is_absolute():
        return data_path
    roots = (_task_project_root(task_path), HDMI_ROOT, Path.cwd())
    for root in roots:
        candidate = root / data_path
        if candidate.exists():
            return candidate
    return roots[0] / data_path


def _task_project_root(task_path: Path) -> Path:
    resolved = task_path.resolve()
    parts = resolved.parts
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "cfg":
            return Path(*parts[:index])
    return HDMI_ROOT


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    data = yaml.safe_load(path.read_text()) or {}
    return _mapping(data, str(path))


def _load_json_mapping(path: Path) -> Mapping[str, Any]:
    _require_file(path)
    return _mapping(json.loads(path.read_text()), str(path))


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping, got {type(value).__name__}.")
    return value


def _required_str(mapping: Mapping[str, Any], key: str, path: Path) -> str:
    value = mapping.get(key)
    if value is None:
        raise ValueError(f"{path}: command.{key} is required for object mapping validation.")
    return str(value)


def _required_path(mapping: Mapping[str, Any], key: str, path: Path) -> Path:
    return Path(_required_str(mapping, key, path))


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _string_list(value: Any, label: str, path: Path) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{path}: {label} must be a sequence of strings, got {type(value).__name__}.")
    return [str(item) for item in value]


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required mapping file: {path}")


def _unique_preserve_order(names: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return tuple(ordered)
