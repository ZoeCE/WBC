from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
HDMI_ROOT = SCRIPT_DIR.parent
DEFAULT_TASK_RELATIVE_DIR = Path("cfg/task/G1/hdmi")
DEFAULT_PYTHON = "python"
SOURCE_REFERENCE = {
    "repository": "https://github.com/LeCAR-Lab/HDMI",
    "scope": "source_contract_manifest",
    "purpose": (
        "Machine-readable HDMI task/motion/MJCF/contact/export contract used to drive "
        "MuJoCo policy migration and parity audits."
    ),
}
SEMANTIC_FIELDS = (
    "name",
    "robot.robot_type",
    "command.data_path",
    "command.root_body_name",
    "command.object_asset_name",
    "command.object_body_name",
    "command.object_joint_name",
    "command.extra_object_names",
    "command.contact_eef_body_name",
    "command.contact_target_pos_offset",
    "command.contact_eef_pos_offset",
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_source_contract_manifest(
        local_root=Path(args.local_root),
        upstream_root=Path(args.upstream_root) if args.upstream_root else None,
        task_relative_dir=Path(args.task_relative_dir),
        expected_task_count=args.expected_task_count,
        python=args.python,
        require_task_count=args.require_task_count,
        require_semantic_match=args.require_semantic_match,
        require_motion_files=args.require_motion_files,
        require_object_assets=args.require_object_assets,
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, sort_keys=True))
    return 0 if report["gate_passed"] else 1


def build_source_contract_manifest(
    *,
    local_root: Path = HDMI_ROOT,
    upstream_root: Path | None = None,
    task_relative_dir: Path = DEFAULT_TASK_RELATIVE_DIR,
    expected_task_count: int | None = None,
    python: str = DEFAULT_PYTHON,
    require_task_count: bool = False,
    require_semantic_match: bool = False,
    require_motion_files: bool = False,
    require_object_assets: bool = False,
) -> dict[str, Any]:
    local_root = local_root.resolve()
    upstream_root = upstream_root.resolve() if upstream_root else None
    local_tasks = _task_files(local_root, task_relative_dir)
    upstream_tasks = _task_files(upstream_root, task_relative_dir) if upstream_root else {}
    task_names = sorted(local_tasks)
    tasks = [
        _build_task_contract(
            local_root=local_root,
            upstream_root=upstream_root,
            local_task_path=local_tasks[task_name],
            upstream_task_path=upstream_tasks.get(task_name),
            python=python,
        )
        for task_name in task_names
    ]

    task_count_gate_passed = expected_task_count is None or (
        len(local_tasks) == expected_task_count
        and (upstream_root is None or len(upstream_tasks) == expected_task_count)
    )
    missing_upstream_tasks = sorted(set(local_tasks) - set(upstream_tasks)) if upstream_root else []
    extra_upstream_tasks = sorted(set(upstream_tasks) - set(local_tasks)) if upstream_root else []
    failures: list[dict[str, Any]] = []
    if require_task_count and (not task_count_gate_passed or missing_upstream_tasks or extra_upstream_tasks):
        failures.append(
            {
                "component": "task_count",
                "reason": "task_inventory_mismatch",
                "expected_task_count": expected_task_count,
                "local_task_count": len(local_tasks),
                "upstream_task_count": len(upstream_tasks) if upstream_root else None,
                "missing_upstream_tasks": missing_upstream_tasks,
                "extra_upstream_tasks": extra_upstream_tasks,
            }
        )
    if require_semantic_match:
        mismatched = [task for task in tasks if task["semantic_match"] is not True]
        if mismatched:
            failures.append(
                {
                    "component": "semantic_match",
                    "reason": "task_semantic_contract_drift",
                    "mismatched_tasks": [task["task_file"] for task in mismatched],
                }
            )
    if require_motion_files:
        missing_motion = [
            task["task_file"]
            for task in tasks
            if not (task["motion"]["has_motion_npz"] and task["motion"]["has_meta_json"])
        ]
        if missing_motion:
            failures.append(
                {
                    "component": "motion_files",
                    "reason": "missing_local_motion_files",
                    "tasks": missing_motion,
                }
            )
    if require_object_assets:
        missing_assets = [task["task_file"] for task in tasks if not task["object"]["local_asset_files"]]
        if missing_assets:
            failures.append(
                {
                    "component": "object_assets",
                    "reason": "missing_local_object_assets",
                    "tasks": missing_assets,
                }
            )

    return {
        "gate_passed": not failures,
        "source_reference": SOURCE_REFERENCE,
        "local_root": str(local_root),
        "upstream_root": str(upstream_root) if upstream_root else None,
        "local_git_head": _git_head(local_root),
        "upstream_git_head": _git_head(upstream_root) if upstream_root else None,
        "task_relative_dir": str(task_relative_dir),
        "expected_task_count": expected_task_count,
        "task_count": len(tasks),
        "local_task_count": len(local_tasks),
        "upstream_task_count": len(upstream_tasks) if upstream_root else None,
        "task_count_gate_passed": task_count_gate_passed,
        "missing_upstream_tasks": missing_upstream_tasks,
        "extra_upstream_tasks": extra_upstream_tasks,
        "tasks": tasks,
        "failures": failures,
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a machine-readable HDMI source contract manifest for MuJoCo migration."
    )
    parser.add_argument("--local-root", default=str(HDMI_ROOT))
    parser.add_argument("--upstream-root", default=None)
    parser.add_argument("--task-relative-dir", default=str(DEFAULT_TASK_RELATIVE_DIR))
    parser.add_argument("--expected-task-count", type=int, default=None)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--require-task-count", action="store_true")
    parser.add_argument("--require-semantic-match", action="store_true")
    parser.add_argument("--require-motion-files", action="store_true")
    parser.add_argument("--require-object-assets", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


def _build_task_contract(
    *,
    local_root: Path,
    upstream_root: Path | None,
    local_task_path: Path,
    upstream_task_path: Path | None,
    python: str,
) -> dict[str, Any]:
    local_cfg = _load_yaml_mapping(local_task_path)
    upstream_cfg = _load_yaml_mapping(upstream_task_path) if upstream_task_path else None
    command = _mapping(local_cfg.get("command"))
    task_stem = local_task_path.stem
    task_name = str(local_cfg.get("name") or task_stem)
    task_override = _task_override_from_path(local_task_path)
    data_path = _optional_str(command.get("data_path"))
    motion = _build_motion_contract(local_root=local_root, data_path=data_path)
    object_contract = _build_object_contract(local_root=local_root, command=command)
    semantic_mismatches = _semantic_mismatches(local_cfg, upstream_cfg, fields=SEMANTIC_FIELDS)
    return {
        "task_name": task_name,
        "task_override": task_override,
        "task_stem": task_stem,
        "task_file": local_task_path.name,
        "local_task_path": str(local_task_path),
        "upstream_task_path": str(upstream_task_path) if upstream_task_path else None,
        "semantic_match": upstream_cfg is not None and not semantic_mismatches,
        "semantic_mismatches": semantic_mismatches,
        "robot": {
            "name": _nested_get(local_cfg, "robot.name"),
            "robot_type": _nested_get(local_cfg, "robot.robot_type"),
        },
        "motion": motion,
        "object": object_contract,
        "contact": {
            "eef_body_names": _list_value(command.get("contact_eef_body_name")),
            "target_pos_offset": _list_value(command.get("contact_target_pos_offset")),
            "eef_pos_offset": _list_value(command.get("contact_eef_pos_offset")),
        },
        "randomization": _mapping(local_cfg.get("randomization")),
        "commands": {
            "policy_export": _policy_export_command(
                python=python,
                task_override=task_override,
            ),
            "mujoco_motion_viewer": _mujoco_motion_viewer_command(
                python=python,
                motion_dir=motion["local_motion_dir"],
            ),
        },
    }


def _build_motion_contract(*, local_root: Path, data_path: str | None) -> dict[str, Any]:
    motion_dir = local_root / data_path if data_path else None
    meta_path = motion_dir / "meta.json" if motion_dir else None
    meta = _load_json_mapping(meta_path) if meta_path and meta_path.exists() else {}
    return {
        "data_path": data_path,
        "local_motion_dir": str(motion_dir) if motion_dir else None,
        "has_motion_npz": bool(motion_dir and (motion_dir / "motion.npz").exists()),
        "has_meta_json": bool(meta_path and meta_path.exists()),
        "body_names": _list_value(meta.get("body_names")),
        "joint_names": _list_value(meta.get("joint_names")),
    }


def _build_object_contract(*, local_root: Path, command: Mapping[str, Any]) -> dict[str, Any]:
    asset_name = _optional_str(command.get("object_asset_name"))
    asset_files = _asset_files(local_root, asset_name)
    return {
        "asset_name": asset_name,
        "body_name": _optional_str(command.get("object_body_name")),
        "joint_name": _optional_str(command.get("object_joint_name")),
        "extra_object_names": _list_value(command.get("extra_object_names")),
        "local_asset_files": [str(path.relative_to(local_root)) for path in asset_files],
    }


def _policy_export_command(*, python: str, task_override: str) -> list[str]:
    return [
        python,
        "scripts/play.py",
        f"task={task_override}",
        "checkpoint_path=<checkpoint_path>",
        "export_policy=true",
        "export_policy_exit=true",
        "export_policy_benchmark_iters=0",
        "headless=true",
        "backend=mujoco",
    ]


def _mujoco_motion_viewer_command(*, python: str, motion_dir: str | None) -> list[list[str]]:
    return [
        [python, "scripts/vis/mujoco_mocap_viewer.py"],
        [python, "scripts/vis/motion_data_publisher.py", motion_dir or "<motion_dir>"],
    ]


def _task_files(root: Path | None, task_relative_dir: Path) -> dict[str, Path]:
    if root is None:
        return {}
    task_dir = root / task_relative_dir
    if not task_dir.exists():
        return {}
    return {path.name: path for path in sorted(task_dir.glob("*.yaml"))}


def _task_override_from_path(task_path: Path) -> str:
    parts = task_path.with_suffix("").parts
    if "task" in parts:
        index = parts.index("task")
        return "/".join(parts[index + 1 :])
    return str(task_path.with_suffix(""))


def _semantic_mismatches(
    local_cfg: Mapping[str, Any],
    upstream_cfg: Mapping[str, Any] | None,
    *,
    fields: Sequence[str],
) -> list[dict[str, Any]]:
    if upstream_cfg is None:
        return [{"field": "*", "local": "present", "upstream": "missing"}]
    mismatches = []
    for field in fields:
        local_value = _nested_get(local_cfg, field)
        upstream_value = _nested_get(upstream_cfg, field)
        if local_value != upstream_value:
            mismatches.append({"field": field, "local": local_value, "upstream": upstream_value})
    return mismatches


def _asset_files(root: Path, asset_name: str | None) -> list[Path]:
    if not asset_name:
        return []
    asset_dir = root / "active_adaptation/assets_mjcf/objects" / asset_name
    if not asset_dir.is_dir():
        return []
    return sorted(path for path in asset_dir.rglob("*.xml") if path.is_file())


def _load_yaml_mapping(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected YAML mapping, got {type(data).__name__}.")
    return data


def _load_json_mapping(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected JSON object, got {type(data).__name__}.")
    return data


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list_value(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _nested_get(data: Mapping[str, Any], dotted_path: str) -> Any:
    value: Any = data
    for part in dotted_path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _git_head(root: Path | None) -> str | None:
    if root is None:
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


if __name__ == "__main__":
    raise SystemExit(main())
