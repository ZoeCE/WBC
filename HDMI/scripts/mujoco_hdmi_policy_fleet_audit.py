from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
HDMI_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, HDMI_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mujoco_policy_export_audit import build_policy_export_audit


SOURCE_REFERENCE = {
    "repository": "https://github.com/LeCAR-Lab/HDMI",
    "policy_export_contract": [
        "python scripts/play.py task=<G1/hdmi/task> checkpoint_path=<teacher_or_student_checkpoint>",
        "export_policy=true",
        "export_policy_exit=true",
        "export_onnx_policy=true",
        "export_onnx_required=true",
        "backend=mujoco",
    ],
    "mujoco_viewer_scope": "motion.npz visualization only; closed-loop policy parity still requires exported policy checkpoints.",
}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_fleet_policy_audit(
        task_dir=args.task_dir,
        task_yamls=args.task_yaml,
        exports_dir=args.exports_dir,
        checkpoint_path=args.checkpoint_path,
        checkpoint_manifest=args.checkpoint_manifest,
        robot_name=args.robot_name,
        require_reference_observation=args.require_reference_observation,
        require_obs_action_smoke=args.require_obs_action_smoke,
        require_onnx_policy=args.require_onnx_policy,
        require_checkpoint_algo=args.require_checkpoint_algo,
        min_checkpoint_total_frames=args.min_checkpoint_total_frames,
        expected_task_count=args.expected_task_count,
        require_task_count=args.require_task_count,
    )
    print(json.dumps(report, sort_keys=True))
    if _gate_requested(args) and not report["gate_passed"]:
        return 1
    return 0


def build_fleet_policy_audit(
    *,
    task_dir: str | Path,
    task_yamls: Sequence[str | Path] | None = None,
    exports_dir: str | Path | None = None,
    checkpoint_path: str | None = None,
    checkpoint_manifest: str | Path | None = None,
    robot_name: str = "g1_29dof",
    require_reference_observation: bool = False,
    require_obs_action_smoke: bool = False,
    require_onnx_policy: bool = False,
    require_checkpoint_algo: str | None = None,
    min_checkpoint_total_frames: int | None = None,
    expected_task_count: int | None = None,
    require_task_count: bool = False,
) -> dict[str, Any]:
    tasks = _resolve_task_yamls(task_dir=Path(task_dir), task_yamls=task_yamls or [])
    checkpoint_by_task = _load_checkpoint_manifest(checkpoint_manifest)
    task_reports: list[dict[str, Any]] = []

    for task_path in tasks:
        task_identity = _task_identity(task_path)
        task_checkpoint = _checkpoint_for_task(
            checkpoint_by_task,
            task_identity=task_identity,
            default_checkpoint_path=checkpoint_path,
        )
        task_reports.append(
            build_policy_export_audit(
                task_yaml=task_path,
                exports_dir=exports_dir,
                checkpoint_path=task_checkpoint,
                robot_name=robot_name,
                require_reference_observation=require_reference_observation,
                require_obs_action_smoke=require_obs_action_smoke,
                require_onnx_policy=require_onnx_policy,
                require_checkpoint_algo=require_checkpoint_algo,
                min_checkpoint_total_frames=min_checkpoint_total_frames,
            )
        )

    missing_by_task = {
        str(task_report["task_name"]): list(task_report["missing_requirements"])
        for task_report in task_reports
        if task_report["missing_requirements"]
    }
    task_count = len(task_reports)
    task_count_gate_passed = expected_task_count is None or task_count == expected_task_count
    task_count_missing = require_task_count and not task_count_gate_passed
    ready_task_count = sum(1 for task_report in task_reports if task_report["gate_passed"])
    gate_passed = not missing_by_task and not task_count_missing

    return {
        "source_reference": SOURCE_REFERENCE,
        "task_dir": str(task_dir),
        "task_count": task_count,
        "expected_task_count": expected_task_count,
        "task_count_gate_passed": task_count_gate_passed,
        "ready_task_count": ready_task_count,
        "not_ready_task_count": task_count - ready_task_count,
        "gate_passed": gate_passed,
        "missing_requirements_by_task": missing_by_task,
        "task_overrides": [str(task_report["task_override"]) for task_report in task_reports],
        "tasks": task_reports,
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit HDMI MuJoCo policy export readiness across a task fleet. "
            "This turns the upstream HDMI policy export contract into a repeatable 13-task gate."
        )
    )
    parser.add_argument(
        "--task-dir",
        default=str(HDMI_ROOT / "cfg" / "task" / "G1" / "hdmi"),
        help="Directory containing HDMI task YAMLs. Ignored for discovery when --task-yaml is provided.",
    )
    parser.add_argument(
        "--task-yaml",
        action="append",
        default=[],
        help="Specific task YAML to audit. Can be passed multiple times for a subset.",
    )
    parser.add_argument("--exports-dir", default=None, help="Policy exports root. Defaults to HDMI/scripts/exports.")
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Fallback checkpoint_path value used for all tasks unless --checkpoint-manifest overrides a task.",
    )
    parser.add_argument(
        "--checkpoint-manifest",
        default=None,
        help=(
            "JSON/YAML mapping from task name, task override, task stem, task filename, or task path "
            "to checkpoint_path."
        ),
    )
    parser.add_argument("--robot-name", default="g1_29dof", help="MuJoCo robot registry name.")
    parser.add_argument("--expected-task-count", type=int, default=None)
    parser.add_argument("--require-task-count", action="store_true")
    parser.add_argument("--require-policy", action="store_true")
    parser.add_argument("--require-reference-observation", action="store_true")
    parser.add_argument("--require-obs-action-smoke", action="store_true")
    parser.add_argument("--require-onnx-policy", action="store_true")
    parser.add_argument("--require-checkpoint-algo", default=None)
    parser.add_argument("--min-checkpoint-total-frames", type=int, default=None)
    return parser.parse_args(argv)


def _resolve_task_yamls(*, task_dir: Path, task_yamls: Sequence[str | Path]) -> list[Path]:
    if task_yamls:
        return sorted((Path(path) for path in task_yamls), key=lambda path: str(path))
    return sorted(task_dir.glob("*.yaml"))


def _load_checkpoint_manifest(path: str | Path | None) -> Mapping[str, str]:
    if path is None:
        return {}
    manifest_path = Path(path)
    with manifest_path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, Mapping):
        raise TypeError(f"{manifest_path}: expected checkpoint manifest mapping, got {type(data).__name__}.")
    return {str(key): str(value) for key, value in data.items()}


def _checkpoint_for_task(
    checkpoint_by_task: Mapping[str, str],
    *,
    task_identity: Mapping[str, str],
    default_checkpoint_path: str | None,
) -> str | None:
    for key in (
        task_identity["task_name"],
        task_identity["task_override"],
        task_identity["task_stem"],
        task_identity["task_file"],
        task_identity["task_path"],
    ):
        if key in checkpoint_by_task:
            return checkpoint_by_task[key]
    return default_checkpoint_path


def _task_identity(task_path: Path) -> dict[str, str]:
    task_cfg = _load_yaml_mapping(task_path)
    task_name = str(task_cfg.get("name") or task_path.stem)
    return {
        "task_name": task_name,
        "task_override": _task_override_from_path(task_path),
        "task_stem": task_path.stem,
        "task_file": task_path.name,
        "task_path": str(task_path),
    }


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


def _gate_requested(args: argparse.Namespace) -> bool:
    return (
        args.require_task_count
        or args.require_policy
        or args.require_reference_observation
        or args.require_obs_action_smoke
        or args.require_onnx_policy
        or args.require_checkpoint_algo is not None
        or args.min_checkpoint_total_frames is not None
    )


if __name__ == "__main__":
    raise SystemExit(main())
