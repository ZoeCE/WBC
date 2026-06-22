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
DEFAULT_TASK_DIR = HDMI_ROOT / "cfg" / "task" / "G1" / "hdmi"
DEFAULT_PYTHON = "/home/zoe/miniconda3/envs/wbc/bin/python"

for path in (SCRIPT_DIR, HDMI_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


SOURCE_REFERENCE = {
    "repository": "https://github.com/LeCAR-Lab/HDMI",
    "policy_export_contract": [
        "python scripts/play.py task=<G1/hdmi/task> checkpoint_path=<teacher_or_student_checkpoint>",
        "export_policy=true",
        "export_policy_exit=true",
        "backend=mujoco",
    ],
    "scope": (
        "This plan starts after original HDMI teacher/student checkpoints are available. "
        "It does not treat existing exported policy filenames as proof of trained checkpoint provenance."
    ),
}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_policy_export_plan(
        task_dir=Path(args.task_dir),
        task_yamls=[Path(path) for path in args.task_yaml],
        checkpoint_manifest=Path(args.checkpoint_manifest) if args.checkpoint_manifest else None,
        exports_dir=Path(args.exports_dir) if args.exports_dir else HDMI_ROOT / "scripts" / "exports",
        python=args.python,
        algo=args.algo,
        expected_task_count=args.expected_task_count,
        require_task_count=args.require_task_count,
        require_all_checkpoints=args.require_all_checkpoints,
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, sort_keys=True))
    if _gate_requested(args) and not report["gate_passed"]:
        return 1
    return 0


def build_policy_export_plan(
    *,
    task_dir: Path = DEFAULT_TASK_DIR,
    task_yamls: Sequence[Path] = (),
    checkpoint_manifest: Path | None = None,
    exports_dir: Path = HDMI_ROOT / "scripts" / "exports",
    python: str = DEFAULT_PYTHON,
    algo: str | None = None,
    expected_task_count: int | None = None,
    require_task_count: bool = False,
    require_all_checkpoints: bool = False,
) -> dict[str, Any]:
    task_paths = _resolve_task_paths(task_dir=task_dir, task_yamls=task_yamls)
    checkpoint_by_task = _load_checkpoint_manifest(checkpoint_manifest)
    tasks = []
    manifest_template: dict[str, str] = {}

    for task_path in task_paths:
        identity = _task_identity(task_path)
        checkpoint_path = _checkpoint_for_task(checkpoint_by_task, identity=identity)
        missing_checkpoint = checkpoint_path is None
        if missing_checkpoint:
            manifest_template[identity["task_override"]] = "<checkpoint_path>"
        tasks.append(
            {
                **identity,
                "checkpoint_path": checkpoint_path,
                "missing_checkpoint": missing_checkpoint,
                "expected_export_dir": str(exports_dir / identity["task_name"]),
                "export_command": _export_command(
                    python=python,
                    task_override=identity["task_override"],
                    checkpoint_path=checkpoint_path,
                    algo=algo,
                ),
                "checkpoint_manifest_keys": [
                    identity["task_name"],
                    identity["task_override"],
                    identity["task_stem"],
                    identity["task_file"],
                    identity["task_path"],
                ],
            }
        )

    task_count = len(tasks)
    provided_count = sum(1 for task in tasks if not task["missing_checkpoint"])
    missing_count = task_count - provided_count
    task_count_gate_passed = expected_task_count is None or task_count == expected_task_count
    ready_to_export = missing_count == 0
    gate_passed = ready_to_export
    if require_task_count and not task_count_gate_passed:
        gate_passed = False
    if require_all_checkpoints and not ready_to_export:
        gate_passed = False

    return {
        "source_reference": SOURCE_REFERENCE,
        "task_dir": str(task_dir),
        "task_count": task_count,
        "expected_task_count": expected_task_count,
        "task_count_gate_passed": task_count_gate_passed,
        "checkpoint_manifest_path": str(checkpoint_manifest) if checkpoint_manifest is not None else None,
        "checkpoint_manifest_template": manifest_template,
        "provided_checkpoint_task_count": provided_count,
        "missing_checkpoint_task_count": missing_count,
        "ready_to_export": ready_to_export,
        "gate_passed": gate_passed,
        "exports_dir": str(exports_dir),
        "tasks": tasks,
        "fleet_policy_audit_command": _fleet_policy_audit_command(
            python=python,
            task_dir=task_dir,
            task_yamls=task_paths if task_yamls else [],
            exports_dir=exports_dir,
            checkpoint_manifest=checkpoint_manifest,
            expected_task_count=expected_task_count,
            require_task_count=require_task_count,
        ),
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the HDMI policy export and fleet-audit command plan that should be run once "
            "original HDMI/Isaac teacher or student checkpoints are available."
        )
    )
    parser.add_argument("--task-dir", default=str(DEFAULT_TASK_DIR))
    parser.add_argument("--task-yaml", action="append", default=[])
    parser.add_argument("--checkpoint-manifest", default=None)
    parser.add_argument("--exports-dir", default=None)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--algo", default=None)
    parser.add_argument("--expected-task-count", type=int, default=None)
    parser.add_argument("--require-task-count", action="store_true")
    parser.add_argument("--require-all-checkpoints", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


def _resolve_task_paths(*, task_dir: Path, task_yamls: Sequence[Path]) -> list[Path]:
    if task_yamls:
        return sorted(task_yamls, key=lambda path: str(path))
    return sorted(task_dir.glob("*.yaml"))


def _load_checkpoint_manifest(path: Path | None) -> Mapping[str, str]:
    if path is None:
        return {}
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, Mapping):
        raise TypeError(f"{path}: expected checkpoint manifest mapping, got {type(data).__name__}.")
    return {str(key): str(value) for key, value in data.items()}


def _task_identity(task_path: Path) -> dict[str, str]:
    cfg = _load_yaml_mapping(task_path)
    task_name = str(cfg.get("name") or task_path.stem)
    return {
        "task_name": task_name,
        "task_override": _task_override_from_path(task_path),
        "task_stem": task_path.stem,
        "task_file": task_path.name,
        "task_path": str(task_path),
    }


def _checkpoint_for_task(checkpoint_by_task: Mapping[str, str], *, identity: Mapping[str, str]) -> str | None:
    for key in (
        identity["task_name"],
        identity["task_override"],
        identity["task_stem"],
        identity["task_file"],
        identity["task_path"],
    ):
        if key in checkpoint_by_task:
            return checkpoint_by_task[key]
    return None


def _export_command(*, python: str, task_override: str, checkpoint_path: str | None, algo: str | None) -> list[str]:
    command = [python, "scripts/play.py"]
    if algo:
        command.append(f"algo={algo}")
    command.extend(
        [
            f"task={task_override}",
            f"checkpoint_path={checkpoint_path or '<checkpoint_path>'}",
            "export_policy=true",
            "export_policy_exit=true",
            "export_policy_benchmark_iters=0",
            "export_onnx_policy=false",
            "headless=true",
            "backend=mujoco",
        ]
    )
    return command


def _fleet_policy_audit_command(
    *,
    python: str,
    task_dir: Path,
    task_yamls: Sequence[Path],
    exports_dir: Path,
    checkpoint_manifest: Path | None,
    expected_task_count: int | None,
    require_task_count: bool,
) -> list[str]:
    command = [python, "scripts/mujoco_hdmi_policy_fleet_audit.py"]
    if task_yamls:
        for task_path in task_yamls:
            command.extend(["--task-yaml", str(task_path)])
    else:
        command.extend(["--task-dir", str(task_dir)])
    command.extend(["--exports-dir", str(exports_dir)])
    if checkpoint_manifest is not None:
        command.extend(["--checkpoint-manifest", str(checkpoint_manifest)])
    command.extend(["--require-policy", "--require-reference-observation", "--require-obs-action-smoke"])
    if expected_task_count is not None:
        command.extend(["--expected-task-count", str(int(expected_task_count))])
    if require_task_count:
        command.append("--require-task-count")
    return command


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
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
    return bool(args.require_task_count or args.require_all_checkpoints)


if __name__ == "__main__":
    raise SystemExit(main())
