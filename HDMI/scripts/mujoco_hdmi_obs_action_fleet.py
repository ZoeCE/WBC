from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
HDMI_ROOT = SCRIPT_DIR.parent
DEFAULT_TASK_DIR = HDMI_ROOT / "cfg" / "task" / "G1" / "hdmi"
DEFAULT_EXPORTS_DIR = HDMI_ROOT / "scripts" / "exports"
DEFAULT_OUTPUT_DIR = Path("/tmp/wbc_hdmi_goal_full_parity_v2/obs_action")
DEFAULT_PYTHON = "/home/zoe/miniconda3/envs/wbc/bin/python"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_obs_action_fleet_report(
        task_dir=Path(args.task_dir),
        task_yamls=[Path(path) for path in args.task_yaml],
        exports_dir=Path(args.exports_dir),
        external_policy_source_report=Path(args.external_policy_source_report)
        if args.external_policy_source_report
        else None,
        existing_obs_action_reports=[Path(path) for path in args.existing_obs_action_report],
        output_dir=Path(args.output_dir),
        python=args.python,
        steps=args.steps,
        action_source=args.action_source,
        max_abs=args.max_abs,
        expected_task_count=args.expected_task_count,
        require_task_count=args.require_task_count,
        require_all_passed=args.require_all_passed,
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if _gate_requested(args):
        return 0 if report["requested_gate_passed"] else 1
    return 0


def build_obs_action_fleet_report(
    *,
    task_dir: Path = DEFAULT_TASK_DIR,
    task_yamls: Sequence[Path] = (),
    exports_dir: Path = DEFAULT_EXPORTS_DIR,
    external_policy_source_report: Path | None = None,
    existing_obs_action_reports: Sequence[Path] = (),
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    python: str = DEFAULT_PYTHON,
    steps: int = 4,
    action_source: str = "zero",
    max_abs: float = 1e-6,
    expected_task_count: int | None = None,
    require_task_count: bool = False,
    require_all_passed: bool = False,
) -> dict[str, Any]:
    task_paths = _resolve_task_paths(task_dir=task_dir, task_yamls=task_yamls)
    existing_by_key = _existing_obs_action_by_key(existing_obs_action_reports)
    external_by_key = _external_policies_by_key(external_policy_source_report)
    tasks = []
    for task_path in task_paths:
        identity = _task_identity(task_path)
        existing = _entry_for_identity(existing_by_key, identity)
        golden_policies = _policy_exports_for_task(exports_dir, identity["task_name"])
        selected_policy = golden_policies[-1] if golden_policies else None
        policy_source = "golden_export" if selected_policy is not None else None
        external_policy = _entry_for_identity(external_by_key, identity)
        if selected_policy is None and external_policy:
            selected_policy = Path(str(external_policy["policy_path"]))
            policy_source = "external"
        tasks.append(
            _task_report(
                identity=identity,
                existing=existing,
                selected_policy=selected_policy,
                policy_source=policy_source,
                python=python,
                output_dir=output_dir,
                steps=steps,
                action_source=action_source,
                max_abs=max_abs,
            )
        )

    task_count = len(tasks)
    passed_count = sum(1 for task in tasks if task["gate_passed"])
    runnable_count = sum(1 for task in tasks if task["status"] == "needs_obs_action_run")
    missing_policy_count = sum(1 for task in tasks if task["status"] == "missing_policy_export")
    failed_count = sum(1 for task in tasks if task["status"] == "failed_existing_obs_action")
    task_count_gate_passed = expected_task_count is None or task_count == expected_task_count
    all_passed = task_count > 0 and passed_count == task_count
    gate_passed = task_count_gate_passed and all_passed
    requested_gate_passed = (not require_task_count or task_count_gate_passed) and (not require_all_passed or all_passed)

    return {
        "task_dir": str(task_dir),
        "exports_dir": str(exports_dir),
        "external_policy_source_report": str(external_policy_source_report)
        if external_policy_source_report
        else None,
        "output_dir": str(output_dir),
        "existing_obs_action_reports": [str(path) for path in existing_obs_action_reports],
        "task_count": task_count,
        "expected_task_count": expected_task_count,
        "task_count_gate_passed": task_count_gate_passed,
        "passed_task_count": passed_count,
        "runnable_task_count": runnable_count,
        "missing_policy_task_count": missing_policy_count,
        "failed_task_count": failed_count,
        "all_passed": all_passed,
        "gate_passed": gate_passed,
        "requested_gate_passed": requested_gate_passed,
        "tasks": tasks,
    }


def _task_report(
    *,
    identity: Mapping[str, str],
    existing: Mapping[str, Any],
    selected_policy: Path | None,
    policy_source: str | None,
    python: str,
    output_dir: Path,
    steps: int,
    action_source: str,
    max_abs: float,
) -> dict[str, Any]:
    if existing:
        gate_passed = existing.get("gate_passed") is True
        status = "passed_existing_obs_action" if gate_passed else "failed_existing_obs_action"
    elif selected_policy is not None:
        gate_passed = False
        status = "needs_obs_action_run"
    else:
        gate_passed = False
        status = "missing_policy_export"

    report_path = output_dir / f"{identity['task_stem']}_obs_action_parity.json"
    metrics = _obs_action_metrics(existing)
    return {
        **identity,
        "status": status,
        "gate_passed": gate_passed,
        "policy_source": policy_source,
        "selected_policy_path": str(selected_policy) if selected_policy is not None else None,
        "existing_report_path": str(existing.get("report_path")) if existing.get("report_path") else None,
        "obs_action_metrics": metrics,
        **metrics,
        "missing_reasons": _missing_reasons(status),
        "planned_report": str(report_path),
        "obs_action_command": _obs_action_command(
            python=python,
            task_yaml=identity["task_path"],
            policy_path=str(selected_policy),
            steps=steps,
            action_source=action_source,
            max_abs=max_abs,
            output=str(report_path),
        )
        if selected_policy is not None
        else None,
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan and aggregate HDMI MuJoCo observation/action parity across the G1/HDMI task fleet."
    )
    parser.add_argument("--task-dir", default=str(DEFAULT_TASK_DIR))
    parser.add_argument("--task-yaml", action="append", default=[])
    parser.add_argument("--exports-dir", default=str(DEFAULT_EXPORTS_DIR))
    parser.add_argument("--external-policy-source-report", default=None)
    parser.add_argument("--existing-obs-action-report", action="append", default=[])
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--action-source", choices=("zero", "ramp"), default="zero")
    parser.add_argument("--max-abs", type=float, default=1e-6)
    parser.add_argument("--expected-task-count", type=int, default=None)
    parser.add_argument("--require-task-count", action="store_true")
    parser.add_argument("--require-all-passed", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


def _resolve_task_paths(*, task_dir: Path, task_yamls: Sequence[Path]) -> list[Path]:
    if task_yamls:
        return sorted(task_yamls, key=lambda path: str(path))
    return sorted(task_dir.glob("*.yaml"))


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
    return str(task_path.with_suffix(""))


def _existing_obs_action_by_key(report_paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for report_path in report_paths:
        if not report_path.is_file():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        entries: list[Any]
        if isinstance(report, Mapping):
            entries = report.get("tasks") if isinstance(report.get("tasks"), list) else [report]
        else:
            entries = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            value = {**entry, "report_path": str(report_path)}
            for field in ("task_name", "task_override", "task_stem", "task", "task_path", "task_yaml"):
                key = value.get(field)
                if key is not None:
                    by_key[str(key)] = value
    return by_key


def _external_policies_by_key(report_path: Path | None) -> dict[str, dict[str, Any]]:
    if report_path is None or not report_path.is_file():
        return {}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    tasks = report.get("tasks") if isinstance(report, Mapping) else None
    if not isinstance(tasks, list):
        return {}
    by_key: dict[str, dict[str, Any]] = {}
    for task in tasks:
        if not isinstance(task, Mapping):
            continue
        if task.get("policy_source_ready") is not True or task.get("policy_path") is None:
            continue
        value = dict(task)
        for field in ("task_name", "task_override", "task_stem", "task", "task_path", "task_file"):
            key = value.get(field)
            if key is not None:
                by_key[str(key)] = value
    return by_key


def _entry_for_identity(
    entries_by_key: Mapping[str, dict[str, Any]],
    identity: Mapping[str, str],
) -> dict[str, Any]:
    for key in (
        identity["task_name"],
        identity["task_override"],
        identity["task_stem"],
        identity["task_file"],
        identity["task_path"],
    ):
        if key in entries_by_key:
            return entries_by_key[key]
    return {}


def _policy_exports_for_task(exports_dir: Path, task_name: str) -> list[Path]:
    task_dir = exports_dir / task_name
    if not task_dir.is_dir():
        return []
    return sorted(task_dir.glob("policy-*.pt"), key=lambda path: str(path))


def _obs_action_metrics(entry: Mapping[str, Any]) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}
    for key in ("max_abs", "mean_abs"):
        value = entry.get(key)
        if value is None:
            continue
        try:
            metrics[key] = float(value)
        except (TypeError, ValueError):
            continue
    for key in ("failure_count", "compared_component_count"):
        value = entry.get(key)
        if value is None:
            continue
        try:
            metrics[key] = int(value)
        except (TypeError, ValueError):
            continue
    return metrics


def _missing_reasons(status: str) -> list[str]:
    if status == "passed_existing_obs_action":
        return []
    if status == "failed_existing_obs_action":
        return ["obs_action_parity_failed"]
    if status == "needs_obs_action_run":
        return ["missing_obs_action_report"]
    if status == "missing_policy_export":
        return ["missing_policy_export"]
    return [status]


def _obs_action_command(
    *,
    python: str,
    task_yaml: str,
    policy_path: str,
    steps: int,
    action_source: str,
    max_abs: float,
    output: str,
) -> list[str]:
    return [
        python,
        "scripts/mujoco_obs_builder_parity.py",
        "--task-yaml",
        task_yaml,
        "--policy-path",
        policy_path,
        "--steps",
        str(steps),
        "--action-source",
        action_source,
        "--max-abs",
        f"{max_abs:g}",
        "--output",
        output,
    ]


def _gate_requested(args: argparse.Namespace) -> bool:
    return args.require_task_count or args.require_all_passed


if __name__ == "__main__":
    raise SystemExit(main())
