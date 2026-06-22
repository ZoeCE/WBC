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
DEFAULT_EXPORTS_DIR = HDMI_ROOT / "scripts" / "exports"
DEFAULT_OUTPUT_DIR = Path("/tmp/wbc_hdmi_goal_full_parity_v2/open_loop")
DEFAULT_PYTHON = "/home/zoe/miniconda3/envs/wbc/bin/python"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_open_loop_fleet_report(
        task_dir=Path(args.task_dir),
        task_yamls=[Path(path) for path in args.task_yaml],
        exports_dir=Path(args.exports_dir),
        existing_open_loop_reports=[Path(path) for path in args.existing_open_loop_report],
        external_policy_source_report=Path(args.external_policy_source_report)
        if args.external_policy_source_report
        else None,
        output_dir=Path(args.output_dir),
        python=args.python,
        steps=args.steps,
        horizon=args.horizon,
        expected_task_count=args.expected_task_count,
        require_task_count=args.require_task_count,
        require_all_passed=args.require_all_passed,
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, sort_keys=True))
    return 0 if report["gate_passed"] else 1 if _gate_requested(args) else 0


def build_open_loop_fleet_report(
    *,
    task_dir: Path = DEFAULT_TASK_DIR,
    task_yamls: Sequence[Path] = (),
    exports_dir: Path = DEFAULT_EXPORTS_DIR,
    existing_open_loop_reports: Sequence[Path] = (),
    external_policy_source_report: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    python: str = DEFAULT_PYTHON,
    steps: str = "0,1",
    horizon: str = "all",
    expected_task_count: int | None = None,
    require_task_count: bool = False,
    require_all_passed: bool = False,
) -> dict[str, Any]:
    task_paths = _resolve_task_paths(task_dir=task_dir, task_yamls=task_yamls)
    existing_by_key = _existing_open_loop_by_key(existing_open_loop_reports)
    external_by_key = _external_policies_by_key(external_policy_source_report)
    tasks = []
    for task_path in task_paths:
        identity = _task_identity(task_path)
        existing = _entry_for_identity(existing_by_key, identity)
        policies = _policy_exports_for_task(exports_dir, identity["task_name"])
        selected_policy = policies[-1] if policies else None
        policy_source = "golden_export" if selected_policy is not None else None
        external_policy = _entry_for_identity(external_by_key, identity)
        if selected_policy is None and external_policy:
            selected_policy = Path(str(external_policy["policy_path"]))
            policy_source = "external"
        tasks.append(
            _task_report(
                identity=identity,
                existing=existing,
                policies=policies,
                selected_policy=selected_policy,
                policy_source=policy_source,
                python=python,
                output_dir=output_dir,
                steps=steps,
                horizon=horizon,
            )
        )

    task_count = len(tasks)
    passed_count = sum(1 for task in tasks if task["gate_passed"])
    runnable_count = sum(1 for task in tasks if task["status"] == "needs_open_loop_run")
    missing_policy_count = sum(1 for task in tasks if task["status"] == "missing_policy_export")
    failed_count = sum(1 for task in tasks if task["status"] == "failed_existing_open_loop")
    task_count_gate_passed = expected_task_count is None or task_count == expected_task_count
    all_passed = task_count > 0 and passed_count == task_count
    gate_passed = (not require_task_count or task_count_gate_passed) and (not require_all_passed or all_passed)

    return {
        "task_dir": str(task_dir),
        "exports_dir": str(exports_dir),
        "output_dir": str(output_dir),
        "external_policy_source_report": str(external_policy_source_report)
        if external_policy_source_report
        else None,
        "existing_open_loop_reports": [str(path) for path in existing_open_loop_reports],
        "task_count": task_count,
        "expected_task_count": expected_task_count,
        "task_count_gate_passed": task_count_gate_passed,
        "passed_task_count": passed_count,
        "runnable_task_count": runnable_count,
        "missing_policy_task_count": missing_policy_count,
        "failed_task_count": failed_count,
        "all_passed": all_passed,
        "gate_passed": gate_passed,
        "tasks": tasks,
    }


def _task_report(
    *,
    identity: Mapping[str, str],
    existing: Mapping[str, Any],
    policies: Sequence[Path],
    selected_policy: Path | None,
    policy_source: str | None,
    python: str,
    output_dir: Path,
    steps: str,
    horizon: str,
) -> dict[str, Any]:
    if existing:
        gate_passed = _open_loop_passed(existing)
        status = "passed_existing_open_loop" if gate_passed else "failed_existing_open_loop"
    elif selected_policy is not None:
        gate_passed = False
        status = "needs_open_loop_run"
    else:
        gate_passed = False
        status = "missing_policy_export"

    trace_path = output_dir / f"{identity['task_stem']}_rollout_trace.json"
    playback_path = output_dir / f"{identity['task_stem']}_rollout.json"
    horizon_path = output_dir / f"{identity['task_stem']}_rollout_horizon_sweep.json"
    metrics = _open_loop_metrics(existing)
    report = {
        **identity,
        "status": status,
        "gate_passed": gate_passed,
        "policy_candidates": [str(path) for path in policies],
        "policy_source": policy_source,
        "selected_policy_path": str(selected_policy) if selected_policy is not None else None,
        "existing_report_path": str(existing.get("report_path")) if existing.get("report_path") else None,
        "open_loop_metrics": metrics,
        **metrics,
        "missing_reasons": _missing_reasons(status),
        "playback_command": None,
        "horizon_sweep_command": None,
        "planned_playback_report": str(playback_path),
        "planned_trace_json": str(trace_path),
        "planned_horizon_report": str(horizon_path),
    }
    if selected_policy is not None:
        report["playback_command"] = _playback_command(
            python=python,
            task_yaml=identity["task_path"],
            policy_path=str(selected_policy),
            trace_path=str(trace_path),
            steps=steps,
        )
        report["horizon_sweep_command"] = _horizon_sweep_command(
            python=python,
            trace_path=str(trace_path),
            horizon=horizon,
        )
    return report


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan and aggregate HDMI MuJoCo open-loop dynamics parity across the G1/HDMI task fleet."
    )
    parser.add_argument("--task-dir", default=str(DEFAULT_TASK_DIR))
    parser.add_argument("--task-yaml", action="append", default=[])
    parser.add_argument("--exports-dir", default=str(DEFAULT_EXPORTS_DIR))
    parser.add_argument("--existing-open-loop-report", action="append", default=[])
    parser.add_argument("--external-policy-source-report", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--steps", default="0,1")
    parser.add_argument("--horizon", default="all")
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


def _existing_open_loop_by_key(report_paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for report_path in report_paths:
        if not report_path.is_file():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        entries = report.get("tasks") if isinstance(report, dict) else None
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            entry = {**entry}
            entry.setdefault("report_path", str(report_path))
            for field in ("task_name", "task_override", "task_stem", "task", "task_path"):
                value = entry.get(field)
                if value is not None:
                    by_key[str(value)] = entry
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


def _policy_exports_for_task(exports_dir: Path, task_name: str) -> list[Path]:
    task_dir = exports_dir / task_name
    if not task_dir.is_dir():
        return []
    return sorted(task_dir.glob("policy-*.pt"), key=lambda path: str(path))


def _open_loop_passed(entry: Mapping[str, Any]) -> bool:
    return entry.get("gate_passed") is True or entry.get("open_loop_passed") is True or entry.get("passed") is True


def _open_loop_metrics(entry: Mapping[str, Any]) -> dict[str, float]:
    metrics = {}
    for key in (
        "policy_rollout_q_l2_max",
        "policy_rollout_body_pos_l2_max",
        "policy_rollout_reward_mean",
    ):
        value = entry.get(key)
        if value is None:
            continue
        try:
            metrics[key] = float(value)
        except (TypeError, ValueError):
            continue
    return metrics


def _missing_reasons(status: str) -> list[str]:
    if status == "missing_policy_export":
        return ["missing_policy_export"]
    if status == "needs_open_loop_run":
        return ["missing_open_loop_report"]
    if status == "failed_existing_open_loop":
        return ["open_loop_gate_failed"]
    return []


def _playback_command(*, python: str, task_yaml: str, policy_path: str, trace_path: str, steps: str) -> list[str]:
    return [
        python,
        "scripts/mujoco_playback_parity.py",
        "--task-yaml",
        task_yaml,
        "--policy-path",
        policy_path,
        "--policy-rollout",
        "--require-reference-observation",
        "--trace-json",
        trace_path,
        "--steps",
        steps,
        "--max-q-l2",
        "1e-6",
        "--max-body-pos-l2",
        "1e-5",
        "--min-reward-mean",
        "0.0",
        "--max-policy-rollout-q-l2",
        "1.0",
        "--max-policy-rollout-body-pos-l2",
        "0.05",
        "--min-policy-rollout-reward-mean",
        "0.0",
    ]


def _horizon_sweep_command(*, python: str, trace_path: str, horizon: str) -> list[str]:
    return [
        python,
        "scripts/mujoco_horizon_sweep.py",
        trace_path,
        "--horizon",
        horizon,
        "--max-q-l2",
        "1e-6",
        "--max-body-pos-l2",
        "1e-5",
        "--min-reward-mean",
        "0.0",
        "--max-policy-rollout-q-l2",
        "1.0",
        "--max-policy-rollout-body-pos-l2",
        "0.05",
        "--min-policy-rollout-reward-mean",
        "0.0",
    ]


def _gate_requested(args: argparse.Namespace) -> bool:
    return bool(args.require_task_count or args.require_all_passed)


if __name__ == "__main__":
    raise SystemExit(main())
