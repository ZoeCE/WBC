#!/usr/bin/env python3
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

from mujoco_hdmi_external_policy_source_audit import DEFAULT_TASK_DIR, build_external_policy_source_audit
from mujoco_hdmi_sim2real_runner import (
    DEFAULT_SIM2REAL_ROOT,
    official_scenario_specs,
    run_scenario,
    run_wbc_export_scenario,
)


DEFAULT_WBC_ROOT = HDMI_ROOT
DEFAULT_WBC_EXPORTS_DIR = HDMI_ROOT / "scripts" / "exports"
DEFAULT_OUTPUT_DIR = Path("/tmp/wbc_hdmi_goal_full_parity_v2/sim2real_fleet")
DEFAULT_WBC_EXPORT_OUTPUT_DIR = Path("/tmp/wbc_hdmi_goal_full_parity_v2/wbc_export_sim2real_fleet")
DEFAULT_WBC_PARITY_THRESHOLDS = {
    "q_ref_l2_mean": 0.5,
    "body_pos_ref_l2_mean": 0.5,
    "object_pos_ref_l2_mean": 0.5,
}
RUNNABLE_POLICY_SET_TO_SCENARIO = {
    "G1Dance1Subject2": "G1Dance1Subject2",
    "G1TrackSuitcase": "G1TrackSuitcase",
    "G1PushDoorHand": "G1PushDoorHand",
    "G1RollBall": "G1RollBall",
}


def build_wbc_export_fleet_plan(
    *,
    task_dir: str | Path = DEFAULT_TASK_DIR,
    exports_dir: str | Path = DEFAULT_WBC_EXPORTS_DIR,
    expected_task_count: int | None = None,
) -> dict[str, Any]:
    task_dir = Path(task_dir)
    exports_dir = Path(exports_dir)
    tasks = [
        _wbc_export_task_row(task_path=task_path, exports_dir=exports_dir)
        for task_path in sorted(task_dir.glob("*.yaml"))
    ]
    task_count = len(tasks)
    task_count_gate_passed = expected_task_count is None or task_count == int(expected_task_count)
    return {
        "task_count": task_count,
        "expected_task_count": expected_task_count,
        "task_count_gate_passed": task_count_gate_passed,
        "ready_task_count": sum(1 for task in tasks if task["ready"]),
        "not_ready_task_count": sum(1 for task in tasks if not task["ready"]),
        "tasks": tasks,
    }


def run_wbc_export_fleet(
    *,
    sim2real_root: str | Path = DEFAULT_SIM2REAL_ROOT,
    wbc_root: str | Path = DEFAULT_WBC_ROOT,
    task_dir: str | Path = DEFAULT_TASK_DIR,
    exports_dir: str | Path = DEFAULT_WBC_EXPORTS_DIR,
    output_dir: str | Path = DEFAULT_WBC_EXPORT_OUTPUT_DIR,
    duration_sec: float = 6.0,
    render_video: bool = False,
    keep_going: bool = True,
    fall_height: float = 0.4,
    wbc_initial_pause_sec: float = 0.0,
    parity_thresholds: Mapping[str, float] = DEFAULT_WBC_PARITY_THRESHOLDS,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = build_wbc_export_fleet_plan(task_dir=task_dir, exports_dir=exports_dir, expected_task_count=13)
    task_reports = []
    for task in plan["tasks"]:
        task_report = dict(task)
        if not task["ready"]:
            task_reports.append(task_report)
            continue
        try:
            summary = run_wbc_export_scenario(
                sim2real_root=sim2real_root,
                wbc_root=wbc_root,
                task_yaml=task["task_yaml"],
                policy_model=task["policy_model"],
                output_dir=output_dir / str(task["task_name"]),
                duration_sec=duration_sec,
                render_video=render_video,
                fall_height=fall_height,
                wbc_initial_state="reference_frame",
                wbc_initial_step=0,
                wbc_initial_pause_sec=wbc_initial_pause_sec,
            )
            task_report = wbc_export_task_report_from_summary(
                task_report,
                summary,
                parity_thresholds=parity_thresholds,
            )
        except Exception as exc:
            task_report["closed_loop_attempted"] = True
            task_report["stability_success"] = False
            task_report["reference_parity_success"] = False
            task_report["status"] = "closed_loop_error"
            task_report["error"] = f"{type(exc).__name__}: {exc}"
            if not keep_going:
                task_reports.append(task_report)
                break
        task_reports.append(task_report)
    aggregate = _aggregate_wbc_export_fleet(plan=plan, task_reports=task_reports)
    output_path = output_dir / "wbc_export_fleet_summary.json"
    output_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")
    aggregate["output_path"] = str(output_path)
    return aggregate


def wbc_export_task_report_from_summary(
    task: Mapping[str, Any],
    closed_loop: Mapping[str, Any],
    *,
    parity_thresholds: Mapping[str, float] = DEFAULT_WBC_PARITY_THRESHOLDS,
) -> dict[str, Any]:
    report = dict(task)
    report["closed_loop"] = dict(closed_loop)
    report["closed_loop_attempted"] = True
    report["stability_success"] = bool(closed_loop.get("heuristic_success"))
    reference_tracking = closed_loop.get("reference_tracking") if isinstance(closed_loop.get("reference_tracking"), Mapping) else {}
    parity_failures = _reference_parity_failures(reference_tracking, parity_thresholds=parity_thresholds)
    report["reference_parity_success"] = bool(reference_tracking.get("available") is True and not parity_failures)
    report["parity_failures"] = parity_failures
    if not report["stability_success"]:
        report["status"] = "stability_failed"
    elif not report["reference_parity_success"]:
        report["status"] = "reference_parity_failed"
    else:
        report["status"] = "reference_parity_success"
    return report


def build_fleet_plan(audit_report: Mapping[str, Any]) -> dict[str, Any]:
    tasks = [_fleet_task_row(task) for task in audit_report.get("tasks", [])]
    return {
        "task_count": len(tasks),
        "runnable_task_count": sum(1 for task in tasks if task["runnable"]),
        "missing_checkpoint_count": sum(1 for task in tasks if task["status"] == "missing_external_policy"),
        "unsupported_policy_count": sum(1 for task in tasks if task["status"] == "unsupported_external_policy"),
        "policy_audit_failed_count": sum(1 for task in tasks if task["status"] == "policy_audit_failed"),
        "tasks": tasks,
    }


def run_fleet(
    *,
    sim2real_root: str | Path = DEFAULT_SIM2REAL_ROOT,
    task_dir: str | Path = DEFAULT_TASK_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    duration_sec: float = 6.0,
    render_video: bool = False,
    keep_going: bool = True,
    fall_height: float = 0.4,
) -> dict[str, Any]:
    sim2real_root = Path(sim2real_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = build_external_policy_source_audit(
        policy_root=sim2real_root / "checkpoints",
        task_dir=Path(task_dir),
        expected_task_count=13,
        require_task_count=True,
        require_reference_observation=True,
        require_obs_action_smoke=True,
    )
    plan = build_fleet_plan(audit)
    scenarios = official_scenario_specs()
    task_reports = []
    for task in plan["tasks"]:
        task_report = dict(task)
        if not task["runnable"]:
            task_reports.append(task_report)
            continue
        try:
            summary = run_scenario(
                root=sim2real_root,
                scenario=scenarios[task["scenario"]],
                output_dir=output_dir,
                duration_sec=duration_sec,
                render_video=render_video,
                fall_height=fall_height,
            )
            task_report["closed_loop"] = summary
            task_report["closed_loop_attempted"] = True
            task_report["closed_loop_success"] = bool(summary.get("heuristic_success"))
            task_report["status"] = "closed_loop_success" if task_report["closed_loop_success"] else "closed_loop_failed"
        except Exception as exc:
            task_report["closed_loop_attempted"] = True
            task_report["closed_loop_success"] = False
            task_report["status"] = "closed_loop_error"
            task_report["error"] = f"{type(exc).__name__}: {exc}"
            if not keep_going:
                task_reports.append(task_report)
                break
        task_reports.append(task_report)
    aggregate = _aggregate_fleet(plan=plan, task_reports=task_reports, audit=audit)
    output_path = output_dir / "fleet_summary.json"
    output_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")
    aggregate["output_path"] = str(output_path)
    return aggregate


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.wbc_export_fleet:
        report = run_wbc_export_fleet(
            sim2real_root=args.sim2real_root,
            wbc_root=args.wbc_root,
            task_dir=args.task_dir,
            exports_dir=args.exports_dir,
            output_dir=args.output_dir,
            duration_sec=args.duration_sec,
            render_video=args.video,
            keep_going=args.keep_going,
            fall_height=args.fall_height,
            wbc_initial_pause_sec=args.wbc_initial_pause_sec,
        )
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(report, sort_keys=True))
        if args.require_all_reference_parity and report["reference_parity_success_count"] != report["ready_task_count"]:
            return 1
        if args.require_all_success and report["stability_success_count"] != report["ready_task_count"]:
            return 1
        return 0

    report = run_fleet(
        sim2real_root=args.sim2real_root,
        task_dir=args.task_dir,
        output_dir=args.output_dir,
        duration_sec=args.duration_sec,
        render_video=args.video,
        keep_going=args.keep_going,
        fall_height=args.fall_height,
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    if args.require_all_runnable and report["runnable_task_count"] != report["task_count"]:
        return 1
    if args.require_all_success and report["closed_loop_success_count"] != report["runnable_task_count"]:
        return 1
    return 0


def _fleet_task_row(task: Mapping[str, Any]) -> dict[str, Any]:
    policy_set = task.get("policy_set")
    missing_requirements = list(task.get("missing_requirements") or [])
    scenario = RUNNABLE_POLICY_SET_TO_SCENARIO.get(str(policy_set)) if policy_set else None
    if scenario and task.get("gate_passed") is True:
        status = "runnable"
        runnable = True
    elif "external_policy_for_task_motion" in missing_requirements or not policy_set:
        status = "missing_external_policy"
        runnable = False
    elif scenario is None:
        status = "unsupported_external_policy"
        runnable = False
    else:
        status = "policy_audit_failed"
        runnable = False
    return {
        "task_name": task.get("task_name"),
        "task_override": task.get("task_override"),
        "task_path": task.get("task_path"),
        "motion_path": task.get("motion_path"),
        "policy_set": policy_set,
        "policy_path": task.get("policy_path"),
        "policy_onnx_path": task.get("policy_onnx_path"),
        "scenario": scenario,
        "runnable": runnable,
        "status": status,
        "missing_requirements": missing_requirements,
    }


def _wbc_export_task_row(*, task_path: Path, exports_dir: Path) -> dict[str, Any]:
    task_identity = _task_identity_from_yaml(task_path)
    export_dir = exports_dir / task_identity["task_name"]
    policy_stem = _select_wbc_policy_stem(export_dir)
    missing_requirements: list[str] = []
    if policy_stem is None:
        missing_requirements.extend(["exported_policy_yaml", "exported_policy_onnx", "exported_policy_json"])
        policy_yaml = None
        policy_model = None
        policy_metadata = None
        policy_pt = None
    else:
        policy_yaml = policy_stem.with_suffix(".yaml")
        policy_model = policy_stem.with_suffix(".onnx")
        policy_metadata = policy_stem.with_suffix(".json")
        policy_pt = policy_stem.with_suffix(".pt")
        if not policy_yaml.exists():
            missing_requirements.append("exported_policy_yaml")
        if not policy_model.exists():
            missing_requirements.append("exported_policy_onnx")
        if not policy_metadata.exists():
            missing_requirements.append("exported_policy_json")
    ready = not missing_requirements
    return {
        "task_name": task_identity["task_name"],
        "task_override": task_identity["task_override"],
        "task_yaml": str(task_path),
        "export_dir": str(export_dir),
        "policy_yaml": str(policy_yaml) if policy_yaml is not None else None,
        "policy_model": str(policy_model) if policy_model is not None else None,
        "policy_metadata": str(policy_metadata) if policy_metadata is not None else None,
        "policy_pt": str(policy_pt) if policy_pt is not None and policy_pt.exists() else None,
        "ready": ready,
        "status": "ready" if ready else "missing_wbc_export",
        "missing_requirements": missing_requirements,
    }


def _task_identity_from_yaml(task_path: Path) -> dict[str, str]:
    task_cfg = _load_yaml_mapping(task_path)
    task_name = str(task_cfg.get("name") or task_path.stem)
    return {
        "task_name": task_name,
        "task_override": _task_override_from_path(task_path),
    }


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected YAML mapping, got {type(data).__name__}.")
    return data


def _task_override_from_path(task_path: Path) -> str:
    parts = task_path.with_suffix("").parts
    if "task" in parts:
        index = parts.index("task")
        return "/".join(parts[index + 1 :])
    return str(task_path)


def _select_wbc_policy_stem(export_dir: Path) -> Path | None:
    if not export_dir.exists():
        return None
    stems = sorted({path.with_suffix("") for path in export_dir.glob("policy-*.yaml")})
    if not stems:
        stems = sorted({path.with_suffix("") for path in export_dir.glob("policy-*.onnx")})
    final_stems = [stem for stem in stems if stem.name.endswith("-final")]
    if final_stems:
        onnx_final_stems = [stem for stem in final_stems if stem.with_suffix(".onnx").exists()]
        return (onnx_final_stems or final_stems)[-1]
    onnx_stems = [stem for stem in stems if stem.with_suffix(".onnx").exists()]
    return (onnx_stems or stems)[-1] if stems else None


def _reference_parity_failures(
    reference_tracking: Mapping[str, Any],
    *,
    parity_thresholds: Mapping[str, float],
) -> dict[str, dict[str, float]]:
    failures: dict[str, dict[str, float]] = {}
    for metric_name, threshold in parity_thresholds.items():
        value = reference_tracking.get(metric_name)
        if value is None:
            failures[metric_name] = {"value": float("inf"), "threshold": float(threshold)}
            continue
        value_float = float(value)
        threshold_float = float(threshold)
        if value_float > threshold_float:
            failures[metric_name] = {"value": value_float, "threshold": threshold_float}
    return failures


def _aggregate_wbc_export_fleet(
    *,
    plan: Mapping[str, Any],
    task_reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "task_count": int(plan["task_count"]),
        "expected_task_count": plan.get("expected_task_count"),
        "task_count_gate_passed": bool(plan["task_count_gate_passed"]),
        "ready_task_count": int(plan["ready_task_count"]),
        "not_ready_task_count": int(plan["not_ready_task_count"]),
        "closed_loop_attempted_count": sum(1 for task in task_reports if task.get("closed_loop_attempted") is True),
        "stability_success_count": sum(1 for task in task_reports if task.get("stability_success") is True),
        "reference_parity_success_count": sum(1 for task in task_reports if task.get("reference_parity_success") is True),
        "reference_parity_failed_count": sum(1 for task in task_reports if task.get("status") == "reference_parity_failed"),
        "closed_loop_error_count": sum(1 for task in task_reports if task.get("status") == "closed_loop_error"),
        "tasks": list(task_reports),
    }


def _aggregate_fleet(*, plan: Mapping[str, Any], task_reports: Sequence[Mapping[str, Any]], audit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_reference": audit.get("source_reference"),
        "task_count": int(plan["task_count"]),
        "runnable_task_count": int(plan["runnable_task_count"]),
        "missing_checkpoint_count": int(plan["missing_checkpoint_count"]),
        "unsupported_policy_count": int(plan["unsupported_policy_count"]),
        "policy_audit_failed_count": int(plan["policy_audit_failed_count"]),
        "closed_loop_attempted_count": sum(1 for task in task_reports if task.get("closed_loop_attempted") is True),
        "closed_loop_success_count": sum(1 for task in task_reports if task.get("closed_loop_success") is True),
        "closed_loop_error_count": sum(1 for task in task_reports if task.get("status") == "closed_loop_error"),
        "tasks": list(task_reports),
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the sim2real HDMI ONNX policy fleet that maps onto WBC G1/HDMI tasks.")
    parser.add_argument("--wbc-export-fleet", action="store_true", help="Run WBC-exported ONNX policies instead of official sim2real checkpoints.")
    parser.add_argument("--sim2real-root", default=str(DEFAULT_SIM2REAL_ROOT))
    parser.add_argument("--wbc-root", default=str(DEFAULT_WBC_ROOT))
    parser.add_argument("--task-dir", default=str(DEFAULT_TASK_DIR))
    parser.add_argument("--exports-dir", default=str(DEFAULT_WBC_EXPORTS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output", default=None)
    parser.add_argument("--duration-sec", type=float, default=6.0)
    parser.add_argument("--fall-height", type=float, default=0.4)
    parser.add_argument("--wbc-initial-pause-sec", type=float, default=0.0)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--require-all-runnable", action="store_true")
    parser.add_argument("--require-all-success", action="store_true")
    parser.add_argument("--require-all-reference-parity", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
