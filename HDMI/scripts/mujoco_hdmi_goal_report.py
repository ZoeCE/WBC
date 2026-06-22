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
GATE_KEYS = ("source_reference", "kinematic", "policy_export", "closed_loop_success")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_goal_report(
        task_dir=Path(args.task_dir),
        task_yamls=[Path(path) for path in args.task_yaml],
        source_reference_report=Path(args.source_reference_report) if args.source_reference_report else None,
        kinematic_report=Path(args.kinematic_report) if args.kinematic_report else None,
        open_loop_report=Path(args.open_loop_report) if args.open_loop_report else None,
        policy_report=Path(args.policy_report) if args.policy_report else None,
        obs_action_report=Path(args.obs_action_report) if args.obs_action_report else None,
        closed_loop_report=Path(args.closed_loop_report) if args.closed_loop_report else None,
        checkpoint_inventory_report=Path(args.checkpoint_inventory_report) if args.checkpoint_inventory_report else None,
        policy_export_plan_report=Path(args.policy_export_plan_report) if args.policy_export_plan_report else None,
        checkpoint_source_report=Path(args.checkpoint_source_report) if args.checkpoint_source_report else None,
        external_policy_source_report=Path(args.external_policy_source_report)
        if args.external_policy_source_report
        else None,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


def build_goal_report(
    *,
    task_dir: Path = DEFAULT_TASK_DIR,
    task_yamls: Sequence[Path] = (),
    source_reference_report: Path | None = None,
    kinematic_report: Path | None = None,
    open_loop_report: Path | None = None,
    policy_report: Path | None = None,
    obs_action_report: Path | None = None,
    closed_loop_report: Path | None = None,
    checkpoint_inventory_report: Path | None = None,
    policy_export_plan_report: Path | None = None,
    checkpoint_source_report: Path | None = None,
    external_policy_source_report: Path | None = None,
) -> dict[str, Any]:
    tasks = [_task_identity(path) for path in _resolve_task_paths(task_dir=task_dir, task_yamls=task_yamls)]
    source = _load_optional_json(source_reference_report)
    kinematic_by_key = _entries_by_identity_key(_load_optional_json(kinematic_report))
    open_loop_loaded = _load_optional_json(open_loop_report)
    open_loop_by_key = _entries_by_identity_key(open_loop_loaded)
    open_loop_gate_enabled = open_loop_report is not None
    policy_by_key = _entries_by_identity_key(_load_optional_json(policy_report))
    obs_action_by_key = _entries_by_identity_key(_load_optional_json(obs_action_report))
    obs_action_gate_enabled = obs_action_report is not None
    closed_loop_by_key = _entries_by_identity_key(_load_optional_json(closed_loop_report))
    checkpoint_inventory = _load_optional_json(checkpoint_inventory_report)
    policy_export_plan = _load_optional_json(policy_export_plan_report)
    checkpoint_source = _load_optional_json(checkpoint_source_report)
    external_policy_source = _load_optional_json(external_policy_source_report)
    source_passed = source.get("gate_passed") is True if source else False

    task_reports = []
    for identity in tasks:
        task_name = identity["task_name"]
        kinematic = _entry_for_identity(kinematic_by_key, identity)
        open_loop = _entry_for_identity(open_loop_by_key, identity)
        policy = _entry_for_identity(policy_by_key, identity)
        obs_action = _entry_for_identity(obs_action_by_key, identity)
        closed_loop = _entry_for_identity(closed_loop_by_key, identity)
        gates = {
            "source_reference": source_passed,
            "kinematic": _kinematic_passed(kinematic),
            "policy_export": _policy_passed(policy),
            "closed_loop_success": _closed_loop_passed(closed_loop),
        }
        gate_keys = list(GATE_KEYS)
        if obs_action_gate_enabled:
            gates["obs_action_parity"] = _obs_action_passed(obs_action)
            gate_keys.insert(gate_keys.index("closed_loop_success"), "obs_action_parity")
        if open_loop_gate_enabled:
            gates["open_loop_dynamics"] = _open_loop_passed(open_loop)
            gate_keys.insert(gate_keys.index("closed_loop_success"), "open_loop_dynamics")
        missing = [gate for gate in gate_keys if not gates[gate]]
        task_reports.append(
            {
                **identity,
                "gates": gates,
                "missing": missing,
                "obs_action_metrics": _obs_action_metrics(obs_action),
                "open_loop_metrics": _open_loop_metrics(open_loop),
                "success": _closed_loop_success(closed_loop),
                "closed_loop_failures": closed_loop.get("failures") or closed_loop.get("candidate_failures") or [],
            }
        )

    missing_by_gate = {
        gate: [task["task_name"] for task in task_reports if not task["gates"][gate]]
        for gate in (
            list(GATE_KEYS[:-1])
            + (["obs_action_parity"] if obs_action_gate_enabled else [])
            + (["open_loop_dynamics"] if open_loop_gate_enabled else [])
            + [GATE_KEYS[-1]]
        )
    }
    return {
        "task_count": len(task_reports),
        "all_tasks_complete": all(not task["missing"] for task in task_reports),
        "source_reference_passed": source_passed,
        "missing_by_gate": missing_by_gate,
        "tasks": task_reports,
        "artifact_readiness": _artifact_readiness(
            checkpoint_inventory=checkpoint_inventory,
            checkpoint_inventory_report=checkpoint_inventory_report,
            policy_export_plan=policy_export_plan,
            policy_export_plan_report=policy_export_plan_report,
            checkpoint_source=checkpoint_source,
            checkpoint_source_report=checkpoint_source_report,
            external_policy_source=external_policy_source,
            external_policy_source_report=external_policy_source_report,
        ),
        "inputs": {
            "task_dir": str(task_dir),
            "task_yamls": [str(path) for path in task_yamls],
            "source_reference_report": str(source_reference_report) if source_reference_report else None,
            "kinematic_report": str(kinematic_report) if kinematic_report else None,
            "open_loop_report": str(open_loop_report) if open_loop_report else None,
            "policy_report": str(policy_report) if policy_report else None,
            "obs_action_report": str(obs_action_report) if obs_action_report else None,
            "closed_loop_report": str(closed_loop_report) if closed_loop_report else None,
            "checkpoint_inventory_report": str(checkpoint_inventory_report) if checkpoint_inventory_report else None,
            "policy_export_plan_report": str(policy_export_plan_report) if policy_export_plan_report else None,
            "checkpoint_source_report": str(checkpoint_source_report) if checkpoint_source_report else None,
            "external_policy_source_report": str(external_policy_source_report)
            if external_policy_source_report
            else None,
        },
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate HDMI MuJoCo goal evidence by task and gate.")
    parser.add_argument("--task-dir", default=str(DEFAULT_TASK_DIR))
    parser.add_argument("--task-yaml", action="append", default=[])
    parser.add_argument("--source-reference-report", default=None)
    parser.add_argument("--kinematic-report", default=None)
    parser.add_argument("--open-loop-report", default=None)
    parser.add_argument("--policy-report", default=None)
    parser.add_argument("--obs-action-report", default=None)
    parser.add_argument("--closed-loop-report", default=None)
    parser.add_argument("--checkpoint-inventory-report", default=None)
    parser.add_argument("--policy-export-plan-report", default=None)
    parser.add_argument("--checkpoint-source-report", default=None)
    parser.add_argument("--external-policy-source-report", default=None)
    return parser.parse_args(argv)


def _resolve_task_paths(*, task_dir: Path, task_yamls: Sequence[Path]) -> list[Path]:
    if task_yamls:
        return list(task_yamls)
    return sorted(Path(task_dir).glob("*.yaml"))


def _task_identity(task_path: Path) -> dict[str, str]:
    cfg = _load_yaml_mapping(task_path)
    task_name = str(cfg.get("name") or task_path.stem)
    return {
        "task_name": task_name,
        "task_override": _task_override_from_path(task_path),
        "task_stem": task_path.stem,
        "task_path": str(task_path),
    }


def _task_override_from_path(task_path: Path) -> str:
    parts = task_path.with_suffix("").parts
    if "task" in parts:
        index = parts.index("task")
        return "/".join(parts[index + 1 :])
    return str(task_path.with_suffix(""))


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected YAML mapping, got {type(data).__name__}.")
    return data


def _load_optional_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected JSON object, got {type(data).__name__}.")
    return data


def _entries_by_identity_key(report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    entries = []
    for field in ("tasks", "rows", "records", "items"):
        value = report.get(field)
        if isinstance(value, list):
            entries.extend(value)
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for field in (
            "task_name",
            "task_override",
            "task_stem",
            "task",
            "task_path",
            "task_file",
            "task_yaml",
            "scenario",
        ):
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
        identity["task_path"],
    ):
        if key in entries_by_key:
            return entries_by_key[key]
    return {}


def _artifact_readiness(
    *,
    checkpoint_inventory: Mapping[str, Any],
    checkpoint_inventory_report: Path | None,
    policy_export_plan: Mapping[str, Any],
    policy_export_plan_report: Path | None,
    checkpoint_source: Mapping[str, Any],
    checkpoint_source_report: Path | None,
    external_policy_source: Mapping[str, Any],
    external_policy_source_report: Path | None,
) -> dict[str, Any]:
    return {
        "checkpoint_inventory": _checkpoint_inventory_readiness(
            checkpoint_inventory,
            report_path=checkpoint_inventory_report,
        ),
        "policy_export_plan": _policy_export_plan_readiness(
            policy_export_plan,
            report_path=policy_export_plan_report,
        ),
        "checkpoint_source": _checkpoint_source_readiness(
            checkpoint_source,
            report_path=checkpoint_source_report,
        ),
        "external_policy_source": _external_policy_source_readiness(
            external_policy_source,
            report_path=external_policy_source_report,
        ),
    }


def _checkpoint_inventory_readiness(
    report: Mapping[str, Any],
    *,
    report_path: Path | None,
) -> dict[str, Any]:
    provided = bool(report)
    golden_candidate_count = _optional_int(report.get("golden_candidate_count")) or 0
    validated_golden_reference_count = _optional_int(report.get("validated_golden_reference_count"))
    missing_golden_reference = report.get("missing_golden_reference")
    missing_validated_golden_reference = report.get("missing_validated_golden_reference")
    if validated_golden_reference_count is None:
        readiness_count = golden_candidate_count
        readiness_missing = missing_golden_reference
    else:
        readiness_count = validated_golden_reference_count
        readiness_missing = missing_validated_golden_reference
    return {
        "provided": provided,
        "report_path": str(report_path) if report_path else None,
        "checkpoint_count": _optional_int(report.get("checkpoint_count")),
        "golden_candidate_count": golden_candidate_count,
        "validated_golden_reference_count": validated_golden_reference_count,
        "weak_mujoco_seed_count": _optional_int(report.get("weak_mujoco_seed_count")),
        "export_count": _optional_int(report.get("export_count")),
        "missing_golden_reference": missing_golden_reference if provided else None,
        "missing_validated_golden_reference": missing_validated_golden_reference if provided else None,
        "gate_passed": provided and readiness_count > 0 and readiness_missing is False,
    }


def _policy_export_plan_readiness(
    report: Mapping[str, Any],
    *,
    report_path: Path | None,
) -> dict[str, Any]:
    provided = bool(report)
    return {
        "provided": provided,
        "report_path": str(report_path) if report_path else None,
        "task_count": _optional_int(report.get("task_count")),
        "task_count_gate_passed": report.get("task_count_gate_passed") if provided else None,
        "provided_checkpoint_task_count": _optional_int(report.get("provided_checkpoint_task_count")),
        "missing_checkpoint_task_count": _optional_int(report.get("missing_checkpoint_task_count")),
        "ready_to_export": report.get("ready_to_export") if provided else None,
        "gate_passed": (report.get("gate_passed") is True) if provided else False,
    }


def _checkpoint_source_readiness(
    report: Mapping[str, Any],
    *,
    report_path: Path | None,
) -> dict[str, Any]:
    provided = bool(report)
    checkpoint_source = report.get("checkpoint_source")
    if not isinstance(checkpoint_source, dict):
        checkpoint_source = {}
    actionable = report.get("actionable_golden_source_available")
    return {
        "provided": provided,
        "report_path": str(report_path) if report_path else None,
        "actionable_golden_source_available": actionable if provided else None,
        "placeholder_run_reference_count": _optional_int(checkpoint_source.get("placeholder_run_reference_count")),
        "concrete_run_reference_count": _optional_int(checkpoint_source.get("concrete_run_reference_count")),
        "direct_checkpoint_link_count": _optional_int(checkpoint_source.get("direct_checkpoint_link_count")),
        "checkpoint_file_count": _optional_int(checkpoint_source.get("checkpoint_file_count")),
        "gate_passed": provided and report.get("gate_passed") is True and actionable is True,
    }


def _external_policy_source_readiness(
    report: Mapping[str, Any],
    *,
    report_path: Path | None,
) -> dict[str, Any]:
    provided = bool(report)
    require_training_provenance = report.get("require_training_provenance") if provided else None
    load_mapping_ready_task_count = _external_load_mapping_ready_task_count(report) if provided else None
    ready_task_count = _optional_int(report.get("ready_task_count")) if provided else None
    golden_ready_task_count = (
        ready_task_count if provided and require_training_provenance is True else 0 if provided else None
    )
    return {
        "provided": provided,
        "report_path": str(report_path) if report_path else None,
        "policy_count": _optional_int(report.get("policy_count")),
        "matched_policy_count": _optional_int(report.get("matched_policy_count")),
        "unmatched_policy_count": _optional_int(report.get("unmatched_policy_count")),
        "load_mapping_ready_task_count": load_mapping_ready_task_count,
        "golden_ready_task_count": golden_ready_task_count,
        "training_provenance_policy_count": _optional_int(report.get("training_provenance_policy_count")),
        "missing_training_provenance_policy_count": _optional_int(
            report.get("missing_training_provenance_policy_count")
        ),
        "require_training_provenance": require_training_provenance,
        "gate_passed": provided
        and require_training_provenance is True
        and report.get("gate_passed") is True,
    }


def _external_load_mapping_ready_task_count(report: Mapping[str, Any]) -> int:
    tasks = report.get("tasks")
    if isinstance(tasks, list):
        return sum(
            1
            for task in tasks
            if isinstance(task, Mapping)
            and (task.get("policy_source_ready") is True or task.get("gate_passed") is True)
        )
    return _optional_int(report.get("ready_task_count")) or 0


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _kinematic_passed(entry: Mapping[str, Any]) -> bool:
    if not entry:
        return False
    if entry.get("parity_passed") is True or entry.get("gate_passed") is True:
        return True
    frame_count = _optional_int(entry.get("render_frame_count")) or 0
    output_size_bytes = _optional_int(entry.get("output_size_bytes")) or 0
    return (
        entry.get("returncode") == 0
        and entry.get("output_exists") is True
        and frame_count > 0
        and output_size_bytes > 0
    )


def _open_loop_passed(entry: Mapping[str, Any]) -> bool:
    if not entry:
        return False
    if entry.get("open_loop_passed") is True:
        return True
    if entry.get("dynamics_passed") is True:
        return True
    return entry.get("gate_passed") is True or entry.get("passed") is True


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


def _obs_action_passed(entry: Mapping[str, Any]) -> bool:
    if not entry:
        return False
    return entry.get("gate_passed") is True or entry.get("passed") is True


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


def _policy_passed(entry: Mapping[str, Any]) -> bool:
    if not entry:
        return False
    return entry.get("gate_passed") is True or entry.get("policy_source_ready") is True


def _closed_loop_passed(entry: Mapping[str, Any]) -> bool:
    if not entry:
        return False
    if entry.get("passed") is True:
        return True
    if entry.get("success_passed") is True:
        return True
    if entry.get("closed_loop_success") is True:
        return True
    closed_loop = entry.get("closed_loop")
    return isinstance(closed_loop, Mapping) and closed_loop.get("heuristic_success") is True


def _closed_loop_success(entry: Mapping[str, Any]) -> float | None:
    for key in ("best_success", "success", "success_metric_value"):
        value = entry.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    closed_loop = entry.get("closed_loop")
    if isinstance(closed_loop, Mapping):
        value = closed_loop.get("success_metric_value")
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


if __name__ == "__main__":
    raise SystemExit(main())
