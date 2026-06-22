from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


HDMI_ROOT = Path(__file__).resolve().parents[1]
if str(HDMI_ROOT) not in sys.path:
    sys.path.insert(0, str(HDMI_ROOT))


SOURCE_REFERENCE = {
    "repository": "https://github.com/LeCAR-Lab/HDMI",
    "closed_loop_metric": "eval/success",
    "scope": "fleet-level evidence that every HDMI task has a MuJoCo closed-loop evaluation summary.",
}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_closed_loop_success_audit(
        task_dir=args.task_dir,
        task_yamls=args.task_yaml,
        summary_paths=[Path(path) for path in args.summary],
        expected_task_count=args.expected_task_count,
        metric_key=args.metric_key,
        min_success=args.min_success,
        min_env_frames=args.min_env_frames,
        require_backend=args.require_backend,
        require_checkpoint=args.require_checkpoint,
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["closed_loop_success_gate_passed"] else 1


def build_closed_loop_success_audit(
    *,
    task_dir: str | Path,
    task_yamls: Sequence[str | Path] | None = None,
    summary_paths: Sequence[Path] = (),
    expected_task_count: int | None = None,
    metric_key: str = "eval/success",
    min_success: float = 0.0,
    min_env_frames: int | None = None,
    require_backend: str | None = "mujoco",
    require_checkpoint: bool = False,
) -> dict[str, Any]:
    tasks = [_task_identity(path) for path in _resolve_task_yamls(task_dir=Path(task_dir), task_yamls=task_yamls or [])]
    summaries = [_load_summary_record(path) for path in summary_paths]
    task_reports = [
        _task_success_report(
            task=task,
            summaries=summaries,
            metric_key=metric_key,
            min_success=float(min_success),
            min_env_frames=min_env_frames,
            require_backend=require_backend,
            require_checkpoint=require_checkpoint,
        )
        for task in tasks
    ]

    failures = []
    if expected_task_count is not None and len(tasks) != expected_task_count:
        failures.append(
            {
                "reason": "task_count_mismatch",
                "actual": len(tasks),
                "expected": int(expected_task_count),
            }
        )
    for task_report in task_reports:
        failures.extend(task_report["failures"])

    return {
        "source_reference": SOURCE_REFERENCE,
        "closed_loop_success_gate_passed": not failures,
        "task_dir": str(task_dir),
        "task_count": len(tasks),
        "expected_task_count": expected_task_count,
        "summary_count": len(summaries),
        "passed_task_count": sum(1 for task in task_reports if task["gate_passed"]),
        "missing_summary_task_count": sum(1 for task in task_reports if task["matched_summary_count"] == 0),
        "metric_key": metric_key,
        "thresholds": {
            "min_success": float(min_success),
            "min_env_frames": min_env_frames,
            "require_backend": require_backend,
            "require_checkpoint": require_checkpoint,
        },
        "tasks": task_reports,
        "failures": failures,
    }


def _task_success_report(
    *,
    task: dict[str, str],
    summaries: Sequence[dict[str, Any]],
    metric_key: str,
    min_success: float,
    min_env_frames: int | None,
    require_backend: str | None,
    require_checkpoint: bool,
) -> dict[str, Any]:
    matches = [summary for summary in summaries if _summary_matches_task(summary, task)]
    candidate_reports = [
        _summary_candidate_report(
            summary,
            metric_key=metric_key,
            min_success=min_success,
            min_env_frames=min_env_frames,
            require_backend=require_backend,
            require_checkpoint=require_checkpoint,
        )
        for summary in matches
    ]
    passing = [candidate for candidate in candidate_reports if candidate["gate_passed"]]
    finite_successes = [
        float(candidate["success"])
        for candidate in candidate_reports
        if candidate["success"] is not None and math.isfinite(float(candidate["success"]))
    ]
    failures = []
    if not matches:
        failures.append(
            {
                "task_name": task["task_name"],
                "task_override": task["task_override"],
                "reason": "summary_missing",
            }
        )
    elif not passing:
        best_success = max(finite_successes) if finite_successes else None
        reason = "success_below_min"
        if best_success is None:
            reason = "success_missing"
        failures.append(
            {
                "task_name": task["task_name"],
                "task_override": task["task_override"],
                "reason": reason,
                "best_success": best_success,
                "limit": min_success,
                "candidate_failures": [
                    failure
                    for candidate in candidate_reports
                    for failure in candidate["failures"]
                ],
            }
        )

    return {
        **task,
        "gate_passed": bool(passing),
        "matched_summary_count": len(matches),
        "passing_summary_count": len(passing),
        "best_success": max(finite_successes) if finite_successes else None,
        "summaries": candidate_reports,
        "failures": failures,
    }


def _summary_candidate_report(
    summary: dict[str, Any],
    *,
    metric_key: str,
    min_success: float,
    min_env_frames: int | None,
    require_backend: str | None,
    require_checkpoint: bool,
) -> dict[str, Any]:
    failures = []
    if summary.get("load_error") is not None:
        failures.append({"reason": "summary_load_error", "error": summary["load_error"]})

    if require_backend is not None and summary.get("backend") != require_backend:
        failures.append(
            {
                "reason": "backend_mismatch",
                "actual": summary.get("backend"),
                "expected": require_backend,
            }
        )

    checkpoint = summary.get("checkpoint_final")
    if require_checkpoint and (not checkpoint or not Path(str(checkpoint)).is_file()):
        failures.append(
            {
                "reason": "checkpoint_missing",
                "checkpoint_final": checkpoint,
            }
        )

    if min_env_frames is not None and not _threshold_passes(summary.get("env_frames"), float(min_env_frames), ">="):
        failures.append(
            {
                "reason": "env_frames_below_min",
                "actual": summary.get("env_frames"),
                "limit": int(min_env_frames),
            }
        )

    success = _nested_metric(summary, "eval", metric_key)
    if not _threshold_passes(success, float(min_success), ">="):
        failures.append(
            {
                "reason": "success_below_min" if success is not None else "success_missing",
                "metric": metric_key,
                "actual": success,
                "limit": float(min_success),
            }
        )

    return {
        "path": summary.get("path"),
        "task": summary.get("task"),
        "backend": summary.get("backend"),
        "checkpoint_final": checkpoint,
        "env_frames": summary.get("env_frames"),
        "success": success,
        "gate_passed": not failures,
        "failures": failures,
    }


def _nested_metric(summary: Mapping[str, Any], group: str, metric_key: str) -> Any:
    metrics = summary.get(group, {})
    if isinstance(metrics, Mapping) and metric_key in metrics:
        return metrics[metric_key]
    if metric_key in summary:
        return summary[metric_key]
    return None


def _summary_matches_task(summary: Mapping[str, Any], task: Mapping[str, str]) -> bool:
    summary_task = summary.get("task")
    if summary_task is None:
        return False
    candidates = {
        task["task_name"],
        task["task_override"],
        task["task_stem"],
        task["task_file"],
        task["task_path"],
    }
    return str(summary_task) in candidates


def _load_summary_record(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError(f"expected JSON object, got {type(data).__name__}")
    except Exception as exc:
        return {"path": str(path), "load_error": f"{type(exc).__name__}: {exc}"}
    data["path"] = str(path)
    return data


def _resolve_task_yamls(*, task_dir: Path, task_yamls: Sequence[str | Path]) -> list[Path]:
    if task_yamls:
        return sorted((Path(path) for path in task_yamls), key=lambda path: str(path))
    return sorted(task_dir.glob("*.yaml"))


def _task_identity(task_path: Path) -> dict[str, str]:
    task_cfg = _load_yaml_mapping(task_path)
    task_name = str(task_cfg.get("name") or task_path.stem)
    return {
        "task_path": str(task_path),
        "task_name": task_name,
        "task_override": _task_override_from_path(task_path),
        "task_stem": task_path.stem,
        "task_file": task_path.name,
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


def _threshold_passes(actual: Any, limit: float, comparison: str) -> bool:
    try:
        actual_value = float(actual)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(actual_value):
        return False
    if comparison == ">=":
        return actual_value >= limit
    if comparison == "<=":
        return actual_value <= limit
    raise ValueError(f"Unsupported comparison {comparison!r}.")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit closed-loop MuJoCo success evidence for every HDMI task."
    )
    parser.add_argument(
        "--task-dir",
        default=str(HDMI_ROOT / "cfg" / "task" / "G1" / "hdmi"),
        help="Directory containing HDMI task YAMLs. Ignored for discovery when --task-yaml is provided.",
    )
    parser.add_argument("--task-yaml", action="append", default=[])
    parser.add_argument("--summary", action="append", default=[], help="Training/evaluation summary JSON.")
    parser.add_argument("--expected-task-count", type=int, default=None)
    parser.add_argument("--metric-key", default="eval/success")
    parser.add_argument("--min-success", type=float, default=0.0)
    parser.add_argument("--min-env-frames", type=int, default=None)
    parser.add_argument("--require-backend", default="mujoco")
    parser.add_argument("--require-checkpoint", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
