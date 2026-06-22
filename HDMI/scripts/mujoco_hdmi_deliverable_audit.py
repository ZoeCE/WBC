from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


VIDEO_SUFFIXES = {".mp4", ".gif", ".webm", ".mov", ".avi", ".mkv"}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_deliverable_audit(
        goal_report=Path(args.goal_report),
        video_roots=[Path(path) for path in args.video_root],
        report_dir=Path(args.report_dir) if args.report_dir else None,
        write_task_reports=args.write_task_reports,
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, sort_keys=True))
    return 0


def build_deliverable_audit(
    *,
    goal_report: Path,
    video_roots: Sequence[Path] = (),
    report_dir: Path | None = None,
    write_task_reports: bool = False,
) -> dict[str, Any]:
    goal = _load_json_mapping(goal_report)
    goal_tasks = goal.get("tasks")
    if not isinstance(goal_tasks, list):
        raise TypeError(f"{goal_report}: expected tasks list.")

    tasks = []
    for raw_task in goal_tasks:
        if not isinstance(raw_task, dict):
            continue
        task = _task_deliverables(
            raw_task,
            video_roots=video_roots,
            report_dir=report_dir,
            write_task_reports=write_task_reports,
        )
        tasks.append(task)

    return {
        "goal_report": str(goal_report),
        "video_roots": [str(path) for path in video_roots],
        "report_dir": str(report_dir) if report_dir else None,
        "task_count": len(tasks),
        "video_task_count": sum(1 for task in tasks if task["video_available"]),
        "metrics_task_count": sum(1 for task in tasks if task["metrics_available"]),
        "success_task_count": sum(1 for task in tasks if task["success_available"]),
        "difference_report_task_count": sum(1 for task in tasks if task["difference_report_available"]),
        "runnable_policy_deliverable_count": sum(1 for task in tasks if task["runnable_policy_deliverable"]),
        "complete_task_count": sum(1 for task in tasks if task["deliverables_complete"]),
        "gate_passed": bool(tasks) and all(task["deliverables_complete"] for task in tasks),
        "tasks": tasks,
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit per-task HDMI MuJoCo deliverables: videos, metrics, success, and "
            "difference-localization reports."
        )
    )
    parser.add_argument("--goal-report", required=True)
    parser.add_argument("--video-root", action="append", default=[])
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--write-task-reports", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


def _task_deliverables(
    task: Mapping[str, Any],
    *,
    video_roots: Sequence[Path],
    report_dir: Path | None,
    write_task_reports: bool,
) -> dict[str, Any]:
    task_name = str(task.get("task_name") or task.get("task_stem") or "unknown")
    task_stem = str(task.get("task_stem") or task_name)
    gates = task.get("gates") if isinstance(task.get("gates"), dict) else {}
    missing = [str(item) for item in task.get("missing", [])] if isinstance(task.get("missing"), list) else []
    success = task.get("success")
    open_loop_metrics = task.get("open_loop_metrics") if isinstance(task.get("open_loop_metrics"), dict) else {}

    video_artifacts = _video_artifacts(task_name=task_name, task_stem=task_stem, roots=video_roots)
    metric_sources = _metric_sources(gates=gates, success=success, open_loop_metrics=open_loop_metrics)
    blocking_reasons = _blocking_reasons(
        gates=gates,
        missing=missing,
        video_available=bool(video_artifacts),
        success_available=success is not None,
    )

    difference_report_path = None
    difference_report_available = False
    if write_task_reports and report_dir is not None:
        report_dir.mkdir(parents=True, exist_ok=True)
        difference_report_path = report_dir / f"{task_stem}.md"
        difference_report_path.write_text(
            _difference_report_markdown(
                task=task,
                metric_sources=metric_sources,
                video_artifacts=video_artifacts,
                blocking_reasons=blocking_reasons,
            ),
            encoding="utf-8",
        )
        difference_report_available = True
    elif report_dir is not None:
        candidate = report_dir / f"{task_stem}.md"
        difference_report_path = candidate
        difference_report_available = candidate.is_file()

    metrics_available = bool(metric_sources)
    success_available = success is not None
    runnable_policy_blocking_reasons = _runnable_policy_blocking_reasons(
        gates=gates,
        video_available=bool(video_artifacts),
        success_available=success_available,
        difference_report_available=difference_report_available,
    )
    runnable_policy_deliverable = not runnable_policy_blocking_reasons
    deliverables_complete = (
        bool(video_artifacts)
        and metrics_available
        and success_available
        and difference_report_available
        and not blocking_reasons
    )
    return {
        "task_name": task_name,
        "task_override": task.get("task_override"),
        "task_stem": task_stem,
        "video_available": bool(video_artifacts),
        "video_artifacts": video_artifacts,
        "metrics_available": metrics_available,
        "metric_sources": metric_sources,
        "success_available": success_available,
        "success": success,
        "difference_report_available": difference_report_available,
        "difference_report_path": str(difference_report_path) if difference_report_path else None,
        "runnable_policy_deliverable": runnable_policy_deliverable,
        "runnable_policy_blocking_reasons": runnable_policy_blocking_reasons,
        "deliverables_complete": deliverables_complete,
        "blocking_reasons": blocking_reasons,
        "gates": gates,
        "missing": missing,
    }


def _load_json_mapping(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected JSON object, got {type(data).__name__}.")
    return data


def _video_artifacts(*, task_name: str, task_stem: str, roots: Sequence[Path]) -> list[str]:
    needles = {_normalize_token(task_name), _normalize_token(task_stem)}
    artifacts = []
    for root in roots:
        if root.is_file():
            candidates = [root]
        elif root.is_dir():
            candidates = [path for path in root.rglob("*") if path.is_file()]
        else:
            candidates = []
        for path in candidates:
            if path.suffix.lower() not in VIDEO_SUFFIXES:
                continue
            normalized = _normalize_token(path.stem)
            if any(needle and needle in normalized for needle in needles):
                artifacts.append(str(path))
    return sorted(dict.fromkeys(artifacts))


def _metric_sources(*, gates: Mapping[str, Any], success: Any, open_loop_metrics: Mapping[str, Any]) -> list[str]:
    sources = []
    if gates.get("kinematic") is True:
        sources.append("kinematic")
    if gates.get("open_loop_dynamics") is True or bool(open_loop_metrics):
        sources.append("open_loop_dynamics")
    if gates.get("closed_loop_success") is True or success is not None:
        sources.append("closed_loop_success")
    return sources


def _blocking_reasons(
    *,
    gates: Mapping[str, Any],
    missing: Sequence[str],
    video_available: bool,
    success_available: bool,
) -> list[str]:
    reasons = []
    if not video_available:
        reasons.append("missing_video")
    if not success_available:
        reasons.append("missing_success")
    for gate in ("policy_export", "open_loop_dynamics", "closed_loop_success"):
        if gate in missing or gates.get(gate) is False:
            reasons.append(f"missing_{gate}")
    return list(dict.fromkeys(reasons))


def _runnable_policy_blocking_reasons(
    *,
    gates: Mapping[str, Any],
    video_available: bool,
    success_available: bool,
    difference_report_available: bool,
) -> list[str]:
    reasons = []
    if not video_available:
        reasons.append("missing_video")
    if not success_available:
        reasons.append("missing_success")
    if not difference_report_available:
        reasons.append("missing_difference_report")
    for gate in ("policy_export", "closed_loop_success"):
        if gates.get(gate) is not True:
            reasons.append(f"missing_{gate}")
    return reasons


def _difference_report_markdown(
    *,
    task: Mapping[str, Any],
    metric_sources: Sequence[str],
    video_artifacts: Sequence[str],
    blocking_reasons: Sequence[str],
) -> str:
    task_name = str(task.get("task_name") or "unknown")
    task_override = str(task.get("task_override") or "")
    gates = task.get("gates") if isinstance(task.get("gates"), dict) else {}
    missing = task.get("missing") if isinstance(task.get("missing"), list) else []
    success = task.get("success")
    open_loop_metrics = task.get("open_loop_metrics") if isinstance(task.get("open_loop_metrics"), dict) else {}

    lines = [
        f"# {task_name}",
        "",
        f"- task_override: `{task_override}`",
        f"- success: `{success}`",
        f"- metric_sources: `{list(metric_sources)}`",
        f"- video_artifacts: `{list(video_artifacts)}`",
        f"- blocking_reasons: `{list(blocking_reasons)}`",
        "",
        "## Gates",
    ]
    for key in sorted(gates):
        lines.append(f"- {key}: `{gates[key]}`")
    lines.extend(["", "## Missing"])
    if missing:
        for item in missing:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    lines.extend(["", "## Open-Loop Metrics"])
    if open_loop_metrics:
        for key in sorted(open_loop_metrics):
            lines.append(f"- {key}: `{open_loop_metrics[key]}`")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def _normalize_token(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


if __name__ == "__main__":
    raise SystemExit(main())
