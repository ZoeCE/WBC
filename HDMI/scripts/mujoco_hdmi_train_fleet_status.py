from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


PROGRESS_RE = re.compile(r"(?P<percent>\d+(?:\.\d+)?)%.*?\|\s*(?P<iteration>\d+)/(?:\s*)?(?P<total>\d+)")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    running_sessions = set(args.running_session) if args.running_session else _tmux_sessions()
    report = build_train_fleet_status(
        manifest_path=Path(args.manifest),
        running_sessions=running_sessions,
        min_success=args.min_success,
        log_tail_lines=args.log_tail_lines,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


def build_train_fleet_status(
    *,
    manifest_path: Path,
    running_sessions: Iterable[str] = (),
    min_success: float = 0.9,
    log_tail_lines: int = 20,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    running = set(running_sessions)
    jobs = [
        _job_status(job, running_sessions=running, min_success=float(min_success), log_tail_lines=log_tail_lines)
        for job in manifest.get("jobs", [])
    ]
    return {
        "manifest_path": str(manifest_path),
        "output_root": manifest.get("output_root"),
        "task_count": len(jobs),
        "completed_task_count": sum(1 for job in jobs if job["status"] == "completed"),
        "running_task_count": sum(1 for job in jobs if job["status"] == "running"),
        "pending_task_count": sum(1 for job in jobs if job["status"] == "pending"),
        "failed_task_count": sum(1 for job in jobs if job["status"] == "completed" and job["failure_reasons"]),
        "min_success": float(min_success),
        "jobs": jobs,
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report status for a MuJoCo HDMI training fleet manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--running-session", action="append", default=[])
    parser.add_argument("--min-success", type=float, default=0.9)
    parser.add_argument("--log-tail-lines", type=int, default=20)
    return parser.parse_args(argv)


def _job_status(
    job: dict[str, Any],
    *,
    running_sessions: set[str],
    min_success: float,
    log_tail_lines: int,
) -> dict[str, Any]:
    summary_path = Path(str(job.get("summary_path", "")))
    log_path = Path(str(job.get("log_path", "")))
    session_name = str(job.get("session_name", ""))
    summary_exists = summary_path.is_file()
    session_running = session_name in running_sessions
    summary = _load_json(summary_path) if summary_exists else {}
    checkpoint = summary.get("checkpoint_final")
    checkpoint_exists = bool(checkpoint) and Path(str(checkpoint)).is_file()
    success = _success_from_summary(summary)
    failure_reasons = _failure_reasons(
        summary_exists=summary_exists,
        session_running=session_running,
        checkpoint_exists=checkpoint_exists,
        success=success,
        min_success=min_success,
    )
    status = "completed" if summary_exists else ("running" if session_running else "pending")
    return {
        "task_name": job.get("task_name"),
        "task_override": job.get("task_override"),
        "session_name": session_name,
        "session_running": session_running,
        "status": status,
        "summary_path": str(summary_path),
        "summary_exists": summary_exists,
        "log_path": str(log_path),
        "log_exists": log_path.is_file(),
        "progress": _progress_from_log(log_path),
        "log_tail": _tail(log_path, log_tail_lines),
        "checkpoint_final": checkpoint,
        "checkpoint_exists": checkpoint_exists,
        "env_frames": summary.get("env_frames"),
        "total_frames": summary.get("total_frames"),
        "total_iters": summary.get("total_iters"),
        "success": success,
        "success_passed": success is not None and math.isfinite(success) and success >= min_success,
        "failure_reasons": failure_reasons,
    }


def _failure_reasons(
    *,
    summary_exists: bool,
    session_running: bool,
    checkpoint_exists: bool,
    success: float | None,
    min_success: float,
) -> list[str]:
    if not summary_exists:
        return [] if session_running else ["summary_missing"]
    reasons = []
    if not checkpoint_exists:
        reasons.append("checkpoint_missing")
    if success is None or not math.isfinite(success):
        reasons.append("success_missing")
    elif success < min_success:
        reasons.append("success_below_min")
    return reasons


def _success_from_summary(summary: dict[str, Any]) -> float | None:
    value = summary.get("eval", {}).get("eval/success") if isinstance(summary.get("eval"), dict) else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _progress_from_log(path: Path) -> dict[str, int | float] | None:
    if not path.is_file():
        return None
    last_match = None
    text = path.read_text(encoding="utf-8", errors="replace")
    for match in PROGRESS_RE.finditer(text):
        last_match = match
    if last_match is None:
        return None
    return {
        "iteration": int(last_match.group("iteration")),
        "total_iterations": int(last_match.group("total")),
        "percent": float(last_match.group("percent")),
    }


def _tail(path: Path, lines: int) -> list[str]:
    if lines <= 0 or not path.is_file():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected JSON object, got {type(data).__name__}.")
    return data


def _tmux_sessions() -> set[str]:
    try:
        result = subprocess.run(
            ["tmux", "list-sessions", "-F", "#{session_name}"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return set()
    if result.returncode != 0:
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


if __name__ == "__main__":
    raise SystemExit(main())
