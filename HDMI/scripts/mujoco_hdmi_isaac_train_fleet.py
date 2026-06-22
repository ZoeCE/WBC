#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
HDMI_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, HDMI_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mujoco_hdmi_external_policy_source_audit import DEFAULT_TASK_DIR  # noqa: E402
from mujoco_hdmi_golden_checkpoint_plan import (  # noqa: E402
    DEFAULT_PYTHON,
    DEFAULT_SIM2REAL_ROOT,
    build_plan_from_sources as build_golden_plan_from_sources,
)


DEFAULT_OUTPUT_ROOT = Path("/tmp/wbc_hdmi_goal_full_parity_v2/isaac_train_fleet")
CommandRunner = Callable[[list[str]], Any]


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    golden_plan = build_golden_plan_from_sources(
        sim2real_root=Path(args.sim2real_root),
        task_dir=Path(args.task_dir),
        python=args.python,
        train_algo=args.train_algo,
        export_algo=args.train_algo,
        wandb_mode=args.wandb_mode,
    )
    report = build_isaac_train_plan(
        golden_plan=golden_plan,
        local_root=Path(args.local_root),
        output_root=Path(args.output_root),
        python=args.python,
        train_algo=args.train_algo,
        total_frames=args.total_frames,
        task_num_envs=args.task_num_envs,
        train_every=args.train_every,
        num_minibatches=args.num_minibatches,
        ppo_epochs=args.ppo_epochs,
        episode_length=args.episode_length,
        wandb_mode=args.wandb_mode,
        session_name=args.session_name,
        include_tasks=args.include_task,
        exclude_tasks=args.exclude_task,
        max_tasks=args.max_tasks,
        skip_completed=not args.no_skip_completed,
        continue_on_failure=args.continue_on_failure,
        launch=args.launch,
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


def build_isaac_train_plan(
    *,
    golden_plan: Mapping[str, Any],
    local_root: Path = HDMI_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    python: str = DEFAULT_PYTHON,
    train_algo: str = "ppo_roa_train",
    total_frames: int = 150_000_000,
    task_num_envs: int = 4096,
    train_every: int = 32,
    num_minibatches: int | None = None,
    ppo_epochs: int | None = None,
    episode_length: int | None = None,
    wandb_mode: str = "online",
    session_name: str = "wbc-hdmi-isaac-golden-train",
    include_tasks: Sequence[str] = (),
    exclude_tasks: Sequence[str] = (),
    max_tasks: int | None = None,
    skip_completed: bool = True,
    continue_on_failure: bool = False,
    launch: bool = False,
    command_runner: CommandRunner | None = None,
) -> dict[str, Any]:
    local_root = local_root.resolve()
    output_root = output_root.resolve()
    log_dir = output_root / "logs"
    summary_dir = output_root / "summaries"
    runner_dir = output_root / "job_scripts"
    for directory in (log_dir, summary_dir, runner_dir):
        directory.mkdir(parents=True, exist_ok=True)

    source_jobs = _filter_source_jobs(
        jobs=golden_plan.get("jobs", []),
        include_tasks=include_tasks,
        exclude_tasks=exclude_tasks,
        max_tasks=max_tasks,
    )
    pending_jobs: list[dict[str, Any]] = []
    skipped_jobs: list[dict[str, Any]] = []
    for source_job in source_jobs:
        job = _build_train_job(
            local_root=local_root,
            job=source_job,
            log_dir=log_dir,
            summary_dir=summary_dir,
            runner_dir=runner_dir,
            python=python,
            train_algo=train_algo,
            total_frames=total_frames,
            task_num_envs=task_num_envs,
            train_every=train_every,
            num_minibatches=num_minibatches,
            ppo_epochs=ppo_epochs,
            episode_length=episode_length,
            wandb_mode=wandb_mode,
        )
        if skip_completed and _completed_summary(Path(job["summary_path"])):
            job["skip_reason"] = "completed_summary_with_checkpoint"
            skipped_jobs.append(job)
        else:
            pending_jobs.append(job)

    run_all_script = output_root / "run_all_isaac_trains.sh"
    _write_run_all_script(
        run_all_script=run_all_script,
        runner_scripts=[Path(job["runner_script"]) for job in pending_jobs],
        continue_on_failure=continue_on_failure,
    )

    report: dict[str, Any] = {
        "source_reference": golden_plan.get("source_reference"),
        "task_count": int(golden_plan.get("task_count", 0)),
        "external_policy_ready_count": int(golden_plan.get("external_policy_ready_count", 0)),
        "missing_checkpoint_task_count": int(golden_plan.get("missing_checkpoint_task_count", len(source_jobs))),
        "selected_task_count": len(source_jobs),
        "pending_task_count": len(pending_jobs),
        "skipped_task_count": len(skipped_jobs),
        "launch_requested": bool(launch),
        "launched": False,
        "local_root": str(local_root),
        "output_root": str(output_root),
        "manifest_path": str(output_root / "isaac_train_manifest.json"),
        "run_all_script": str(run_all_script),
        "session_name": _safe_session_name(session_name),
        "backend": "isaac",
        "train_algo": train_algo,
        "task_num_envs": int(task_num_envs),
        "train_every": int(train_every),
        "frames_per_batch": int(task_num_envs) * int(train_every),
        "total_frames": int(total_frames),
        "num_minibatches": None if num_minibatches is None else int(num_minibatches),
        "ppo_epochs": None if ppo_epochs is None else int(ppo_epochs),
        "episode_length": None if episode_length is None else int(episode_length),
        "wandb_mode": wandb_mode,
        "continue_on_failure": bool(continue_on_failure),
        "jobs": pending_jobs,
        "skipped_jobs": skipped_jobs,
    }
    if launch and pending_jobs:
        runner = command_runner or _subprocess_runner
        command = ["tmux", "new-session", "-d", "-s", report["session_name"], f"bash {run_all_script}"]
        runner(command)
        report["launched"] = True
        report["launch_command"] = command
    _write_manifest(report)
    return report


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and optionally launch a sequential Isaac PPO training queue for HDMI tasks missing golden checkpoints."
    )
    parser.add_argument("--local-root", default=str(HDMI_ROOT))
    parser.add_argument("--sim2real-root", default=str(DEFAULT_SIM2REAL_ROOT))
    parser.add_argument("--task-dir", default=str(DEFAULT_TASK_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output", default=None)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--train-algo", default="ppo_roa_train")
    parser.add_argument("--total-frames", type=int, default=150_000_000)
    parser.add_argument("--task-num-envs", type=int, default=4096)
    parser.add_argument("--train-every", type=int, default=32)
    parser.add_argument("--num-minibatches", type=int, default=None)
    parser.add_argument("--ppo-epochs", type=int, default=None)
    parser.add_argument("--episode-length", type=int, default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--include-task", action="append", default=[])
    parser.add_argument("--exclude-task", action="append", default=[])
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--no-skip-completed", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--session-name", default="wbc-hdmi-isaac-golden-train")
    parser.add_argument("--launch", action="store_true")
    return parser.parse_args(argv)


def _filter_source_jobs(
    *,
    jobs: Sequence[Mapping[str, Any]],
    include_tasks: Sequence[str],
    exclude_tasks: Sequence[str],
    max_tasks: int | None,
) -> list[Mapping[str, Any]]:
    include = {str(item) for item in include_tasks}
    exclude = {str(item) for item in exclude_tasks}
    selected = []
    for job in jobs:
        if include and not _task_matches(job, include):
            continue
        if exclude and _task_matches(job, exclude):
            continue
        selected.append(job)
    if max_tasks is not None:
        selected = selected[: max(0, int(max_tasks))]
    return selected


def _task_matches(job: Mapping[str, Any], filters: set[str]) -> bool:
    candidates = {
        str(job.get("task_name", "")),
        str(job.get("task_override", "")),
        str(job.get("task_path", "")),
        Path(str(job.get("task_path", ""))).name,
        Path(str(job.get("task_path", ""))).stem,
    }
    return any(candidate in filters for candidate in candidates if candidate)


def _build_train_job(
    *,
    local_root: Path,
    job: Mapping[str, Any],
    log_dir: Path,
    summary_dir: Path,
    runner_dir: Path,
    python: str,
    train_algo: str,
    total_frames: int,
    task_num_envs: int,
    train_every: int,
    num_minibatches: int | None,
    ppo_epochs: int | None,
    episode_length: int | None,
    wandb_mode: str,
) -> dict[str, Any]:
    task_name = str(job.get("task_name") or job.get("task_override") or "task")
    task_override = str(job["task_override"])
    safe_task = _safe_name(task_name)
    summary_path = summary_dir / f"{safe_task}_summary.json"
    log_path = log_dir / f"{safe_task}.log"
    runner_script = runner_dir / f"{safe_task}.sh"
    command = [
        python,
        "scripts/train.py",
        "backend=isaac",
        f"algo={train_algo}",
        f"task={task_override}",
        f"task.num_envs={int(task_num_envs)}",
        f"algo.train_every={int(train_every)}",
    ]
    if num_minibatches is not None:
        command.append(f"algo.num_minibatches={int(num_minibatches)}")
    if ppo_epochs is not None:
        command.append(f"algo.ppo_epochs={int(ppo_epochs)}")
    command.append(f"total_frames={int(total_frames)}")
    if episode_length is not None:
        command.append(f"task.max_episode_length={int(episode_length)}")
    command.extend(
        [
            f"wandb.mode={wandb_mode}",
            "eval_render=false",
            "headless=true",
            f"train_summary_path={summary_path}",
        ]
    )
    _write_runner_script(
        runner_script=runner_script,
        local_root=local_root,
        command=command,
        log_path=log_path,
    )
    return {
        "task_name": task_name,
        "task_override": task_override,
        "motion_path": job.get("motion_path"),
        "task_path": job.get("task_path"),
        "summary_path": str(summary_path),
        "log_path": str(log_path),
        "runner_script": str(runner_script),
        "command": command,
    }


def _write_runner_script(*, runner_script: Path, local_root: Path, command: Sequence[str], log_path: Path) -> None:
    runner_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -uo pipefail",
                f"mkdir -p {shlex.quote(str(log_path.parent))}",
                f"cd {shlex.quote(str(local_root))}",
                "export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}",
                "set +e",
                f"{shlex.join(command)} 2>&1 | tee -a {shlex.quote(str(log_path))}",
                "status=${PIPESTATUS[0]}",
                "set -e",
                f"echo ISAAC_TRAIN_EXIT:$status | tee -a {shlex.quote(str(log_path))}",
                'exit "$status"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    os.chmod(runner_script, 0o755)


def _write_run_all_script(
    *,
    run_all_script: Path,
    runner_scripts: Sequence[Path],
    continue_on_failure: bool,
) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -uo pipefail",
        "failures=0",
    ]
    for script in runner_scripts:
        lines.extend(
            [
                f"bash {shlex.quote(str(script))}",
                "status=$?",
                'if [ "$status" -ne 0 ]; then failures=$((failures + 1)); fi',
            ]
        )
        if not continue_on_failure:
            lines.extend(
                [
                    'if [ "$status" -ne 0 ]; then',
                    '  echo "ISAAC_TRAIN_FAILURES:$failures"',
                    '  exit "$failures"',
                    "fi",
                ]
            )
    lines.extend(['echo "ISAAC_TRAIN_FAILURES:$failures"', 'exit "$failures"', ""])
    run_all_script.write_text("\n".join(lines), encoding="utf-8")
    os.chmod(run_all_script, 0o755)


def _completed_summary(summary_path: Path) -> bool:
    if not summary_path.is_file():
        return False
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    checkpoint = data.get("checkpoint_final") if isinstance(data, dict) else None
    return bool(checkpoint) and Path(str(checkpoint)).is_file()


def _write_manifest(report: Mapping[str, Any]) -> None:
    path = Path(str(report["manifest_path"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return safe or "task"


def _safe_session_name(value: str) -> str:
    return _safe_name(value)[:80]


def _subprocess_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True)


if __name__ == "__main__":
    raise SystemExit(main())
