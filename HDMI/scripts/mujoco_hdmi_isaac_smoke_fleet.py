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

from mujoco_hdmi_golden_checkpoint_plan import (  # noqa: E402
    DEFAULT_PYTHON,
    DEFAULT_SIM2REAL_ROOT,
    build_plan_from_sources as build_golden_plan_from_sources,
)
from mujoco_hdmi_external_policy_source_audit import DEFAULT_TASK_DIR  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path("/tmp/wbc_hdmi_goal_full_parity_v2/isaac_smoke_fleet")
CommandRunner = Callable[[list[str]], Any]


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    golden_plan = build_golden_plan_from_sources(
        sim2real_root=Path(args.sim2real_root),
        task_dir=Path(args.task_dir),
        python=args.python,
        train_algo=args.train_algo,
        export_algo=args.train_algo,
        wandb_mode=args.train_wandb_mode,
    )
    report = build_isaac_smoke_plan(
        golden_plan=golden_plan,
        local_root=Path(args.local_root),
        output_root=Path(args.output_root),
        python=args.python,
        train_algo=args.train_algo,
        task_num_envs=args.task_num_envs,
        train_every=args.train_every,
        num_minibatches=args.num_minibatches,
        ppo_epochs=args.ppo_epochs,
        timeout_sec=args.timeout_sec,
        kill_after_sec=args.kill_after_sec,
        episode_length=args.episode_length,
        wandb_mode=args.wandb_mode,
        session_name=args.session_name,
        max_tasks=args.max_tasks,
        launch=args.launch,
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


def build_isaac_smoke_plan(
    *,
    golden_plan: Mapping[str, Any],
    local_root: Path = HDMI_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    python: str = DEFAULT_PYTHON,
    train_algo: str = "ppo_roa_train",
    task_num_envs: int = 1,
    train_every: int = 1,
    num_minibatches: int = 1,
    ppo_epochs: int = 1,
    timeout_sec: int = 240,
    kill_after_sec: int = 30,
    episode_length: int = 32,
    wandb_mode: str = "disabled",
    session_name: str = "wbc-hdmi-isaac-smoke",
    max_tasks: int | None = None,
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

    source_jobs = list(golden_plan.get("jobs", []))
    if max_tasks is not None:
        source_jobs = source_jobs[: max(0, int(max_tasks))]
    jobs = [
        _build_smoke_job(
            local_root=local_root,
            job=job,
            log_dir=log_dir,
            summary_dir=summary_dir,
            runner_dir=runner_dir,
            python=python,
            train_algo=train_algo,
            task_num_envs=task_num_envs,
            train_every=train_every,
            num_minibatches=num_minibatches,
            ppo_epochs=ppo_epochs,
            timeout_sec=timeout_sec,
            kill_after_sec=kill_after_sec,
            episode_length=episode_length,
            wandb_mode=wandb_mode,
        )
        for job in source_jobs
    ]
    run_all_script = output_root / "run_all_smokes.sh"
    _write_run_all_script(run_all_script=run_all_script, runner_scripts=[Path(job["runner_script"]) for job in jobs])
    report: dict[str, Any] = {
        "source_reference": golden_plan.get("source_reference"),
        "task_count": int(golden_plan.get("task_count", 0)),
        "external_policy_ready_count": int(golden_plan.get("external_policy_ready_count", 0)),
        "missing_checkpoint_task_count": int(golden_plan.get("missing_checkpoint_task_count", len(jobs))),
        "smoke_task_count": len(jobs),
        "launch_requested": bool(launch),
        "launched": False,
        "local_root": str(local_root),
        "output_root": str(output_root),
        "manifest_path": str(output_root / "isaac_smoke_manifest.json"),
        "run_all_script": str(run_all_script),
        "session_name": _safe_session_name(session_name),
        "task_num_envs": int(task_num_envs),
        "train_every": int(train_every),
        "total_frames_per_smoke": int(task_num_envs) * int(train_every),
        "timeout_sec": int(timeout_sec),
        "kill_after_sec": int(kill_after_sec),
        "episode_length": int(episode_length),
        "jobs": jobs,
    }
    if launch and jobs:
        runner = command_runner or _subprocess_runner
        command = ["tmux", "new-session", "-d", "-s", report["session_name"], f"bash {run_all_script}"]
        runner(command)
        report["launched"] = True
        report["launch_command"] = command
    _write_manifest(report)
    return report


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and optionally launch sequential Isaac smoke jobs for HDMI tasks missing golden checkpoints."
    )
    parser.add_argument("--local-root", default=str(HDMI_ROOT))
    parser.add_argument("--sim2real-root", default=str(DEFAULT_SIM2REAL_ROOT))
    parser.add_argument("--task-dir", default=str(DEFAULT_TASK_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output", default=None)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--train-algo", default="ppo_roa_train")
    parser.add_argument("--train-wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")
    parser.add_argument("--task-num-envs", type=int, default=1)
    parser.add_argument("--train-every", type=int, default=1)
    parser.add_argument("--num-minibatches", type=int, default=1)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--timeout-sec", type=int, default=240)
    parser.add_argument("--kill-after-sec", type=int, default=30)
    parser.add_argument("--episode-length", type=int, default=32)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--session-name", default="wbc-hdmi-isaac-smoke")
    parser.add_argument("--launch", action="store_true")
    return parser.parse_args(argv)


def _build_smoke_job(
    *,
    local_root: Path,
    job: Mapping[str, Any],
    log_dir: Path,
    summary_dir: Path,
    runner_dir: Path,
    python: str,
    train_algo: str,
    task_num_envs: int,
    train_every: int,
    num_minibatches: int,
    ppo_epochs: int,
    timeout_sec: int,
    kill_after_sec: int,
    episode_length: int,
    wandb_mode: str,
) -> dict[str, Any]:
    task_name = str(job.get("task_name") or job.get("task_override") or "task")
    task_override = str(job["task_override"])
    safe_task = _safe_name(task_name)
    summary_path = summary_dir / f"{safe_task}_summary.json"
    log_path = log_dir / f"{safe_task}.log"
    runner_script = runner_dir / f"{safe_task}.sh"
    total_frames = int(task_num_envs) * int(train_every)
    command = [
        "timeout",
        f"--kill-after={int(kill_after_sec)}s",
        str(int(timeout_sec)),
        python,
        "scripts/train.py",
        "backend=isaac",
        f"algo={train_algo}",
        f"task={task_override}",
        f"task.num_envs={int(task_num_envs)}",
        f"algo.train_every={int(train_every)}",
        f"algo.num_minibatches={int(num_minibatches)}",
        f"algo.ppo_epochs={int(ppo_epochs)}",
        f"total_frames={total_frames}",
        f"task.max_episode_length={int(episode_length)}",
        f"wandb.mode={wandb_mode}",
        "eval_render=false",
        "headless=true",
        f"train_summary_path={summary_path}",
    ]
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
                f"echo SMOKE_EXIT:$status | tee -a {shlex.quote(str(log_path))}",
                'exit "$status"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    os.chmod(runner_script, 0o755)


def _write_run_all_script(*, run_all_script: Path, runner_scripts: Sequence[Path]) -> None:
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
    lines.extend(['echo "SMOKE_FAILURES:$failures"', 'exit "$failures"', ""])
    run_all_script.write_text("\n".join(lines), encoding="utf-8")
    os.chmod(run_all_script, 0o755)


def _write_manifest(report: Mapping[str, Any]) -> None:
    manifest_path = Path(str(report["manifest_path"]))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-") or "task"


def _safe_session_name(value: str) -> str:
    return _safe_name(value).replace(".", "-")[:80]


def _subprocess_runner(command: list[str]) -> None:
    subprocess.run(command, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
