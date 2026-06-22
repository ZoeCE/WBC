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

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
HDMI_ROOT = SCRIPT_DIR.parent
DEFAULT_TASK_DIR = HDMI_ROOT / "cfg" / "task" / "G1" / "hdmi"
DEFAULT_PYTHON = "/home/zoe/miniconda3/envs/wbc/bin/python"
DEFAULT_OUTPUT_ROOT = Path("/tmp/wbc_hdmi_goal_full_parity_v2/training_fleet")

CommandRunner = Callable[[list[str]], Any]


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_train_fleet(
        local_root=Path(args.local_root),
        task_dir=Path(args.task_dir),
        task_yamls=[Path(path) for path in args.task_yaml],
        output_root=Path(args.output_root),
        python=args.python,
        algo=args.algo,
        checkpoint_manifest=Path(args.checkpoint_manifest) if args.checkpoint_manifest else None,
        total_frames=args.total_frames,
        task_num_envs=args.task_num_envs,
        train_every=args.train_every,
        num_minibatches=args.num_minibatches,
        ppo_epochs=args.ppo_epochs,
        wandb_mode=args.wandb_mode,
        session_prefix=args.session_prefix,
        min_success=args.min_success,
        launch=args.launch,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


def build_train_fleet(
    *,
    local_root: Path = HDMI_ROOT,
    task_dir: Path = DEFAULT_TASK_DIR,
    task_yamls: Sequence[Path] = (),
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    python: str = DEFAULT_PYTHON,
    algo: str = "ppo_roa_finetune",
    checkpoint_manifest: Path | None = None,
    total_frames: int = 65_536,
    task_num_envs: int = 64,
    train_every: int = 64,
    num_minibatches: int = 4,
    ppo_epochs: int = 2,
    wandb_mode: str = "online",
    session_prefix: str = "wbc-hdmi",
    min_success: float = 0.9,
    launch: bool = False,
    command_runner: CommandRunner | None = None,
) -> dict[str, Any]:
    local_root = local_root.resolve()
    task_paths = _resolve_task_paths(task_dir=task_dir, task_yamls=task_yamls)
    checkpoint_by_task = _load_checkpoint_manifest(checkpoint_manifest)
    output_root = output_root.resolve()
    log_dir = output_root / "logs"
    summary_dir = output_root / "summaries"
    runner_dir = output_root / "job_scripts"
    for directory in (log_dir, summary_dir, runner_dir):
        directory.mkdir(parents=True, exist_ok=True)

    jobs = []
    for task_path in task_paths:
        identity = _task_identity(task_path)
        checkpoint_path = _checkpoint_for_task(checkpoint_by_task, identity=identity)
        job = _build_job(
            local_root=local_root,
            task_path=task_path,
            identity=identity,
            output_root=output_root,
            log_dir=log_dir,
            summary_dir=summary_dir,
            runner_dir=runner_dir,
            python=python,
            algo=algo,
            checkpoint_path=checkpoint_path,
            total_frames=total_frames,
            task_num_envs=task_num_envs,
            train_every=train_every,
            num_minibatches=num_minibatches,
            ppo_epochs=ppo_epochs,
            wandb_mode=wandb_mode,
            session_prefix=session_prefix,
        )
        jobs.append(job)

    audit_command = _build_closed_loop_success_audit_command(
        python=python,
        task_paths=task_paths,
        summary_paths=[Path(job["summary_path"]) for job in jobs],
        min_success=min_success,
    )
    report: dict[str, Any] = {
        "launch_requested": bool(launch),
        "launched_task_count": 0,
        "task_count": len(jobs),
        "local_root": str(local_root),
        "output_root": str(output_root),
        "manifest_path": str(output_root / "fleet_manifest.json"),
        "closed_loop_success_audit_command": audit_command,
        "jobs": jobs,
    }
    _write_manifest(report)

    if launch:
        runner = command_runner or _subprocess_runner
        launched = 0
        for job in jobs:
            command = [
                "tmux",
                "new-session",
                "-d",
                "-s",
                job["session_name"],
                f"bash {job['runner_script']}",
            ]
            runner(command)
            job["launch_command"] = command
            job["launched"] = True
            launched += 1
        report["launched_task_count"] = launched
        _write_manifest(report)
    return report


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and optionally launch tmux MuJoCo training jobs for the HDMI G1 task fleet."
    )
    parser.add_argument("--local-root", default=str(HDMI_ROOT))
    parser.add_argument("--task-dir", default=str(DEFAULT_TASK_DIR))
    parser.add_argument("--task-yaml", action="append", default=[])
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--algo", default="ppo_roa_finetune")
    parser.add_argument("--checkpoint-manifest", default=None)
    parser.add_argument("--total-frames", type=int, default=65_536)
    parser.add_argument("--task-num-envs", type=int, default=64)
    parser.add_argument("--train-every", type=int, default=64)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--session-prefix", default="wbc-hdmi")
    parser.add_argument("--min-success", type=float, default=0.9)
    parser.add_argument("--launch", action="store_true")
    return parser.parse_args(argv)


def _resolve_task_paths(*, task_dir: Path, task_yamls: Sequence[Path]) -> list[Path]:
    if task_yamls:
        return sorted(task_yamls, key=lambda path: str(path))
    return sorted(Path(task_dir).glob("*.yaml"))


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


def _build_job(
    *,
    local_root: Path,
    task_path: Path,
    identity: Mapping[str, str],
    output_root: Path,
    log_dir: Path,
    summary_dir: Path,
    runner_dir: Path,
    python: str,
    algo: str,
    checkpoint_path: str | None,
    total_frames: int,
    task_num_envs: int,
    train_every: int,
    num_minibatches: int,
    ppo_epochs: int,
    wandb_mode: str,
    session_prefix: str,
) -> dict[str, Any]:
    task_name = identity["task_name"]
    safe_task = _safe_name(task_name)
    summary_path = summary_dir / f"{safe_task}_summary.json"
    log_path = log_dir / f"{safe_task}.log"
    runner_script = runner_dir / f"{safe_task}.sh"
    command = [
        python,
        "scripts/train.py",
        "backend=mujoco",
        f"algo={algo}",
        f"task={identity['task_override']}",
        f"checkpoint_path={checkpoint_path if checkpoint_path is not None else 'null'}",
        f"task.num_envs={int(task_num_envs)}",
        f"algo.train_every={int(train_every)}",
        f"algo.num_minibatches={int(num_minibatches)}",
        f"algo.ppo_epochs={int(ppo_epochs)}",
        f"total_frames={int(total_frames)}",
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
        "task_override": identity["task_override"],
        "task_yaml": str(task_path),
        "checkpoint_path": checkpoint_path,
        "session_name": _safe_session_name(f"{session_prefix}-{task_name}"),
        "summary_path": str(summary_path),
        "log_path": str(log_path),
        "runner_script": str(runner_script),
        "command": command,
        "launched": False,
    }


def _write_runner_script(*, runner_script: Path, local_root: Path, command: Sequence[str], log_path: Path) -> None:
    runner_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"mkdir -p {shlex.quote(str(log_path.parent))}",
                f"cd {local_root}",
                f"{shlex.join(command)} 2>&1 | tee -a {shlex.quote(str(log_path))}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    os.chmod(runner_script, 0o755)


def _build_closed_loop_success_audit_command(
    *,
    python: str,
    task_paths: Sequence[Path],
    summary_paths: Sequence[Path],
    min_success: float,
) -> list[str]:
    command = [python, "scripts/mujoco_closed_loop_success_audit.py"]
    for path in task_paths:
        command.extend(["--task-yaml", str(path)])
    for path in summary_paths:
        command.extend(["--summary", str(path)])
    command.extend(["--min-success", str(float(min_success)).rstrip("0").rstrip(".")])
    command.extend(["--require-backend", "mujoco", "--require-checkpoint"])
    return command


def _write_manifest(report: Mapping[str, Any]) -> None:
    manifest_path = Path(str(report["manifest_path"]))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")


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


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-") or "task"


def _safe_session_name(value: str) -> str:
    safe = _safe_name(value).replace(".", "-")
    return safe[:80]


def _subprocess_runner(command: list[str]) -> None:
    subprocess.run(command, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
