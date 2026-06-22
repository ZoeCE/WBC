#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
HDMI_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, HDMI_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mujoco_hdmi_external_policy_source_audit import DEFAULT_TASK_DIR, build_external_policy_source_audit


DEFAULT_SIM2REAL_ROOT = Path("/home/zoe/Workspace/sim2real-hdmi")
DEFAULT_PYTHON = "/home/zoe/miniconda3/envs/wbc/bin/python"
DEFAULT_OUTPUT = Path("/tmp/wbc_hdmi_goal_full_parity_v2/golden_checkpoint_plan.json")

SOURCE_REFERENCE = {
    "golden_reference": "original HDMI/Isaac trained checkpoint",
    "upstream_repository": "https://github.com/LeCAR-Lab/HDMI",
    "external_deployment_repository": "https://github.com/EGalahad/sim2real@hdmi",
    "reason": "Tasks without a matched external deployment policy need original Isaac teacher/student checkpoints before MuJoCo parity can be claimed.",
}


def build_golden_checkpoint_plan(
    *,
    audit_report: Mapping[str, Any],
    python: str = DEFAULT_PYTHON,
    train_algo: str = "ppo_roa_train",
    export_algo: str | None = None,
    wandb_mode: str = "online",
) -> dict[str, Any]:
    export_algo = export_algo or train_algo
    tasks = list(audit_report.get("tasks", []))
    ready = [task for task in tasks if task.get("gate_passed") is True and task.get("policy_set")]
    missing = [_missing_task_job(task, python=python, train_algo=train_algo, export_algo=export_algo, wandb_mode=wandb_mode) for task in tasks if _needs_checkpoint(task)]
    return {
        "source_reference": SOURCE_REFERENCE,
        "task_count": len(tasks),
        "external_policy_ready_count": len(ready),
        "missing_checkpoint_task_count": len(missing),
        "checkpoint_manifest_template": {
            str(job["task_override"]): "<teacher_or_student_checkpoint>"
            for job in missing
        },
        "jobs": missing,
    }


def build_plan_from_sources(
    *,
    sim2real_root: Path = DEFAULT_SIM2REAL_ROOT,
    task_dir: Path = DEFAULT_TASK_DIR,
    python: str = DEFAULT_PYTHON,
    train_algo: str = "ppo_roa_train",
    export_algo: str | None = None,
    wandb_mode: str = "online",
) -> dict[str, Any]:
    audit = build_external_policy_source_audit(
        policy_root=sim2real_root / "checkpoints",
        task_dir=task_dir,
        expected_task_count=13,
        require_task_count=True,
        require_reference_observation=True,
        require_obs_action_smoke=True,
    )
    plan = build_golden_checkpoint_plan(
        audit_report=audit,
        python=python,
        train_algo=train_algo,
        export_algo=export_algo,
        wandb_mode=wandb_mode,
    )
    plan["external_policy_audit"] = {
        "policy_root": audit.get("policy_root"),
        "task_count": audit.get("task_count"),
        "ready_task_count": audit.get("ready_task_count"),
        "not_ready_task_count": audit.get("not_ready_task_count"),
        "matched_policy_count": audit.get("matched_policy_count"),
        "missing_requirements_by_task": audit.get("missing_requirements_by_task"),
    }
    return plan


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    plan = build_plan_from_sources(
        sim2real_root=Path(args.sim2real_root),
        task_dir=Path(args.task_dir),
        python=args.python,
        train_algo=args.train_algo,
        export_algo=args.export_algo,
        wandb_mode=args.wandb_mode,
    )
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = DEFAULT_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    plan["output_path"] = str(output_path)
    print(json.dumps(plan, sort_keys=True))
    return 0 if not args.require_no_missing or plan["missing_checkpoint_task_count"] == 0 else 1


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Isaac golden-checkpoint acquisition and policy export commands for missing HDMI tasks.")
    parser.add_argument("--sim2real-root", default=str(DEFAULT_SIM2REAL_ROOT))
    parser.add_argument("--task-dir", default=str(DEFAULT_TASK_DIR))
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--train-algo", default="ppo_roa_train")
    parser.add_argument("--export-algo", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--require-no-missing", action="store_true")
    return parser.parse_args(argv)


def _needs_checkpoint(task: Mapping[str, Any]) -> bool:
    missing = task.get("missing_requirements") or []
    return "external_policy_for_task_motion" in missing or not task.get("policy_set")


def _missing_task_job(
    task: Mapping[str, Any],
    *,
    python: str,
    train_algo: str,
    export_algo: str,
    wandb_mode: str,
) -> dict[str, Any]:
    task_override = str(task.get("task_override"))
    checkpoint_placeholder = "<teacher_or_student_checkpoint>"
    return {
        "task_name": task.get("task_name"),
        "task_override": task_override,
        "task_path": task.get("task_path"),
        "motion_path": task.get("motion_path"),
        "status": "needs_isaac_golden_checkpoint",
        "missing_requirements": list(task.get("missing_requirements") or []),
        "train_command": [
            python,
            "scripts/train.py",
            "backend=isaac",
            f"algo={train_algo}",
            f"task={task_override}",
            f"wandb.mode={wandb_mode}",
        ],
        "checkpoint_manifest_entry": {
            task_override: checkpoint_placeholder,
        },
        "export_command_template": [
            python,
            "scripts/play.py",
            "backend=mujoco",
            f"algo={export_algo}",
            f"task={task_override}",
            f"checkpoint_path={checkpoint_placeholder}",
            "export_policy=true",
            "export_policy_exit=true",
            "export_policy_benchmark_iters=0",
            "export_onnx_policy=false",
            "headless=true",
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())
