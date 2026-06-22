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

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
HDMI_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, HDMI_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mujoco_hdmi_external_policy_source_audit import DEFAULT_TASK_DIR  # noqa: E402
from mujoco_hdmi_golden_checkpoint_plan import DEFAULT_PYTHON  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path("/tmp/wbc_hdmi_goal_full_parity_v2/isaac_export_fleet")
CommandRunner = Callable[[list[str]], Any]


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_isaac_export_plan(
        local_root=Path(args.local_root),
        task_dir=Path(args.task_dir),
        train_manifests=[Path(path) for path in args.train_manifest],
        summary_paths=[Path(path) for path in args.summary],
        output_root=Path(args.output_root),
        exports_dir=Path(args.exports_dir) if args.exports_dir else HDMI_ROOT / "scripts" / "exports",
        python=args.python,
        algo=args.algo,
        session_name=args.session_name,
        launch=args.launch,
        require_all_ready=args.require_all_ready,
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if not args.require_all_ready or report["all_ready"] else 1


def build_isaac_export_plan(
    *,
    local_root: Path = HDMI_ROOT,
    task_dir: Path = DEFAULT_TASK_DIR,
    train_manifests: Sequence[Path] = (),
    summary_paths: Sequence[Path] = (),
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    exports_dir: Path = HDMI_ROOT / "scripts" / "exports",
    python: str = DEFAULT_PYTHON,
    algo: str = "ppo_roa_train",
    session_name: str = "wbc-hdmi-isaac-export",
    launch: bool = False,
    require_all_ready: bool = False,
    command_runner: CommandRunner | None = None,
) -> dict[str, Any]:
    local_root = local_root.resolve()
    output_root = output_root.resolve()
    exports_dir = exports_dir.resolve()
    log_dir = output_root / "logs"
    runner_dir = output_root / "job_scripts"
    for directory in (log_dir, runner_dir):
        directory.mkdir(parents=True, exist_ok=True)

    task_identities = _task_identities(task_dir)
    summary_specs = _summary_specs_from_sources(
        train_manifests=train_manifests,
        summary_paths=summary_paths,
        task_identities=task_identities,
    )

    ready_jobs: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    checkpoint_manifest: dict[str, str] = {}
    for spec in summary_specs:
        status = _summary_status(spec)
        if status["ready"]:
            identity = spec["identity"]
            checkpoint = status["checkpoint_final"]
            checkpoint_manifest[identity["task_override"]] = checkpoint
            ready_jobs.append(
                _build_export_job(
                    local_root=local_root,
                    identity=identity,
                    checkpoint_path=checkpoint,
                    log_dir=log_dir,
                    runner_dir=runner_dir,
                    python=python,
                    algo=algo,
                )
            )
        else:
            pending.append({**_public_spec(spec), **status})

    checkpoint_manifest_path = output_root / "checkpoints_from_isaac_summaries.yaml"
    checkpoint_manifest_path.write_text(yaml.safe_dump(checkpoint_manifest, sort_keys=True), encoding="utf-8")

    run_all_script = output_root / "run_all_exports.sh"
    _write_run_all_script(run_all_script=run_all_script, runner_scripts=[Path(job["runner_script"]) for job in ready_jobs])

    report: dict[str, Any] = {
        "local_root": str(local_root),
        "task_dir": str(task_dir),
        "output_root": str(output_root),
        "exports_dir": str(exports_dir),
        "manifest_path": str(output_root / "isaac_export_manifest.json"),
        "checkpoint_manifest_path": str(checkpoint_manifest_path),
        "run_all_script": str(run_all_script),
        "session_name": _safe_session_name(session_name),
        "launch_requested": bool(launch),
        "launched": False,
        "require_all_ready": bool(require_all_ready),
        "summary_source_count": len(summary_specs),
        "ready_task_count": len(ready_jobs),
        "pending_task_count": len(pending),
        "all_ready": len(pending) == 0 and len(summary_specs) > 0,
        "checkpoint_manifest": checkpoint_manifest,
        "jobs": ready_jobs,
        "pending": pending,
        "policy_fleet_audit_command": _policy_fleet_audit_command(
            python=python,
            task_paths=[Path(job["task_path"]) for job in ready_jobs],
            exports_dir=exports_dir,
            checkpoint_manifest_path=checkpoint_manifest_path,
        ),
    }
    if launch and ready_jobs:
        runner = command_runner or _subprocess_runner
        command = ["tmux", "new-session", "-d", "-s", report["session_name"], f"bash {run_all_script}"]
        runner(command)
        report["launched"] = True
        report["launch_command"] = command
    _write_manifest(report)
    return report


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and optionally launch MuJoCo export jobs from completed Isaac HDMI training summaries."
    )
    parser.add_argument("--local-root", default=str(HDMI_ROOT))
    parser.add_argument("--task-dir", default=str(DEFAULT_TASK_DIR))
    parser.add_argument("--train-manifest", action="append", default=[])
    parser.add_argument("--summary", action="append", default=[])
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--exports-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--algo", default="ppo_roa_train")
    parser.add_argument("--session-name", default="wbc-hdmi-isaac-export")
    parser.add_argument("--require-all-ready", action="store_true")
    parser.add_argument("--launch", action="store_true")
    return parser.parse_args(argv)


def _summary_specs_from_sources(
    *,
    train_manifests: Sequence[Path],
    summary_paths: Sequence[Path],
    task_identities: Mapping[str, Mapping[str, str]],
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for manifest_path in train_manifests:
        manifest = _load_json(manifest_path)
        for job in manifest.get("jobs", []):
            identity = _identity_for_job(job, task_identities)
            if identity is None:
                specs.append(
                    {
                        "summary_path": Path(str(job.get("summary_path", ""))),
                        "identity": _unknown_identity(job),
                        "source": str(manifest_path),
                        "identity_missing": True,
                    }
                )
                continue
            key = identity["task_override"]
            if key in seen:
                continue
            seen.add(key)
            specs.append(
                {
                    "summary_path": Path(str(job.get("summary_path", ""))),
                    "identity": identity,
                    "source": str(manifest_path),
                    "identity_missing": False,
                }
            )

    for summary_path in summary_paths:
        summary = _load_json(summary_path) if summary_path.is_file() else {}
        identity = _identity_for_summary(summary, summary_path=summary_path, task_identities=task_identities)
        key = identity["task_override"] if identity is not None else str(summary_path)
        if key in seen:
            continue
        seen.add(key)
        specs.append(
            {
                "summary_path": summary_path,
                "identity": identity or _unknown_identity({"summary_path": str(summary_path), "task_name": summary.get("task")}),
                "source": "summary",
                "identity_missing": identity is None,
            }
        )
    return specs


def _summary_status(spec: Mapping[str, Any]) -> dict[str, Any]:
    summary_path = Path(spec["summary_path"])
    reasons: list[str] = []
    summary: dict[str, Any] = {}
    if spec.get("identity_missing"):
        reasons.append("task_identity_missing")
    if not summary_path.is_file():
        reasons.append("summary_missing")
    else:
        try:
            summary = _load_json(summary_path)
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            reasons.append("summary_load_error")
            summary = {"load_error": str(exc)}
    checkpoint = summary.get("checkpoint_final") if isinstance(summary, dict) else None
    if summary_path.is_file() and not checkpoint:
        reasons.append("checkpoint_final_missing")
    checkpoint_exists = bool(checkpoint) and Path(str(checkpoint)).is_file()
    if checkpoint and not checkpoint_exists:
        reasons.append("checkpoint_final_file_missing")
    return {
        "ready": not reasons,
        "failure_reasons": reasons,
        "checkpoint_final": str(checkpoint) if checkpoint else None,
        "checkpoint_exists": checkpoint_exists,
        "summary_task": summary.get("task") if isinstance(summary, dict) else None,
        "summary_backend": summary.get("backend") if isinstance(summary, dict) else None,
        "summary_env_frames": summary.get("env_frames") if isinstance(summary, dict) else None,
        "summary_total_frames": summary.get("total_frames") if isinstance(summary, dict) else None,
    }


def _build_export_job(
    *,
    local_root: Path,
    identity: Mapping[str, str],
    checkpoint_path: str,
    log_dir: Path,
    runner_dir: Path,
    python: str,
    algo: str,
) -> dict[str, Any]:
    task_name = identity["task_name"]
    safe_task = _safe_name(task_name)
    log_path = log_dir / f"{safe_task}.log"
    runner_script = runner_dir / f"{safe_task}.sh"
    command = [
        python,
        "scripts/play.py",
        f"algo={algo}",
        f"task={identity['task_override']}",
        "algo.phase=adapt",
        f"checkpoint_path={checkpoint_path}",
        "export_policy=true",
        "export_policy_exit=true",
        "export_policy_benchmark_iters=0",
        "export_onnx_policy=true",
        "export_onnx_required=true",
        "headless=true",
        "backend=mujoco",
    ]
    _write_runner_script(runner_script=runner_script, local_root=local_root, command=command, log_path=log_path)
    return {
        "task_name": task_name,
        "task_override": identity["task_override"],
        "task_path": identity["task_path"],
        "checkpoint_path": checkpoint_path,
        "expected_export_dir": str(HDMI_ROOT / "scripts" / "exports" / task_name),
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
                f"echo ISAAC_EXPORT_EXIT:$status | tee -a {shlex.quote(str(log_path))}",
                'exit "$status"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    os.chmod(runner_script, 0o755)


def _write_run_all_script(*, run_all_script: Path, runner_scripts: Sequence[Path]) -> None:
    lines = ["#!/usr/bin/env bash", "set -uo pipefail", "failures=0"]
    for script in runner_scripts:
        lines.extend(
            [
                f"bash {shlex.quote(str(script))}",
                "status=$?",
                'if [ "$status" -ne 0 ]; then failures=$((failures + 1)); fi',
                'if [ "$status" -ne 0 ]; then',
                '  echo "ISAAC_EXPORT_FAILURES:$failures"',
                '  exit "$failures"',
                "fi",
            ]
        )
    lines.extend(['echo "ISAAC_EXPORT_FAILURES:$failures"', 'exit "$failures"', ""])
    run_all_script.write_text("\n".join(lines), encoding="utf-8")
    os.chmod(run_all_script, 0o755)


def _policy_fleet_audit_command(
    *,
    python: str,
    task_paths: Sequence[Path],
    exports_dir: Path,
    checkpoint_manifest_path: Path,
) -> list[str]:
    command = [python, "scripts/mujoco_hdmi_policy_fleet_audit.py"]
    for task_path in task_paths:
        command.extend(["--task-yaml", str(task_path)])
    command.extend(
        [
            "--exports-dir",
            str(exports_dir),
            "--checkpoint-manifest",
            str(checkpoint_manifest_path),
            "--require-policy",
            "--require-reference-observation",
            "--require-obs-action-smoke",
            "--require-onnx-policy",
        ]
    )
    return command


def _task_identities(task_dir: Path) -> dict[str, dict[str, str]]:
    identities: dict[str, dict[str, str]] = {}
    for task_path in sorted(task_dir.glob("*.yaml")):
        cfg = _load_yaml(task_path)
        task_name = str(cfg.get("name") or task_path.stem)
        identity = {
            "task_name": task_name,
            "task_override": _task_override_from_path(task_path),
            "task_stem": task_path.stem,
            "task_file": task_path.name,
            "task_path": str(task_path),
        }
        for key in (task_name, identity["task_override"], identity["task_stem"], identity["task_file"], identity["task_path"]):
            identities[str(key)] = identity
    return identities


def _identity_for_job(job: Mapping[str, Any], task_identities: Mapping[str, Mapping[str, str]]) -> Mapping[str, str] | None:
    for key in (
        job.get("task_name"),
        job.get("task_override"),
        Path(str(job.get("task_path", ""))).stem,
        Path(str(job.get("task_path", ""))).name,
        job.get("task_path"),
    ):
        if key is not None and str(key) in task_identities:
            return task_identities[str(key)]
    return None


def _identity_for_summary(
    summary: Mapping[str, Any],
    *,
    summary_path: Path,
    task_identities: Mapping[str, Mapping[str, str]],
) -> Mapping[str, str] | None:
    for key in (summary.get("task"), summary_path.stem.replace("_summary", ""), summary_path.parent.name):
        if key is not None and str(key) in task_identities:
            return task_identities[str(key)]
    return None


def _unknown_identity(source: Mapping[str, Any]) -> dict[str, str]:
    task_name = str(source.get("task_name") or source.get("task") or "unknown")
    return {
        "task_name": task_name,
        "task_override": str(source.get("task_override") or task_name),
        "task_stem": task_name,
        "task_file": "",
        "task_path": str(source.get("task_path") or ""),
    }


def _public_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    identity = spec["identity"]
    return {
        "task_name": identity.get("task_name"),
        "task_override": identity.get("task_override"),
        "task_path": identity.get("task_path"),
        "summary_path": str(spec["summary_path"]),
        "source": spec.get("source"),
    }


def _task_override_from_path(task_path: Path) -> str:
    parts = task_path.with_suffix("").parts
    if "task" in parts:
        index = parts.index("task")
        return "/".join(parts[index + 1 :])
    return str(task_path)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected YAML mapping, got {type(data).__name__}.")
    return data


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected JSON object, got {type(data).__name__}.")
    return data


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
