from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


HDMI_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (HDMI_ROOT, SCRIPT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from active_adaptation.mujoco.external_payloads import (  # noqa: E402
    audit_external_payloads,
    load_external_payload_manifest,
)
from active_adaptation.mujoco.task_mapping import (  # noqa: E402
    TaskMotionMappingReport,
    validate_all_task_motion_mappings,
)
from mujoco_component_audit import COMPONENT_SPECS  # noqa: E402
from mujoco_train_summary_gate import build_gate_report  # noqa: E402


DEFAULT_PAYLOAD_MANIFEST = HDMI_ROOT / "mujoco_external_payloads.yaml"
DEFAULT_TASK_DIR = HDMI_ROOT / "cfg/task/G1/hdmi"
REQUIRED_COMPONENT_REPORT_COVERAGE = tuple(COMPONENT_SPECS)
REQUIRED_PLAYBACK_THRESHOLD_KEYS = ("max_q_l2", "max_body_pos_l2", "min_reward_mean")
REQUIRED_PLAYBACK_METRIC_KEYS = ("q_l2_max", "body_pos_l2_max", "reward_mean")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_migration_audit(
        payload_manifest=Path(args.payload_manifest),
        payload_root=Path(args.payload_root),
        verify_payload_sha256=args.verify_payload_sha256,
        require_payloads=args.require_payloads,
        task_dir=Path(args.task_dir),
        robot_name=args.robot_name,
        require_task_mappings=args.require_task_mappings,
        min_task_mappings=args.min_task_mappings,
        training_summaries=[Path(path) for path in args.training_summary],
        require_training_summaries=args.require_training_summaries,
        training_backend=args.training_backend,
        min_training_env_frames=args.min_training_env_frames,
        min_training_summaries=args.min_training_summaries,
        min_training_eval_metrics=_parse_metric_thresholds(args.min_training_eval_metric),
        max_training_eval_metrics=_parse_metric_thresholds(args.max_training_eval_metric),
        min_training_train_metrics=_parse_metric_thresholds(args.min_training_train_metric),
        max_training_train_metrics=_parse_metric_thresholds(args.max_training_train_metric),
        policy_export_reports=[Path(path) for path in args.policy_export_report],
        require_policy_export=args.require_policy_export,
        playback_parity_reports=[Path(path) for path in args.playback_parity_report],
        require_playback_parity=args.require_playback_parity,
        component_reports=[Path(path) for path in args.component_report],
        require_component_reports=args.require_component_reports,
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["migration_passed"] else 1


def build_migration_audit(
    *,
    payload_manifest: Path = DEFAULT_PAYLOAD_MANIFEST,
    payload_root: Path = HDMI_ROOT,
    verify_payload_sha256: bool = False,
    require_payloads: bool = False,
    task_dir: Path = DEFAULT_TASK_DIR,
    robot_name: str = "g1_29dof",
    require_task_mappings: bool = False,
    min_task_mappings: int | None = None,
    training_summaries: Sequence[Path] = (),
    require_training_summaries: bool = False,
    training_backend: str | None = "mujoco",
    min_training_env_frames: int | None = None,
    min_training_summaries: int | None = None,
    min_training_eval_metrics: dict[str, float] | None = None,
    max_training_eval_metrics: dict[str, float] | None = None,
    min_training_train_metrics: dict[str, float] | None = None,
    max_training_train_metrics: dict[str, float] | None = None,
    policy_export_reports: Sequence[Path] = (),
    require_policy_export: bool = False,
    playback_parity_reports: Sequence[Path] = (),
    require_playback_parity: bool = False,
    component_reports: Sequence[Path] = (),
    require_component_reports: bool = False,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []

    payloads = _build_payload_report(
        manifest_path=payload_manifest,
        root=payload_root,
        verify_sha256=verify_payload_sha256,
    )
    if require_payloads and not payloads["gate_passed"]:
        failures.append(
            {
                "component": "payloads",
                "reason": "payload_gate_failed",
                "failures": payloads.get("failures", []),
            }
        )

    task_mapping = _build_task_mapping_report(
        task_dir=task_dir,
        robot_name=robot_name,
        min_task_mappings=min_task_mappings if min_task_mappings is not None else (1 if require_task_mappings else None),
    )
    if (require_task_mappings or min_task_mappings is not None) and not task_mapping["gate_passed"]:
        failures.append(
            {
                "component": "task_mapping",
                "reason": "task_mapping_gate_failed",
                "failures": task_mapping.get("failures", []),
            }
        )

    training_required = _training_gate_requested(
        training_summaries=training_summaries,
        require_training_summaries=require_training_summaries,
        min_training_env_frames=min_training_env_frames,
        min_training_summaries=min_training_summaries,
        min_training_eval_metrics=min_training_eval_metrics or {},
        max_training_eval_metrics=max_training_eval_metrics or {},
        min_training_train_metrics=min_training_train_metrics or {},
        max_training_train_metrics=max_training_train_metrics or {},
    )
    training = _build_training_report(
        summaries=training_summaries,
        require_backend=training_backend if training_required else None,
        require_checkpoint=training_required,
        min_env_frames=min_training_env_frames,
        min_num_summaries=(
            min_training_summaries
            if min_training_summaries is not None
            else (1 if require_training_summaries else None)
        ),
        min_eval_metrics=min_training_eval_metrics or {},
        max_eval_metrics=max_training_eval_metrics or {},
        min_train_metrics=min_training_train_metrics or {},
        max_train_metrics=max_training_train_metrics or {},
    )
    if training_required and not training["gate_passed"]:
        failures.append(
            {
                "component": "training",
                "reason": "training_gate_failed",
                "failures": training.get("failures", []),
            }
        )

    policy_export_required = require_policy_export or bool(policy_export_reports)
    policy_export = _build_json_report_gate(
        report_paths=policy_export_reports,
        pass_key="gate_passed",
        require_reports=require_policy_export,
    )
    if policy_export_required and not policy_export["gate_passed"]:
        failures.append(
            {
                "component": "policy_export",
                "reason": "policy_export_gate_failed",
                "failures": policy_export.get("failures", []),
            }
        )

    playback_parity_required = require_playback_parity or bool(playback_parity_reports)
    playback_parity = _build_playback_parity_report_gate(
        report_paths=playback_parity_reports,
        require_reports=require_playback_parity,
    )
    if playback_parity_required and not playback_parity["gate_passed"]:
        failures.append(
            {
                "component": "playback_parity",
                "reason": "playback_parity_gate_failed",
                "failures": playback_parity.get("failures", []),
            }
        )

    component_reports_required = require_component_reports or bool(component_reports)
    component_report = _build_component_report_gate(
        report_paths=component_reports,
        require_reports=require_component_reports,
        required_components=REQUIRED_COMPONENT_REPORT_COVERAGE,
    )
    if component_reports_required and not component_report["gate_passed"]:
        failures.append(
            {
                "component": "component_reports",
                "reason": "component_report_gate_failed",
                "failures": component_report.get("failures", []),
            }
        )

    return {
        "migration_passed": not failures,
        "failures": failures,
        "payloads": payloads,
        "task_mapping": task_mapping,
        "training": training,
        "policy_export": policy_export,
        "playback_parity": playback_parity,
        "component_reports": component_report,
    }


def _build_payload_report(*, manifest_path: Path, root: Path, verify_sha256: bool) -> dict[str, Any]:
    try:
        manifest = load_external_payload_manifest(manifest_path)
        report = audit_external_payloads(manifest, root=root, verify_sha256=verify_sha256)
    except Exception as exc:
        return {
            "gate_passed": False,
            "manifest_path": str(manifest_path),
            "root": str(root),
            "verify_sha256": verify_sha256,
            "failures": [
                {
                    "reason": "payload_audit_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            ],
        }

    failures = []
    for reason in ("required_missing", "size_mismatch", "sha256_mismatch"):
        entries = report.get(reason) or []
        if entries:
            failures.append({"reason": reason, "entries": entries})
    return {
        "gate_passed": not failures,
        "manifest_path": str(manifest_path),
        "root": str(root),
        "verify_sha256": verify_sha256,
        "failures": failures,
        **report,
    }


def _build_task_mapping_report(
    *,
    task_dir: Path,
    robot_name: str,
    min_task_mappings: int | None,
) -> dict[str, Any]:
    try:
        reports = validate_all_task_motion_mappings(task_dir, robot_name=robot_name)
    except Exception as exc:
        return {
            "gate_passed": False,
            "task_dir": str(task_dir),
            "robot_name": robot_name,
            "num_tasks": 0,
            "min_task_mappings": min_task_mappings,
            "tasks": [],
            "failures": [
                {
                    "reason": "task_mapping_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            ],
        }

    failures = []
    if min_task_mappings is not None and len(reports) < min_task_mappings:
        failures.append(
            {
                "reason": "task_mappings_below_min",
                "actual": len(reports),
                "limit": int(min_task_mappings),
            }
        )
    return {
        "gate_passed": not failures,
        "task_dir": str(task_dir),
        "robot_name": robot_name,
        "num_tasks": len(reports),
        "min_task_mappings": min_task_mappings,
        "tasks": [_task_mapping_record(report) for report in reports],
        "failures": failures,
    }


def _task_mapping_record(report: TaskMotionMappingReport) -> dict[str, Any]:
    return {
        "task_path": str(report.task_path),
        "motion_dir": str(report.motion_dir),
        "object_asset_name": report.object_asset_name,
        "object_type": report.object_type,
        "task_object_body_name": report.task_object_body_name,
        "reference_object_body_name": report.reference_object_body_name,
        "object_joint_name": report.object_joint_name,
        "asset_object_body_names": list(report.asset_object_body_names),
        "asset_object_joint_names": list(report.asset_object_joint_names),
        "extra_object_names": list(report.extra_object_names),
        "num_motion_body_names": len(report.motion_body_names),
        "num_motion_joint_names": len(report.motion_joint_names),
        "body_name_mapping": [entry.to_dict() for entry in report.body_name_mapping],
        "joint_name_mapping": [entry.to_dict() for entry in report.joint_name_mapping],
    }


def _build_training_report(
    *,
    summaries: Sequence[Path],
    require_backend: str | None,
    require_checkpoint: bool,
    min_env_frames: int | None,
    min_num_summaries: int | None,
    min_eval_metrics: dict[str, float],
    max_eval_metrics: dict[str, float],
    min_train_metrics: dict[str, float],
    max_train_metrics: dict[str, float],
) -> dict[str, Any]:
    return build_gate_report(
        summary_paths=summaries,
        require_backend=require_backend,
        require_checkpoint=require_checkpoint,
        min_env_frames=min_env_frames,
        min_num_summaries=min_num_summaries,
        min_eval_metrics=min_eval_metrics,
        max_eval_metrics=max_eval_metrics,
        min_train_metrics=min_train_metrics,
        max_train_metrics=max_train_metrics,
    )


def _build_json_report_gate(
    *,
    report_paths: Sequence[Path],
    pass_key: str,
    require_reports: bool,
) -> dict[str, Any]:
    reports = []
    failures: list[dict[str, Any]] = []

    for path in report_paths:
        report, load_failure = _load_json_report(path)
        if load_failure is not None:
            failures.append(load_failure)
            continue
        reports.append({"path": str(path), "report": report})
        if report.get(pass_key) is not True:
            failures.append(
                {
                    "report_path": str(path),
                    "reason": f"{pass_key}_not_true",
                    "actual": report.get(pass_key),
                    "missing_requirements": report.get("missing_requirements"),
                    "threshold_failures": report.get("threshold_failures"),
                    "component_failures": report.get("component_failures"),
                }
            )

    if require_reports and not reports:
        failures.append(
            {
                "reason": "num_reports_below_min",
                "actual": len(reports),
                "limit": 1,
            }
        )

    return {
        "gate_passed": not failures,
        "pass_key": pass_key,
        "num_reports": len(reports),
        "report_paths": [str(path) for path in report_paths],
        "reports": reports,
        "failures": failures,
    }


def _build_playback_parity_report_gate(
    *,
    report_paths: Sequence[Path],
    require_reports: bool,
) -> dict[str, Any]:
    gate = _build_json_report_gate(
        report_paths=report_paths,
        pass_key="parity_passed",
        require_reports=require_reports,
    )
    failures = list(gate["failures"])
    required_thresholds = list(REQUIRED_PLAYBACK_THRESHOLD_KEYS)
    required_metrics = list(REQUIRED_PLAYBACK_METRIC_KEYS)

    for entry in gate["reports"]:
        report = entry["report"]
        missing_metrics = [key for key in required_metrics if report.get(key) is None]
        if missing_metrics:
            failures.append(
                {
                    "report_path": entry["path"],
                    "reason": "missing_playback_metrics",
                    "missing_metrics": missing_metrics,
                }
            )

        thresholds = report.get("thresholds")
        threshold_keys = set(thresholds) if isinstance(thresholds, dict) else set()
        missing_thresholds = [key for key in required_thresholds if key not in threshold_keys]
        if missing_thresholds:
            failures.append(
                {
                    "report_path": entry["path"],
                    "reason": "missing_playback_thresholds",
                    "missing_thresholds": missing_thresholds,
                }
            )

    return {
        **gate,
        "gate_passed": not failures,
        "required_metrics": required_metrics,
        "required_thresholds": required_thresholds,
        "failures": failures,
    }


def _build_component_report_gate(
    *,
    report_paths: Sequence[Path],
    require_reports: bool,
    required_components: Sequence[str],
) -> dict[str, Any]:
    gate = _build_json_report_gate(
        report_paths=report_paths,
        pass_key="component_gate_passed",
        require_reports=require_reports,
    )
    failures = list(gate["failures"])
    required = list(required_components)
    for entry in gate["reports"]:
        report = entry["report"]
        covered = report.get("covered_components")
        covered_set = set(covered) if isinstance(covered, list) else set()
        missing_components = [component for component in required if component not in covered_set]
        if missing_components:
            failures.append(
                {
                    "report_path": entry["path"],
                    "reason": "covered_components_missing",
                    "missing_components": missing_components,
                }
            )
    return {
        **gate,
        "gate_passed": not failures,
        "required_components": required,
        "failures": failures,
    }


def _load_json_report(path: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, {
            "report_path": str(path),
            "reason": "report_load_error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    if not isinstance(report, dict):
        return {}, {
            "report_path": str(path),
            "reason": "report_not_mapping",
            "actual_type": type(report).__name__,
        }
    return report, None


def _training_gate_requested(
    *,
    training_summaries: Sequence[Path],
    require_training_summaries: bool,
    min_training_env_frames: int | None,
    min_training_summaries: int | None,
    min_training_eval_metrics: dict[str, float],
    max_training_eval_metrics: dict[str, float],
    min_training_train_metrics: dict[str, float],
    max_training_train_metrics: dict[str, float],
) -> bool:
    return any(
        (
            training_summaries,
            require_training_summaries,
            min_training_env_frames is not None,
            min_training_summaries is not None,
            min_training_eval_metrics,
            max_training_eval_metrics,
            min_training_train_metrics,
            max_training_train_metrics,
        )
    )


def _parse_metric_thresholds(raw: Sequence[Sequence[str]] | None) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for metric, value in raw or []:
        thresholds[metric] = float(value)
    return thresholds


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate MuJoCo migration evidence across assets, mappings, training, playback, and components."
    )
    parser.add_argument("--payload-manifest", default=str(DEFAULT_PAYLOAD_MANIFEST))
    parser.add_argument("--payload-root", default=str(HDMI_ROOT))
    parser.add_argument("--verify-payload-sha256", action="store_true")
    parser.add_argument("--require-payloads", action="store_true")

    parser.add_argument("--task-dir", default=str(DEFAULT_TASK_DIR))
    parser.add_argument("--robot-name", default="g1_29dof")
    parser.add_argument("--require-task-mappings", action="store_true")
    parser.add_argument("--min-task-mappings", type=int, default=None)

    parser.add_argument("--training-summary", action="append", default=[], metavar="PATH")
    parser.add_argument("--require-training-summaries", action="store_true")
    parser.add_argument("--training-backend", default="mujoco")
    parser.add_argument("--min-training-env-frames", type=int, default=None)
    parser.add_argument("--min-training-summaries", type=int, default=None)
    parser.add_argument("--min-training-eval-metric", nargs=2, action="append", default=[], metavar=("KEY", "VALUE"))
    parser.add_argument("--max-training-eval-metric", nargs=2, action="append", default=[], metavar=("KEY", "VALUE"))
    parser.add_argument("--min-training-train-metric", nargs=2, action="append", default=[], metavar=("KEY", "VALUE"))
    parser.add_argument("--max-training-train-metric", nargs=2, action="append", default=[], metavar=("KEY", "VALUE"))

    parser.add_argument("--policy-export-report", action="append", default=[], metavar="PATH")
    parser.add_argument("--require-policy-export", action="store_true")
    parser.add_argument("--playback-parity-report", action="append", default=[], metavar="PATH")
    parser.add_argument("--require-playback-parity", action="store_true")
    parser.add_argument("--component-report", action="append", default=[], metavar="PATH")
    parser.add_argument("--require-component-reports", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
