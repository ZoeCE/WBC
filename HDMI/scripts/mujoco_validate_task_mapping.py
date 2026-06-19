import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


HDMI_ROOT = Path(__file__).resolve().parents[1]
if str(HDMI_ROOT) not in sys.path:
    sys.path.insert(0, str(HDMI_ROOT))

from active_adaptation.mujoco.task_mapping import (
    validate_all_task_motion_mappings,
    validate_policy_task_motion_mapping,
    validate_task_motion_mapping,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.task_yaml is not None:
        if args.policy_config is not None:
            report = validate_policy_task_motion_mapping(
                args.policy_config,
                args.task_yaml,
                robot_name=args.robot_name,
            )
            payload = _policy_report_summary(report) if args.summary else report.to_dict()
        else:
            report = validate_task_motion_mapping(args.task_yaml, robot_name=args.robot_name)
            payload = _task_report_summary(report) if args.summary else report.to_dict()
        print(json.dumps(payload, sort_keys=True))
        return 0

    reports = validate_all_task_motion_mappings(args.task_dir, robot_name=args.robot_name)
    if args.summary:
        payload = _task_reports_summary(reports)
    else:
        payload = {
            "num_tasks": len(reports),
            "reports": [report.to_dict() for report in reports],
        }
    print(json.dumps(payload, sort_keys=True))
    if args.require_nonempty and not reports:
        return 1
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate HDMI task YAML, motion meta, and MuJoCo asset name mapping before MuJoCo playback/training."
        )
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--task-yaml",
        default=None,
        help="Single HDMI task YAML to validate.",
    )
    target.add_argument(
        "--task-dir",
        default=None,
        help="Directory of HDMI task YAML files. Only object tasks are validated.",
    )
    parser.add_argument(
        "--robot-name",
        default="g1_29dof",
        help="MuJoCo robot registry name used to resolve assets_mjcf scenes.",
    )
    parser.add_argument(
        "--policy-config",
        default=None,
        help="Optional exported policy YAML to validate against the task motion and MuJoCo MJCF names.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print compact counts suitable for CI logs instead of full per-name mappings.",
    )
    parser.add_argument(
        "--require-nonempty",
        action="store_true",
        help="Return exit code 1 when --task-dir contains no object task mappings.",
    )
    return parser.parse_args(argv)


def _task_reports_summary(reports) -> dict:
    return {
        "num_tasks": len(reports),
        "tasks": [_task_report_summary(report) for report in reports],
    }


def _task_report_summary(report) -> dict:
    return {
        "task_path": str(report.task_path),
        "motion_dir": str(report.motion_dir),
        "object_asset_name": report.object_asset_name,
        "object_type": report.object_type,
        "task_object_body_name": report.task_object_body_name,
        "reference_object_body_name": report.reference_object_body_name,
        "object_joint_name": report.object_joint_name,
        "extra_object_names": list(report.extra_object_names),
        "num_motion_bodies": len(report.motion_body_names),
        "num_motion_joints": len(report.motion_joint_names),
        "num_body_mappings": len(report.body_name_mapping),
        "num_joint_mappings": len(report.joint_name_mapping),
    }


def _policy_report_summary(report) -> dict:
    payload = _task_report_summary(report.task_report)
    payload.update(
        {
            "policy_config_path": str(report.policy_config_path),
            "num_policy_bodies": len(report.policy_body_names),
            "num_policy_joints": len(report.policy_joint_names),
            "num_policy_body_mappings": len(report.policy_body_name_mapping),
            "num_policy_joint_mappings": len(report.policy_joint_name_mapping),
        }
    )
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
