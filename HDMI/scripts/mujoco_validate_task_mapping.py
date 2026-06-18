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
    validate_task_motion_mapping,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.task_yaml is not None:
        report = validate_task_motion_mapping(args.task_yaml, robot_name=args.robot_name)
        print(json.dumps(report.to_dict(), sort_keys=True))
        return 0

    reports = validate_all_task_motion_mappings(args.task_dir, robot_name=args.robot_name)
    print(
        json.dumps(
            {
                "num_tasks": len(reports),
                "reports": [report.to_dict() for report in reports],
            },
            sort_keys=True,
        )
    )
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
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
