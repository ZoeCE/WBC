import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


HDMI_ROOT = Path(__file__).resolve().parents[1]
if str(HDMI_ROOT) not in sys.path:
    sys.path.insert(0, str(HDMI_ROOT))

from active_adaptation.mujoco.external_payloads import (  # noqa: E402
    audit_external_payloads,
    discover_task_motion_payloads,
    load_external_payload_manifest,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = load_external_payload_manifest(args.manifest)
    report = audit_external_payloads(
        manifest,
        root=args.root,
        verify_sha256=args.verify_sha256,
    )
    if args.include_task_motion:
        report["task_motion_payloads"] = {
            path.as_posix(): [task.as_posix() for task in tasks]
            for path, tasks in discover_task_motion_payloads(args.task_root).items()
        }
    print(json.dumps(report, sort_keys=True))
    if args.require_present and (report["required_missing"] or report["size_mismatch"] or report["sha256_mismatch"]):
        return 1
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit external MuJoCo migration payload files.")
    parser.add_argument(
        "--manifest",
        default=str(HDMI_ROOT / "mujoco_external_payloads.yaml"),
        help="Path to the tracked external payload manifest.",
    )
    parser.add_argument(
        "--root",
        default=str(HDMI_ROOT),
        help="HDMI root used to resolve manifest-relative payload paths.",
    )
    parser.add_argument(
        "--task-root",
        default=str(HDMI_ROOT / "cfg/task"),
        help="Task config root used when --include-task-motion is set.",
    )
    parser.add_argument("--include-task-motion", action="store_true")
    parser.add_argument("--verify-sha256", action="store_true")
    parser.add_argument("--require-present", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
