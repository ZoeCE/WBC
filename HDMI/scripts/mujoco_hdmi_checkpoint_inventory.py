from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
HDMI_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, HDMI_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mujoco_policy_export_audit import _local_checkpoint_provenance


SOURCE_REFERENCE = {
    "repository": "https://github.com/LeCAR-Lab/HDMI",
    "golden_reference_definition": (
        "A checkpoint can be a golden reference candidate only when its saved cfg proves "
        "it is an Isaac policy checkpoint. It becomes a validated golden reference only when "
        "closed-loop evidence for the same task succeeds."
    ),
    "export_policy_contract": "python scripts/play.py checkpoint_path=<teacher_or_student_checkpoint> export_policy=true",
}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_checkpoint_inventory(
        checkpoint_roots=args.checkpoint_root,
        exports_dir=args.exports_dir,
        min_golden_total_frames=args.min_golden_total_frames,
        closed_loop_report=Path(args.closed_loop_report) if args.closed_loop_report else None,
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, sort_keys=True))
    return 0


def build_checkpoint_inventory(
    *,
    checkpoint_roots: Sequence[str | Path],
    exports_dir: str | Path | None = None,
    min_golden_total_frames: int = 100_000_000,
    closed_loop_report: Path | None = None,
) -> dict[str, Any]:
    closed_loop_success_by_task = _closed_loop_success_by_task(closed_loop_report)
    checkpoints = [
        _checkpoint_entry(
            path,
            min_golden_total_frames=int(min_golden_total_frames),
            closed_loop_success_by_task=closed_loop_success_by_task,
        )
        for path in _discover_checkpoints(checkpoint_roots)
    ]
    exports = _discover_policy_exports(Path(exports_dir) if exports_dir is not None else HDMI_ROOT / "scripts" / "exports")
    golden_candidates = [entry for entry in checkpoints if entry["golden_candidate"]]
    validated_golden_references = [entry for entry in checkpoints if entry["validated_golden_reference"]]
    weak_mujoco_seeds = [entry for entry in checkpoints if entry["provenance_class"] == "weak_mujoco_seed"]
    by_task: dict[str, list[dict[str, Any]]] = {}
    for entry in checkpoints:
        task_name = entry.get("checkpoint_task_name") or "unknown"
        by_task.setdefault(str(task_name), []).append(entry)

    return {
        "source_reference": SOURCE_REFERENCE,
        "checkpoint_roots": [str(Path(root)) for root in checkpoint_roots],
        "exports_dir": str(Path(exports_dir) if exports_dir is not None else HDMI_ROOT / "scripts" / "exports"),
        "min_golden_total_frames": int(min_golden_total_frames),
        "closed_loop_report": str(closed_loop_report) if closed_loop_report else None,
        "checkpoint_count": len(checkpoints),
        "golden_candidate_count": len(golden_candidates),
        "validated_golden_reference_count": len(validated_golden_references),
        "weak_mujoco_seed_count": len(weak_mujoco_seeds),
        "export_count": len(exports),
        "missing_golden_reference": not bool(golden_candidates),
        "missing_validated_golden_reference": not bool(validated_golden_references),
        "tasks_with_checkpoints": sorted(by_task),
        "checkpoints": checkpoints,
        "exports": exports,
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inventory local HDMI checkpoints and exported policies, classifying which artifacts can "
            "credibly serve as original HDMI/Isaac golden-reference candidates for MuJoCo sim2sim parity."
        )
    )
    parser.add_argument(
        "--checkpoint-root",
        action="append",
        default=[],
        help=(
            "Directory or checkpoint file to scan. Can be passed multiple times. "
            "Defaults to HDMI/outputs when omitted."
        ),
    )
    parser.add_argument("--exports-dir", default=None, help="Policy exports root. Defaults to HDMI/scripts/exports.")
    parser.add_argument("--min-golden-total-frames", type=int, default=100_000_000)
    parser.add_argument(
        "--closed-loop-report",
        default=None,
        help="Optional rollout aggregate JSON used to validate golden candidates by task success.",
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser.parse_args(argv)


def _discover_checkpoints(roots: Sequence[str | Path]) -> list[Path]:
    scan_roots = [Path(root) for root in roots] if roots else [HDMI_ROOT / "outputs"]
    paths: list[Path] = []
    seen: set[Path] = set()
    for root in scan_roots:
        candidates: list[Path]
        if root.is_file():
            candidates = [root]
        elif root.is_dir():
            candidates = sorted(root.rglob("checkpoint*.pt"))
        else:
            candidates = []
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(candidate)
    return sorted(paths, key=lambda path: str(path))


def _checkpoint_entry(
    path: Path,
    *,
    min_golden_total_frames: int,
    closed_loop_success_by_task: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    provenance = _local_checkpoint_provenance(path)
    rejection_reasons = _golden_rejection_reasons(provenance, min_total_frames=int(min_golden_total_frames))
    validation_rejection_reasons = _validated_golden_rejection_reasons(
        provenance,
        rejection_reasons,
        closed_loop_success_by_task=closed_loop_success_by_task,
    )
    provenance_class = _provenance_class(provenance, rejection_reasons)
    return {
        "path": str(path),
        "exists": path.is_file(),
        **provenance,
        "golden_candidate": not rejection_reasons,
        "validated_golden_reference": not validation_rejection_reasons,
        "provenance_class": provenance_class,
        "rejection_reasons": rejection_reasons,
        "validation_rejection_reasons": validation_rejection_reasons,
    }


def _validated_golden_rejection_reasons(
    provenance: dict[str, Any],
    candidate_rejection_reasons: Sequence[str],
    *,
    closed_loop_success_by_task: Mapping[str, bool] | None,
) -> list[str]:
    if candidate_rejection_reasons:
        return ["not_golden_candidate"]
    if closed_loop_success_by_task is None:
        return ["closed_loop_success_missing"]
    task_name = provenance.get("checkpoint_task_name")
    if task_name is None or str(task_name) not in closed_loop_success_by_task:
        return ["closed_loop_success_missing"]
    if closed_loop_success_by_task[str(task_name)] is not True:
        return ["closed_loop_success_false"]
    return []


def _closed_loop_success_by_task(path: Path | None) -> dict[str, bool] | None:
    if path is None:
        return None
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: list[Any] = []
    if isinstance(data, Mapping):
        for key in ("items", "rows", "tasks"):
            value = data.get(key)
            if isinstance(value, list):
                rows.extend(value)
    elif isinstance(data, list):
        rows = list(data)
    by_task: dict[str, bool] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        task_name = _optional_str(row.get("task_name") or row.get("scenario") or row.get("task"))
        if task_name is None:
            continue
        success = row.get("heuristic_success")
        if success is None:
            success = row.get("success")
        if success is None:
            success = row.get("closed_loop_success")
        by_task[task_name] = bool(success)
    return by_task


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _golden_rejection_reasons(provenance: dict[str, Any], *, min_total_frames: int) -> list[str]:
    reasons: list[str] = []
    if provenance.get("checkpoint_cfg_loadable") is not True:
        reasons.append("cfg_not_loadable")
        return reasons

    backend = provenance.get("checkpoint_backend")
    total_frames = provenance.get("checkpoint_total_frames")
    if backend != "isaac":
        reasons.append("not_isaac_backend")
    if not _at_least(total_frames, min_total_frames):
        reasons.append("insufficient_total_frames")
    return reasons


def _provenance_class(provenance: dict[str, Any], rejection_reasons: Sequence[str]) -> str:
    if not rejection_reasons:
        return "golden_reference_candidate"
    if provenance.get("checkpoint_cfg_loadable") is not True:
        return "invalid_or_unreadable_checkpoint"
    backend = provenance.get("checkpoint_backend")
    total_frames = provenance.get("checkpoint_total_frames")
    if backend == "mujoco" and not _at_least(total_frames, 100_000_000):
        return "weak_mujoco_seed"
    if backend == "mujoco":
        return "mujoco_trained_checkpoint"
    if backend == "isaac":
        return "isaac_checkpoint_below_threshold"
    return "unknown_checkpoint"


def _discover_policy_exports(exports_dir: Path) -> list[dict[str, Any]]:
    if not exports_dir.is_dir():
        return []
    exports = []
    for path in sorted(exports_dir.glob("*/policy-*.pt"), key=lambda item: str(item)):
        task_name = path.parent.name
        wandb_id, checkpoint_label = _parse_policy_export_name(path)
        exports.append(
            {
                "path": str(path),
                "task_name": task_name,
                "yaml_path": str(path.with_suffix(".yaml")),
                "yaml_exists": path.with_suffix(".yaml").is_file(),
                "wandb_id": wandb_id,
                "checkpoint_label": checkpoint_label,
                "trusted_as_golden_reference": False,
                "trust_note": "exported policy metadata alone does not prove trained checkpoint provenance",
            }
        )
    return exports


def _parse_policy_export_name(path: Path) -> tuple[str | None, str | None]:
    match = re.fullmatch(r"policy-(?P<run>.+)-(?P<label>[^-]+)\.pt", path.name)
    if not match:
        return None, None
    run_id = match.group("run")
    label = match.group("label")
    return (None if run_id == "unknown" else run_id, None if label == "unknown" else label)


def _at_least(actual: Any, limit: int) -> bool:
    try:
        return float(actual) >= float(limit)
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
