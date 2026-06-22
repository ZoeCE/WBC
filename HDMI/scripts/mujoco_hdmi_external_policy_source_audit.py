from __future__ import annotations

import argparse
import json
import sys
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
HDMI_ROOT = SCRIPT_DIR.parent
DEFAULT_TASK_DIR = HDMI_ROOT / "cfg" / "task" / "G1" / "hdmi"
SOURCE_REFERENCE = {
    "repository": "https://github.com/EGalahad/sim2real",
    "tag": "hdmi",
    "scope": "external_deployment_policy_source",
}

for path in (SCRIPT_DIR, HDMI_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mujoco_policy_export_audit import build_policy_export_audit


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_external_policy_source_audit(
        policy_root=Path(args.policy_root),
        task_dir=Path(args.task_dir),
        task_yamls=[Path(path) for path in args.task_yaml],
        robot_name=args.robot_name,
        expected_task_count=args.expected_task_count,
        require_task_count=args.require_task_count,
        require_reference_observation=args.require_reference_observation,
        require_obs_action_smoke=args.require_obs_action_smoke,
        require_training_provenance=args.require_training_provenance,
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, sort_keys=True))
    if _gate_requested(args) and not report["gate_passed"]:
        return 1
    return 0


def build_external_policy_source_audit(
    *,
    policy_root: Path,
    task_dir: Path = DEFAULT_TASK_DIR,
    task_yamls: Sequence[Path] = (),
    robot_name: str = "g1_29dof",
    expected_task_count: int | None = None,
    require_task_count: bool = False,
    require_reference_observation: bool = False,
    require_obs_action_smoke: bool = False,
    require_training_provenance: bool = False,
) -> dict[str, Any]:
    policy_root = policy_root.resolve()
    tasks = [_task_identity(path) for path in _resolve_task_yamls(task_dir=task_dir, task_yamls=task_yamls)]
    policies = _discover_policy_candidates(policy_root)
    policies_by_motion_path = _policies_by_motion_path(policies)

    task_reports: list[dict[str, Any]] = []
    matched_policy_paths: set[str] = set()
    for task in tasks:
        matches = policies_by_motion_path.get(task["motion_path"], [])
        selected = matches[0] if matches else None
        if selected is not None:
            matched_policy_paths.add(selected["policy_path"])
            audit = build_policy_export_audit(
                task_yaml=task["task_path"],
                policy_path=selected["policy_path"],
                robot_name=robot_name,
                require_reference_observation=require_reference_observation,
                require_obs_action_smoke=require_obs_action_smoke,
            )
            missing_requirements = list(audit["missing_requirements"])
            provenance = selected["training_provenance"]
            training_provenance_ready = provenance["has_actionable_training_provenance"]
            if require_training_provenance and not training_provenance_ready:
                missing_requirements.append("external_policy_training_provenance")
            gate_passed = audit["gate_passed"] is True and (
                training_provenance_ready or not require_training_provenance
            )
            task_reports.append(
                {
                    **task,
                    "policy_source": "external",
                    "policy_source_ready": audit["gate_passed"] is True,
                    "policy_set": selected["policy_set"],
                    "policy_path": selected["policy_path"],
                    "policy_config_path": selected["policy_config_path"],
                    "policy_metadata_path": selected["policy_metadata_path"],
                    "policy_onnx_path": selected["policy_onnx_path"],
                    "candidate_policy_count": len(matches),
                    "candidate_policy_paths": [candidate["policy_path"] for candidate in matches],
                    "training_provenance_ready": training_provenance_ready,
                    "training_provenance": provenance,
                    "gate_passed": gate_passed,
                    "missing_requirements": missing_requirements,
                    "audit": audit,
                }
            )
            continue
        task_reports.append(
            {
                **task,
                "policy_source": "external",
                "policy_source_ready": False,
                "policy_set": None,
                "policy_path": None,
                "policy_config_path": None,
                "policy_metadata_path": None,
                "policy_onnx_path": None,
                "candidate_policy_count": 0,
                "candidate_policy_paths": [],
                "gate_passed": False,
                "missing_requirements": ["external_policy_for_task_motion"],
                "audit": None,
            }
        )

    unmatched = [
        policy for policy in policies if policy["policy_path"] not in matched_policy_paths
    ]
    task_count = len(task_reports)
    ready_task_count = sum(1 for task in task_reports if task["gate_passed"])
    task_count_gate_passed = expected_task_count is None or task_count == expected_task_count
    gate_passed = ready_task_count == task_count
    if require_task_count and not task_count_gate_passed:
        gate_passed = False

    return {
        "source_reference": SOURCE_REFERENCE,
        "policy_root": str(policy_root),
        "task_dir": str(task_dir),
        "task_count": task_count,
        "expected_task_count": expected_task_count,
        "task_count_gate_passed": task_count_gate_passed,
        "policy_count": len(policies),
        "matched_policy_count": len(matched_policy_paths),
        "unmatched_policy_count": len(unmatched),
        "ready_task_count": ready_task_count,
        "not_ready_task_count": task_count - ready_task_count,
        "require_training_provenance": require_training_provenance,
        "training_provenance_policy_count": sum(
            1 for policy in policies if policy["training_provenance"]["has_actionable_training_provenance"]
        ),
        "missing_training_provenance_policy_count": sum(
            1 for policy in policies if not policy["training_provenance"]["has_actionable_training_provenance"]
        ),
        "gate_passed": gate_passed,
        "missing_requirements_by_task": {
            task["task_name"]: task["missing_requirements"]
            for task in task_reports
            if task["missing_requirements"]
        },
        "tasks": task_reports,
        "policies": policies,
        "unmatched_policies": unmatched,
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit external HDMI deployment policy sources, such as EGalahad/sim2real@hdmi, "
            "by matching exported policy YAML motion_path fields to local HDMI task YAMLs."
        )
    )
    parser.add_argument("--policy-root", required=True)
    parser.add_argument("--task-dir", default=str(DEFAULT_TASK_DIR))
    parser.add_argument("--task-yaml", action="append", default=[])
    parser.add_argument("--robot-name", default="g1_29dof")
    parser.add_argument("--expected-task-count", type=int, default=None)
    parser.add_argument("--require-task-count", action="store_true")
    parser.add_argument("--require-reference-observation", action="store_true")
    parser.add_argument("--require-obs-action-smoke", action="store_true")
    parser.add_argument("--require-training-provenance", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


def _resolve_task_yamls(*, task_dir: Path, task_yamls: Sequence[Path]) -> list[Path]:
    if task_yamls:
        return sorted(task_yamls, key=lambda path: str(path))
    return sorted(task_dir.glob("*.yaml"))


def _task_identity(task_path: Path) -> dict[str, Any]:
    cfg = _load_yaml_mapping(task_path)
    task_name = str(cfg.get("name") or task_path.stem)
    command = _mapping(cfg.get("command"))
    motion_path = _optional_str(command.get("data_path"))
    return {
        "task_name": task_name,
        "task_override": _task_override_from_path(task_path),
        "task_stem": task_path.stem,
        "task_file": task_path.name,
        "task_path": str(task_path),
        "motion_path": motion_path,
    }


def _discover_policy_candidates(policy_root: Path) -> list[dict[str, Any]]:
    candidates = []
    for policy_path in sorted(policy_root.rglob("policy-*.pt"), key=lambda path: str(path)):
        config_path = policy_path.with_suffix(".yaml")
        metadata_path = policy_path.with_suffix(".json")
        onnx_path = policy_path.with_suffix(".onnx")
        cfg = _load_yaml_mapping(config_path) if config_path.is_file() else {}
        metadata = _load_json_mapping(metadata_path) if metadata_path.is_file() else {}
        filename_meta = _policy_filename_metadata(policy_path)
        candidates.append(
            {
                "policy_set": policy_path.parent.name,
                "policy_path": str(policy_path),
                "policy_config_path": str(config_path) if config_path.is_file() else None,
                "policy_metadata_path": str(metadata_path) if metadata_path.is_file() else None,
                "policy_onnx_path": str(onnx_path) if onnx_path.is_file() else None,
                "policy_size_bytes": policy_path.stat().st_size,
                "policy_config_size_bytes": config_path.stat().st_size if config_path.is_file() else None,
                "policy_metadata_size_bytes": metadata_path.stat().st_size if metadata_path.is_file() else None,
                "policy_onnx_size_bytes": onnx_path.stat().st_size if onnx_path.is_file() else None,
                "policy_filename_run_id": filename_meta["run_id"],
                "policy_filename_checkpoint_label": filename_meta["checkpoint_label"],
                "policy_config_top_level_keys": sorted(cfg.keys()),
                "policy_metadata_top_level_keys": sorted(metadata.keys()),
                "motion_paths": _unique_sorted(_collect_values_by_key(cfg, "motion_path")),
                "in_keys": metadata.get("in_keys"),
                "in_shapes": metadata.get("in_shapes"),
                "training_provenance": _training_provenance(policy_path=policy_path, cfg=cfg, metadata=metadata),
            }
        )
    return candidates


def _policies_by_motion_path(policies: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_motion_path: dict[str, list[dict[str, Any]]] = {}
    for policy in policies:
        for motion_path in policy["motion_paths"]:
            by_motion_path.setdefault(str(motion_path), []).append(dict(policy))
    for matches in by_motion_path.values():
        matches.sort(key=lambda policy: policy["policy_path"])
    return by_motion_path


def _collect_values_by_key(value: Any, key: str) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for item_key, item_value in value.items():
            if str(item_key) == key and item_value is not None:
                found.append(str(item_value))
            found.extend(_collect_values_by_key(item_value, key))
    elif isinstance(value, list):
        for item in value:
            found.extend(_collect_values_by_key(item, key))
    return found


def _unique_sorted(values: Sequence[str]) -> list[str]:
    return sorted(set(values))


def _policy_filename_metadata(policy_path: Path) -> dict[str, str | None]:
    match = re.match(r"^policy-(?P<run_id>.+)-(?P<label>[^.]+)\.pt$", policy_path.name)
    if match is None:
        return {"run_id": None, "checkpoint_label": None}
    return {
        "run_id": match.group("run_id"),
        "checkpoint_label": match.group("label"),
    }


def _training_provenance(*, policy_path: Path, cfg: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    combined = {"policy_yaml": cfg, "policy_json": metadata}
    git_commit_candidates = _unique_sorted(
        _collect_values_by_key_predicate(
            combined,
            lambda key: "commit" in key.lower() or key.lower() in {"git_sha", "git_hash", "sha", "head_sha"},
        )
    )
    backend_candidates = _unique_sorted(
        _collect_values_by_key_predicate(combined, lambda key: key.lower() == "backend")
    )
    checkpoint_candidates = _unique_sorted(
        _collect_values_by_key_predicate(
            combined,
            lambda key: "checkpoint" in key.lower() or key.lower() in {"ckpt", "resume", "resume_from"},
        )
    )
    wandb_candidates = _unique_sorted(
        _collect_values_by_key_predicate(
            combined,
            lambda key: "wandb" in key.lower() or key.lower() in {"run", "run_id", "run_path", "project", "entity"},
        )
        + _collect_strings_matching(combined, _looks_like_wandb_reference)
    )
    concrete_wandb_run_paths = [
        value for value in wandb_candidates if _looks_like_concrete_wandb_run(value)
    ]
    filename_meta = _policy_filename_metadata(policy_path)
    missing_fields = []
    if not git_commit_candidates:
        missing_fields.append("git_commit")
    if not checkpoint_candidates:
        missing_fields.append("training_checkpoint")
    if not concrete_wandb_run_paths:
        missing_fields.append("wandb_run_path")
    if not backend_candidates:
        missing_fields.append("training_backend")
    has_actionable = bool(git_commit_candidates or checkpoint_candidates or concrete_wandb_run_paths)
    return {
        "policy_filename_run_id": filename_meta["run_id"],
        "policy_filename_checkpoint_label": filename_meta["checkpoint_label"],
        "git_commit_candidates": git_commit_candidates,
        "checkpoint_candidates": checkpoint_candidates,
        "wandb_candidates": wandb_candidates,
        "concrete_wandb_run_paths": concrete_wandb_run_paths,
        "backend_candidates": backend_candidates,
        "missing_fields": missing_fields,
        "has_actionable_training_provenance": has_actionable,
    }


def _collect_values_by_key_predicate(value: Any, predicate) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for item_key, item_value in value.items():
            if predicate(str(item_key)):
                found.extend(_string_leaf_values(item_value))
            found.extend(_collect_values_by_key_predicate(item_value, predicate))
    elif isinstance(value, list):
        for item in value:
            found.extend(_collect_values_by_key_predicate(item, predicate))
    return found


def _collect_strings_matching(value: Any, predicate) -> list[str]:
    found: list[str] = []
    if isinstance(value, str):
        if predicate(value):
            found.append(value)
    elif isinstance(value, Mapping):
        for item in value.values():
            found.extend(_collect_strings_matching(item, predicate))
    elif isinstance(value, list):
        for item in value:
            found.extend(_collect_strings_matching(item, predicate))
    return found


def _string_leaf_values(value: Any) -> list[str]:
    if isinstance(value, (str, int, float, bool)):
        return [str(value)]
    if isinstance(value, Mapping):
        result: list[str] = []
        for nested in value.values():
            result.extend(_string_leaf_values(nested))
        return result
    if isinstance(value, list):
        result: list[str] = []
        for nested in value:
            result.extend(_string_leaf_values(nested))
        return result
    return []


def _looks_like_wandb_reference(value: str) -> bool:
    return "wandb.ai/" in value or value.startswith("run:")


def _looks_like_concrete_wandb_run(value: str) -> bool:
    if "wandb_run_path" in value or "<" in value or "your_" in value.lower():
        return False
    if value.startswith("run:"):
        return len(value.split(":")[-1].split("/")) >= 3
    return "wandb.ai/" in value and "/runs/" in value


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected YAML mapping, got {type(data).__name__}.")
    return data


def _load_json_mapping(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected JSON object, got {type(data).__name__}.")
    return data


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _task_override_from_path(task_path: Path) -> str:
    parts = task_path.with_suffix("").parts
    if "task" in parts:
        index = parts.index("task")
        return "/".join(parts[index + 1 :])
    return str(task_path.with_suffix(""))


def _gate_requested(args: argparse.Namespace) -> bool:
    return (
        args.require_task_count
        or args.require_reference_observation
        or args.require_obs_action_smoke
        or args.require_training_provenance
    )


if __name__ == "__main__":
    raise SystemExit(main())
