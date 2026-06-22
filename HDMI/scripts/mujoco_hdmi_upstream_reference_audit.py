from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
HDMI_ROOT = SCRIPT_DIR.parent
DEFAULT_TASK_RELATIVE_DIR = Path("cfg/task/G1/hdmi")
UPSTREAM_REFERENCE = {
    "repository": "https://github.com/LeCAR-Lab/HDMI",
    "contracts": {
        "policy_export": "scripts/play.py with export_policy=true",
        "mujoco_motion_viewer": [
            "scripts/vis/mujoco_mocap_viewer.py",
            "scripts/vis/motion_data_publisher.py",
        ],
        "task_source": "cfg/task/G1/hdmi/*.yaml",
        "scope": "MuJoCo README viewer covers motion.npz playback; closed-loop policy parity requires exported checkpoints.",
    },
}
README_REQUIRED_PATTERNS = {
    "readme_play_entry": "scripts/play.py",
    "readme_export_policy": "export_policy=true",
    "readme_mujoco_viewer": "scripts/vis/mujoco_mocap_viewer.py",
    "readme_motion_publisher": "scripts/vis/motion_data_publisher.py",
}
VIEWER_FILES = (
    Path("scripts/vis/mujoco_mocap_viewer.py"),
    Path("scripts/vis/motion_data_publisher.py"),
)
EXPORT_FILE = Path("scripts/play.py")
TASK_SEMANTIC_FIELDS = (
    "name",
    "robot.robot_type",
    "command.data_path",
    "command.object_asset_name",
    "command.object_body_name",
    "command.object_joint_name",
    "command.contact_eef_body_name",
    "command.extra_object_names",
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_upstream_reference_audit(
        local_root=Path(args.local_root),
        upstream_root=Path(args.upstream_root),
        task_relative_dir=Path(args.task_relative_dir),
        expected_task_count=args.expected_task_count,
        require_task_count=args.require_task_count,
        require_readme_contract=args.require_readme_contract,
        require_viewer_contract=args.require_viewer_contract,
        require_export_contract=args.require_export_contract,
        require_task_semantic_fields=args.require_task_semantic_fields,
        require_task_yaml_identical=args.require_task_yaml_identical,
        require_object_assets=args.require_object_assets,
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["gate_passed"] else 1


def build_upstream_reference_audit(
    *,
    local_root: Path,
    upstream_root: Path,
    task_relative_dir: Path = DEFAULT_TASK_RELATIVE_DIR,
    expected_task_count: int | None = None,
    require_task_count: bool = False,
    require_readme_contract: bool = False,
    require_viewer_contract: bool = False,
    require_export_contract: bool = False,
    require_task_semantic_fields: bool = False,
    require_task_yaml_identical: bool = False,
    require_object_assets: bool = False,
) -> dict[str, Any]:
    local_root = local_root.resolve()
    upstream_root = upstream_root.resolve()
    local_tasks = _task_files(local_root, task_relative_dir)
    upstream_tasks = _task_files(upstream_root, task_relative_dir)
    inventory = _build_task_inventory(
        local_tasks=local_tasks,
        upstream_tasks=upstream_tasks,
        expected_task_count=expected_task_count,
    )
    readme_contract = _build_readme_contract(upstream_root)
    viewer_contract = _build_file_contract(local_root, upstream_root, VIEWER_FILES)
    export_contract = _build_export_contract(local_root, upstream_root)
    task_semantic_fields = _build_task_semantic_field_report(local_tasks, upstream_tasks)
    task_yaml_digests = _build_task_yaml_digest_report(local_tasks, upstream_tasks)
    object_assets = _build_object_asset_report(local_root, upstream_root, local_tasks, upstream_tasks)

    failures: list[dict[str, Any]] = []
    if require_task_count and not inventory["gate_passed"]:
        failures.append(
            {
                "component": "task_inventory",
                "reason": "task_inventory_gate_failed",
                "missing_local_tasks": inventory["missing_local_tasks"],
                "missing_upstream_tasks": inventory["missing_upstream_tasks"],
                "expected_task_count": expected_task_count,
                "local_task_count": inventory["local_task_count"],
                "upstream_task_count": inventory["upstream_task_count"],
            }
        )
    if require_readme_contract and not readme_contract["gate_passed"]:
        failures.append(
            {
                "component": "readme_contract",
                "reason": "readme_contract_gate_failed",
                "missing_patterns": readme_contract["missing_patterns"],
            }
        )
    if require_viewer_contract and not viewer_contract["gate_passed"]:
        failures.append(
            {
                "component": "viewer_contract",
                "reason": "viewer_contract_gate_failed",
                "missing_local_files": viewer_contract["missing_local_files"],
                "missing_upstream_files": viewer_contract["missing_upstream_files"],
            }
        )
    if require_export_contract and not export_contract["gate_passed"]:
        failures.append(
            {
                "component": "export_contract",
                "reason": "export_contract_gate_failed",
                "missing_local_file": export_contract["missing_local_file"],
                "missing_upstream_file": export_contract["missing_upstream_file"],
                "missing_local_patterns": export_contract["missing_local_patterns"],
                "missing_upstream_patterns": export_contract["missing_upstream_patterns"],
            }
        )
    if require_task_semantic_fields and not task_semantic_fields["gate_passed"]:
        failures.append(
            {
                "component": "task_semantic_fields",
                "reason": "task_semantic_fields_gate_failed",
                "mismatches": task_semantic_fields["mismatches"],
            }
        )
    if require_task_yaml_identical and not task_yaml_digests["gate_passed"]:
        failures.append(
            {
                "component": "task_yaml_digests",
                "reason": "task_yaml_digest_gate_failed",
                "changed_task_yamls": task_yaml_digests["changed_task_yamls"],
            }
        )
    if require_object_assets and not object_assets["gate_passed"]:
        failures.append(
            {
                "component": "object_assets",
                "reason": "object_asset_gate_failed",
                "missing_local_assets": object_assets["missing_local_assets"],
                "missing_upstream_assets": object_assets["missing_upstream_assets"],
            }
        )

    return {
        "gate_passed": not failures,
        "upstream_reference": UPSTREAM_REFERENCE,
        "local_root": str(local_root),
        "upstream_root": str(upstream_root),
        "task_relative_dir": str(task_relative_dir),
        "local_git_head": _git_head(local_root),
        "upstream_git_head": _git_head(upstream_root),
        "task_inventory": inventory,
        "readme_contract": readme_contract,
        "viewer_contract": viewer_contract,
        "export_contract": export_contract,
        "task_semantic_fields": task_semantic_fields,
        "task_yaml_digests": task_yaml_digests,
        "object_assets": object_assets,
        "failures": failures,
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the local HDMI MuJoCo migration against the upstream LeCAR-Lab/HDMI source contracts."
        )
    )
    parser.add_argument("--local-root", default=str(HDMI_ROOT))
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--task-relative-dir", default=str(DEFAULT_TASK_RELATIVE_DIR))
    parser.add_argument("--expected-task-count", type=int, default=None)
    parser.add_argument("--require-task-count", action="store_true")
    parser.add_argument("--require-readme-contract", action="store_true")
    parser.add_argument("--require-viewer-contract", action="store_true")
    parser.add_argument("--require-export-contract", action="store_true")
    parser.add_argument("--require-task-semantic-fields", action="store_true")
    parser.add_argument("--require-task-yaml-identical", action="store_true")
    parser.add_argument("--require-object-assets", action="store_true")
    return parser.parse_args(argv)


def _task_files(root: Path, task_relative_dir: Path) -> dict[str, Path]:
    task_dir = root / task_relative_dir
    if not task_dir.exists():
        return {}
    return {path.name: path for path in sorted(task_dir.glob("*.yaml"))}


def _build_task_inventory(
    *,
    local_tasks: Mapping[str, Path],
    upstream_tasks: Mapping[str, Path],
    expected_task_count: int | None,
) -> dict[str, Any]:
    local_names = set(local_tasks)
    upstream_names = set(upstream_tasks)
    missing_local = sorted(upstream_names - local_names)
    missing_upstream = sorted(local_names - upstream_names)
    matched = sorted(local_names & upstream_names)
    expected_passed = expected_task_count is None or (
        len(local_names) == expected_task_count and len(upstream_names) == expected_task_count
    )
    return {
        "gate_passed": not missing_local and not missing_upstream and expected_passed,
        "expected_task_count": expected_task_count,
        "local_task_count": len(local_names),
        "upstream_task_count": len(upstream_names),
        "matched_task_count": len(matched),
        "matched_tasks": matched,
        "local_tasks": sorted(local_names),
        "upstream_tasks": sorted(upstream_names),
        "missing_local_tasks": missing_local,
        "missing_upstream_tasks": missing_upstream,
    }


def _build_readme_contract(upstream_root: Path) -> dict[str, Any]:
    readme_path = upstream_root / "README.md"
    text = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
    missing = [key for key, pattern in README_REQUIRED_PATTERNS.items() if pattern not in text]
    return {
        "gate_passed": readme_path.exists() and not missing,
        "readme_path": str(readme_path),
        "readme_exists": readme_path.exists(),
        "required_patterns": README_REQUIRED_PATTERNS,
        "missing_patterns": missing,
    }


def _build_file_contract(local_root: Path, upstream_root: Path, relative_files: Sequence[Path]) -> dict[str, Any]:
    missing_local = [str(path) for path in relative_files if not (local_root / path).exists()]
    missing_upstream = [str(path) for path in relative_files if not (upstream_root / path).exists()]
    return {
        "gate_passed": not missing_local and not missing_upstream,
        "required_files": [str(path) for path in relative_files],
        "missing_local_files": missing_local,
        "missing_upstream_files": missing_upstream,
    }


def _build_export_contract(local_root: Path, upstream_root: Path) -> dict[str, Any]:
    local_path = local_root / EXPORT_FILE
    upstream_path = upstream_root / EXPORT_FILE
    local_text = local_path.read_text(encoding="utf-8") if local_path.exists() else ""
    upstream_text = upstream_path.read_text(encoding="utf-8") if upstream_path.exists() else ""
    required_patterns = {"export_policy": "export_policy"}
    missing_local_patterns = [key for key, pattern in required_patterns.items() if pattern not in local_text]
    missing_upstream_patterns = [key for key, pattern in required_patterns.items() if pattern not in upstream_text]
    return {
        "gate_passed": local_path.exists()
        and upstream_path.exists()
        and not missing_local_patterns
        and not missing_upstream_patterns,
        "required_file": str(EXPORT_FILE),
        "missing_local_file": not local_path.exists(),
        "missing_upstream_file": not upstream_path.exists(),
        "missing_local_patterns": missing_local_patterns,
        "missing_upstream_patterns": missing_upstream_patterns,
    }


def _build_task_semantic_field_report(
    local_tasks: Mapping[str, Path],
    upstream_tasks: Mapping[str, Path],
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    matched = sorted(set(local_tasks) & set(upstream_tasks))
    for task_name in matched:
        local_cfg = _load_yaml_mapping(local_tasks[task_name])
        upstream_cfg = _load_yaml_mapping(upstream_tasks[task_name])
        for field in TASK_SEMANTIC_FIELDS:
            local_value = _nested_get(local_cfg, field)
            upstream_value = _nested_get(upstream_cfg, field)
            if local_value != upstream_value:
                mismatches.append(
                    {
                        "task": task_name,
                        "field": field,
                        "local": local_value,
                        "upstream": upstream_value,
                    }
                )
    return {
        "gate_passed": not mismatches,
        "checked_fields": list(TASK_SEMANTIC_FIELDS),
        "checked_task_count": len(matched),
        "mismatches": mismatches,
    }


def _build_task_yaml_digest_report(
    local_tasks: Mapping[str, Path],
    upstream_tasks: Mapping[str, Path],
) -> dict[str, Any]:
    changed = []
    for task_name in sorted(set(local_tasks) & set(upstream_tasks)):
        local_sha = _sha256(local_tasks[task_name])
        upstream_sha = _sha256(upstream_tasks[task_name])
        if local_sha != upstream_sha:
            changed.append(
                {
                    "task": task_name,
                    "local_sha256": local_sha,
                    "upstream_sha256": upstream_sha,
                }
            )
    return {"gate_passed": not changed, "changed_task_yamls": changed}


def _build_object_asset_report(
    local_root: Path,
    upstream_root: Path,
    local_tasks: Mapping[str, Path],
    upstream_tasks: Mapping[str, Path],
) -> dict[str, Any]:
    local_assets = _task_object_assets(local_tasks)
    upstream_assets = _task_object_assets(upstream_tasks)
    missing_local = [asset for asset in local_assets if not _asset_exists(local_root, asset)]
    missing_upstream = [asset for asset in upstream_assets if not _asset_exists(upstream_root, asset)]
    return {
        "gate_passed": not missing_local and not missing_upstream,
        "local_assets": local_assets,
        "upstream_assets": upstream_assets,
        "missing_local_assets": missing_local,
        "missing_upstream_assets": missing_upstream,
    }


def _task_object_assets(tasks: Mapping[str, Path]) -> list[str]:
    assets = set()
    for path in tasks.values():
        asset = _nested_get(_load_yaml_mapping(path), "command.object_asset_name")
        if asset is not None:
            assets.add(str(asset))
    return sorted(assets)


def _asset_exists(root: Path, asset_name: str) -> bool:
    asset_dir = root / "active_adaptation/assets_mjcf/objects" / asset_name
    return asset_dir.exists() and any(asset_dir.iterdir())


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected YAML mapping, got {type(data).__name__}.")
    return data


def _nested_get(data: Mapping[str, Any], dotted_path: str) -> Any:
    value: Any = data
    for part in dotted_path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


if __name__ == "__main__":
    raise SystemExit(main())
