import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_upstream_audit_module():
    script_path = ROOT / "scripts/mujoco_hdmi_upstream_reference_audit.py"
    spec = importlib.util.spec_from_file_location("mujoco_hdmi_upstream_reference_audit", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_reference_tree(
    root: Path,
    *,
    task_stem: str = "push_box",
    task_name: str = "G1PushBox",
    object_asset_name: str = "box",
    object_body_name: str = "box",
    robot_type: str = "g1_29dof_nohand-feet_sphere-eef_L-body_capsule",
    data_path: str = "data/motion/g1/push_box/example",
    readme_contract: bool = True,
    viewer_files: bool = True,
    play_file: bool = True,
) -> Path:
    task_path = root / "cfg/task/G1/hdmi" / f"{task_stem}.yaml"
    task_path.parent.mkdir(parents=True, exist_ok=True)
    task_path.write_text(
        "\n".join(
            [
                f"name: {task_name}",
                "robot:",
                f"  robot_type: {robot_type}",
                "command:",
                f"  data_path: {data_path}",
                f"  object_asset_name: {object_asset_name!r}",
                f"  object_body_name: {object_body_name!r}",
                "  contact_eef_body_name: ['left_wrist_yaw_link']",
            ]
        ),
        encoding="utf-8",
    )
    if readme_contract:
        (root / "README.md").write_text(
            "\n".join(
                [
                    "python scripts/play.py algo=ppo_roa_train task=G1/hdmi/move_suitcase",
                    "python scripts/vis/mujoco_mocap_viewer.py",
                    "python scripts/vis/motion_data_publisher.py <path-to-motion-folder>",
                    "To export trained policies, add `export_policy=true` to the play script.",
                ]
            ),
            encoding="utf-8",
        )
    if viewer_files:
        vis_dir = root / "scripts/vis"
        vis_dir.mkdir(parents=True, exist_ok=True)
        (vis_dir / "mujoco_mocap_viewer.py").write_text("# viewer\n", encoding="utf-8")
        (vis_dir / "motion_data_publisher.py").write_text("# publisher\n", encoding="utf-8")
    if play_file:
        play_path = root / "scripts/play.py"
        play_path.parent.mkdir(parents=True, exist_ok=True)
        play_path.write_text("if cfg.export_policy:\n    pass\n", encoding="utf-8")
    asset_dir = root / "active_adaptation/assets_mjcf/objects" / object_asset_name
    asset_dir.mkdir(parents=True, exist_ok=True)
    (asset_dir / f"{object_asset_name}.xml").write_text("<mujoco/>\n", encoding="utf-8")
    return task_path


def test_upstream_reference_audit_passes_when_contracts_and_task_fields_match(tmp_path, capsys):
    module = _load_upstream_audit_module()
    local_root = tmp_path / "local"
    upstream_root = tmp_path / "upstream"
    _write_reference_tree(local_root)
    _write_reference_tree(upstream_root)

    exit_code = module.main(
        [
            "--local-root",
            str(local_root),
            "--upstream-root",
            str(upstream_root),
            "--expected-task-count",
            "1",
            "--require-task-count",
            "--require-readme-contract",
            "--require-viewer-contract",
            "--require-export-contract",
            "--require-task-semantic-fields",
            "--require-object-assets",
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["gate_passed"] is True
    assert report["upstream_reference"]["repository"] == "https://github.com/LeCAR-Lab/HDMI"
    assert report["task_inventory"]["matched_task_count"] == 1
    assert report["readme_contract"]["gate_passed"] is True
    assert report["viewer_contract"]["gate_passed"] is True
    assert report["export_contract"]["gate_passed"] is True
    assert report["task_semantic_fields"]["gate_passed"] is True
    assert report["object_assets"]["gate_passed"] is True
    assert report["failures"] == []


def test_upstream_reference_audit_rejects_semantic_field_drift(tmp_path, capsys):
    module = _load_upstream_audit_module()
    local_root = tmp_path / "local"
    upstream_root = tmp_path / "upstream"
    _write_reference_tree(local_root, object_asset_name="crate")
    _write_reference_tree(upstream_root, object_asset_name="box")

    exit_code = module.main(
        [
            "--local-root",
            str(local_root),
            "--upstream-root",
            str(upstream_root),
            "--expected-task-count",
            "1",
            "--require-task-count",
            "--require-task-semantic-fields",
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert report["gate_passed"] is False
    assert report["task_semantic_fields"]["gate_passed"] is False
    assert report["task_semantic_fields"]["mismatches"] == [
        {
            "task": "push_box.yaml",
            "field": "command.object_asset_name",
            "local": "crate",
            "upstream": "box",
        }
    ]
    assert report["failures"] == [
        {
            "component": "task_semantic_fields",
            "reason": "task_semantic_fields_gate_failed",
            "mismatches": report["task_semantic_fields"]["mismatches"],
        }
    ]
