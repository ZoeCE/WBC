import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_manifest_module():
    script_path = ROOT / "scripts/mujoco_hdmi_source_contract_manifest.py"
    spec = importlib.util.spec_from_file_location("mujoco_hdmi_source_contract_manifest", script_path)
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
    object_joint_name: str = "box_root",
    contact_eef_body_name: list[str] | None = None,
    data_path: str = "data/motion/g1/push_box/example",
) -> Path:
    contact_eef_body_name = contact_eef_body_name or ["left_wrist_yaw_link"]
    task_path = root / "cfg/task/G1/hdmi" / f"{task_stem}.yaml"
    task_path.parent.mkdir(parents=True, exist_ok=True)
    task_path.write_text(
        "\n".join(
            [
                f"name: {task_name}",
                "robot:",
                "  robot_type: g1_29dof_nohand-feet_sphere-eef_L-body_capsule",
                "command:",
                f"  data_path: {data_path}",
                "  root_body_name: pelvis",
                f"  object_asset_name: {object_asset_name!r}",
                f"  object_body_name: {object_body_name!r}",
                f"  object_joint_name: {object_joint_name!r}",
                "  extra_object_names: ['support0']",
                f"  contact_eef_body_name: {contact_eef_body_name!r}",
                "  contact_target_pos_offset:",
                "    - [0.0, -0.2, 0.8]",
                "  contact_eef_pos_offset:",
                "    - [0.1, 0.0, 0.0]",
                "randomization:",
                "  object_body_randomization:",
                "    mass_range: [6.0, 12.0]",
            ]
        ),
        encoding="utf-8",
    )

    motion_dir = root / data_path
    motion_dir.mkdir(parents=True, exist_ok=True)
    (motion_dir / "motion.npz").write_bytes(b"motion")
    (motion_dir / "meta.json").write_text(
        json.dumps(
            {
                "body_names": ["pelvis", "left_wrist_yaw_link", object_body_name],
                "joint_names": ["left_hip_pitch_joint", object_joint_name],
            }
        ),
        encoding="utf-8",
    )

    asset_dir = root / "active_adaptation/assets_mjcf/objects" / object_asset_name
    asset_dir.mkdir(parents=True, exist_ok=True)
    (asset_dir / f"{object_asset_name}.xml").write_text("<mujoco/>\n", encoding="utf-8")
    return task_path


def test_source_contract_manifest_exports_task_motion_asset_and_command_contract(tmp_path, capsys):
    module = _load_manifest_module()
    local_root = tmp_path / "local"
    upstream_root = tmp_path / "upstream"
    _write_reference_tree(local_root)
    _write_reference_tree(upstream_root)
    output = tmp_path / "manifest.json"

    exit_code = module.main(
        [
            "--local-root",
            str(local_root),
            "--upstream-root",
            str(upstream_root),
            "--expected-task-count",
            "1",
            "--require-task-count",
            "--require-semantic-match",
            "--require-motion-files",
            "--require-object-assets",
            "--output",
            str(output),
        ]
    )
    report = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text())

    assert exit_code == 0
    assert written == report
    assert report["gate_passed"] is True
    assert report["task_count"] == 1
    assert report["source_reference"]["repository"] == "https://github.com/LeCAR-Lab/HDMI"
    assert report["source_reference"]["scope"] == "source_contract_manifest"

    task = report["tasks"][0]
    assert task["task_name"] == "G1PushBox"
    assert task["task_override"] == "G1/hdmi/push_box"
    assert task["semantic_match"] is True
