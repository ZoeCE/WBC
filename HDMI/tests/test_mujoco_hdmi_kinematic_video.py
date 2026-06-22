import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_video_module():
    script_path = ROOT / "scripts/mujoco_hdmi_kinematic_video.py"
    spec = importlib.util.spec_from_file_location("mujoco_hdmi_kinematic_video", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_kinematic_video_dry_run_reports_mujoco_motion_mapping(tmp_path, capsys):
    module = _load_video_module()
    task_path = ROOT / "cfg/task/G1/hdmi/push_box.yaml"
    output_path = tmp_path / "push_box_kinematic.mp4"

    exit_code = module.main(
        [
            "--task-yaml",
            str(task_path),
            "--output",
            str(output_path),
            "--max-frames",
            "12",
            "--dry-run",
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["task_name"] == "G1PushBox"
    assert report["task_stem"] == "push_box"
    assert report["output_path"] == str(output_path)
    assert report["dry_run"] is True
    assert report["max_frames"] == 12
    assert report["motion_frame_count"] > 0
    assert report["render_frame_count"] == 12
    assert report["mjcf_path"].endswith("g1_29dof_nohand-box.xml")
    assert report["free_joint_mappings"] == [
        {"joint_name": "box_root", "body_name": "box", "qpos_address": 0, "motion_body_index": 28},
        {"joint_name": "pelvis_root", "body_name": "pelvis", "qpos_address": 7, "motion_body_index": 0},
    ]
    assert report["hinge_joint_count"] == 25
    assert "left_hip_pitch_joint" in report["mapped_hinge_joint_names"]
