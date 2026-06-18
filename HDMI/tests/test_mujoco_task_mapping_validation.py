import importlib.util
import json
from pathlib import Path

from active_adaptation.mujoco.task_mapping import (
    validate_all_task_motion_mappings,
    validate_task_motion_mapping,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_mapping_cli_module():
    script_path = ROOT / "scripts/mujoco_validate_task_mapping.py"
    spec = importlib.util.spec_from_file_location("mujoco_validate_task_mapping_script", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_task_motion_asset_mapping_resolves_reference_object_body_for_door_panel():
    report = validate_task_motion_mapping(ROOT / "cfg/task/G1/hdmi/open_door-feet.yaml")

    assert report.object_asset_name == "door"
    assert report.task_object_body_name == "door_panel"
    assert report.reference_object_body_name == "door"
    assert report.object_joint_name == "door_joint"
    assert tuple(report.asset_object_body_names) == ("door", "door_panel")
    assert report.motion_body_names[-1] == "door"
    assert report.motion_joint_names[-1] == "door_joint"


def test_task_motion_asset_mapping_reports_ordered_mujoco_name_indices():
    report = validate_task_motion_mapping(ROOT / "cfg/task/G1/hdmi/open_door-feet.yaml")

    door_motion_index = report.motion_body_names.index("door")
    door_mapping = report.body_name_mapping[door_motion_index]
    assert door_mapping.name == "door"
    assert door_mapping.motion_index == door_motion_index
    assert isinstance(door_mapping.mujoco_index, int)

    joint_motion_index = report.motion_joint_names.index("door_joint")
    joint_mapping = report.joint_name_mapping[joint_motion_index]
    assert joint_mapping.name == "door_joint"
    assert joint_mapping.motion_index == joint_motion_index
    assert isinstance(joint_mapping.mujoco_index, int)

    summary = report.to_dict()
    assert summary["body_name_mapping"][door_motion_index] == {
        "name": "door",
        "motion_index": door_motion_index,
        "mujoco_index": door_mapping.mujoco_index,
    }
    assert summary["joint_name_mapping"][joint_motion_index]["name"] == "door_joint"


def test_all_hdmi_object_tasks_have_motion_asset_name_mapping():
    reports = validate_all_task_motion_mappings(ROOT / "cfg/task/G1/hdmi")

    assert len(reports) >= 10
    assert {report.task_path.name for report in reports} >= {
        "open_door-feet.yaml",
        "open_foldchair-sit.yaml",
        "push_box.yaml",
    }
    for report in reports:
        assert report.reference_object_body_name in report.motion_body_names


def test_task_mapping_cli_prints_json_summary(capsys):
    module = _load_mapping_cli_module()
    task_path = ROOT / "cfg/task/G1/hdmi/open_door-feet.yaml"

    exit_code = module.main(["--task-yaml", str(task_path)])
    captured = capsys.readouterr()
    summary = json.loads(captured.out)

    assert exit_code == 0
    assert summary["object_asset_name"] == "door"
    assert summary["task_object_body_name"] == "door_panel"
    assert summary["reference_object_body_name"] == "door"
