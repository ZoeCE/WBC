import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_plan_module():
    script_path = ROOT / "scripts/mujoco_hdmi_policy_export_plan.py"
    spec = importlib.util.spec_from_file_location("mujoco_hdmi_policy_export_plan_script", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_policy_export_plan_reports_missing_checkpoint_manifest_entries(tmp_path, capsys):
    module = _load_plan_module()
    task_path = ROOT / "cfg/task/G1/hdmi/push_box.yaml"

    exit_code = module.main(
        [
            "--task-yaml",
            str(task_path),
            "--python",
            "/envs/wbc/bin/python",
            "--exports-dir",
            str(tmp_path / "exports"),
            "--output",
            str(tmp_path / "plan.json"),
        ]
    )
    report = json.loads(capsys.readouterr().out)
    saved = json.loads((tmp_path / "plan.json").read_text())

    assert exit_code == 0
    assert saved == report
    assert report["task_count"] == 1
    assert report["provided_checkpoint_task_count"] == 0
    assert report["missing_checkpoint_task_count"] == 1
    assert report["ready_to_export"] is False
    assert report["checkpoint_manifest_template"] == {"G1/hdmi/push_box": "<checkpoint_path>"}
    task = report["tasks"][0]
    assert task["task_name"] == "G1PushBox"
    assert task["task_override"] == "G1/hdmi/push_box"
    assert task["checkpoint_path"] is None
    assert task["missing_checkpoint"] is True
    assert task["export_command"] == [
        "/envs/wbc/bin/python",
        "scripts/play.py",
        "task=G1/hdmi/push_box",
        "checkpoint_path=<checkpoint_path>",
        "export_policy=true",
        "export_policy_exit=true",
        "export_policy_benchmark_iters=0",
        "export_onnx_policy=false",
        "headless=true",
        "backend=mujoco",
    ]


def test_policy_export_plan_uses_checkpoint_manifest_for_export_and_audit_commands(tmp_path, capsys):
    module = _load_plan_module()
    task_path = ROOT / "cfg/task/G1/hdmi/push_box.yaml"
    checkpoint_manifest = tmp_path / "checkpoints.yaml"
    checkpoint_manifest.write_text("G1/hdmi/push_box: run:entity/hdmi/push_box_teacher\n")

    exit_code = module.main(
        [
            "--task-yaml",
            str(task_path),
            "--checkpoint-manifest",
            str(checkpoint_manifest),
            "--python",
            "/envs/wbc/bin/python",
            "--exports-dir",
            str(tmp_path / "exports"),
            "--expected-task-count",
            "1",
            "--require-task-count",
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["task_count_gate_passed"] is True
    assert report["provided_checkpoint_task_count"] == 1
    assert report["missing_checkpoint_task_count"] == 0
    assert report["ready_to_export"] is True
    task = report["tasks"][0]
    assert task["checkpoint_path"] == "run:entity/hdmi/push_box_teacher"
    assert task["export_command"] == [
        "/envs/wbc/bin/python",
        "scripts/play.py",
        "task=G1/hdmi/push_box",
        "checkpoint_path=run:entity/hdmi/push_box_teacher",
        "export_policy=true",
        "export_policy_exit=true",
        "export_policy_benchmark_iters=0",
        "export_onnx_policy=false",
        "headless=true",
        "backend=mujoco",
    ]
    assert report["fleet_policy_audit_command"] == [
        "/envs/wbc/bin/python",
        "scripts/mujoco_hdmi_policy_fleet_audit.py",
        "--task-yaml",
        str(task_path),
        "--exports-dir",
        str(tmp_path / "exports"),
        "--checkpoint-manifest",
        str(checkpoint_manifest),
        "--require-policy",
        "--require-reference-observation",
        "--require-obs-action-smoke",
        "--expected-task-count",
        "1",
        "--require-task-count",
    ]
