import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_fleet_audit_module():
    script_path = ROOT / "scripts/mujoco_hdmi_policy_fleet_audit.py"
    spec = importlib.util.spec_from_file_location("mujoco_hdmi_policy_fleet_audit_script", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fleet_policy_audit_reports_missing_export_with_export_command(tmp_path, capsys):
    module = _load_fleet_audit_module()
    task_path = ROOT / "cfg/task/G1/hdmi/push_box.yaml"

    exit_code = module.main(
        [
            "--task-yaml",
            str(task_path),
            "--exports-dir",
            str(tmp_path / "exports"),
            "--checkpoint-path",
            "run:entity/project/push_box_teacher",
            "--require-policy",
            "--require-reference-observation",
            "--require-onnx-policy",
        ]
    )
    summary = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert summary["source_reference"]["repository"] == "https://github.com/LeCAR-Lab/HDMI"
    assert summary["task_count"] == 1
    assert summary["gate_passed"] is False
    assert summary["ready_task_count"] == 0
    assert summary["missing_requirements_by_task"] == {
        "G1PushBox": [
            "exported_policy_pt",
            "exported_policy_yaml",
            "policy_loadable",
            "policy_task_motion_mjcf_mapping",
            "policy_reference_observation",
            "exported_policy_onnx",
            "exported_policy_onnx_json",
            "policy_onnx_loadable",
        ]
    }
    task_summary = summary["tasks"][0]
    assert task_summary["task_name"] == "G1PushBox"
    assert task_summary["task_override"] == "G1/hdmi/push_box"
    assert task_summary["checkpoint_path"] == "run:entity/project/push_box_teacher"
    assert task_summary["export_command"] == [
        "python",
        "scripts/play.py",
        "task=G1/hdmi/push_box",
        "checkpoint_path=run:entity/project/push_box_teacher",
        "export_policy=true",
        "export_policy_exit=true",
        "export_policy_benchmark_iters=0",
        "export_onnx_policy=true",
        "export_onnx_required=true",
        "headless=true",
        "backend=mujoco",
    ]


def test_fleet_policy_audit_uses_checkpoint_manifest_by_task_override(tmp_path, capsys):
    module = _load_fleet_audit_module()
    task_path = ROOT / "cfg/task/G1/hdmi/push_box.yaml"
    checkpoint_manifest = tmp_path / "checkpoints.yaml"
    checkpoint_manifest.write_text("G1/hdmi/push_box: /mnt/checkpoints/push_box_teacher.pt\n")

    exit_code = module.main(
        [
            "--task-yaml",
            str(task_path),
            "--exports-dir",
            str(tmp_path / "exports"),
            "--checkpoint-manifest",
            str(checkpoint_manifest),
        ]
    )
    summary = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    task_summary = summary["tasks"][0]
