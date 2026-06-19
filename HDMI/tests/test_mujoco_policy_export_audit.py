import importlib.util
import json
from pathlib import Path

import torch
import yaml
from tensordict.nn import TensorDictModule


ROOT = Path(__file__).resolve().parents[1]


def _load_audit_cli_module():
    script_path = ROOT / "scripts/mujoco_policy_export_audit.py"
    spec = importlib.util.spec_from_file_location("mujoco_policy_export_audit_script", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_policy_export(export_dir: Path) -> Path:
    export_dir.mkdir(parents=True)
    policy_path = export_dir / "policy-test-final.pt"
    module = TensorDictModule(
        torch.nn.Linear(1, 1, bias=False),
        in_keys=["policy"],
        out_keys=["action"],
    )
    torch.save(module, policy_path)
    policy_path.with_suffix(".yaml").write_text(
        yaml.safe_dump(
            {
                "observation": {
                    "policy": {
                        "joint_pos_history": {
                            "joint_names": ["left_hip_pitch_joint"],
                            "history_steps": [0],
                        }
                    }
                },
                "action_scale": 1.0,
                "policy_joint_names": ["left_hip_pitch_joint"],
                "isaac_joint_names": ["left_hip_pitch_joint"],
                "isaac_body_names": ["pelvis"],
                "default_joint_pos": 0.0,
            }
        )
    )
    return policy_path


def test_policy_export_audit_require_policy_fails_when_export_is_missing(tmp_path, capsys):
    module = _load_audit_cli_module()
    task_path = ROOT / "cfg/task/G1/hdmi/push_box.yaml"

    exit_code = module.main(
        [
            "--task-yaml",
            str(task_path),
            "--exports-dir",
            str(tmp_path / "exports"),
            "--checkpoint-path",
            "run:entity/project/runid",
            "--require-policy",
        ]
    )
    summary = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert summary["task_name"] == "G1PushBox"
    assert summary["task_override"] == "G1/hdmi/push_box"
    assert summary["policy_exists"] is False
    assert summary["policy_config_exists"] is False
    assert summary["gate_passed"] is False
    assert summary["missing_requirements"] == [
        "exported_policy_pt",
        "exported_policy_yaml",
        "policy_loadable",
        "policy_task_motion_mjcf_mapping",
    ]
    assert summary["checkpoint_kind"] == "wandb_run"
    assert summary["checkpoint_exists"] is None
    assert summary["expected_export_dir"] == str(tmp_path / "exports/G1PushBox")
    assert summary["export_command"] == [
        "python",
        "scripts/play.py",
        "task=G1/hdmi/push_box",
        "checkpoint_path=run:entity/project/runid",
        "export_policy=true",
        "export_policy_exit=true",
        "headless=true",
        "backend=isaac",
    ]


def test_policy_export_audit_loads_policy_and_validates_task_mapping(tmp_path, capsys):
    module = _load_audit_cli_module()
    task_path = ROOT / "cfg/task/G1/hdmi/push_box.yaml"
    policy_path = _write_policy_export(tmp_path / "exports/G1PushBox")

    exit_code = module.main(
        [
            "--task-yaml",
            str(task_path),
            "--policy-path",
            str(policy_path),
            "--require-policy",
        ]
    )
    summary = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert summary["gate_passed"] is True
    assert summary["missing_requirements"] == []
