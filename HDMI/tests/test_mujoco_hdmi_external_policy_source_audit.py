import importlib.util
import json
from pathlib import Path

import torch
import yaml
from tensordict.nn import TensorDictModule


ROOT = Path(__file__).resolve().parents[1]


def _load_external_policy_audit_module():
    script_path = ROOT / "scripts/mujoco_hdmi_external_policy_source_audit.py"
    spec = importlib.util.spec_from_file_location("mujoco_hdmi_external_policy_source_audit", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_external_policy(
    root: Path,
    policy_set: str,
    *,
    motion_path: str,
    joint_name: str = "left_hip_pitch_joint",
    body_name: str = "pelvis",
    metadata_extra: dict | None = None,
    config_extra: dict | None = None,
) -> Path:
    policy_dir = root / policy_set
    policy_dir.mkdir(parents=True)
    policy_path = policy_dir / "policy-test-final.pt"
    module = TensorDictModule(
        torch.nn.Linear(1, 1, bias=False),
        in_keys=["policy"],
        out_keys=["action"],
    )
    torch.save(module, policy_path)
    config = {
        "observation": {
            "policy": {
                "joint_pos_history": {
                    "joint_names": [joint_name],
                    "history_steps": [0],
                },
            },
            "command": {
                "ref_motion_phase": {"motion_path": motion_path},
                "ref_body_pos_future_local": {
                    "motion_path": motion_path,
                    "body_names": [body_name],
                    "future_steps": [1],
                    "root_body_name": body_name,
                },
            },
        },
        "action_scale": 1.0,
        "policy_joint_names": [joint_name],
        "isaac_joint_names": [joint_name],
        "isaac_body_names": [body_name],
        "default_joint_pos": 0.0,
    }
    if config_extra:
        config.update(config_extra)
    policy_path.with_suffix(".yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    metadata = {
        "in_keys": ["command", "policy"],
        "in_shapes": [[[1, 1], [1, 1]]],
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    policy_path.with_suffix(".json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    policy_path.with_suffix(".onnx").write_bytes(b"onnx")
    return policy_path


def test_external_policy_source_audit_matches_policy_by_motion_path_and_reuses_export_audit(
    tmp_path,
    capsys,
):
    module = _load_external_policy_audit_module()
    policy_root = tmp_path / "sim2real"
    task_path = ROOT / "cfg/task/G1/hdmi/push_box.yaml"
    _write_external_policy(
        policy_root,
        "G1PushBoxCandidate",
        motion_path="data/motion/g1/push_box/push_box-VID_20250423_220958-light-high-adjust_root_height",
    )
    _write_external_policy(
        policy_root,
        "G1Dance",
        motion_path="data/motion/lafan/dance1_subject2-270_1000",
    )
    output = tmp_path / "external_policy_source.json"

    exit_code = module.main(
        [
            "--policy-root",
            str(policy_root),
            "--task-yaml",
            str(task_path),
            "--require-reference-observation",
            "--require-obs-action-smoke",
            "--output",
            str(output),
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert json.loads(output.read_text()) == report
    assert report["source_reference"] == {
        "repository": "https://github.com/EGalahad/sim2real",
        "tag": "hdmi",
        "scope": "external_deployment_policy_source",
    }
    assert report["policy_count"] == 2
    assert report["matched_policy_count"] == 1
    assert report["unmatched_policy_count"] == 1
    assert report["task_count"] == 1
    assert report["ready_task_count"] == 1
    assert report["not_ready_task_count"] == 0
    assert report["gate_passed"] is True
    assert report["unmatched_policies"][0]["policy_set"] == "G1Dance"
    assert report["unmatched_policies"][0]["motion_paths"] == [
        "data/motion/lafan/dance1_subject2-270_1000"
    ]


def test_external_policy_source_audit_reports_missing_training_provenance(
    tmp_path,
    capsys,
):
    module = _load_external_policy_audit_module()
    policy_root = tmp_path / "sim2real"
    task_path = ROOT / "cfg/task/G1/hdmi/push_box.yaml"
    _write_external_policy(
        policy_root,
        "G1PushBoxCandidate",
        motion_path="data/motion/g1/push_box/push_box-VID_20250423_220958-light-high-adjust_root_height",
    )

    exit_code = module.main(
        [
            "--policy-root",
            str(policy_root),
            "--task-yaml",
            str(task_path),
            "--require-reference-observation",
            "--require-obs-action-smoke",
            "--require-training-provenance",
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert report["gate_passed"] is False
    assert report["training_provenance_policy_count"] == 0
    assert report["missing_training_provenance_policy_count"] == 1
    task = report["tasks"][0]
    assert task["training_provenance_ready"] is False
    assert "external_policy_training_provenance" in task["missing_requirements"]
    assert task["training_provenance"]["policy_filename_run_id"] == "test"
    assert task["training_provenance"]["missing_fields"] == [
        "git_commit",
        "training_checkpoint",
        "wandb_run_path",
        "training_backend",
    ]


def test_external_policy_source_audit_accepts_actionable_training_provenance(
    tmp_path,
    capsys,
):
    module = _load_external_policy_audit_module()
    policy_root = tmp_path / "sim2real"
    task_path = ROOT / "cfg/task/G1/hdmi/push_box.yaml"
    _write_external_policy(
        policy_root,
        "G1PushBoxCandidate",
        motion_path="data/motion/g1/push_box/push_box-VID_20250423_220958-light-high-adjust_root_height",
        metadata_extra={
            "wandb": {"run_path": "run:entity/project/teacher123"},
            "git": {"commit": "abc123"},
            "checkpoint_path": "/tmp/checkpoint_final.pt",
            "backend": "isaac",
        },
    )

    exit_code = module.main(
        [
            "--policy-root",
            str(policy_root),
            "--task-yaml",
            str(task_path),
            "--require-reference-observation",
            "--require-obs-action-smoke",
            "--require-training-provenance",
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["gate_passed"] is True
    task = report["tasks"][0]
    assert task["training_provenance_ready"] is True
    provenance = task["training_provenance"]
    assert provenance["git_commit_candidates"] == ["abc123"]
    assert provenance["checkpoint_candidates"] == ["/tmp/checkpoint_final.pt"]
    assert provenance["concrete_wandb_run_paths"] == ["run:entity/project/teacher123"]
    assert provenance["backend_candidates"] == ["isaac"]
