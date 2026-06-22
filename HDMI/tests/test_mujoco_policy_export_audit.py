import importlib.util
import json
from pathlib import Path

import torch
import yaml
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

import active_adaptation.utils.export as export_utils


ROOT = Path(__file__).resolve().parents[1]


class _MultiInputActionModule(torch.nn.Module):
    def __init__(self, action_dim: int = 1):
        super().__init__()
        self.action_dim = int(action_dim)

    def forward(self, *inputs):
        batch = inputs[0].shape[0]
        total = torch.zeros(batch, 1, dtype=inputs[0].dtype, device=inputs[0].device)
        for tensor in inputs:
            total = total + tensor.reshape(batch, -1).sum(dim=-1, keepdim=True) * 0.0
        return total.expand(batch, self.action_dim)


def _load_audit_cli_module():
    script_path = ROOT / "scripts/mujoco_policy_export_audit.py"
    spec = importlib.util.spec_from_file_location("mujoco_policy_export_audit_script", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_policy_export(export_dir: Path, *, reference_observation: bool = False, onnx: bool = False) -> Path:
    export_dir.mkdir(parents=True)
    policy_path = export_dir / "policy-test-final.pt"
    module = TensorDictModule(
        torch.nn.Linear(1, 1, bias=False),
        in_keys=["policy"],
        out_keys=["action"],
    )
    torch.save(module, policy_path)
    observation = {
        "policy": {
            "joint_pos_history": {
                "joint_names": ["left_hip_pitch_joint"],
                "history_steps": [0],
            }
        }
    }
    if reference_observation:
        observation["command"] = {"ref_motion_phase": {}}
    policy_path.with_suffix(".yaml").write_text(
        yaml.safe_dump(
            {
                "observation": observation,
                "action_scale": 1.0,
                "policy_joint_names": ["left_hip_pitch_joint"],
                "isaac_joint_names": ["left_hip_pitch_joint"],
                "isaac_body_names": ["pelvis"],
                "default_joint_pos": 0.0,
            }
        )
    )
    if onnx:
        export_utils.export_onnx(
            module,
            TensorDict({"policy": torch.zeros(1, 1)}, batch_size=[1]),
            str(policy_path.with_suffix(".onnx")),
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
        "export_policy_benchmark_iters=0",
        "export_onnx_policy=true",
        "export_onnx_required=true",
        "headless=true",
        "backend=mujoco",
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
    assert summary["policy_observation_groups"] == ["policy"]
    assert summary["policy_observation_keys"] == {"policy": ["joint_pos_history"]}
    assert summary["policy_has_reference_observation"] is False
    assert summary["policy_reference_observation_keys"] == []


def test_policy_export_audit_can_require_reference_observation(tmp_path, capsys):
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
            "--require-reference-observation",
        ]
    )
    summary = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert summary["gate_passed"] is False
    assert summary["policy_has_reference_observation"] is False
    assert summary["missing_requirements"] == ["policy_reference_observation"]


def test_policy_export_audit_passes_when_reference_observation_is_present(tmp_path, capsys):
    module = _load_audit_cli_module()
    task_path = ROOT / "cfg/task/G1/hdmi/push_box.yaml"
    policy_path = _write_policy_export(tmp_path / "exports/G1PushBox", reference_observation=True)

    exit_code = module.main(
        [
            "--task-yaml",
            str(task_path),
            "--policy-path",
            str(policy_path),
            "--require-policy",
            "--require-reference-observation",
        ]
    )
    summary = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert summary["gate_passed"] is True
    assert summary["missing_requirements"] == []
    assert summary["policy_has_reference_observation"] is True
    assert summary["policy_reference_observation_keys"] == ["ref_motion_phase"]


def test_policy_export_audit_reports_local_checkpoint_provenance(tmp_path, capsys):
    module = _load_audit_cli_module()
    task_path = ROOT / "cfg/task/G1/hdmi/push_box.yaml"
    policy_path = _write_policy_export(tmp_path / "exports/G1PushBox", reference_observation=True)
    checkpoint_path = tmp_path / "checkpoint_final.pt"
    torch.save(
        {
            "cfg": {
                "backend": "mujoco",
                "total_frames": 8192,
                "checkpoint_path": "run:entity/hdmi/teacher",
                "algo": {
                    "name": "ppo_roa",
                    "_target_": "active_adaptation.learning.ppo.ppo_roa.PPOROA",
                },
                "task": {"name": "G1PushBox", "num_envs": 64},
            },
            "wandb": {"id": "trained1", "name": "trained-run"},
        },
        checkpoint_path,
    )

    exit_code = module.main(
        [
            "--task-yaml",
            str(task_path),
            "--policy-path",
            str(policy_path),
            "--checkpoint-path",
            str(checkpoint_path),
            "--require-policy",
            "--require-reference-observation",
        ]
    )
    summary = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert summary["gate_passed"] is True
    assert summary["checkpoint_kind"] == "local"
    assert summary["checkpoint_exists"] is True
    assert summary["checkpoint_cfg_loadable"] is True
    assert summary["checkpoint_algo_name"] == "ppo_roa"
    assert summary["checkpoint_algo_target"] == "active_adaptation.learning.ppo.ppo_roa.PPOROA"
    assert summary["checkpoint_backend"] == "mujoco"
    assert summary["checkpoint_total_frames"] == 8192
    assert summary["checkpoint_task_name"] == "G1PushBox"
    assert summary["checkpoint_num_envs"] == 64
    assert summary["checkpoint_source_checkpoint_path"] == "run:entity/hdmi/teacher"
    assert summary["checkpoint_wandb_id"] == "trained1"


def test_policy_export_audit_can_gate_checkpoint_provenance(tmp_path, capsys):
    module = _load_audit_cli_module()
    task_path = ROOT / "cfg/task/G1/hdmi/push_box.yaml"
    policy_path = _write_policy_export(tmp_path / "exports/G1PushBox", reference_observation=True)
    checkpoint_path = tmp_path / "checkpoint_final.pt"
    torch.save(
        {
            "cfg": {
                "backend": "mujoco",
                "total_frames": 2,
                "algo": {"name": "ppo"},
                "task": {"name": "G1PushBox"},
            },
            "wandb": {"id": "smoke2"},
        },
        checkpoint_path,
    )

    exit_code = module.main(
        [
            "--task-yaml",
            str(task_path),
            "--policy-path",
            str(policy_path),
            "--checkpoint-path",
            str(checkpoint_path),
            "--require-policy",
            "--require-reference-observation",
            "--require-checkpoint-algo",
            "ppo_roa",
            "--min-checkpoint-total-frames",
            "1024",
        ]
    )
    summary = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert summary["gate_passed"] is False
    assert summary["checkpoint_algo_name"] == "ppo"
    assert summary["checkpoint_total_frames"] == 2
    assert summary["missing_requirements"] == [
        "checkpoint_algo",
        "checkpoint_total_frames",
    ]


def test_policy_export_audit_can_require_obs_action_smoke(tmp_path, capsys):
    module = _load_audit_cli_module()
    task_path = ROOT / "cfg/task/G1/hdmi/push_box.yaml"
    policy_path = _write_policy_export(tmp_path / "exports/G1PushBox", reference_observation=True)

    exit_code = module.main(
        [
            "--task-yaml",
            str(task_path),
            "--policy-path",
            str(policy_path),
            "--require-policy",
            "--require-reference-observation",
            "--require-obs-action-smoke",
        ]
    )
    summary = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert summary["gate_passed"] is True
    assert summary["missing_requirements"] == []
    assert summary["policy_obs_action_smoke_ok"] is True
    assert summary["policy_smoke_observation_shapes"] == {
        "policy": [1, 1],
        "command": [1, 1],
    }
    assert summary["policy_smoke_action_shape"] == [1, 1]
    assert summary["policy_smoke_joint_target_shape"] == [1, 1]


def test_policy_export_audit_can_require_onnx_policy(tmp_path, capsys):
    module = _load_audit_cli_module()
    task_path = ROOT / "cfg/task/G1/hdmi/push_box.yaml"
    policy_path = _write_policy_export(tmp_path / "exports/G1PushBox", reference_observation=True)

    exit_code = module.main(
        [
            "--task-yaml",
            str(task_path),
            "--policy-path",
            str(policy_path),
            "--require-policy",
            "--require-onnx-policy",
        ]
    )
    summary = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert summary["policy_onnx_exists"] is False
    assert summary["policy_onnx_metadata_exists"] is False
    assert summary["missing_requirements"] == [
        "exported_policy_onnx",
        "exported_policy_onnx_json",
        "policy_onnx_loadable",
    ]

    policy_path = _write_policy_export(
        tmp_path / "exports_with_onnx/G1PushBox",
        reference_observation=True,
        onnx=True,
    )
    capsys.readouterr()
    exit_code = module.main(
        [
            "--task-yaml",
            str(task_path),
            "--policy-path",
            str(policy_path),
            "--require-policy",
            "--require-onnx-policy",
        ]
    )
    summary = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert summary["gate_passed"] is True
    assert summary["policy_onnx_loadable"] is True
    assert summary["policy_onnx_input_count"] == 1
    assert summary["policy_onnx_output_count"] == 1


def test_policy_export_audit_obs_action_smoke_uses_reference_observation_specific_dims(tmp_path, capsys):
    module = _load_audit_cli_module()
    task_path = ROOT / "cfg/task/G1/hdmi/push_box.yaml"
    policy_path = tmp_path / "exports/G1PushBox/policy-command-final.pt"
    policy_path.parent.mkdir(parents=True)
    command_dim = 2 * 2 * 3 + 2 * 1 + 1
    policy = TensorDictModule(
        torch.nn.Linear(command_dim, 1, bias=False),
        in_keys=["command"],
        out_keys=["action"],
    )
    torch.save(policy, policy_path)
    policy_path.with_suffix(".yaml").write_text(
        yaml.safe_dump(
            {
                "observation": {
                    "command": {
                        "ref_body_pos_future_local": {
                            "future_steps": [1, 2],
                            "body_names": ["pelvis", "left_elbow_link"],
                            "joint_names": ["left_elbow_joint"],
                            "root_body_name": "pelvis",
                        },
                        "ref_joint_pos_future": {
                            "future_steps": [1, 2],
                            "body_names": ["pelvis", "left_elbow_link"],
                            "joint_names": ["left_elbow_joint"],
                            "root_body_name": "pelvis",
                        },
                        "ref_motion_phase": {
                            "future_steps": [1, 2],
                            "body_names": ["pelvis", "left_elbow_link"],
                            "joint_names": ["left_elbow_joint"],
                            "root_body_name": "pelvis",
                        },
                    }
                },
                "action_scale": 1.0,
                "policy_joint_names": ["left_elbow_joint"],
                "isaac_joint_names": [
                    "left_hip_pitch_joint",
                    "left_hip_roll_joint",
                    "left_hip_yaw_joint",
                    "left_elbow_joint",
                ],
                "isaac_body_names": ["pelvis", "left_elbow_link", "left_wrist_yaw_link"],
                "default_joint_pos": 0.0,
            }
        )
    )

    exit_code = module.main(
        [
            "--task-yaml",
            str(task_path),
            "--policy-path",
            str(policy_path),
            "--require-policy",
            "--require-reference-observation",
            "--require-obs-action-smoke",
        ]
    )
    summary = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert summary["policy_obs_action_smoke_ok"] is True
    assert summary["policy_smoke_observation_shapes"] == {"command": [1, command_dim]}
    assert summary["policy_smoke_action_shape"] == [1, 1]


def test_policy_export_audit_smoke_builds_teacher_privileged_wbc_observations(tmp_path, capsys):
    module = _load_audit_cli_module()
    task_path = ROOT / "cfg/task/G1/hdmi/push_box.yaml"
    policy_path = tmp_path / "exports/G1PushBox/policy-teacher-final.pt"
    policy_path.parent.mkdir(parents=True)
    policy = TensorDictModule(
        _MultiInputActionModule(action_dim=1),
        in_keys=["policy", "command", "object", "priv", "ref_joint_pos_"],
        out_keys=["action"],
    )
    torch.save(policy, policy_path)
    policy_path.with_suffix(".yaml").write_text(
        yaml.safe_dump(
            {
                "observation": {
                    "policy": {
                        "joint_pos_history": {
                            "joint_names": ["left_hip_pitch_joint"],
                            "history_steps": [0],
                        },
                        "prev_actions": {"steps": 2},
                    },
                    "command": {
                        "ref_root_pos_future_b": {"future_steps": [1, 2]},
                        "ref_root_ori_future_b": {"future_steps": [1, 2]},
                        "ref_motion_phase": {"future_steps": [1, 2]},
                    },
                    "object": {
                        "object_pos_b": {"object_name": "box"},
                        "object_ori_b": {"object_name": "box"},
                        "object_joint_pos": {},
                        "object_joint_vel": {},
                        "object_joint_torque": {},
                    },
                    "priv": {
                        "ref_body_pos_future_local": {
                            "future_steps": [1, 2],
                            "body_names": ["pelvis", "left_elbow_link"],
                            "joint_names": ["left_hip_pitch_joint"],
                            "root_body_name": "pelvis",
                        },
                        "diff_body_pos_future_local": {
                            "future_steps": [1, 2],
                            "body_names": ["pelvis", "left_elbow_link"],
                            "root_body_name": "pelvis",
                        },
                        "diff_body_ori_future_local": {
                            "future_steps": [1, 2],
                            "body_names": ["pelvis", "left_elbow_link"],
                            "root_body_name": "pelvis",
                        },
                        "diff_body_lin_vel_future_local": {
                            "future_steps": [1, 2],
                            "body_names": ["pelvis", "left_elbow_link"],
                            "root_body_name": "pelvis",
                        },
                        "diff_body_ang_vel_future_local": {
                            "future_steps": [1, 2],
                            "body_names": ["pelvis", "left_elbow_link"],
                            "root_body_name": "pelvis",
                        },
                        "root_linvel_b": {},
                        "body_pos_b": {"body_names": ["pelvis", "left_elbow_link"]},
                        "body_vel_b": {"body_names": ["pelvis", "left_elbow_link"]},
                        "body_height": {"body_names": ["pelvis", "left_elbow_link"]},
                        "applied_torque": {"joint_names": ["left_hip_pitch_joint"]},
                        "diff_object_pos_future": {"future_steps": [1, 2]},
                        "diff_object_ori_future": {"future_steps": [1, 2]},
                        "ref_object_contact_future": {"future_steps": [1, 2]},
                    },
                    "ref_joint_pos_": {
                        "ref_joint_pos_action_policy": {},
                    },
                },
                "action_scale": 1.0,
                "policy_joint_names": ["left_hip_pitch_joint"],
                "isaac_joint_names": ["left_hip_pitch_joint"],
                "isaac_body_names": ["pelvis", "left_elbow_link"],
                "default_joint_pos": 0.0,
            }
        )
    )

    exit_code = module.main(
        [
            "--task-yaml",
            str(task_path),
            "--policy-path",
            str(policy_path),
            "--require-policy",
            "--require-reference-observation",
            "--require-obs-action-smoke",
        ]
    )
    summary = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert summary["policy_obs_action_smoke_ok"] is True
    assert summary["policy_smoke_action_shape"] == [1, 1]
    assert set(summary["policy_smoke_observation_shapes"]) == {
        "policy",
        "command",
        "object",
        "priv",
        "ref_joint_pos_",
    }
