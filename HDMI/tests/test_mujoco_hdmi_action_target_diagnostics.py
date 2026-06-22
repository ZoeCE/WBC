import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    script_path = ROOT / "scripts/mujoco_hdmi_action_target_diagnostics.py"
    spec = importlib.util.spec_from_file_location("mujoco_hdmi_action_target_diagnostics", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_action_target_summary_flags_near_zero_action_with_far_reference_target():
    module = _load_module()

    report = module.summarize_action_target_metrics(
        task_identity={
            "task_name": "G1PushBox",
            "task_stem": "push_box",
            "task_path": "cfg/task/G1/hdmi/push_box.yaml",
        },
        policy_path="/tmp/policy.pt",
        policy_config_path="/tmp/policy.yaml",
        steps=torch.tensor([0, 1]),
        decimation=10,
        actions=torch.tensor(
            [
                [[0.01, -0.02]],
                [[0.30, -0.10]],
            ],
            dtype=torch.float32,
        ),
        joint_position_targets=torch.tensor(
            [
                [[0.0, 0.0]],
                [[0.3, 0.4]],
            ],
            dtype=torch.float32,
        ),
        reference_joint_pos=torch.tensor(
            [
                [[1.0, 0.0]],
                [[0.3, 0.4]],
            ],
            dtype=torch.float32,
        ),
        policy_joint_names=["left_hip_pitch_joint", "right_elbow_joint"],
        q_l2=torch.tensor([[0.52], [0.01]], dtype=torch.float32),
        body_pos_l2=torch.tensor([[0.04], [0.002]], dtype=torch.float32),
    )

    assert report["task_name"] == "G1PushBox"
    assert report["decimation"] == 10
    assert report["step_count"] == 2
    assert report["first_step"]["step"] == 0
    assert report["first_step"]["raw_action_abs_max"] == 0.02


def test_action_target_summary_reports_top_joint_target_reference_errors():
    module = _load_module()

    report = module.summarize_action_target_metrics(
        task_identity={
            "task_name": "G1PushBox",
            "task_stem": "push_box",
            "task_path": "cfg/task/G1/hdmi/push_box.yaml",
        },
        policy_path="/tmp/policy.pt",
        policy_config_path="/tmp/policy.yaml",
        steps=torch.tensor([0, 1]),
        decimation=10,
        actions=torch.zeros((2, 1, 3), dtype=torch.float32),
        joint_position_targets=torch.tensor(
            [
                [[0.0, 2.0, 0.1]],
                [[0.0, 3.0, 1.6]],
            ],
            dtype=torch.float32,
        ),
        reference_joint_pos=torch.tensor(
            [
                [[0.0, 0.5, 0.0]],
                [[0.0, 1.0, 0.4]],
            ],
            dtype=torch.float32,
        ),
        policy_joint_names=["left_hip_pitch_joint", "right_elbow_joint", "right_wrist_roll_joint"],
        q_l2=torch.tensor([[0.52], [0.01]], dtype=torch.float32),
        body_pos_l2=torch.tensor([[0.04], [0.002]], dtype=torch.float32),
    )

    assert report["top_joint_target_ref_errors"][0] == {
        "name": "right_elbow_joint",
        "mean_abs_error": 1.75,
        "max_abs_error": 2.0,
        "max_step": 1,
    }
    assert report["top_joint_target_ref_errors"][1] == {
        "name": "right_wrist_roll_joint",
        "mean_abs_error": 0.65,
        "max_abs_error": 1.2,
        "max_step": 1,
    }


def test_action_target_cli_parses_initial_state():
    module = _load_module()

    default_args = module._parse_args([
        "--task-yaml",
        "cfg/task/G1/hdmi/move_suitcase.yaml",
        "--policy-path",
        "/tmp/policy.pt",
    ])
    scene_default_args = module._parse_args([
        "--task-yaml",
        "cfg/task/G1/hdmi/move_suitcase.yaml",
        "--policy-path",
        "/tmp/policy.pt",
        "--initial-state",
        "scene_default",
    ])

    assert default_args.initial_state == "reference_frame"
    assert default_args.robot_name is None
    assert scene_default_args.initial_state == "scene_default"


def test_action_target_diagnostic_resolves_scene_and_policy_inputs_from_task_cfg():
    module = _load_module()
    task_cfg = {
        "robot": {
            "robot_type": "g1_29dof_rubberhand-feet_box-eef_box-body_capsule",
            "override_params": {
                "init_state": {
                    "joint_pos": {
                        "left_wrist_yaw_joint": -0.4,
                        "right_wrist_yaw_joint": 0.4,
                    }
                }
            },
        },
        "command": {
            "object_asset_name": "foam",
            "extra_object_names": ["stool_support"],
        },
    }

    inputs = module.resolve_diagnostic_scene_inputs(
        robot_name=None,
        object_type=None,
        task_cfg=task_cfg,
        command_cfg=task_cfg["command"],
    )

    assert inputs["robot_name"] == "g1_29dof_rubberhand"
    assert inputs["object_type"] == "foam_with_support"
    assert inputs["policy_config_overrides"]["default_joint_pos"] == {
        "left_wrist_yaw_joint": -0.4,
        "right_wrist_yaw_joint": 0.4,
    }


def test_action_target_diagnostic_defaults_two_contact_offsets_to_wrist_eefs():
    module = _load_module()

    names = module._contact_eef_body_names_from_command(
        {
            "contact_target_pos_offset": [[0.0, 0.17, 0.14], [0.0, -0.17, 0.14]],
            "contact_eef_pos_offset": [[0.05, 0.0, 0.0], [0.05, 0.0, 0.0]],
        }
    )

    assert names == ["left_wrist_yaw_link", "right_wrist_yaw_link"]
