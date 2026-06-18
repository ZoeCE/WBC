import importlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from tensordict.nn import TensorDictModule


ROOT = Path(__file__).resolve().parents[1]


def _load_cli_module():
    script_path = ROOT / "scripts/mujoco_playback_parity.py"
    spec = importlib.util.spec_from_file_location("mujoco_playback_parity_script", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_motion_dir(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    module = importlib.import_module("active_adaptation.envs.mujoco")
    from active_adaptation.assets_mjcf import ROBOTS

    class SceneCfg:
        robot = ROBOTS.with_object("g1_29dof", object_asset_name="door")

    scene = module.MJScene(SceneCfg(), num_envs=1, launch_viewer=False)
    robot = scene["robot"]
    door = scene["door"]

    body_names = [robot.body_names[0], door.body_names[0]]
    joint_names = [robot.joint_names[0], door.joint_names[0]]
    body_pos_w = np.array(
        [
            [[0.0, 0.0, 0.80], [1.0, 0.0, 0.20]],
            [[0.1, 0.0, 0.82], [1.2, 0.1, 0.25]],
        ],
        dtype=np.float32,
    )
    body_quat_w = np.zeros((2, 2, 4), dtype=np.float32)
    body_quat_w[..., 0] = 1.0
    joint_pos = np.array(
        [
            [0.10, 0.40],
            [0.30, 0.80],
        ],
        dtype=np.float32,
    )
    np.savez_compressed(
        tmp_path / "motion.npz",
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        joint_pos=joint_pos,
        object_contact=np.array([[True], [False]], dtype=np.bool_),
    )
    (tmp_path / "meta.json").write_text(
        json.dumps({"body_names": body_names, "joint_names": joint_names, "fps": 50.0})
    )
    return door.body_names[0], door.joint_names[0]


def _write_robot_motion_dir(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    module = importlib.import_module("active_adaptation.envs.mujoco")
    from active_adaptation.assets_mjcf import ROBOTS

    class SceneCfg:
        robot = ROBOTS["g1_29dof"]

    scene = module.MJScene(SceneCfg(), num_envs=1, launch_viewer=False)
    robot = scene["robot"]

    body_names = [robot.body_names[0]]
    joint_names = [robot.joint_names[0]]
    body_pos_w = np.array(
        [
            [[0.0, 0.0, 0.80]],
            [[0.1, 0.0, 0.82]],
        ],
        dtype=np.float32,
    )
    body_quat_w = np.zeros((2, 1, 4), dtype=np.float32)
    body_quat_w[..., 0] = 1.0
    joint_pos = np.array([[0.10], [0.30]], dtype=np.float32)
    np.savez_compressed(
        tmp_path / "motion.npz",
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        joint_pos=joint_pos,
    )
    (tmp_path / "meta.json").write_text(
        json.dumps({"body_names": body_names, "joint_names": joint_names, "fps": 50.0})
    )
    return body_names[0], joint_names[0]


def _write_policy_bundle(tmp_path, action_dim):
    module = TensorDictModule(
        torch.nn.Linear(action_dim, action_dim, bias=False),
        in_keys=["policy"],
        out_keys=["action"],
    )
    module.module.weight.data.copy_(torch.eye(action_dim))
    policy_path = tmp_path / "policy-test-final.pt"
    torch.save(module, policy_path)
    (tmp_path / "policy-test-final.yaml").write_text(
        yaml.safe_dump(
            {
                "observation": {
                    "policy": {
                        "applied_action": {},
                    },
                },
                "action_scale": 0.5,
                "policy_joint_names": [f"j{i}" for i in range(action_dim)],
                "default_joint_pos": 0.0,
            }
        )
    )
    return policy_path


def _write_rollout_policy_bundle(tmp_path, joint_name):
    module = TensorDictModule(
        torch.nn.Linear(1, 1, bias=False),
        in_keys=["policy"],
        out_keys=["action"],
    )
    module.module.weight.data.zero_()
    policy_path = tmp_path / "policy-rollout-final.pt"
    torch.save(module, policy_path)
    (tmp_path / "policy-rollout-final.yaml").write_text(
        yaml.safe_dump(
            {
                "observation": {
                    "policy": {
                        "applied_action": {},
                    },
                },
                "action_scale": 0.5,
                "policy_joint_names": [joint_name],
                "default_joint_pos": {joint_name: 0.0},
            }
        )
    )
    return policy_path


def _write_object_policy_bundle(tmp_path, object_body_name):
    module = TensorDictModule(
        torch.nn.Linear(7, 2, bias=False),
        in_keys=["object"],
        out_keys=["action"],
    )
    module.module.weight.data.zero_()
    module.module.weight.data[0, 0] = 1.0
    module.module.weight.data[1, -1] = 1.0
    policy_path = tmp_path / "policy-object-final.pt"
    torch.save(module, policy_path)
    (tmp_path / "policy-object-final.yaml").write_text(
        yaml.safe_dump(
            {
                "observation": {
                    "object": {
                        "object_xy_b": {"object_name": object_body_name},
                        "object_heading_b": {"object_name": object_body_name},
                        "ref_contact_pos_b": {
                            "object_name": object_body_name,
                            "contact_target_pos_offset": [[0.0, 1.0, 0.0]],
                            "yaw_only": True,
                        },
                    },
                },
                "action_scale": 1.0,
                "policy_joint_names": ["j0", "j1"],
                "default_joint_pos": 0.0,
            }
        )
    )
    return policy_path


def _write_full_object_pose_policy_bundle(tmp_path, object_body_name):
    module = TensorDictModule(
        torch.nn.Linear(12, 2, bias=False),
        in_keys=["object"],
        out_keys=["action"],
    )
    module.module.weight.data.zero_()
    module.module.weight.data[0, 0] = 1.0
    module.module.weight.data[1, -1] = 1.0
    policy_path = tmp_path / "policy-object-pose-final.pt"
    torch.save(module, policy_path)
    (tmp_path / "policy-object-pose-final.yaml").write_text(
        yaml.safe_dump(
            {
                "observation": {
                    "object": {
                        "object_pos_b": {"object_name": object_body_name},
                        "object_ori_b": {"object_name": object_body_name},
                    },
                },
                "action_scale": 1.0,
                "policy_joint_names": ["j0", "j1"],
                "default_joint_pos": 0.0,
            }
        )
    )
    return policy_path


def _write_command_future_policy_bundle(tmp_path, body_name, joint_name):
    module = TensorDictModule(
        torch.nn.Linear(13, 1, bias=False),
        in_keys=["command"],
        out_keys=["action"],
    )
    module.module.weight.data.zero_()
    policy_path = tmp_path / "policy-command-future-final.pt"
    torch.save(module, policy_path)
    (tmp_path / "policy-command-future-final.yaml").write_text(
        yaml.safe_dump(
            {
                "observation": {
                    "command": {
                        "ref_body_pos_future_local": {
                            "body_names": [body_name],
                            "future_steps": [0, 1, 1],
                            "root_body_name": body_name,
                        },
                        "ref_joint_pos_future": {
                            "joint_names": [joint_name],
                            "future_steps": [0, 1, 1],
                        },
                        "ref_motion_phase": {},
                    },
                },
                "action_scale": 0.5,
                "policy_joint_names": [joint_name],
                "default_joint_pos": {joint_name: 0.0},
            }
        )
    )
    return policy_path


def test_mujoco_playback_parity_cli_prints_json_summary(tmp_path, capsys):
    object_body_name, object_joint_name = _write_motion_dir(tmp_path)
    reward_cfg_path = tmp_path / "reward.json"
    reward_cfg_path.write_text(
        json.dumps(
            {
                "tracking": {
                    "joint_pos_tracking_product": {"weight": 1.0, "sigma": 0.25},
                },
                "object_tracking": {
                    "object_joint_pos_tracking": {"weight": 1.0, "sigma": 0.25},
                },
            }
        )
    )
    script = _load_cli_module()

    exit_code = script.main(
        [
            "--motion-dir",
            str(tmp_path),
            "--object-name",
            "door",
            "--object-body-name",
            object_body_name,
            "--object-joint-name",
            object_joint_name,
            "--reward-config-json",
            str(reward_cfg_path),
            "--steps",
            "0,1",
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["steps"] == 2
    assert summary["envs"] == 1
    assert summary["reward_shape"] == [2, 1, 2]
    assert summary["q_l2_max"] < 1e-5
    assert summary["body_pos_l2_max"] >= 0.0
    assert summary["body_pos_l2_mean"] <= summary["body_pos_l2_max"]
    assert summary["reward_mean"] > 0.99


def test_mujoco_playback_parity_cli_loads_task_yaml_reward(tmp_path, capsys):
    object_body_name, object_joint_name = _write_motion_dir(tmp_path)
    cfg_dir = tmp_path / "cfg/task"
    (cfg_dir / "base").mkdir(parents=True)
    (cfg_dir / "base/test-base.yaml").write_text(
        """
reward:
  tracking:
    joint_pos_tracking_product: {weight: 1.0, sigma: 0.25}
    joint_vel_tracking_product: {weight: 1.0, sigma: 0.25}
  object_tracking:
    object_pos_tracking: {enabled: false}
""".strip()
    )
    task_yaml = cfg_dir / "door.yaml"
    task_yaml.write_text(
        """
defaults:
  - base/test-base
  - _self_
reward:
  object_tracking:
    object_joint_pos_tracking: {weight: 1.0, sigma: 0.25}
""".strip()
    )
    script = _load_cli_module()

    exit_code = script.main(
        [
            "--motion-dir",
            str(tmp_path),
            "--object-name",
            "door",
            "--object-body-name",
            object_body_name,
            "--object-joint-name",
            object_joint_name,
            "--task-yaml",
            str(task_yaml),
            "--steps",
            "0,1",
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["reward_shape"] == [2, 1, 2]
    assert summary["reward_terms_used"] == [
        "tracking.joint_pos_tracking_product",
        "tracking.joint_vel_tracking_product",
        "object_tracking.object_joint_pos_tracking",
    ]
    assert summary["reward_terms_skipped"] == []


def test_mujoco_playback_parity_cli_infers_inputs_from_task_yaml(tmp_path, capsys):
    motion_dir = tmp_path / "data/motion/test_door"
    object_body_name, object_joint_name = _write_motion_dir(motion_dir)
    task_yaml = tmp_path / "cfg/task/G1/hdmi/door.yaml"
    task_yaml.parent.mkdir(parents=True)
    task_yaml.write_text(
        f"""
command:
  data_path: data/motion/test_door
  root_body_name: pelvis
  object_asset_name: door
  object_body_name: {object_body_name}
  object_joint_name: {object_joint_name}
reward:
  object_tracking:
    object_joint_pos_tracking: {{weight: 1.0, sigma: 0.25}}
""".strip()
    )
    script = _load_cli_module()

    exit_code = script.main(["--task-yaml", str(task_yaml), "--steps", "0,1"])

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["motion_dir"] == str(motion_dir)
    assert summary["object_name"] == "door"
    assert summary["object_body_name"] == object_body_name
    assert summary["object_joint_name"] == object_joint_name
    assert summary["root_body_name"] == "pelvis"
    assert summary["steps"] == 2
    assert summary["envs"] == 1
    assert summary["reward_shape"] == [2, 1, 1]
    assert summary["q_l2_max"] < 1e-5


def test_mujoco_playback_parity_cli_reports_policy_motion_mjcf_name_mapping(tmp_path, capsys):
    motion_dir = tmp_path / "data/motion/test_door"
    object_body_name, object_joint_name = _write_motion_dir(motion_dir)
    meta = json.loads((motion_dir / "meta.json").read_text())
    policy_path = _write_rollout_policy_bundle(tmp_path, meta["joint_names"][0])
    policy_cfg_path = policy_path.with_suffix(".yaml")
    policy_cfg = yaml.safe_load(policy_cfg_path.read_text())
    policy_cfg["isaac_body_names"] = [meta["body_names"][0]]
    policy_cfg["isaac_joint_names"] = [meta["joint_names"][0]]
    policy_cfg_path.write_text(yaml.safe_dump(policy_cfg))
    task_yaml = tmp_path / "cfg/task/G1/hdmi/door.yaml"
    task_yaml.parent.mkdir(parents=True)
    task_yaml.write_text(
        f"""
command:
  data_path: data/motion/test_door
  root_body_name: {meta["body_names"][0]}
  object_asset_name: door
  object_body_name: {object_body_name}
  object_joint_name: {object_joint_name}
reward:
  object_tracking:
    object_joint_pos_tracking: {{weight: 1.0, sigma: 0.25}}
""".strip()
    )
    script = _load_cli_module()

    exit_code = script.main(
        [
            "--task-yaml",
            str(task_yaml),
            "--policy-path",
            str(policy_path),
            "--steps",
            "0,1",
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["policy_config_path"] == str(policy_cfg_path)
    assert summary["policy_body_name_mapping"][0]["name"] == meta["body_names"][0]
    assert summary["policy_body_name_mapping"][0]["policy_index"] == 0
    assert summary["policy_body_name_mapping"][0]["motion_index"] == 0
    assert isinstance(summary["policy_body_name_mapping"][0]["mujoco_index"], int)
    assert summary["policy_joint_name_mapping"][0]["name"] == meta["joint_names"][0]
    assert summary["policy_joint_name_mapping"][0]["policy_index"] == 0
    assert summary["policy_joint_name_mapping"][0]["motion_index"] == 0
    assert isinstance(summary["policy_joint_name_mapping"][0]["mujoco_index"], int)


def test_mujoco_playback_parity_cli_reports_task_motion_body_mapping(tmp_path, capsys):
    motion_dir = tmp_path / "data/motion/test_door"
    reference_object_body_name, object_joint_name = _write_motion_dir(motion_dir)
    task_yaml = tmp_path / "cfg/task/G1/hdmi/door.yaml"
    task_yaml.parent.mkdir(parents=True)
    task_yaml.write_text(
        f"""
command:
  data_path: data/motion/test_door
  root_body_name: pelvis
  object_asset_name: door
  object_body_name: door_panel
  object_joint_name: {object_joint_name}
reward:
  object_tracking:
    object_joint_pos_tracking: {{weight: 1.0, sigma: 0.25}}
""".strip()
    )
    script = _load_cli_module()

    exit_code = script.main(["--task-yaml", str(task_yaml), "--steps", "0,1"])

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["object_name"] == "door"
    assert summary["object_body_name"] == "door_panel"
    assert summary["task_object_body_name"] == "door_panel"
    assert summary["reference_object_body_name"] == reference_object_body_name
    assert summary["asset_object_body_names"] == ["door", "door_panel"]
    assert summary["object_joint_name"] == object_joint_name
    assert summary["q_l2_max"] < 1e-5
    assert summary["body_pos_l2_max"] < 1e-5


def test_mujoco_playback_parity_cli_uses_task_yaml_contact_reward(tmp_path, capsys):
    motion_dir = tmp_path / "data/motion/test_door"
    object_body_name, object_joint_name = _write_motion_dir(motion_dir)
    meta = json.loads((motion_dir / "meta.json").read_text())
    eef_body_name = meta["body_names"][0]
    task_yaml = tmp_path / "cfg/task/G1/hdmi/door.yaml"
    task_yaml.parent.mkdir(parents=True)
    task_yaml.write_text(
        f"""
command:
  data_path: data/motion/test_door
  root_body_name: {eef_body_name}
  object_asset_name: door
  object_body_name: {object_body_name}
  object_joint_name: {object_joint_name}
  contact_eef_body_name: [{eef_body_name}]
  contact_target_pos_offset: [[0.0, 0.0, 0.0]]
  contact_eef_pos_offset: [[0.0, 0.0, 0.0]]
reward:
  object_tracking:
    eef_contact_exp: {{weight: 1.0, pos_sigma: 1.0, frc_sigma: 1.0, frc_thres: 0.0}}
""".strip()
    )
    script = _load_cli_module()

    exit_code = script.main(["--task-yaml", str(task_yaml), "--steps", "0,1"])

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["reward_shape"] == [2, 1, 1]
    assert summary["reward_terms_used"] == ["object_tracking.eef_contact_exp"]
    assert summary["reward_terms_skipped"] == []



def test_mujoco_playback_parity_cli_rolls_out_task_yaml_contact_reward(tmp_path, capsys):
    motion_dir = tmp_path / "data/motion/test_door"
    object_body_name, object_joint_name = _write_motion_dir(motion_dir)
    meta = json.loads((motion_dir / "meta.json").read_text())
    eef_body_name = meta["body_names"][0]
    joint_name = meta["joint_names"][0]
    policy_path = _write_rollout_policy_bundle(tmp_path, joint_name)
    task_yaml = tmp_path / "cfg/task/G1/hdmi/door.yaml"
    task_yaml.parent.mkdir(parents=True)
    task_yaml.write_text(
        f"""
command:
  data_path: data/motion/test_door
  root_body_name: {eef_body_name}
  object_asset_name: door
  object_body_name: {object_body_name}
  object_joint_name: {object_joint_name}
  contact_eef_body_name: [{eef_body_name}]
  contact_target_pos_offset: [[0.0, 0.0, 0.0]]
  contact_eef_pos_offset: [[0.0, 0.0, 0.0]]
reward:
  object_tracking:
    eef_contact_exp: {{weight: 1.0, pos_sigma: 1.0, frc_sigma: 1.0, frc_thres: 0.0}}
""".strip()
    )
    script = _load_cli_module()

    exit_code = script.main(
        [
            "--task-yaml",
            str(task_yaml),
            "--policy-path",
            str(policy_path),
            "--policy-rollout",
            "--steps",
            "0,1",
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["reward_terms_used"] == ["object_tracking.eef_contact_exp"]
    assert summary["contact_eef_body_names"] == [eef_body_name]
    assert summary["contact_target_pos_offset"] == [[0.0, 0.0, 0.0]]
    assert summary["contact_eef_pos_offset"] == [[0.0, 0.0, 0.0]]
    assert summary["policy_rollout_reward_shape"] == [2, 1, 1]
    assert summary["policy_rollout_reward_min"] <= summary["policy_rollout_reward_max"]

def test_mujoco_playback_parity_cli_uses_supported_loco_rewards(tmp_path, capsys):
    motion_dir = tmp_path / "data/motion/test_robot"
    root_body_name, joint_name = _write_robot_motion_dir(motion_dir)
    task_yaml = tmp_path / "cfg/task/G1/hdmi/robot.yaml"
    task_yaml.parent.mkdir(parents=True)
    task_yaml.write_text(
        f"""
command:
  data_path: data/motion/test_robot
  root_body_name: {root_body_name}
reward:
  loco:
    survival: {{weight: 1.0}}
    joint_vel_l2: {{weight: 5.0e-4, joint_names: [{joint_name}]}}
    joint_pos_limits: {{weight: 10.0, joint_names: [{joint_name}], soft_factor: 0.9}}
    action_rate_l2: {{weight: 0.01}}
    joint_torque_limits: {{weight: 0.01, joint_names: [{joint_name}], soft_factor: 0.9}}
""".strip()
    )
    script = _load_cli_module()

    exit_code = script.main(["--task-yaml", str(task_yaml), "--steps", "0,1"])

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["reward_shape"] == [2, 1, 1]
    assert summary["reward_terms_used"] == [
        "loco.survival",
        "loco.joint_vel_l2",
        "loco.joint_pos_limits",
        "loco.action_rate_l2",
        "loco.joint_torque_limits",
    ]
    assert summary["reward_terms_skipped"] == []


def test_mujoco_playback_parity_cli_uses_supported_feet_rewards(tmp_path, capsys):
    motion_dir = tmp_path / "data/motion/test_robot"
    body_name, _ = _write_robot_motion_dir(motion_dir)
    task_yaml = tmp_path / "cfg/task/G1/hdmi/robot.yaml"
    task_yaml.parent.mkdir(parents=True)
    task_yaml.write_text(
        f"""
command:
  data_path: data/motion/test_robot
  root_body_name: {body_name}
reward:
  feet:
    feet_slip: {{weight: 1.0, body_names: [{body_name}], tolerance: 0.0}}
    impact_force_l2: {{weight: 1.0, body_names: [{body_name}]}}
    feet_air_time: {{weight: 1.0, body_names: [{body_name}], thres: 0.2}}
    feet_stumble: {{weight: 1.0, body_names: [{body_name}]}}
""".strip()
    )
    script = _load_cli_module()

    exit_code = script.main(["--task-yaml", str(task_yaml), "--steps", "0,1"])

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["reward_shape"] == [2, 1, 1]
    assert summary["reward_terms_used"] == [
        "feet.feet_slip",
        "feet.impact_force_l2",
        "feet.feet_air_time",
        "feet.feet_stumble",
    ]
    assert summary["reward_terms_skipped"] == []


def test_all_g1_hdmi_task_rewards_have_mujoco_playback_terms():
    script = _load_cli_module()
    missing = {}
    for task_yaml in sorted((ROOT / "cfg/task/G1/hdmi").glob("*.yaml")):
        reward_cfg = script.load_task_reward_config(task_yaml)
        _, _, skipped = script.filter_kinematic_reward_config(reward_cfg)
        if skipped:
            missing[task_yaml.name] = skipped

    assert missing == {}


def test_mujoco_playback_parity_cli_reports_policy_action_summary(tmp_path, capsys):
    object_body_name, object_joint_name = _write_motion_dir(tmp_path)
    policy_path = _write_policy_bundle(tmp_path, action_dim=2)
    script = _load_cli_module()

    exit_code = script.main(
        [
            "--motion-dir",
            str(tmp_path),
            "--object-name",
            "door",
            "--object-body-name",
            object_body_name,
            "--object-joint-name",
            object_joint_name,
            "--policy-path",
            str(policy_path),
            "--steps",
            "0,1",
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["policy_path"] == str(policy_path)
    assert summary["policy_action_shape"] == [2, 1, 2]
    assert summary["policy_joint_target_shape"] == [2, 1, 2]
    assert summary["policy_action_max_abs"] == 0.0


def test_mujoco_playback_parity_cli_uses_exported_policy_reference_future_steps(tmp_path, capsys):
    object_body_name, object_joint_name = _write_motion_dir(tmp_path)
    meta = json.loads((tmp_path / "meta.json").read_text())
    policy_path = _write_command_future_policy_bundle(tmp_path, meta["body_names"][0], meta["joint_names"][0])
    script = _load_cli_module()

    exit_code = script.main(
        [
            "--motion-dir",
            str(tmp_path),
            "--object-name",
            "door",
            "--object-body-name",
            object_body_name,
            "--object-joint-name",
            object_joint_name,
            "--policy-path",
            str(policy_path),
            "--steps",
            "0,1",
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["policy_action_shape"] == [2, 1, 1]
    assert summary["policy_joint_target_shape"] == [2, 1, 1]
    assert summary["policy_action_dim"] == 1
    assert summary["policy_reference_body_names"] == [meta["body_names"][0]]
    assert summary["policy_reference_joint_names"] == [meta["joint_names"][0]]
    assert summary["policy_reference_root_body_name"] == meta["body_names"][0]
    assert summary["policy_reference_future_steps"] == [0, 1, 1]


def test_mujoco_playback_parity_cli_reports_closed_loop_policy_rollout(tmp_path, capsys):
    object_body_name, object_joint_name = _write_motion_dir(tmp_path)
    joint_name = json.loads((tmp_path / "meta.json").read_text())["joint_names"][0]
    policy_path = _write_rollout_policy_bundle(tmp_path, joint_name)
    reward_cfg_path = tmp_path / "rollout-reward.json"
    reward_cfg_path.write_text(json.dumps({"loco": {"survival": {"weight": 1.0}}}))
    script = _load_cli_module()

    exit_code = script.main(
        [
            "--motion-dir",
            str(tmp_path),
            "--object-name",
            "door",
            "--object-body-name",
            object_body_name,
            "--object-joint-name",
            object_joint_name,
            "--policy-path",
            str(policy_path),
            "--policy-rollout",
            "--num-envs",
            "2",
            "--reward-config-json",
            str(reward_cfg_path),
            "--steps",
            "0,1",
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["policy_path"] == str(policy_path)
    assert summary["envs"] == 2
    assert summary["policy_rollout_q_l2_shape"] == [2, 2]
    assert summary["policy_rollout_body_pos_l2_shape"] == [2, 2]
    assert summary["policy_rollout_action_shape"] == [2, 2, 1]
    assert summary["policy_rollout_joint_target_shape"] == [2, 2, 1]
    assert summary["policy_rollout_q_l2_max"] >= 0.0
    assert summary["policy_rollout_q_l2_mean"] >= 0.0
    assert summary["policy_rollout_body_pos_l2_max"] >= 0.0
    assert summary["policy_rollout_body_pos_l2_mean"] >= 0.0
    assert summary["policy_rollout_action_rate_l2_shape"] == [2, 2, 1]
    assert summary["policy_rollout_action_rate_l2_max"] >= 0.0
    assert summary["policy_rollout_reward_shape"] == [2, 2, 1]
    assert summary["policy_rollout_reward_mean"] == 1.0
    assert summary["policy_rollout_reward_min"] == 1.0
    assert summary["policy_rollout_reward_max"] == 1.0



def test_mujoco_playback_parity_cli_writes_per_step_trace_json(tmp_path, capsys):
    object_body_name, object_joint_name = _write_motion_dir(tmp_path)
    joint_name = json.loads((tmp_path / "meta.json").read_text())["joint_names"][0]
    policy_path = _write_rollout_policy_bundle(tmp_path, joint_name)
    reward_cfg_path = tmp_path / "trace-reward.json"
    reward_cfg_path.write_text(json.dumps({"loco": {"survival": {"weight": 1.0}}}))
    trace_path = tmp_path / "trace/mujoco-trace.json"
    script = _load_cli_module()

    try:
        exit_code = script.main(
            [
                "--motion-dir",
                str(tmp_path),
                "--object-name",
                "door",
                "--object-body-name",
                object_body_name,
                "--object-joint-name",
                object_joint_name,
                "--policy-path",
                str(policy_path),
                "--policy-rollout",
                "--num-envs",
                "2",
                "--reward-config-json",
                str(reward_cfg_path),
                "--trace-json",
                str(trace_path),
                "--steps",
                "0,1",
            ]
        )
    except SystemExit as exc:
        exit_code = exc.code

    assert exit_code == 0
    _ = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    trace = json.loads(trace_path.read_text())
    assert trace["playback"]["q_l2"]["shape"] == [2, 2]
    assert trace["playback"]["body_pos_l2"]["shape"] == [2, 2]
    assert trace["playback"]["reward"]["shape"] == [2, 2, 1]
    assert trace["policy_rollout"]["reward"]["shape"] == [2, 2, 1]
    assert trace["policy_rollout"]["action_rate_l2"]["shape"] == [2, 2, 1]
    assert trace["policy_rollout"]["actions"]["shape"] == [2, 2, 1]
    assert trace["policy_rollout"]["reward"]["values"] == [[[1.0], [1.0]], [[1.0], [1.0]]]
    assert trace["playback"]["reward_terms"]["loco.survival"]["shape"] == [2, 2, 1]
    assert trace["playback"]["reward_terms"]["loco.survival"]["values"] == [
        [[1.0], [1.0]],
        [[1.0], [1.0]],
    ]
    assert trace["policy_rollout"]["reward_terms"]["loco.survival"]["shape"] == [2, 2, 1]
    assert trace["policy_rollout"]["reward_terms"]["loco.survival"]["values"] == [
        [[1.0], [1.0]],
        [[1.0], [1.0]],
    ]

def test_mujoco_playback_parity_cli_fills_object_policy_observations_from_reference(tmp_path, capsys):
    object_body_name, object_joint_name = _write_motion_dir(tmp_path)
    policy_path = _write_object_policy_bundle(tmp_path, object_body_name)
    script = _load_cli_module()

    exit_code = script.main(
        [
            "--motion-dir",
            str(tmp_path),
            "--object-name",
            "door",
            "--object-body-name",
            object_body_name,
            "--object-joint-name",
            object_joint_name,
            "--policy-path",
            str(policy_path),
            "--steps",
            "0,1",
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["policy_action_shape"] == [2, 1, 2]
    assert summary["policy_joint_target_shape"] == [2, 1, 2]
    assert summary["policy_action_max_abs"] > 0.5


def test_mujoco_playback_parity_cli_fills_full_object_pose_policy_observations_from_reference(tmp_path, capsys):
    object_body_name, object_joint_name = _write_motion_dir(tmp_path)
    policy_path = _write_full_object_pose_policy_bundle(tmp_path, object_body_name)
    script = _load_cli_module()

    exit_code = script.main(
        [
            "--motion-dir",
            str(tmp_path),
            "--object-name",
            "door",
            "--object-body-name",
            object_body_name,
            "--object-joint-name",
            object_joint_name,
            "--policy-path",
            str(policy_path),
            "--steps",
            "0,1",
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["policy_action_shape"] == [2, 1, 2]
    assert summary["policy_joint_target_shape"] == [2, 1, 2]
    assert summary["policy_action_max_abs"] > 0.5
