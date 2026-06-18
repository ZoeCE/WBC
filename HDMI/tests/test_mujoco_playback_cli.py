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
    )
    (tmp_path / "meta.json").write_text(
        json.dumps({"body_names": body_names, "joint_names": joint_names, "fps": 50.0})
    )
    return door.body_names[0], door.joint_names[0]


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
        "object_tracking.object_joint_pos_tracking",
    ]
    assert summary["reward_terms_skipped"] == ["tracking.joint_vel_tracking_product"]


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
