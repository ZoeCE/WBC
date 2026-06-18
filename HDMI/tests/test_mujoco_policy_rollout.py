import json
import importlib
from pathlib import Path

import numpy as np
import torch
import yaml
from tensordict.nn import TensorDictModule

from active_adaptation.mujoco import MujocoMotionReference
from active_adaptation.mujoco import policy_rollout
from active_adaptation.mujoco.policy import MujocoPolicyBundle
from active_adaptation.mujoco.policy_rollout import run_mujoco_policy_rollout


def _write_motion_dir(motion_dir: Path, body_names: list[str], joint_names: list[str]) -> None:
    motion_dir.mkdir(parents=True, exist_ok=True)
    body_pos_w = np.array(
        [
            [[0.0, 0.0, 0.80], [1.0, 0.0, 0.20]],
            [[0.1, 0.0, 0.82], [1.2, 0.1, 0.25]],
        ],
        dtype=np.float32,
    )
    body_quat_w = np.zeros((2, 2, 4), dtype=np.float32)
    body_quat_w[..., 0] = 1.0
    joint_start = np.linspace(
        0.10,
        0.10 + 0.20 * (len(joint_names) - 1),
        num=len(joint_names),
        dtype=np.float32,
    )
    joint_pos = np.stack((joint_start, joint_start + 0.20), axis=0)
    np.savez_compressed(
        motion_dir / "motion.npz",
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        joint_pos=joint_pos,
    )
    (motion_dir / "meta.json").write_text(
        json.dumps({"body_names": body_names, "joint_names": joint_names, "fps": 50.0})
    )


def _write_zero_policy_bundle(tmp_path: Path, joint_name: str) -> MujocoPolicyBundle:
    module = TensorDictModule(
        torch.nn.Linear(1, 1, bias=False),
        in_keys=["policy"],
        out_keys=["action"],
    )
    module.module.weight.data.zero_()
    policy_path = tmp_path / "policy-zero-final.pt"
    torch.save(module, policy_path)
    (tmp_path / "policy-zero-final.yaml").write_text(
        yaml.safe_dump(
            {
                "observation": {"policy": {"applied_action": {}}},
                "action_scale": 0.5,
                "policy_joint_names": [joint_name],
                "default_joint_pos": {joint_name: 0.0},
            }
        )
    )
    return MujocoPolicyBundle.load(policy_path)


def _write_object_policy_bundle(tmp_path: Path, joint_name: str, object_body_name: str) -> MujocoPolicyBundle:
    module = TensorDictModule(
        torch.nn.Linear(7, 1, bias=False),
        in_keys=["object"],
        out_keys=["action"],
    )
    module.module.weight.data.zero_()
    module.module.weight.data[0, 0] = 1.0
    module.module.weight.data[0, -1] = 1.0
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
                "action_scale": 0.5,
                "policy_joint_names": [joint_name],
                "default_joint_pos": {joint_name: 0.0},
            }
        )
    )
    return MujocoPolicyBundle.load(policy_path)


def _write_full_object_pose_policy_bundle(
    tmp_path: Path,
    joint_name: str,
    object_body_name: str,
) -> MujocoPolicyBundle:
    module = TensorDictModule(
        torch.nn.Linear(12, 1, bias=False),
        in_keys=["object"],
        out_keys=["action"],
    )
    module.module.weight.data.zero_()
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
                "action_scale": 0.5,
                "policy_joint_names": [joint_name],
                "default_joint_pos": {joint_name: 0.0},
            }
        )
    )
    return MujocoPolicyBundle.load(policy_path)


def test_reference_frame_write_initializes_object_root_and_joints(tmp_path):
    module = importlib.import_module("active_adaptation.envs.mujoco")
    from active_adaptation.assets_mjcf import ROBOTS

    class SceneCfg:
        robot = ROBOTS.with_object("g1_29dof", object_asset_name="door")

    scene = module.MJScene(SceneCfg(), num_envs=2, launch_viewer=False)
    robot = scene["robot"]
    door = scene["door"]
    object_body_name = door.body_names[0]
    robot_joint_name = robot.joint_names[0]
    object_joint_name = door.joint_names[0]
    body_names = [robot.body_names[0], object_body_name]
    joint_names = [robot_joint_name, object_joint_name]
    motion_dir = tmp_path / "motion-object-reference-frame"
    _write_motion_dir(motion_dir, body_names, joint_names)
    reference = MujocoMotionReference.from_motion_dir(
        motion_dir=motion_dir,
        body_names=body_names,
        joint_names=joint_names,
        root_body_name=robot.body_names[0],
        future_steps=[0],
    )

    policy_rollout._write_reference_frame_to_scene(scene, reference, step=0)
    scene.update(0.0)

    object_body_index = reference.body_names.index(object_body_name)
    expected_object_pos = (
        reference.body_pos_w[0, object_body_index].unsqueeze(0).expand(scene.num_envs, -1)
    )
    expected_object_quat = (
        reference.body_quat_w[0, object_body_index].unsqueeze(0).expand(scene.num_envs, -1)
    )
    object_joint_index = reference.joint_names.index(object_joint_name)
    expected_object_joint_pos = reference.joint_pos[0, object_joint_index].expand(scene.num_envs)

    assert torch.allclose(door.data.body_link_pos_w[:, 0], expected_object_pos, atol=1e-5)
    assert torch.allclose(door.data.body_link_quat_w[:, 0], expected_object_quat, atol=1e-5)
    assert torch.allclose(door.data.joint_pos[:, 0], expected_object_joint_pos, atol=1e-5)


def test_policy_rollout_steps_mujoco_scene_and_reports_parity(tmp_path):
    module = importlib.import_module("active_adaptation.envs.mujoco")
    from active_adaptation.assets_mjcf import ROBOTS

    class SceneCfg:
        robot = ROBOTS.with_object("g1_29dof", object_asset_name="door")

    scene = module.MJScene(SceneCfg(), num_envs=1, launch_viewer=False)
    robot = scene["robot"]
    door = scene["door"]
    body_names = [robot.body_names[0], door.body_names[0]]
    joint_name = robot.joint_names[0]
    motion_dir = tmp_path / "motion"
    _write_motion_dir(motion_dir, body_names, [joint_name])
    reference = MujocoMotionReference.from_motion_dir(
        motion_dir=motion_dir,
        body_names=body_names,
        joint_names=[joint_name],
        root_body_name=robot.body_names[0],
        future_steps=[0],
    )
    policy_bundle = _write_zero_policy_bundle(tmp_path, joint_name)

    metrics = run_mujoco_policy_rollout(
        scene=scene,
        policy_bundle=policy_bundle,
        reference=reference,
        steps=[0, 1],
        decimation=1,
    )

    assert metrics.actions.shape == (2, 1, 1)
    assert metrics.joint_position_targets.shape == (2, 1, 1)
    assert metrics.q_l2.shape == (2, 1)
    assert metrics.body_pos_l2.shape == (2, 1)
    assert torch.isfinite(metrics.q_l2).all()
    assert torch.isfinite(metrics.body_pos_l2).all()
    assert torch.allclose(metrics.actions, torch.zeros_like(metrics.actions))
    assert metrics.action_rate_l2.shape == (2, 1, 1)
    assert torch.allclose(metrics.action_rate_l2, torch.zeros_like(metrics.action_rate_l2))


def test_policy_rollout_computes_closed_loop_reward_from_spec(tmp_path):
    module = importlib.import_module("active_adaptation.envs.mujoco")
    from active_adaptation.assets_mjcf import ROBOTS

    class SceneCfg:
        robot = ROBOTS.with_object("g1_29dof", object_asset_name="door")

    scene = module.MJScene(SceneCfg(), num_envs=1, launch_viewer=False)
    robot = scene["robot"]
    door = scene["door"]
    body_names = [robot.body_names[0], door.body_names[0]]
    joint_name = robot.joint_names[0]
    motion_dir = tmp_path / "motion-rollout-reward"
    _write_motion_dir(motion_dir, body_names, [joint_name])
    reference = MujocoMotionReference.from_motion_dir(
        motion_dir=motion_dir,
        body_names=body_names,
        joint_names=[joint_name],
        root_body_name=robot.body_names[0],
        future_steps=[0],
    )
    policy_bundle = _write_zero_policy_bundle(tmp_path, joint_name)
    reward_cfg = {
        "loco": {
            "survival": {
                "weight": 1.0,
                "enabled": True,
            }
        }
    }

    metrics = run_mujoco_policy_rollout(
        scene=scene,
        policy_bundle=policy_bundle,
        reference=reference,
        reward_cfg=reward_cfg,
        steps=[0, 1],
        decimation=1,
    )

    assert metrics.reward is not None
    assert metrics.reward.shape == (2, 1, 1)
    assert torch.allclose(metrics.reward, torch.ones_like(metrics.reward))


def test_policy_rollout_fills_object_observations_from_mujoco_scene(tmp_path):
    module = importlib.import_module("active_adaptation.envs.mujoco")
    from active_adaptation.assets_mjcf import ROBOTS

    class SceneCfg:
        robot = ROBOTS.with_object("g1_29dof", object_asset_name="door")

    scene = module.MJScene(SceneCfg(), num_envs=1, launch_viewer=False)
    robot = scene["robot"]
    door = scene["door"]
    object_body_name = door.body_names[0]
    body_names = [robot.body_names[0], object_body_name]
    joint_name = robot.joint_names[0]
    motion_dir = tmp_path / "motion-object"
    _write_motion_dir(motion_dir, body_names, [joint_name])
    reference = MujocoMotionReference.from_motion_dir(
        motion_dir=motion_dir,
        body_names=body_names,
        joint_names=[joint_name],
        root_body_name=robot.body_names[0],
        future_steps=[0],
    )
    policy_bundle = _write_object_policy_bundle(tmp_path, joint_name, object_body_name)

    metrics = run_mujoco_policy_rollout(
        scene=scene,
        policy_bundle=policy_bundle,
        reference=reference,
        steps=[0, 1],
        decimation=1,
    )

    assert metrics.actions.shape == (2, 1, 1)
    assert metrics.joint_position_targets.shape == (2, 1, 1)
    assert torch.isfinite(metrics.actions).all()


def test_policy_rollout_fills_full_object_pose_observations_from_mujoco_scene(tmp_path):
    module = importlib.import_module("active_adaptation.envs.mujoco")
    from active_adaptation.assets_mjcf import ROBOTS

    class SceneCfg:
        robot = ROBOTS.with_object("g1_29dof", object_asset_name="door")

    scene = module.MJScene(SceneCfg(), num_envs=1, launch_viewer=False)
    robot = scene["robot"]
    door = scene["door"]
    object_body_name = door.body_names[0]
    body_names = [robot.body_names[0], object_body_name]
    joint_name = robot.joint_names[0]
    motion_dir = tmp_path / "motion-object-pose"
    _write_motion_dir(motion_dir, body_names, [joint_name])
    reference = MujocoMotionReference.from_motion_dir(
        motion_dir=motion_dir,
        body_names=body_names,
        joint_names=[joint_name],
        root_body_name=robot.body_names[0],
        future_steps=[0],
    )
    policy_bundle = _write_full_object_pose_policy_bundle(tmp_path, joint_name, object_body_name)

    metrics = run_mujoco_policy_rollout(
        scene=scene,
        policy_bundle=policy_bundle,
        reference=reference,
        steps=[0, 1],
        decimation=1,
    )

    assert metrics.actions.shape == (2, 1, 1)
    assert torch.isfinite(metrics.actions).all()
