import json
import importlib
from pathlib import Path

import numpy as np
import torch
import yaml
from tensordict.nn import TensorDictModule

from active_adaptation.mujoco import MujocoMotionReference
from active_adaptation.mujoco import policy_rollout
from active_adaptation.mujoco.policy import MujocoPolicyBundle, resolve_named_values
from active_adaptation.mujoco.policy_rollout import MujocoActionAdapterConfig, run_mujoco_policy_rollout


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


class _ConstantAction(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.register_buffer("value", torch.tensor([[value]], dtype=torch.float32))

    def forward(self, policy: torch.Tensor) -> torch.Tensor:
        return self.value.expand(policy.shape[0], -1)


def _write_constant_policy_bundle(tmp_path: Path, joint_name: str, value: float) -> MujocoPolicyBundle:
    module = TensorDictModule(
        _ConstantAction(value),
        in_keys=["policy"],
        out_keys=["action"],
    )
    policy_path = tmp_path / "policy-constant-final.pt"
    torch.save(module, policy_path)
    (tmp_path / "policy-constant-final.yaml").write_text(
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


def _write_zero_policy_bundle_with_joint_gains(
    tmp_path: Path,
    joint_name: str,
    *,
    stiffness: float,
    damping: float,
) -> MujocoPolicyBundle:
    module = TensorDictModule(
        torch.nn.Linear(1, 1, bias=False),
        in_keys=["policy"],
        out_keys=["action"],
    )
    module.module.weight.data.zero_()
    policy_path = tmp_path / "policy-gains-final.pt"
    torch.save(module, policy_path)
    (tmp_path / "policy-gains-final.yaml").write_text(
        yaml.safe_dump(
            {
                "observation": {"policy": {"applied_action": {}}},
                "action_scale": 0.5,
                "policy_joint_names": [joint_name],
                "default_joint_pos": {joint_name: 0.0},
                "joint_kp": {".*": stiffness},
                "joint_kd": {".*": damping},
            }
        )
    )
    return MujocoPolicyBundle.load(policy_path)


def _write_joint_pos_policy_bundle(
    tmp_path: Path,
    joint_name: str,
    *,
    default_joint_pos: float,
) -> MujocoPolicyBundle:
    module = TensorDictModule(
        torch.nn.Linear(1, 1, bias=False),
        in_keys=["policy"],
        out_keys=["action"],
    )
    module.module.weight.data.fill_(1.0)
    policy_path = tmp_path / "policy-joint-pos-final.pt"
    torch.save(module, policy_path)
    (tmp_path / "policy-joint-pos-final.yaml").write_text(
        yaml.safe_dump(
            {
                "observation": {
                    "policy": {
                        "joint_pos_history": {
                            "joint_names": [joint_name],
                            "history_steps": [0],
                        }
                    }
                },
                "action_scale": 1.0,
                "policy_joint_names": [joint_name],
                "default_joint_pos": {joint_name: default_joint_pos},
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

def _write_future_priv_policy_bundle(tmp_path: Path, joint_name: str, body_name: str) -> MujocoPolicyBundle:
    module = TensorDictModule(
        _ConstantAction(0.0),
        in_keys=["priv"],
        out_keys=["action"],
    )
    policy_path = tmp_path / "policy-future-priv-final.pt"
    torch.save(module, policy_path)
    (tmp_path / "policy-future-priv-final.yaml").write_text(
        yaml.safe_dump(
            {
                "observation": {
                    "priv": {
                        "ref_root_pos_future_b": {},
                        "ref_root_ori_future_b": {},
                        "ref_joint_pos_action_policy": {},
                        "diff_body_pos_future_local": {},
                        "diff_body_ori_future_local": {},
                        "diff_body_lin_vel_future_local": {},
                        "diff_body_ang_vel_future_local": {},
                        "root_linvel_b": {},
                        "body_pos_b": {"body_names": [body_name]},
                        "body_vel_b": {"body_names": [body_name]},
                        "body_height": {"body_names": [body_name]},
                    }
                },
                "action_scale": 0.5,
                "policy_joint_names": [joint_name],
                "default_joint_pos": {joint_name: 0.0},
            }
        )
    )
    return MujocoPolicyBundle.load(policy_path)





def _write_object_future_priv_policy_bundle(
    tmp_path: Path,
    joint_name: str,
    object_body_name: str,
    object_joint_name: str,
) -> MujocoPolicyBundle:
    module = TensorDictModule(
        _ConstantAction(0.0),
        in_keys=["priv"],
        out_keys=["action"],
    )
    policy_path = tmp_path / "policy-object-future-priv-final.pt"
    torch.save(module, policy_path)
    (tmp_path / "policy-object-future-priv-final.yaml").write_text(
        yaml.safe_dump(
            {
                "observation": {
                    "priv": {
                        "object_pos_b": {"object_name": object_body_name},
                        "object_ori_b": {"object_name": object_body_name},
                        "diff_object_pos_future": {},
                        "diff_object_ori_future": {},
                        "ref_object_contact_future": {},
                        "diff_contact_pos_b": {},
                        "object_joint_pos": {},
                        "object_joint_vel": {},
                        "object_joint_torque": {},
                    },
                    "object": {
                        "ref_contact_pos_b": {
                            "object_name": object_body_name,
                            "contact_target_pos_offset": [[0.0, 0.0, 0.0]],
                        }
                    },
                },
                "action_scale": 0.5,
                "policy_joint_names": [joint_name],
                "default_joint_pos": {joint_name: 0.0},
                "object_joint_name": object_joint_name,
            }
        )
    )
    return MujocoPolicyBundle.load(policy_path)

def test_policy_bundle_load_applies_task_robot_overrides_without_regex_conflicts(tmp_path):
    module = TensorDictModule(
        torch.nn.Linear(1, 1, bias=False),
        in_keys=["policy"],
        out_keys=["action"],
    )
    module.module.weight.data.zero_()
    policy_path = tmp_path / "policy-overrides-final.pt"
    torch.save(module, policy_path)
    isaac_joint_names = [
        "left_elbow_joint",
        "left_wrist_yaw_joint",
        "right_wrist_yaw_joint",
    ]
    (tmp_path / "policy-overrides-final.yaml").write_text(
        yaml.safe_dump(
            {
                "observation": {
                    "policy": {
                        "joint_pos_history": {
                            "joint_names": isaac_joint_names,
                            "history_steps": [0],
                        },
                    }
                },
                "action_scale": {"left_elbow_joint": 0.5},
                "policy_joint_names": ["left_elbow_joint"],
                "isaac_joint_names": isaac_joint_names,
                "default_joint_pos": {".*": 0.0},
                "joint_kp": {
                    ".*_elbow_joint": 40.0,
                    ".*_wrist_yaw_joint": 20.0,
                },
            }
        )
    )

    bundle = MujocoPolicyBundle.load(
        policy_path,
        policy_config_overrides={
            "default_joint_pos": {
                "left_wrist_yaw_joint": -0.4,
                "right_wrist_yaw_joint": 0.4,
            },
            "joint_kp": {
                ".*_wrist_yaw_joint": 4.0,
            },
        },
    )

    assert torch.allclose(bundle.default_joint_pos, torch.tensor([0.0]))
    assert torch.allclose(
        bundle.observation_default_joint_pos,
        torch.tensor([0.0, -0.4, 0.4]),
    )
    assert resolve_named_values(
        bundle.config["joint_kp"],
        isaac_joint_names,
        field_name="joint_kp",
        require_all=True,
    ).tolist() == [40.0, 4.0, 4.0]


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


def test_policy_rollout_can_start_from_scene_default_without_reference_reset(tmp_path):
    module = importlib.import_module("active_adaptation.envs.mujoco")
    from active_adaptation.assets_mjcf import ROBOTS

    class SceneCfg:
        robot = ROBOTS.with_object("g1_29dof", object_asset_name="door")

    scene = module.MJScene(SceneCfg(), num_envs=1, launch_viewer=False)
    robot = scene["robot"]
    body_names = [robot.body_names[0]]
    joint_name = robot.joint_names[0]
    motion_dir = tmp_path / "motion-scene-default-start"
    motion_dir.mkdir(parents=True)
    body_pos_w = np.array([[[2.0, 0.0, 1.4]]], dtype=np.float32)
    body_quat_w = np.array([[[1.0, 0.0, 0.0, 0.0]]], dtype=np.float32)
    joint_pos = np.array([[0.25]], dtype=np.float32)
    np.savez_compressed(
        motion_dir / "motion.npz",
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        joint_pos=joint_pos,
    )
    (motion_dir / "meta.json").write_text(
        json.dumps({"body_names": body_names, "joint_names": [joint_name], "fps": 50.0})
    )
    reference = MujocoMotionReference.from_motion_dir(
        motion_dir=motion_dir,
        body_names=body_names,
        joint_names=[joint_name],
        root_body_name=robot.body_names[0],
        future_steps=[0],
    )
    policy_bundle = _write_zero_policy_bundle(tmp_path, joint_name)
    scene.update(0.0)
    default_root_pos = robot.data.root_link_pos_w.clone()

    run_mujoco_policy_rollout(
        scene=scene,
        policy_bundle=policy_bundle,
        reference=reference,
        steps=[0],
        decimation=1,
        initial_state="scene_default",
    )

    ref_root_pos = reference.body_pos_w[0, reference.root_body_index]
    current_root_pos = robot.data.root_link_pos_w
    assert torch.linalg.vector_norm(current_root_pos - ref_root_pos).item() > 1.0
    assert torch.linalg.vector_norm(current_root_pos - default_root_pos).item() < 0.05


def test_policy_rollout_applies_hdmi_action_delay_and_low_pass_during_decimation(tmp_path):
    module = importlib.import_module("active_adaptation.envs.mujoco")
    from active_adaptation.assets_mjcf import ROBOTS

    class SceneCfg:
        robot = ROBOTS.with_object("g1_29dof", object_asset_name="door")

    scene = module.MJScene(SceneCfg(), num_envs=1, launch_viewer=False)
    robot = scene["robot"]
    body_names = [robot.body_names[0]]
    joint_name = robot.joint_names[0]
    motion_dir = tmp_path / "motion-action-adapter"
    _write_motion_dir(motion_dir, body_names, [joint_name])
    reference = MujocoMotionReference.from_motion_dir(
        motion_dir=motion_dir,
        body_names=body_names,
        joint_names=[joint_name],
        root_body_name=robot.body_names[0],
        future_steps=[0],
    )
    policy_bundle = _write_constant_policy_bundle(tmp_path, joint_name, value=1.0)

    metrics = run_mujoco_policy_rollout(
        scene=scene,
        policy_bundle=policy_bundle,
        reference=reference,
        steps=[0],
        decimation=4,
        action_adapter_config=MujocoActionAdapterConfig(delay=2, alpha=0.5),
    )

    # HDMI JointPosition semantics: substeps 0/1 still use delayed zero action,
    # substeps 2/3 low-pass toward action=1.0: 0 -> 0.5 -> 0.75.
    expected_target = torch.tensor(0.75 * 0.5)
    joint_id = robot.joint_names.index(joint_name)
    assert torch.allclose(robot.data.joint_pos_target[0, joint_id], expected_target)
    assert torch.allclose(metrics.joint_position_targets[0, 0, 0], expected_target)


def test_policy_rollout_applies_exported_joint_gains_to_mujoco_robot(tmp_path):
    module = importlib.import_module("active_adaptation.envs.mujoco")
    from active_adaptation.assets_mjcf import ROBOTS

    class SceneCfg:
        robot = ROBOTS.with_object("g1_29dof", object_asset_name="door")

    scene = module.MJScene(SceneCfg(), num_envs=1, launch_viewer=False)
    robot = scene["robot"]
    body_names = [robot.body_names[0]]
    joint_name = robot.joint_names[0]
    joint_id = robot.joint_names.index(joint_name)
    motion_dir = tmp_path / "motion-policy-gains"
    _write_motion_dir(motion_dir, body_names, [joint_name])
    reference = MujocoMotionReference.from_motion_dir(
        motion_dir=motion_dir,
        body_names=body_names,
        joint_names=[joint_name],
        root_body_name=robot.body_names[0],
        future_steps=[0],
    )
    policy_bundle = _write_zero_policy_bundle_with_joint_gains(
        tmp_path,
        joint_name,
        stiffness=12.0,
        damping=0.75,
    )

    run_mujoco_policy_rollout(
        scene=scene,
        policy_bundle=policy_bundle,
        reference=reference,
        steps=[0],
        decimation=1,
    )

    assert torch.allclose(robot.data.joint_stiffness[0, joint_id], torch.tensor(12.0))
    assert torch.allclose(robot.data.joint_damping[0, joint_id], torch.tensor(0.75))


def test_policy_rollout_offsets_joint_pos_history_by_exported_default_joint_pos(tmp_path):
    module = importlib.import_module("active_adaptation.envs.mujoco")
    from active_adaptation.assets_mjcf import ROBOTS

    class SceneCfg:
        robot = ROBOTS.with_object("g1_29dof", object_asset_name="door")

    scene = module.MJScene(SceneCfg(), num_envs=1, launch_viewer=False)
    robot = scene["robot"]
    body_names = [robot.body_names[0]]
    joint_name = robot.joint_names[0]
    motion_dir = tmp_path / "motion-joint-offset"
    _write_motion_dir(motion_dir, body_names, [joint_name])
    reference = MujocoMotionReference.from_motion_dir(
        motion_dir=motion_dir,
        body_names=body_names,
        joint_names=[joint_name],
        root_body_name=robot.body_names[0],
        future_steps=[0],
    )
    policy_bundle = _write_joint_pos_policy_bundle(
        tmp_path,
        joint_name,
        default_joint_pos=0.5,
    )

    metrics = run_mujoco_policy_rollout(
        scene=scene,
        policy_bundle=policy_bundle,
        reference=reference,
        steps=[0],
        decimation=1,
    )

    expected_observation = reference.joint_pos[0, 0] - 0.5
    assert torch.allclose(metrics.actions[0, 0, 0], expected_observation, atol=1e-5)


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


def test_policy_rollout_computes_object_joint_reward_from_full_motion_reference(tmp_path):
    module = importlib.import_module("active_adaptation.envs.mujoco")
    from active_adaptation.assets_mjcf import ROBOTS

    class SceneCfg:
        robot = ROBOTS.with_object("g1_29dof", object_asset_name="door")

    scene = module.MJScene(SceneCfg(), num_envs=1, launch_viewer=False)
    robot = scene["robot"]
    door = scene["door"]
    body_names = [robot.body_names[0], door.body_names[0]]
    robot_joint_name = robot.joint_names[0]
    object_joint_name = door.joint_names[0]
    motion_dir = tmp_path / "motion-rollout-object-joint-reward"
    _write_motion_dir(motion_dir, body_names, [robot_joint_name, object_joint_name])
    reference = MujocoMotionReference.from_motion_dir(
        motion_dir=motion_dir,
        body_names=body_names,
        joint_names=[robot_joint_name],
        root_body_name=robot.body_names[0],
        future_steps=[0],
    )
    policy_bundle = _write_zero_policy_bundle(tmp_path, robot_joint_name)
    reward_cfg = {
        "object_tracking": {
            "object_joint_pos_tracking": {
                "weight": 1.0,
                "sigma": 0.25,
            }
        }
    }

    metrics = run_mujoco_policy_rollout(
        scene=scene,
        policy_bundle=policy_bundle,
        reference=reference,
        reward_cfg=reward_cfg,
        object_name="door",
        object_body_name=door.body_names[0],
        object_joint_name=object_joint_name,
        steps=[0, 1],
        decimation=1,
    )

    assert metrics.reward is not None
    assert metrics.reward.shape == (2, 1, 1)
    assert torch.isfinite(metrics.reward).all()


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


def test_policy_rollout_fills_hdmi_future_priv_observations_from_reference_and_scene(tmp_path):
    module = importlib.import_module("active_adaptation.envs.mujoco")
    from active_adaptation.assets_mjcf import ROBOTS

    class SceneCfg:
        robot = ROBOTS.with_object("g1_29dof", object_asset_name="door")

    scene = module.MJScene(SceneCfg(), num_envs=1, launch_viewer=False)
    robot = scene["robot"]
    root_body_name = robot.body_names[0]
    joint_name = robot.joint_names[0]
    motion_dir = tmp_path / "motion-future-priv"
    _write_motion_dir(motion_dir, [root_body_name], [joint_name])
    reference = MujocoMotionReference.from_motion_dir(
        motion_dir=motion_dir,
        body_names=[root_body_name],
        joint_names=[joint_name],
        root_body_name=root_body_name,
        future_steps=[0, 1],
    )
    policy_bundle = _write_future_priv_policy_bundle(tmp_path, joint_name, root_body_name)

    metrics = run_mujoco_policy_rollout(
        scene=scene,
        policy_bundle=policy_bundle,
        reference=reference,
        steps=[0],
        decimation=1,
    )

    assert metrics.actions.shape == (1, 1, 1)
    assert torch.isfinite(metrics.actions).all()


def test_policy_rollout_fills_object_future_contact_and_joint_observations(tmp_path):
    module = importlib.import_module("active_adaptation.envs.mujoco")
    from active_adaptation.assets_mjcf import ROBOTS

    class SceneCfg:
        robot = ROBOTS.with_object("g1_29dof", object_asset_name="door")

    scene = module.MJScene(SceneCfg(), num_envs=1, launch_viewer=False)
    robot = scene["robot"]
    door = scene["door"]
    root_body_name = robot.body_names[0]
    object_body_name = door.body_names[0]
    robot_joint_name = robot.joint_names[0]
    object_joint_name = door.joint_names[0]
    motion_dir = tmp_path / "motion-object-future-priv"
    _write_motion_dir(motion_dir, [root_body_name, object_body_name], [robot_joint_name, object_joint_name])
    reference = MujocoMotionReference.from_motion_dir(
        motion_dir=motion_dir,
        body_names=[root_body_name],
        joint_names=[robot_joint_name],
        root_body_name=root_body_name,
        future_steps=[0, 1],
    )
    policy_bundle = _write_object_future_priv_policy_bundle(
        tmp_path,
        robot_joint_name,
        object_body_name,
        object_joint_name,
    )

    metrics = run_mujoco_policy_rollout(
        scene=scene,
        policy_bundle=policy_bundle,
        reference=reference,
        object_name="door",
        object_body_name=object_body_name,
        object_joint_name=object_joint_name,
        contact_eef_body_names=[root_body_name],
        contact_target_pos_offset=[[0.0, 0.0, 0.0]],
        contact_eef_pos_offset=[[0.0, 0.0, 0.0]],
        steps=[0],
        decimation=1,
    )

    assert metrics.actions.shape == (1, 1, 1)
    assert torch.isfinite(metrics.actions).all()
