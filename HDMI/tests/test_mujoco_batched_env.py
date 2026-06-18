import importlib

import numpy as np
import torch


def _mujoco_env_module():
    return importlib.import_module("active_adaptation.envs.mujoco")


def _robot_cfg():
    from active_adaptation.assets_mjcf import ROBOTS

    return ROBOTS["g1_29dof"]


def test_mujoco_env_module_imports_without_eager_omni_runtime():
    module = _mujoco_env_module()

    assert module.MJArticulationCfg.__name__ == "MJArticulationCfg"
    assert module.MJArticulation.__name__ == "MJArticulation"


def test_mj_articulation_keeps_independent_batched_state_buffers():
    module = _mujoco_env_module()
    robot = module.MJArticulation(_robot_cfg(), num_envs=2)

    assert robot.num_instances == 2
    assert len(robot.mj_datas) == 2
    assert robot.data.default_joint_pos.shape == (2, robot.num_joints)
    assert robot.data.default_joint_vel.shape == (2, robot.num_joints)
    assert robot.data.applied_torque.shape == (2, robot.num_joints)
    assert robot._external_force_b.shape == (2, robot.num_bodies, 3)

    target = robot.data.default_joint_pos.clone()
    target[:, 0] = torch.tensor([0.15, -0.25])
    robot.set_joint_position_target(target)
    robot.write_data_to_sim()

    expected_torque = robot.data.joint_stiffness[:, 0] * (target[:, 0] - robot.data.joint_pos[:, 0])
    assert torch.allclose(robot.data.joint_pos_target[:, 0], torch.tensor([0.15, -0.25]))
    assert torch.allclose(robot.data.applied_torque[:, 0], expected_torque)
    assert not torch.allclose(robot.data.applied_torque[0], robot.data.applied_torque[1])


def test_mj_articulation_writes_joint_and_root_state_only_to_selected_envs():
    module = _mujoco_env_module()
    robot = module.MJArticulation(_robot_cfg(), num_envs=2)
    env_ids = torch.tensor([1])

    root_state = robot.data.default_root_state[:1].clone()
    root_state[0, :3] = torch.tensor([0.5, -0.25, 1.2])
    robot.write_root_state_to_sim(root_state, env_ids=env_ids)

    assert np.allclose(robot.mj_datas[1].qpos[:3], root_state[0, :3].numpy())
    assert not np.allclose(robot.mj_datas[0].qpos[:3], root_state[0, :3].numpy())

    joint_ids = [0, 1]
    joint_pos = torch.tensor([[0.2, -0.3]])
    joint_vel = torch.tensor([[0.4, -0.5]])
    robot.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=joint_ids, env_ids=env_ids)

    assert np.allclose(robot.mj_datas[1].qpos[robot.joint_qposadr_read[joint_ids]], joint_pos[0].numpy())
    assert np.allclose(robot.mj_datas[1].qvel[robot.joint_qveladr_read[joint_ids]], joint_vel[0].numpy())
    assert not np.allclose(robot.mj_datas[0].qpos[robot.joint_qposadr_read[joint_ids]], joint_pos[0].numpy())


def test_mj_scene_uses_requested_num_envs_without_viewer():
    module = _mujoco_env_module()

    class SceneCfg:
        robot = _robot_cfg()
        contact_forces = "robot"

    scene = module.MJScene(SceneCfg(), num_envs=2, launch_viewer=False)
    sim = module.MJSim(scene, realtime=False)
    scene.update(0.0)

    assert scene.num_envs == 2
    assert scene.env_origins.shape == (2, 3)
    assert scene.viewer is None
    assert sim.has_gui() is False
    assert scene["robot"].num_instances == 2
    assert scene["contact_forces"].data.net_forces_w.shape == (2, scene["robot"].num_bodies, 3)
