import importlib
from pathlib import Path

import numpy as np
import active_adaptation as aa
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch


def _mujoco_env_module():
    return importlib.import_module("active_adaptation.envs.mujoco")


def test_mujoco_articulation_uses_independent_models_per_env_for_model_randomization():
    module = _mujoco_env_module()
    from active_adaptation.assets_mjcf import ROBOTS

    robot = module.MJArticulation(ROBOTS.with_object("g1_29dof", object_asset_name="box"), num_envs=2)

    assert len(robot.mj_models) == 2
    assert robot.mj_models[0] is not robot.mj_models[1]
    original_env0_mass = float(robot.mj_models[0].body_mass[robot.body_adrs_read[0]])
    robot.mj_models[1].body_mass[robot.body_adrs_read[0]] = original_env0_mass + 10.0

    assert float(robot.mj_models[0].body_mass[robot.body_adrs_read[0]]) == original_env0_mass
    assert float(robot.mj_models[1].body_mass[robot.body_adrs_read[0]]) == original_env0_mass + 10.0


def test_mujoco_articulation_applies_motor_parameter_randomization_per_env():
    module = _mujoco_env_module()
    from active_adaptation.assets_mjcf import ROBOTS

    robot = module.MJArticulation(ROBOTS["g1_29dof"], num_envs=2)
    joint_ids = torch.tensor([0, 1])
    stiffness = torch.tensor([[11.0, 12.0], [21.0, 22.0]])
    damping = torch.tensor([[1.1, 1.2], [2.1, 2.2]])
    armature = torch.tensor([[0.01, 0.02], [0.03, 0.04]])
    friction = torch.tensor([[0.5, 0.6], [0.7, 0.8]])

    robot.write_joint_stiffness_to_sim(stiffness, joint_ids=joint_ids, env_ids=torch.arange(2))
    robot.write_joint_damping_to_sim(damping, joint_ids=joint_ids, env_ids=torch.arange(2))
    robot.write_joint_armature_to_sim(armature, joint_ids=joint_ids, env_ids=torch.arange(2))
    robot.write_joint_friction_coefficient_to_sim(friction, joint_ids=joint_ids, env_ids=torch.arange(2))

    dof_adrs = robot.joint_qveladr_read[joint_ids.numpy()]
    assert torch.allclose(robot.data.joint_stiffness[:, joint_ids], stiffness)
    assert torch.allclose(robot.data.joint_damping[:, joint_ids], damping)
    assert torch.allclose(
        torch.as_tensor(
            np.stack([robot.mj_models[0].dof_armature[dof_adrs], robot.mj_models[1].dof_armature[dof_adrs]])
        ),
        armature.double(),
    )
    assert torch.allclose(
        torch.as_tensor(
            np.stack(
                [robot.mj_models[0].dof_frictionloss[dof_adrs], robot.mj_models[1].dof_frictionloss[dof_adrs]]
            )
        ),
        friction.double(),
    )


def test_mujoco_root_physx_view_sets_dof_friction_coefficients_per_env():
    module = _mujoco_env_module()
    from active_adaptation.assets_mjcf import ROBOTS

    robot = module.MJArticulation(ROBOTS["g1_29dof"], num_envs=2)
    frictions = torch.zeros(2, robot.num_joints)
    frictions[0, :2] = torch.tensor([0.11, 0.12])
    frictions[1, :2] = torch.tensor([0.21, 0.22])

    robot.root_physx_view.set_dof_friction_coefficients(frictions, indices=torch.arange(2))

    dof_adrs = robot.joint_qveladr_read[:2]
    assert np.allclose(robot.mj_models[0].dof_frictionloss[dof_adrs], frictions[0, :2].numpy())
    assert np.allclose(robot.mj_models[1].dof_frictionloss[dof_adrs], frictions[1, :2].numpy())


def test_mujoco_object_root_physx_view_applies_body_randomization_per_env():
    module = _mujoco_env_module()
    from active_adaptation.assets_mjcf import ROBOTS

    class SceneCfg:
        robot = ROBOTS.with_object("g1_29dof", object_asset_name="box")

    scene = module.MJScene(SceneCfg(), num_envs=2, launch_viewer=False)
    box = scene.rigid_objects["box"]
    view = box.root_physx_view

    assert view.max_shapes >= 1
    default_masses = view.get_masses()
    default_inertias = view.get_inertias()
    default_materials = view.get_material_properties()
    assert default_masses.shape == (2, box.num_bodies)
    assert default_inertias.shape == (2, box.num_bodies, 3)
    assert default_materials.shape == (2, view.max_shapes, 3)

    masses = default_masses.clone()
    masses[0, 0] = 3.0
    masses[1, 0] = 7.0
    view.set_masses(masses, torch.arange(2))

    inertias = default_inertias.clone()
    inertias[0, 0] = torch.tensor([0.1, 0.2, 0.3])
    inertias[1, 0] = torch.tensor([0.4, 0.5, 0.6])
    view.set_inertias(inertias, torch.arange(2))

    materials = default_materials.clone()
    materials[0, :, :] = torch.tensor([1.2, 0.8, 0.1])
    materials[1, :, :] = torch.tensor([1.5, 0.4, 0.2])
    view.set_material_properties(materials.flatten(), torch.arange(2))

    assert torch.allclose(view.get_masses(), masses)
    assert torch.allclose(view.get_inertias(), inertias)
    assert torch.allclose(view.get_material_properties(), materials)
    assert box.mj_models[0].body_mass[box.body_adrs_read[0]] == 3.0
    assert box.mj_models[1].body_mass[box.body_adrs_read[0]] == 7.0
    assert torch.isclose(torch.tensor(box.mj_models[0].geom_friction[view.geom_adrs[0], 0]), torch.tensor(0.8, dtype=torch.float64))
    assert torch.isclose(torch.tensor(box.mj_models[1].geom_friction[view.geom_adrs[0], 0]), torch.tensor(0.4, dtype=torch.float64))


def test_mujoco_object_view_applies_per_env_body_scale_to_object_geometry():
    module = _mujoco_env_module()
    from active_adaptation.assets_mjcf import ROBOTS

    class SceneCfg:
        robot = ROBOTS.with_object("g1_29dof", object_asset_name="box")

    scene = module.MJScene(SceneCfg(), num_envs=2, launch_viewer=False)
    box = scene.rigid_objects["box"]

    default_geom_size = [
        torch.as_tensor(model.geom_size[box.geom_adrs].copy(), dtype=torch.float32)
        for model in box.mj_models
    ]
    default_geom_pos = [
        torch.as_tensor(model.geom_pos[box.geom_adrs].copy(), dtype=torch.float32)
        for model in box.mj_models
    ]
    default_body_pos = [
        torch.as_tensor(model.body_pos[box.body_adrs_read].copy(), dtype=torch.float32)
        for model in box.mj_models
    ]
    scale = torch.tensor([[1.0, 1.0, 1.0], [2.0, 0.5, 1.5]])

    box.apply_body_scale(scale)

    assert torch.allclose(
        torch.as_tensor(box.mj_models[0].geom_size[box.geom_adrs], dtype=torch.float32),
        default_geom_size[0],
    )
    assert torch.allclose(
        torch.as_tensor(box.mj_models[1].geom_size[box.geom_adrs], dtype=torch.float32),
        default_geom_size[1] * scale[1],
    )
    assert torch.allclose(
        torch.as_tensor(box.mj_models[1].geom_pos[box.geom_adrs], dtype=torch.float32),
        default_geom_pos[1] * scale[1],
    )
    assert torch.allclose(
        torch.as_tensor(box.mj_models[1].body_pos[box.body_adrs_read], dtype=torch.float32),
        default_body_pos[1],
    )
    assert torch.allclose(box.cfg.spawn.scale, scale)


def test_mujoco_simple_env_applies_body_scale_randomization_from_task_cfg():
    root = Path(__file__).resolve().parents[1]
    aa.set_backend("mujoco")
    env = None
    try:
        with initialize_config_dir(config_dir=str((root / "cfg").resolve()), version_base=None):
            cfg = compose(
                config_name="train",
                overrides=[
                    "backend=mujoco",
                    "task=G1/hdmi/push_box",
                    "task.num_envs=2",
                    "task.max_episode_length=4",
                    "task.viewer.env_spacing=0",
                    "task.randomization.body_scale.scale_range=[1.25,1.25]",
                    "~task.observation.depth",
                ],
            )
        OmegaConf.resolve(cfg)
        OmegaConf.set_struct(cfg, False)
        from active_adaptation.envs import SimpleEnv

        env = SimpleEnv(cfg.task)
        box = env.scene["box"]
        scale = box.cfg.spawn.scale
        geom_id = int(box.geom_adrs[0])
        assert torch.allclose(scale, torch.tensor([[1.0, 1.0, 1.0], [1.25, 1.25, 1.25]]))
        assert torch.allclose(
            torch.as_tensor(box.mj_models[1].geom_size[geom_id], dtype=torch.float32),
            torch.as_tensor(box.mj_models[0].geom_size[geom_id], dtype=torch.float32) * 1.25,
        )
    finally:
        if env is not None:
            env.close()
        aa.set_backend("isaac")


def test_mujoco_simple_env_applies_object_randomization_from_task_cfg():
    root = Path(__file__).resolve().parents[1]
    aa.set_backend("mujoco")
    env = None
    try:
        with initialize_config_dir(config_dir=str((root / "cfg").resolve()), version_base=None):
            cfg = compose(
                config_name="train",
                overrides=[
                    "backend=mujoco",
                    "task=G1/hdmi/open_door-feet",
                    "task.num_envs=2",
                    "task.max_episode_length=4",
                    "task.viewer.env_spacing=0",
                    "task.randomization.body_scale.scale_range=[1.0,1.0]",
                    "task.randomization.object_body_randomization.mass_range=[9.0,9.0]",
                    "task.randomization.object_body_randomization.dynamic_friction_range=[0.6,0.6]",
                    "task.randomization.object_body_randomization.static_dynamic_friction_ratio_range=[2.0,2.0]",
                    "task.randomization.object_body_randomization.restitution_range=[0.1,0.1]",
                    "task.randomization.object_joint_randomization.armature_range=[0.03,0.03]",
                    "task.randomization.object_joint_randomization.friction_range=[0.7,0.7]",
                    "task.randomization.object_joint_randomization.damping_range=[2.5,2.5]",
                    "~task.observation.depth",
                ],
            )
        OmegaConf.resolve(cfg)
        OmegaConf.set_struct(cfg, False)
        from active_adaptation.envs import SimpleEnv

        env = SimpleEnv(cfg.task)
        door = env.scene["door"]
        materials = door.root_physx_view.get_material_properties()
        masses = door.root_physx_view.get_masses()
        dof_addr = int(door.joint_qveladr_read[0])
        armatures = torch.tensor(
            [door.mj_models[env_id].dof_armature[dof_addr] for env_id in range(door.num_instances)]
        )

        env.reset()

        assert torch.allclose(masses, torch.full_like(masses, 9.0))
        assert torch.allclose(materials[..., 0], torch.full_like(materials[..., 0], 1.2))
        assert torch.allclose(materials[..., 1], torch.full_like(materials[..., 1], 0.6))
        assert torch.allclose(materials[..., 2], torch.full_like(materials[..., 2], 0.1))
        assert torch.allclose(armatures, torch.full_like(armatures, 0.03))
        assert torch.allclose(door._custom_friction, torch.full_like(door._custom_friction, 0.7))
        assert torch.allclose(door._custom_damping, torch.full_like(door._custom_damping, 2.5))
    finally:
        if env is not None:
            env.close()
        aa.set_backend("isaac")


def test_mujoco_articulated_object_applies_joint_armature_and_custom_terms_per_env():
    module = _mujoco_env_module()
    from active_adaptation.assets_mjcf import ROBOTS

    class SceneCfg:
        robot = ROBOTS.with_object("g1_29dof", object_asset_name="door")

    scene = module.MJScene(SceneCfg(), num_envs=2, launch_viewer=False)
    door = scene.articulations["door"]

    door.write_joint_armature_to_sim(torch.tensor([[0.01], [0.03]]), joint_ids=[0])
    dof_addr = int(door.joint_qveladr_read[0])

    assert torch.isclose(torch.tensor(door.mj_models[0].dof_armature[dof_addr]), torch.tensor(0.01, dtype=torch.float64))
    assert torch.isclose(torch.tensor(door.mj_models[1].dof_armature[dof_addr]), torch.tensor(0.03, dtype=torch.float64))
    assert door._custom_friction.shape == (2,)
    assert door._custom_damping.shape == (2,)


def test_mujoco_articulated_object_applies_custom_friction_and_damping_torque():
    module = _mujoco_env_module()
    from active_adaptation.assets_mjcf import ROBOTS

    class SceneCfg:
        robot = ROBOTS.with_object("g1_29dof", object_asset_name="door")

    scene = module.MJScene(SceneCfg(), num_envs=2, launch_viewer=False)
    door = scene.articulations["door"]

    door.write_joint_state_to_sim(
        torch.zeros(2, 1),
        torch.tensor([[0.2], [-0.3]]),
        joint_ids=[0],
    )
    door._custom_friction[:] = torch.tensor([0.5, 0.7])
    door._custom_damping[:] = torch.tensor([2.0, 3.0])

    scene.write_data_to_sim()
    dof_addr = int(door.joint_qveladr_read[0])

    applied = torch.tensor([
        door.mj_datas[0].qfrc_applied[dof_addr],
        door.mj_datas[1].qfrc_applied[dof_addr],
    ])
    expected = torch.tensor([
        -0.5 - 0.2 * 2.0,
        0.7 + 0.3 * 3.0,
    ], dtype=applied.dtype)
    assert torch.allclose(applied, expected)
