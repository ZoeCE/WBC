import importlib
from pathlib import Path

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
