import importlib

import torch


def _mujoco_env_module():
    return importlib.import_module("active_adaptation.envs.mujoco")


def test_robot_registry_resolves_object_scene_without_polluting_robot_policy_names():
    from active_adaptation.assets_mjcf import ROBOTS

    base_cfg = ROBOTS["g1_29dof"]
    door_cfg = ROBOTS.with_object("g1_29dof", object_asset_name="door")

    assert door_cfg.mjcf_path.endswith("g1_29dof_nohand-door.xml")
    assert list(door_cfg.body_names_isaac) == list(base_cfg.body_names_isaac)
    assert list(door_cfg.joint_names_isaac) == list(base_cfg.joint_names_isaac)
    assert "door_panel" not in door_cfg.body_names_isaac
    assert "door_joint" not in door_cfg.joint_names_isaac

    door_spec = door_cfg.object_specs["door"]
    assert tuple(door_spec.body_names) == ("door", "door_panel")
    assert tuple(door_spec.joint_names) == ("door_joint",)


def test_mj_scene_exposes_rigid_object_view_from_object_scene():
    module = _mujoco_env_module()
    from active_adaptation.assets_mjcf import ROBOTS

    class SceneCfg:
        robot = ROBOTS.with_object("g1_29dof", object_asset_name="box")
        contact_forces = "robot"

    scene = module.MJScene(SceneCfg(), num_envs=2, launch_viewer=False)

    assert "box" in scene
    assert "box" in scene.rigid_objects
    assert "box" not in scene.articulations
    assert "box" not in scene["robot"].body_names
    assert scene["box"].body_names == ["box"]
    assert scene["box"].joint_names == []
    assert scene["box"].data.root_link_pos_w.shape == (2, 3)
    assert scene["box"].data.body_link_pos_w.shape == (2, 1, 3)


def test_mj_scene_exposes_articulated_object_view_and_filtered_contact_sensor():
    module = _mujoco_env_module()
    from active_adaptation.assets_mjcf import ROBOTS

    class SceneCfg:
        robot = ROBOTS.with_object("g1_29dof", object_asset_name="door")
        contact_forces = "robot"
        right_wrist_yaw_link_door_contact_forces = module.MJContactSensorCfg(
            target="robot",
            body_names=["right_wrist_yaw_link"],
            filter_body_names=["door_panel"],
        )

    scene = module.MJScene(SceneCfg(), num_envs=2, launch_viewer=False)
    door = scene.articulations["door"]
    sensor = scene.sensors["right_wrist_yaw_link_door_contact_forces"]

    assert "door" in scene
    assert door.body_names == ["door", "door_panel"]
    assert door.joint_names == ["door_joint"]
    assert door.data.root_link_pos_w.shape == (2, 3)
    assert door.data.body_link_quat_w.shape == (2, 2, 4)
    assert door.data.joint_pos.shape == (2, 1)
    assert sensor.body_names == ["right_wrist_yaw_link"]
    assert sensor.data.force_matrix_w.shape == (2, 1, 1, 3)

    joint_pos = torch.tensor([[-0.5], [-0.2]])
    joint_vel = torch.tensor([[0.1], [0.3]])
    door.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=[0])

    assert torch.allclose(door.data.joint_pos, joint_pos)
    assert torch.allclose(door.data.joint_vel, joint_vel)


def test_robot_root_pose_write_uses_robot_free_joint_when_object_precedes_robot():
    module = _mujoco_env_module()
    from active_adaptation.assets_mjcf import ROBOTS

    class SceneCfg:
        robot = ROBOTS.with_object("g1_29dof", object_asset_name="box")

    scene = module.MJScene(SceneCfg(), num_envs=2, launch_viewer=False)
    robot = scene["robot"]
    box = scene["box"]
    box_root_before = box.data.root_link_pos_w.clone()
    root_pose = torch.tensor(
        [
            [1.25, -0.75, 0.91, 1.0, 0.0, 0.0, 0.0],
            [-0.50, 0.80, 0.87, 1.0, 0.0, 0.0, 0.0],
        ]
    )

    robot.write_root_link_pose_to_sim(root_pose)
    scene.update(0.0)

    assert torch.allclose(robot.data.root_link_pos_w, root_pose[:, :3])
    assert torch.allclose(box.data.root_link_pos_w, box_root_before)
