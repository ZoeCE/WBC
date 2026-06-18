import pytest
import torch
from types import SimpleNamespace

from active_adaptation.mujoco.motion_reference import MujocoMotionReference
from active_adaptation.mujoco import playback_parity
from active_adaptation.mujoco.playback_parity import (
    compute_playback_parity,
)


def _mujoco_env_module():
    import importlib

    return importlib.import_module("active_adaptation.envs.mujoco")


def test_playback_parity_computes_joint_and_body_l2_per_env():
    metrics = compute_playback_parity(
        q_mujoco=torch.tensor([[1.0, 2.0, 4.0]]),
        q_ref=torch.tensor([[1.0, 0.0, 0.0]]),
        body_pos_mujoco_w=torch.tensor([[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]]]),
        body_pos_ref_w=torch.tensor([[[1.0, 1.0, 0.0], [1.0, 3.0, 3.0]]]),
        reward=torch.tensor([0.75]),
    )

    assert torch.allclose(metrics.q_l2, torch.tensor([20.0**0.5]))
    assert torch.allclose(metrics.body_pos_l2, torch.tensor([5.0**0.5]))
    assert torch.equal(metrics.reward, torch.tensor([0.75]))


def test_playback_parity_preserves_time_and_env_axes_for_motion_rollouts():
    q_mujoco = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )
    q_ref = torch.zeros_like(q_mujoco)
    body_pos_mujoco_w = torch.tensor(
        [
            [
                [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
                [[0.0, 0.0, 3.0], [4.0, 0.0, 0.0]],
            ],
            [
                [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 2.0, 2.0], [0.0, 0.0, 1.0]],
            ],
        ]
    )
    reward = torch.tensor([[0.1, 0.2], [0.3, 0.4]])

    metrics = compute_playback_parity(
        q_mujoco,
        q_ref,
        body_pos_mujoco_w,
        torch.zeros_like(body_pos_mujoco_w),
        reward,
    )

    assert torch.allclose(metrics.q_l2, torch.linalg.vector_norm(q_mujoco, dim=-1))
    assert torch.allclose(
        metrics.body_pos_l2,
        torch.tensor([[5.0**0.5, 25.0**0.5], [2.0**0.5, 9.0**0.5]]),
    )
    assert torch.equal(metrics.reward, reward)


def test_playback_parity_accepts_motion_reference_future_slice():
    ref_joint_pos_future = torch.tensor([[[0.2, 0.4], [9.0, 9.0]]])
    ref_body_pos_future_w = torch.tensor([[[[0.0, 0.0, 1.0]], [[9.0, 9.0, 9.0]]]])

    metrics = compute_playback_parity(
        q_mujoco=torch.tensor([[0.1, 0.1]]),
        q_ref=ref_joint_pos_future[:, 0],
        body_pos_mujoco_w=torch.tensor([[[0.0, 0.0, 0.0]]]),
        body_pos_ref_w=ref_body_pos_future_w[:, 0],
    )

    assert torch.allclose(metrics.q_l2, torch.tensor([(0.1**2 + 0.3**2) ** 0.5]))
    assert torch.allclose(metrics.body_pos_l2, torch.tensor([1.0]))
    assert metrics.reward is None


def test_playback_parity_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="q_mujoco shape.*q_ref shape"):
        compute_playback_parity(
            q_mujoco=torch.zeros(1, 2),
            q_ref=torch.zeros(1, 3),
            body_pos_mujoco_w=torch.zeros(1, 1, 3),
            body_pos_ref_w=torch.zeros(1, 1, 3),
        )


def test_reward_from_spec_matches_hdmi_group_sum_and_aliases():
    state = playback_parity.MujocoRewardState(
        actual_body_pos_w=torch.tensor([[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]]),
        ref_body_pos_w=torch.tensor([[[0.3, 0.4, 0.0]], [[1.0, 0.0, 0.0]]]),
        joint_pos=torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
        ref_joint_pos=torch.tensor([[0.2, 1.0], [1.5, 3.5]]),
        object_pos_w=torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
        ref_object_pos_w=torch.tensor([[0.5, 0.0, 0.0], [1.0, 1.5, 0.0]]),
    )
    reward_cfg = {
        "tracking": {
            "tracking_root_pos(keypoint_pos_tracking_product)": {"weight": 0.5, "sigma": 0.5},
            "joint_pos_tracking_product": {"weight": 2.0, "sigma": 0.25},
            "disabled_object": {"enabled": False, "weight": 100.0},
        },
        "object_tracking": {
            "object_pos_tracking": {"weight": 3.0, "sigma": 0.5},
        },
    }

    reward = playback_parity.compute_reward_from_spec(reward_cfg, state)

    keypoint = torch.exp(
        -((state.ref_body_pos_w - state.actual_body_pos_w).norm(dim=-1).mean(dim=1)) / 0.5
    ).unsqueeze(1)
    joint = torch.exp(-((state.ref_joint_pos - state.joint_pos).abs().mean(dim=1)) / 0.25).unsqueeze(1)
    object_pos = torch.exp(-((state.ref_object_pos_w - state.object_pos_w).norm(dim=-1)) / 0.5).unsqueeze(1)
    expected = torch.cat([0.5 * keypoint + 2.0 * joint, 3.0 * object_pos], dim=1)
    assert torch.allclose(reward, expected)

    metrics = compute_playback_parity(
        q_mujoco=state.joint_pos,
        q_ref=state.ref_joint_pos,
        body_pos_mujoco_w=state.actual_body_pos_w,
        body_pos_ref_w=state.ref_body_pos_w,
        reward=reward,
    )
    assert torch.allclose(metrics.reward, expected)


def test_reward_from_spec_supports_velocity_and_orientation_tracking_terms():
    state = playback_parity.MujocoRewardState(
        actual_body_quat_w=torch.tensor(
            [
                [[1.0, 0.0, 0.0, 0.0]],
                [[2**-0.5, 0.0, 0.0, 2**-0.5]],
            ]
        ),
        ref_body_quat_w=torch.tensor(
            [
                [[2**-0.5, 0.0, 0.0, 2**-0.5]],
                [[2**-0.5, 0.0, 0.0, 2**-0.5]],
            ]
        ),
        actual_body_lin_vel_w=torch.tensor([[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]]),
        ref_body_lin_vel_w=torch.tensor([[[0.3, 0.4, 0.0]], [[1.0, 0.0, 0.0]]]),
        actual_body_ang_vel_w=torch.tensor([[[0.0, 0.0, 0.0]], [[0.0, 2.0, 0.0]]]),
        ref_body_ang_vel_w=torch.tensor([[[0.0, 0.0, 0.2]], [[0.0, 1.5, 0.0]]]),
        joint_vel=torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
        ref_joint_vel=torch.tensor([[0.5, 1.0], [1.0, 3.2]]),
    )
    reward_cfg = {
        "tracking": {
            "body_ori(keypoint_ori_tracking_product)": {"weight": 0.5, "sigma": 0.25},
            "body_lin_vel(keypoint_lin_vel_tracking_product)": {"weight": 2.0, "sigma": 0.5},
            "body_ang_vel(keypoint_ang_vel_tracking_product)": {"weight": 3.0, "sigma": 0.5},
            "joint_vel_tracking_product": {"weight": 4.0, "sigma": 0.25},
        }
    }

    reward = playback_parity.compute_reward_from_spec(reward_cfg, state)

    ori_error = torch.tensor([torch.pi / 2, 0.0])
    ori = torch.exp(-ori_error / 0.25).unsqueeze(1)
    lin = torch.exp(-torch.tensor([0.5, 0.0]) / 0.5).unsqueeze(1)
    ang = torch.exp(-torch.tensor([0.2, 0.5]) / 0.5).unsqueeze(1)
    joint = torch.exp(-torch.tensor([0.25, 0.6]) / 0.25).unsqueeze(1)
    expected = 0.5 * ori + 2.0 * lin + 3.0 * ang + 4.0 * joint
    assert torch.allclose(reward, expected, atol=1e-6)


def test_reward_from_spec_supports_loco_survival_joint_velocity_and_position_limits():
    joint_pos = torch.tensor([[-0.9, 0.0, 1.8], [0.6, 1.6, 0.2]])
    joint_vel = torch.tensor([[1.0, -3.0, 10.0], [0.5, -0.5, 0.0]])
    joint_pos_limits = torch.tensor(
        [
            [[-1.0, 1.0], [-2.0, 2.0], [0.0, 2.0]],
            [[-1.0, 1.0], [-2.0, 2.0], [0.0, 2.0]],
        ]
    )
    state = playback_parity.MujocoRewardState(
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        joint_pos_limits=joint_pos_limits,
    )
    reward_cfg = {
        "loco": {
            "survival": {"weight": 2.0},
            "joint_vel_l2": {"weight": 0.1},
            "joint_pos_limits": {"weight": 3.0, "soft_factor": 0.5},
        }
    }

    reward = playback_parity.compute_reward_from_spec(reward_cfg, state)

    survival = torch.ones(2, 1)
    joint_vel_penalty = -joint_vel.square().clamp_max(5.0).sum(dim=1, keepdim=True)
    jpos_mean = joint_pos_limits.mean(dim=-1)
    jpos_range = joint_pos_limits[..., 1] - joint_pos_limits[..., 0]
    soft_lower = jpos_mean - 0.5 * jpos_range * 0.5
    soft_upper = jpos_mean + 0.5 * jpos_range * 0.5
    joint_limit_penalty = -(
        (soft_lower - joint_pos).clamp_min(0.0)
        + (joint_pos - soft_upper).clamp_min(0.0)
    ).sum(dim=1, keepdim=True)
    expected = 2.0 * survival + 0.1 * joint_vel_penalty + 3.0 * joint_limit_penalty
    assert torch.allclose(reward, expected)


def test_reward_from_spec_supports_feet_rewards_with_body_name_mapping():
    state = playback_parity.MujocoRewardState(
        body_names=["left_foot", "right_foot", "torso"],
        contact_body_names=["right_foot", "left_foot"],
        actual_body_lin_vel_w=torch.tensor(
            [
                [[0.5, 0.0, 0.0], [0.0, 0.3, 0.0], [10.0, 0.0, 0.0]],
                [[1.5, 0.0, 0.0], [0.2, 0.2, 0.0], [10.0, 0.0, 0.0]],
            ]
        ),
        contact_current_contact_time=torch.tensor([[0.01, 0.03], [0.04, 0.10]]),
        contact_last_air_time=torch.tensor([[0.50, 0.20], [0.40, 0.10]]),
        contact_first_contact=torch.tensor([[False, True], [True, False]]),
        contact_net_forces_w_history=torch.tensor(
            [
                [
                    [[0.0, 0.0, 0.0], [30.0, 0.0, 0.0]],
                    [[0.0, 20.0, 0.0], [10.0, 0.0, 0.0]],
                ],
                [
                    [[50.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[10.0, 0.0, 0.0], [0.0, 40.0, 0.0]],
                ],
            ]
        ),
        default_mass_total=torch.tensor([20.0, 10.0]),
    )
    reward_cfg = {
        "feet": {
            "feet_slip": {"weight": 2.0, "body_names": ["left_foot", "right_foot"], "tolerance": 0.1},
            "impact_force_l2": {"weight": 0.5, "body_names": ["left_foot", "right_foot"]},
            "feet_air_time": {"weight": 3.0, "body_names": ["left_foot", "right_foot"], "thres": 0.30},
        }
    }

    reward = playback_parity.compute_reward_from_spec(reward_cfg, state)

    body_lin_vel_w = state.actual_body_lin_vel_w[:, [0, 1]]
    contact_order = [1, 0]
    current_contact_time = state.contact_current_contact_time[:, contact_order]
    last_air_time = state.contact_last_air_time[:, contact_order]
    first_contact = state.contact_first_contact[:, contact_order]
    net_forces_w_history = state.contact_net_forces_w_history[:, :, contact_order]
    in_contact = current_contact_time > 0.02
    feet_vel = (body_lin_vel_w[..., :2].norm(dim=-1) - 0.1).clamp(min=0.0, max=1.0)
    slip = -(in_contact * feet_vel).sum(dim=1, keepdim=True)
    contact_forces = net_forces_w_history.norm(dim=-1).mean(dim=1)
    force = contact_forces / state.default_mass_total[:, None]
    impact = -(force.square() * first_contact).clamp_max(10.0).sum(dim=1, keepdim=True)
    air_time = ((last_air_time - 0.30).clamp_max(0.0) * first_contact).sum(dim=1, keepdim=True)
    expected = 2.0 * slip + 0.5 * impact + 3.0 * air_time
    assert torch.allclose(reward, expected)


def test_reward_from_spec_matches_hdmi_multiplicative_group():
    state = playback_parity.MujocoRewardState(
        object_pos_w=torch.tensor([[0.0, 0.0, 0.0]]),
        ref_object_pos_w=torch.tensor([[0.2, 0.0, 0.0]]),
        object_joint_pos=torch.tensor([0.4]),
        ref_object_joint_pos=torch.tensor([0.1]),
    )
    reward_cfg = {
        "object_tracking": {
            "_multiplicative": True,
            "object_pos_tracking": {"weight": 2.0, "sigma": 0.5},
            "object_joint_pos_tracking": {"weight": 0.5, "sigma": 0.25},
        }
    }

    reward = playback_parity.compute_reward_from_spec(reward_cfg, state)

    object_pos = torch.exp(-torch.tensor([[0.2]]) / 0.5)
    object_joint = torch.exp(-torch.tensor([[0.3]]) / 0.25)
    expected = (2.0 * object_pos) * (0.5 * object_joint)
    assert torch.allclose(reward, expected)


def test_contact_forces_w_sums_all_object_filter_body_forces():
    eef_body_name = "right_wrist_yaw_link"
    force_matrix_w = torch.zeros(2, 1, 2, 3)
    force_matrix_w[:, 0, 1] = torch.tensor([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])

    class Sensor:
        data = SimpleNamespace(force_matrix_w=force_matrix_w)

        def find_bodies(self, name_keys, preserve_order=False):
            assert list(name_keys) == [eef_body_name]
            return [0], [eef_body_name]

    scene = SimpleNamespace(
        sensors={
            f"{eef_body_name}_door_contact_forces": Sensor(),
        }
    )
    object_view = SimpleNamespace(spec=SimpleNamespace(asset_name="door"))

    forces_w = playback_parity._contact_forces_w(
        scene=scene,
        eef_body_names=[eef_body_name],
        object_name="door",
        object_view=object_view,
        dtype=torch.float32,
        device=torch.device("cpu"),
        num_envs=2,
    )

    expected = torch.tensor([[[3.0, 0.0, 0.0]], [[0.0, 4.0, 0.0]]])
    assert torch.allclose(forces_w, expected)


def test_reward_state_from_mujoco_scene_feeds_reward_spec():
    module = _mujoco_env_module()
    from active_adaptation.assets_mjcf import ROBOTS

    class SceneCfg:
        robot = ROBOTS.with_object("g1_29dof", object_asset_name="door")

    scene = module.MJScene(SceneCfg(), num_envs=2, launch_viewer=False)
    robot = scene["robot"]
    door = scene["door"]

    body_names = [robot.body_names[0], robot.body_names[1]]
    joint_names = [robot.joint_names[0], robot.joint_names[1]]
    body_ids = robot.find_bodies(body_names, preserve_order=True)[0]
    joint_ids = robot.find_joints(joint_names, preserve_order=True)[0]

    robot_joint_pos = robot.data.joint_pos.clone()
    robot_joint_pos[:, joint_ids] = torch.tensor([[0.1, -0.2], [0.3, -0.4]])
    robot.write_joint_state_to_sim(robot_joint_pos, torch.zeros_like(robot_joint_pos))
    door.write_joint_state_to_sim(
        torch.tensor([[0.4], [0.8]]),
        torch.zeros(2, 1),
        joint_ids=[0],
    )
    scene.update(0.0)

    ref_body_pos_w = robot.data.body_link_pos_w[:, body_ids].clone()
    ref_body_pos_w[:, 0, 0] += torch.tensor([0.2, 0.0])
    ref_joint_pos = robot.data.joint_pos[:, joint_ids].clone()
    ref_joint_pos[:, 0] += torch.tensor([0.1, -0.1])
    ref_object_pos_w = door.data.root_link_pos_w.clone()
    ref_object_pos_w[:, 2] += torch.tensor([0.3, 0.0])
    ref_object_joint_pos = door.data.joint_pos[:, 0] + torch.tensor([0.2, -0.2])

    state = playback_parity.build_reward_state_from_scene(
        scene,
        body_names=body_names,
        joint_names=joint_names,
        ref_body_pos_w=ref_body_pos_w,
        ref_joint_pos=ref_joint_pos,
        object_name="door",
        ref_object_pos_w=ref_object_pos_w,
        ref_object_joint_pos=ref_object_joint_pos,
    )
    reward_cfg = {
        "tracking": {
            "tracking_root_pos(keypoint_pos_tracking_product)": {"weight": 1.0, "sigma": 0.5},
            "joint_pos_tracking_product": {"weight": 1.0, "sigma": 0.25},
        },
        "object_tracking": {
            "_multiplicative": True,
            "object_pos_tracking": {"weight": 2.0, "sigma": 0.5},
            "object_joint_pos_tracking": {"weight": 0.5, "sigma": 0.25},
        },
    }

    reward = playback_parity.compute_reward_from_spec(reward_cfg, state)

    assert torch.allclose(state.actual_body_pos_w, robot.data.body_link_pos_w[:, body_ids])
    assert torch.allclose(state.joint_pos, robot.data.joint_pos[:, joint_ids])
    assert torch.allclose(state.object_pos_w, door.data.root_link_pos_w)
    assert torch.allclose(state.object_joint_pos, door.data.joint_pos[:, 0])
    assert reward.shape == (2, 2)


def test_reward_state_from_mujoco_scene_preserves_resolved_joint_names_for_loco_rewards():
    module = _mujoco_env_module()
    from active_adaptation.assets_mjcf import ROBOTS

    class SceneCfg:
        robot = ROBOTS["g1_29dof"]

    scene = module.MJScene(SceneCfg(), num_envs=2, launch_viewer=False)
    robot = scene["robot"]
    joint_name_pattern = ".*hip_pitch_joint"
    joint_ids, resolved_joint_names = robot.find_joints(joint_name_pattern)
    joint_vel = robot.data.joint_vel.clone()
    joint_vel[:, joint_ids] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    robot.write_joint_state_to_sim(robot.data.joint_pos, joint_vel)

    state = playback_parity.build_reward_state_from_scene(scene, joint_names=joint_name_pattern)
    reward = playback_parity.compute_reward_from_spec(
        {"loco": {"joint_vel_l2": {"weight": 1.0, "joint_names": joint_name_pattern}}},
        state,
    )

    expected = -joint_vel[:, joint_ids].square().clamp_max(5.0).sum(dim=1, keepdim=True)
    assert state.joint_names == resolved_joint_names
    assert torch.allclose(reward, expected)


def test_kinematic_motion_playback_parity_writes_robot_and_object_reference_order():
    module = _mujoco_env_module()
    from active_adaptation.assets_mjcf import ROBOTS

    class SceneCfg:
        robot = ROBOTS.with_object("g1_29dof", object_asset_name="door")

    scene = module.MJScene(SceneCfg(), num_envs=1, launch_viewer=False)
    robot = scene["robot"]
    door = scene["door"]

    body_names = [robot.body_names[0], door.body_names[0]]
    joint_names = [robot.joint_names[0], door.joint_names[0], robot.joint_names[1]]
    body_pos_w = torch.tensor(
        [
            [[0.0, 0.0, 0.80], [1.0, 0.0, 0.20]],
            [[0.1, 0.0, 0.82], [1.2, 0.1, 0.25]],
        ]
    )
    body_quat_w = torch.zeros(2, 2, 4)
    body_quat_w[..., 0] = 1.0
    joint_pos = torch.tensor(
        [
            [0.10, 0.40, -0.20],
            [0.30, 0.80, -0.40],
        ]
    )
    reference = MujocoMotionReference(
        body_names=body_names,
        joint_names=joint_names,
        requested_body_names=body_names,
        requested_joint_names=joint_names,
        root_body_name=body_names[0],
        future_steps=torch.tensor([0]),
        body_indices=torch.arange(len(body_names)),
        joint_indices=torch.arange(len(joint_names)),
        root_body_index=0,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        joint_pos=joint_pos,
        fps=50.0,
    )
    reward_cfg = {
        "tracking": {
            "joint_pos_tracking_product": {"weight": 1.0, "sigma": 0.25},
        },
        "object_tracking": {
            "object_joint_pos_tracking": {"weight": 1.0, "sigma": 0.25},
        },
    }

    metrics = playback_parity.compute_kinematic_motion_playback_parity(
        scene,
        reference,
        steps=[0, 1],
        object_name="door",
        object_body_name=door.body_names[0],
        object_joint_name=door.joint_names[0],
        reward_cfg=reward_cfg,
    )

    assert metrics.reward.shape == (2, 1, 2)
    assert metrics.q_l2.max() < 1e-5


def test_kinematic_motion_playback_parity_builds_contact_reward_from_motion_reference():
    module = _mujoco_env_module()
    from active_adaptation.assets_mjcf import ROBOTS

    class SceneCfg:
        robot = ROBOTS.with_object("g1_29dof", object_asset_name="door")

    scene = module.MJScene(SceneCfg(), num_envs=2, launch_viewer=False)
    robot = scene["robot"]
    door = scene["door"]

    eef_body_name = robot.body_names[0]
    object_body_name = door.body_names[0]
    body_names = [eef_body_name, object_body_name]
    joint_names = [robot.joint_names[0], door.joint_names[0]]
    body_pos_w = torch.tensor(
        [
            [[0.0, 0.0, 0.80], [1.0, 0.0, 0.20]],
            [[0.1, 0.0, 0.82], [1.2, 0.1, 0.25]],
        ]
    )
    body_quat_w = torch.zeros(2, 2, 4)
    body_quat_w[..., 0] = 1.0
    joint_pos = torch.tensor(
        [
            [0.10, 0.40],
            [0.30, 0.80],
        ]
    )
    reference = MujocoMotionReference(
        body_names=body_names,
        joint_names=joint_names,
        requested_body_names=body_names,
        requested_joint_names=joint_names,
        root_body_name=eef_body_name,
        future_steps=torch.tensor([0]),
        body_indices=torch.arange(len(body_names)),
        joint_indices=torch.arange(len(joint_names)),
        root_body_index=0,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        joint_pos=joint_pos,
        fps=50.0,
        object_contact=torch.tensor([[True], [False]]),
    )
    reward_cfg = {
        "object_tracking": {
            "eef_contact_exp": {
                "weight": 1.0,
                "pos_sigma": 1.0,
                "frc_sigma": 1.0,
                "frc_thres": 0.0,
            },
        },
    }

    metrics = playback_parity.compute_kinematic_motion_playback_parity(
        scene,
        reference,
        steps=[0, 1],
        object_name="door",
        object_body_name=object_body_name,
        object_joint_name=door.joint_names[0],
        reward_cfg=reward_cfg,
        contact_eef_body_names=[eef_body_name],
        contact_target_pos_offset=[[0.0, 0.0, 0.0]],
        contact_eef_pos_offset=[[0.0, 0.0, 0.0]],
    )

    assert metrics.reward.shape == (2, 2, 1)
    assert torch.isfinite(metrics.reward).all()


def test_reward_from_spec_supports_action_rate_and_joint_torque_limits():
    state = playback_parity.MujocoRewardState(
        action_buf=torch.tensor(
            [
                [[1.0, 0.5], [-1.0, -0.25]],
                [[0.0, 0.5], [2.0, 1.5]],
            ]
        ),
        joint_names=["hip", "knee", "elbow"],
        applied_torque=torch.tensor(
            [
                [8.0, -12.0, 2.0],
                [-15.0, 3.0, 20.0],
            ]
        ),
        joint_effort_limits=torch.tensor(
            [
                [10.0, 10.0, 5.0],
                [10.0, 10.0, 10.0],
            ]
        ),
    )
    reward_cfg = {
        "loco": {
            "action_rate_l2": {"weight": 0.25},
            "joint_torque_limits": {
                "weight": 2.0,
                "joint_names": ["hip", "elbow"],
                "soft_factor": 0.6,
            },
        }
    }

    reward = playback_parity.compute_reward_from_spec(reward_cfg, state)

    action_diff = state.action_buf[:, :, 0] - state.action_buf[:, :, 1]
    action_rate = -action_diff.square().sum(dim=-1, keepdim=True)
    joint_ids = [0, 2]
    applied_torque = state.applied_torque[:, joint_ids]
    soft_limits = state.joint_effort_limits[:, joint_ids] * 0.6
    violation_high = (applied_torque / soft_limits - 1.0).clamp_min(0.0)
    violation_low = (-applied_torque / soft_limits - 1.0).clamp_min(0.0)
    torque_limit = -(violation_high + violation_low).sum(dim=1, keepdim=True)
    expected = 0.25 * action_rate + 2.0 * torque_limit
    assert torch.allclose(reward, expected)
