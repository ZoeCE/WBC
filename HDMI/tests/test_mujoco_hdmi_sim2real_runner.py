import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_runner_module():
    script_path = ROOT / "scripts/mujoco_hdmi_sim2real_runner.py"
    spec = importlib.util.spec_from_file_location("mujoco_hdmi_sim2real_runner", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_sim2real_root_is_bundled_inside_project(monkeypatch):
    monkeypatch.delenv("HDMI_SIM2REAL_ROOT", raising=False)

    module = _load_runner_module()

    assert module.DEFAULT_SIM2REAL_ROOT == ROOT / "third_party/sim2real_hdmi"
    assert (module.DEFAULT_SIM2REAL_ROOT / "checkpoints/G1Dance1Subject2").is_dir()


def test_official_sim2real_scenarios_cover_four_checkpoint_sets():
    module = _load_runner_module()

    scenarios = module.official_scenario_specs()

    assert list(scenarios) == [
        "G1Dance1Subject2",
        "G1TrackSuitcase",
        "G1PushDoorHand",
        "G1RollBall",
    ]
    assert scenarios["G1TrackSuitcase"].scene_config == "config/scene/g1_29dof_rubberhand-suitcase.yaml"
    assert scenarios["G1PushDoorHand"].policy_config.endswith("policy-xg6644nr-final.yaml")
    assert scenarios["G1RollBall"].success_metric == "ball_xy_displacement"


def test_select_scenarios_accepts_all_and_aliases():
    module = _load_runner_module()

    all_scenarios = module.select_scenarios(["all"])
    selected = module.select_scenarios(["suitcase", "door"])

    assert [scenario.name for scenario in all_scenarios] == [
        "G1Dance1Subject2",
        "G1TrackSuitcase",
        "G1PushDoorHand",
        "G1RollBall",
    ]
    assert [scenario.name for scenario in selected] == ["G1TrackSuitcase", "G1PushDoorHand"]


def test_resolve_existing_sim2real_paths_reports_required_files(tmp_path):
    module = _load_runner_module()
    root = tmp_path / "sim2real"
    (root / "config/robot").mkdir(parents=True)
    (root / "config/scene").mkdir(parents=True)
    (root / "checkpoints/G1Dance1Subject2").mkdir(parents=True)
    (root / "config/robot/g1.yaml").write_text("ROBOT_TYPE: g1_29dof\n", encoding="utf-8")
    (root / "config/scene/g1_29dof_rubberhand.yaml").write_text("ROBOT_SCENE: robot.xml\n", encoding="utf-8")
    (root / "checkpoints/G1Dance1Subject2/policy-1781wsjf-final.yaml").write_text("{}\n", encoding="utf-8")
    (root / "checkpoints/G1Dance1Subject2/policy-1781wsjf-final.onnx").write_bytes(b"onnx")
    (root / "checkpoints/G1Dance1Subject2/policy-1781wsjf-final.json").write_text("{}\n", encoding="utf-8")

    paths = module.resolve_sim2real_scenario_paths(root, module.select_scenarios(["dance"])[0])

    assert paths["robot_config"] == root / "config/robot/g1.yaml"
    assert paths["scene_config"] == root / "config/scene/g1_29dof_rubberhand.yaml"
    assert paths["policy_config"].name == "policy-1781wsjf-final.yaml"
    assert paths["policy_model"].suffix == ".onnx"
    assert paths["policy_metadata"].suffix == ".json"


def test_apply_wbc_robot_overrides_expands_non_overlapping_deploy_mappings():
    module = _load_runner_module()
    policy_config = {
        "isaac_joint_names": [
            "left_wrist_yaw_joint",
            "right_wrist_yaw_joint",
            "left_elbow_joint",
        ],
        "default_joint_pos": {".*": 0.0},
        "joint_kp": {
            ".*_wrist_yaw_joint": 20.0,
            ".*_elbow_joint": 40.0,
        },
        "joint_kd": {
            ".*_wrist_yaw_joint": 1.0,
            ".*_elbow_joint": 1.5,
        },
    }
    robot_cfg = {
        "override_params": {
            "init_state": {
                "joint_pos": {
                    "left_wrist_yaw_joint": -0.4,
                    "right_wrist_yaw_joint": 0.4,
                }
            },
            "actuators": {
                "arms": {
                    "stiffness": {
                        ".*_wrist_yaw_joint": 4.0,
                    },
                }
            },
        }
    }

    module._apply_wbc_robot_overrides_to_policy_config(policy_config, robot_cfg)

    assert policy_config["default_joint_pos"] == {
        "left_wrist_yaw_joint": -0.4,
        "right_wrist_yaw_joint": 0.4,
        "left_elbow_joint": 0.0,
    }
    assert policy_config["joint_kp"] == {
        "left_wrist_yaw_joint": 4.0,
        "right_wrist_yaw_joint": 4.0,
        "left_elbow_joint": 40.0,
    }
    assert policy_config["joint_kd"] == {
        ".*_wrist_yaw_joint": 1.0,
        ".*_elbow_joint": 1.5,
    }


def test_summarize_rollout_uses_scenario_success_metric(tmp_path):
    module = _load_runner_module()
    scenario = module.select_scenarios(["suitcase"])[0]
    video_path = tmp_path / "suitcase.mp4"
    samples = [
        {"time": 0.0, "pelvis_z": 0.79, "pelvis_xy": [0.0, 0.0], "suitcase_xy": [0.0, 0.0]},
        {"time": 6.0, "pelvis_z": 0.65, "pelvis_xy": [0.2, 0.0], "suitcase_xy": [1.2, 0.0]},
    ]

    summary = module.summarize_rollout(
        scenario=scenario,
        duration_sec=6.0,
        policy_steps=300,
        sim_steps=1200,
        decimation=4,
        samples=samples,
        actions_abs_max=[1.0, 2.0],
        actions_abs_mean=[0.5, 0.25],
        q_target_abs_max=[1.1, 1.2],
        video_path=video_path,
    )

    assert summary["finite"] is True
    assert summary["heuristic_success"] is True
    assert summary["success_metric"] == "suitcase_xy_displacement"
    assert summary["object_metrics"]["suitcase_xy_displacement"] == 1.2
    assert summary["video_path"] == str(video_path)
    assert json.loads(json.dumps(summary))["scenario"] == "G1TrackSuitcase"



def test_summarize_rollout_distinguishes_transient_crouch_from_final_fall(tmp_path):
    module = _load_runner_module()
    scenario = module.select_scenarios(["suitcase"])[0]
    samples = [
        {"time": 0.0, "pelvis_z": 0.79, "pelvis_up_z": 1.0, "pelvis_xy": [0.0, 0.0], "suitcase_xy": [0.0, 0.0]},
        {"time": 3.0, "pelvis_z": 0.29, "pelvis_up_z": 0.92, "pelvis_xy": [0.1, 0.4], "suitcase_xy": [0.7, 0.0]},
        {"time": 6.0, "pelvis_z": 0.65, "pelvis_up_z": 0.95, "pelvis_xy": [0.2, 1.0], "suitcase_xy": [1.2, 0.0]},
    ]

    summary = module.summarize_rollout(
        scenario=scenario,
        duration_sec=6.0,
        policy_steps=300,
        sim_steps=1200,
        decimation=4,
        samples=samples,
        actions_abs_max=[1.0],
        actions_abs_mean=[0.5],
        q_target_abs_max=[1.2],
        video_path=tmp_path / "suitcase.mp4",
    )

    assert summary["low_pelvis_observed"] is True
    assert summary["final_posture_ok"] is True
    assert summary["not_fallen"] is True
    assert summary["heuristic_success"] is True


def test_summarize_rollout_includes_reference_tracking_metrics(tmp_path):
    module = _load_runner_module()
    scenario = module.select_scenarios(["dance"])[0]
    samples = [
        {
            "time": 0.02,
            "pelvis_z": 0.79,
            "pelvis_up_z": 1.0,
            "pelvis_xy": [0.0, 0.0],
            "reference_step": 1,
            "q_ref_l2": 0.10,
            "q_ref_linf": 0.08,
            "body_pos_ref_l2": 0.20,
            "body_pos_ref_linf": 0.15,
            "object_pos_ref_l2": 0.30,
            "object_pos_ref_linf": 0.25,
        },
        {
            "time": 0.04,
            "pelvis_z": 0.80,
            "pelvis_up_z": 1.0,
            "pelvis_xy": [0.0, 0.0],
            "reference_step": 2,
            "q_ref_l2": 0.40,
            "q_ref_linf": 0.35,
            "body_pos_ref_l2": 0.60,
            "body_pos_ref_linf": 0.50,
            "object_pos_ref_l2": 0.90,
            "object_pos_ref_linf": 0.70,
        },
    ]

    summary = module.summarize_rollout(
        scenario=scenario,
        duration_sec=0.04,
        policy_steps=2,
        sim_steps=2,
        decimation=1,
        samples=samples,
        actions_abs_max=[0.0],
        actions_abs_mean=[0.0],
        q_target_abs_max=[0.0],
        video_path=tmp_path / "dance.mp4",
    )

    tracking = summary["reference_tracking"]
    assert tracking["available"] is True
    assert tracking["reference_step_first"] == 1
    assert tracking["reference_step_last"] == 2
    assert tracking["q_ref_l2_mean"] == 0.25
    assert tracking["q_ref_l2_max"] == 0.4
    assert tracking["body_pos_ref_l2_mean"] == 0.4
    assert tracking["body_pos_ref_linf_max"] == 0.5
    assert tracking["object_pos_ref_l2_mean"] == 0.6
    assert tracking["object_pos_ref_linf_max"] == 0.7


def test_q_target_reference_tracking_is_sampled_and_summarized(tmp_path):
    module = _load_runner_module()
    scenario = module.select_scenarios(["dance"])[0]
    reference_tracker = {
        "initial_step": 0,
        "num_steps": 3,
        "motion": {
            "joint_pos": np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [-1.0, 0.5, 2.0],
                    [-1.5, 0.75, 2.25],
                ],
                dtype=np.float64,
            )
        },
        "joint_motion_indices": np.asarray([1, 0], dtype=np.int64),
        "joint_target_indices": np.asarray([0, 2], dtype=np.int64),
        "tracked_joint_names": ["left_hip_pitch_joint", "right_knee_joint"],
    }
    sample = {
        "time": 0.02,
        "pelvis_z": 0.79,
        "pelvis_up_z": 1.0,
        "pelvis_xy": [0.0, 0.0],
    }
    sample.update(
        module.sample_wbc_q_target_reference_tracking(
            q_target=np.asarray([0.25, 99.0, -0.25], dtype=np.float64),
            reference_tracker=reference_tracker,
            policy_step=0,
        )
    )

    summary = module.summarize_rollout(
        scenario=scenario,
        duration_sec=0.02,
        policy_steps=1,
        sim_steps=1,
        decimation=1,
        samples=[sample],
        actions_abs_max=[0.0],
        actions_abs_mean=[0.0],
        q_target_abs_max=[99.0],
        video_path=tmp_path / "dance.mp4",
    )

    tracking = summary["reference_tracking"]
    assert tracking["q_target_ref_l2_mean"] == 0.79056942
    assert tracking["q_target_ref_linf_max"] == 0.75
    assert tracking["top_q_target_ref_joint_errors"][0]["name"] == "right_knee_joint"


def test_summarize_rollout_reports_top_reference_error_contributors(tmp_path):
    module = _load_runner_module()
    scenario = module.select_scenarios(["dance"])[0]
    samples = [
        {
            "time": 0.02,
            "pelvis_z": 0.79,
            "pelvis_up_z": 1.0,
            "pelvis_xy": [0.0, 0.0],
            "reference_step": 1,
            "q_ref_l2": 0.5,
            "body_pos_ref_l2": 0.6,
            "q_ref_joint_error_top": [
                {"name": "left_hip_pitch_joint", "abs_error": 0.20, "signed_error": -0.20},
                {"name": "right_wrist_yaw_joint", "abs_error": 0.10, "signed_error": 0.10},
            ],
            "body_pos_ref_error_top": [
                {"name": "pelvis", "l2": 0.30, "linf": 0.20, "error_xyz": [0.1, 0.2, 0.0]},
                {"name": "torso_link", "l2": 0.10, "linf": 0.08, "error_xyz": [0.0, 0.08, 0.0]},
            ],
            "object_pos_ref_error_top": [
                {"name": "suitcase", "l2": 0.40, "linf": 0.30, "error_xyz": [0.3, 0.0, 0.0]},
            ],
        },
        {
            "time": 0.04,
            "pelvis_z": 0.80,
            "pelvis_up_z": 1.0,
            "pelvis_xy": [0.0, 0.0],
            "reference_step": 2,
            "q_ref_l2": 0.8,
            "body_pos_ref_l2": 0.9,
            "q_ref_joint_error_top": [
                {"name": "left_hip_pitch_joint", "abs_error": 0.40, "signed_error": -0.40},
                {"name": "right_wrist_yaw_joint", "abs_error": 0.60, "signed_error": 0.60},
            ],
            "body_pos_ref_error_top": [
                {"name": "pelvis", "l2": 0.50, "linf": 0.40, "error_xyz": [0.4, 0.0, 0.0]},
                {"name": "torso_link", "l2": 0.20, "linf": 0.15, "error_xyz": [0.0, 0.15, 0.0]},
            ],
            "object_pos_ref_error_top": [
                {"name": "suitcase", "l2": 0.70, "linf": 0.50, "error_xyz": [0.5, 0.0, 0.0]},
            ],
        },
    ]

    summary = module.summarize_rollout(
        scenario=scenario,
        duration_sec=0.04,
        policy_steps=2,
        sim_steps=2,
        decimation=1,
        samples=samples,
        actions_abs_max=[0.0],
        actions_abs_mean=[0.0],
        q_target_abs_max=[0.0],
        video_path=tmp_path / "dance.mp4",
    )

    tracking = summary["reference_tracking"]
    assert tracking["top_q_ref_joint_errors"][0] == {
        "name": "right_wrist_yaw_joint",
        "mean_abs_error": 0.35,
        "max_abs_error": 0.6,
        "max_reference_step": 2,
    }
    assert tracking["top_body_pos_ref_errors"][0] == {
        "name": "pelvis",
        "mean_l2": 0.4,
        "max_l2": 0.5,
        "max_reference_step": 2,
    }
    assert tracking["top_object_pos_ref_errors"][0] == {
        "name": "suitcase",
        "mean_l2": 0.55,
        "max_l2": 0.7,
        "max_reference_step": 2,
    }


def test_summarize_rollout_reports_best_reference_phase_offset(tmp_path):
    module = _load_runner_module()
    scenario = module.select_scenarios(["dance"])[0]
    samples = [
        {
            "time": 0.02,
            "pelvis_z": 0.79,
            "pelvis_up_z": 1.0,
            "pelvis_xy": [0.0, 0.0],
            "reference_step": 10,
            "reference_offset_errors": [
                {"offset": -1, "q_ref_l2": 0.6, "body_pos_ref_l2": 0.8, "object_pos_ref_l2": 0.4},
                {"offset": 0, "q_ref_l2": 1.2, "body_pos_ref_l2": 1.5, "object_pos_ref_l2": 0.9},
                {"offset": 1, "q_ref_l2": 0.9, "body_pos_ref_l2": 1.1, "object_pos_ref_l2": 0.7},
            ],
        },
        {
            "time": 0.04,
            "pelvis_z": 0.80,
            "pelvis_up_z": 1.0,
            "pelvis_xy": [0.0, 0.0],
            "reference_step": 11,
            "reference_offset_errors": [
                {"offset": -1, "q_ref_l2": 0.4, "body_pos_ref_l2": 0.7, "object_pos_ref_l2": 0.2},
                {"offset": 0, "q_ref_l2": 1.0, "body_pos_ref_l2": 1.2, "object_pos_ref_l2": 0.8},
                {"offset": 1, "q_ref_l2": 0.8, "body_pos_ref_l2": 1.0, "object_pos_ref_l2": 0.6},
            ],
        },
    ]

    summary = module.summarize_rollout(
        scenario=scenario,
        duration_sec=0.04,
        policy_steps=2,
        sim_steps=2,
        decimation=1,
        samples=samples,
        actions_abs_max=[0.0],
        actions_abs_mean=[0.0],
        q_target_abs_max=[0.0],
        video_path=tmp_path / "dance.mp4",
    )

    phase = summary["reference_tracking"]["phase_offset_diagnostic"]
    assert phase["best_offset"] == -1
    assert phase["best_q_ref_l2_mean"] == 0.5
    assert phase["zero_offset_q_ref_l2_mean"] == 1.1
    assert phase["best_body_pos_ref_l2_mean"] == 0.75
    assert phase["best_object_pos_ref_l2_mean"] == 0.3
    assert phase["q_ref_l2_improvement_vs_zero"] == 0.6


def test_control_latency_delays_policy_target_by_rl_step():
    module = _load_runner_module()
    pending = []
    hold = np.array([0.0, 0.0], dtype=np.float32)
    first_policy_target = np.array([1.0, 1.5], dtype=np.float32)
    second_policy_target = np.array([2.0, 2.5], dtype=np.float32)

    first_active = module.select_control_q_target(
        pending,
        first_policy_target,
        hold,
        control_latency_steps=1,
    )
    second_active = module.select_control_q_target(
        pending,
        second_policy_target,
        hold,
        control_latency_steps=1,
    )
    zero_latency_active = module.select_control_q_target(
        [],
        first_policy_target,
        hold,
        control_latency_steps=0,
    )

    np.testing.assert_allclose(first_active, hold)
    np.testing.assert_allclose(second_active, first_policy_target)
    np.testing.assert_allclose(zero_latency_active, first_policy_target)
    assert len(pending) == 1
    np.testing.assert_allclose(pending[0], second_policy_target)


def test_wbc_reference_clock_holds_during_initial_pause_then_advances():
    module = _load_runner_module()

    steps = [
        module.wbc_reference_step_for_policy_step(policy_step, initial_pause_steps=3, num_steps=10)
        for policy_step in range(6)
    ]
    sample_steps = [
        module.wbc_tracking_reference_step_for_policy_step(policy_step, initial_pause_steps=3, num_steps=10)
        for policy_step in range(6)
    ]

    assert steps == [0, 0, 0, 0, 1, 2]
    assert sample_steps == [1, 1, 1, 1, 2, 3]


def test_wbc_direct_observation_bridge_offsets_and_prefills_joint_history():
    import mujoco

    module = _load_runner_module()
    model = mujoco.MjModel.from_xml_string(
        "\n".join(
            [
                "<mujoco>",
                "  <worldbody>",
                "    <body name='pelvis' pos='0 0 1'>",
                "      <freejoint name='pelvis_root'/>",
                "      <geom type='sphere' size='0.1' mass='1'/>",
                "      <body name='joint_body_a' pos='0 0 0.1'>",
                "        <joint name='joint_a' type='hinge' axis='0 0 1'/>",
                "        <geom type='sphere' size='0.05' mass='0.1'/>",
                "        <body name='joint_body_b' pos='0 0 0.1'>",
                "          <joint name='joint_b' type='hinge' axis='0 1 0'/>",
                "          <geom type='sphere' size='0.05' mass='0.1'/>",
                "        </body>",
                "      </body>",
                "    </body>",
                "  </worldbody>",
                "</mujoco>",
            ]
        )
    )
    data = mujoco.MjData(model)
    data.qpos[7] = 0.60
    data.qpos[8] = -0.10
    mujoco.mj_forward(model, data)
    policy_config = {
        "observation": {
            "policy": {
                "joint_pos_history": {
                    "history_steps": [0, 1, 2],
                    "joint_names": ["joint_a", "joint_b"],
                }
            }
        },
        "policy_joint_names": ["joint_a", "joint_b"],
        "isaac_joint_names": ["joint_a", "joint_b"],
        "default_joint_pos": {"joint_a": 0.50, "joint_b": -0.25},
        "action_scale": 1.0,
    }
    bridge = module.WBCDirectObservationBridge(
        model=model,
        policy_config=policy_config,
        policy_joint_names=["joint_a", "joint_b"],
        action_dim=2,
        object_names=[],
    )

    obs = bridge.build(model, data)

    np.testing.assert_allclose(
        obs["policy"],
        np.asarray([[0.10, 0.15, 0.10, 0.15, 0.10, 0.15]], dtype=np.float32),
        atol=1e-6,
    )



def test_wbc_direct_observation_bridge_builds_teacher_privileged_groups(tmp_path):
    import mujoco

    module = _load_runner_module()
    model = mujoco.MjModel.from_xml_string(
        "\n".join(
            [
                "<mujoco>",
                "  <worldbody>",
                "    <body name='pelvis' pos='0 0 1'>",
                "      <freejoint name='pelvis_root'/>",
                "      <geom type='sphere' size='0.1' mass='1'/>",
                "      <body name='left_ankle_roll_link' pos='0.1 0 0'>",
                "        <joint name='hip_joint' type='hinge' axis='0 0 1'/>",
                "        <geom type='sphere' size='0.05' mass='0.1'/>",
                "      </body>",
                "      <body name='torso_link' pos='0 0 0.2'>",
                "        <geom type='sphere' size='0.05' mass='0.1'/>",
                "      </body>",
                "    </body>",
                "    <body name='door' pos='1 0 0'>",
                "      <joint name='door_joint' type='hinge' axis='0 0 1'/>",
                "      <geom type='box' size='0.1 0.1 0.1' mass='1'/>",
                "    </body>",
                "  </worldbody>",
                "  <actuator>",
                "    <motor name='hip_joint' joint='hip_joint' gear='1'/>",
                "    <motor name='door_joint' joint='door_joint' gear='1'/>",
                "  </actuator>",
                "</mujoco>",
            ]
        )
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    motion_dir = tmp_path / "motion"
    motion_dir.mkdir()
    (motion_dir / "meta.json").write_text(
        json.dumps(
            {
                "fps": 50,
                "body_names": ["pelvis", "left_ankle_roll_link", "torso_link", "door"],
                "joint_names": ["hip_joint", "door_joint"],
            }
        ),
        encoding="utf-8",
    )
    body_pos = np.zeros((4, 4, 3), dtype=np.float32)
    body_pos[:, 0] = [0.0, 0.0, 1.0]
    body_pos[:, 1] = [0.1, 0.0, 1.0]
    body_pos[:, 2] = [0.0, 0.0, 1.2]
    body_pos[:, 3] = [1.0, 0.0, 0.0]
    body_quat = np.zeros((4, 4, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    np.savez_compressed(
        motion_dir / "motion.npz",
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=np.zeros_like(body_pos),
        body_ang_vel_w=np.zeros_like(body_pos),
        joint_pos=np.asarray([[0.0, 0.0], [0.2, 0.1], [0.3, 0.2], [0.4, 0.3]], dtype=np.float32),
        joint_vel=np.zeros((4, 2), dtype=np.float32),
        object_contact=np.asarray([[False], [True], [False], [False]]),
    )
    ref_cfg = {
        "motion_path": str(motion_dir),
        "future_steps": [1, 2],
        "body_names": ["pelvis", "left_ankle_roll_link"],
        "joint_names": ["hip_joint"],
        "root_body_name": "pelvis",
    }
    policy_config = {
        "observation": {
            "priv": {
                "root_ang_vel_history": {"history_steps": [0]},
                "projected_gravity_history": {"history_steps": [0]},
                "joint_pos_history": {"history_steps": [0], "joint_names": ["hip_joint"]},
                "ref_root_pos_future_b": {},
                "ref_root_ori_future_b": {},
                "diff_body_pos_future_local": {},
                "diff_body_ori_future_local": {},
                "diff_body_lin_vel_future_local": {},
                "diff_body_ang_vel_future_local": {},
                "root_linvel_b": {},
                "body_pos_b": {"body_names": ["left_ankle_roll_link", "torso_link"]},
                "body_vel_b": {"body_names": ["left_ankle_roll_link", "torso_link"]},
                "body_height": {"body_names": ["left_ankle_roll_link", "torso_link"]},
                "applied_action": {},
                "applied_torque": {"joint_names": ["hip_joint"]},
                "object_pos_b": {},
                "object_ori_b": {},
                "diff_object_pos_future": {},
                "diff_object_ori_future": {},
                "ref_object_contact_future": {},
                "diff_contact_pos_b": {},
                "object_joint_pos": {},
                "object_joint_vel": {},
                "object_joint_torque": {},
            },
            "object": {
                "object_xy_b": {"object_name": "door", "root_body_name": "pelvis"},
                "object_heading_b": {"object_name": "door", "root_body_name": "pelvis"},
                "ref_contact_pos_b": {
                    "object_name": "door",
                    "root_body_name": "pelvis",
                    "contact_target_pos_offset": [[0.0, 0.0, 0.0]],
                    "contact_eef_body_name": ["left_ankle_roll_link"],
                    "contact_eef_pos_offset": [[0.0, 0.0, 0.0]],
                    "yaw_only": True,
                },
            },
            "command": {
                "ref_body_pos_future_local": dict(ref_cfg),
                "ref_joint_pos_future": dict(ref_cfg),
                "ref_motion_phase": dict(ref_cfg),
            },
            "policy": {
                "root_ang_vel_history": {"history_steps": [0]},
                "projected_gravity_history": {"history_steps": [0]},
                "joint_pos_history": {"history_steps": [0], "joint_names": ["hip_joint"]},
                "prev_actions": {"steps": 3},
            },
            "ref_joint_pos_": {"ref_joint_pos_action_policy": {}},
        },
        "policy_joint_names": ["hip_joint"],
        "isaac_joint_names": ["hip_joint"],
        "default_joint_pos": {"hip_joint": 0.0},
        "action_scale": {"hip_joint": 0.5},
    }
    bridge = module.WBCDirectObservationBridge(
        model=model,
        policy_config=policy_config,
        policy_joint_names=["hip_joint"],
        action_dim=1,
        object_names=["door", "pelvis"],
    )

    obs = bridge.build(model, data)

    assert set(obs) == {"priv", "object", "command", "policy", "ref_joint_pos_"}
    assert obs["object"].shape == (1, 7)
    assert obs["ref_joint_pos_"].shape == (1, 1)
    assert obs["priv"].shape[0] == 1
    assert np.isfinite(obs["priv"]).all()


def test_wbc_direct_observation_bridge_accepts_external_reference_step(tmp_path):
    import mujoco

    module = _load_runner_module()
    model = mujoco.MjModel.from_xml_string(
        "\n".join(
            [
                "<mujoco>",
                "  <worldbody>",
                "    <body name='pelvis' pos='0 0 1'>",
                "      <freejoint name='pelvis_root'/>",
                "      <geom type='sphere' size='0.1' mass='1'/>",
                "      <body name='torso_link' pos='0 0 0.2'>",
                "        <joint name='hip_joint' type='hinge' axis='0 0 1'/>",
                "        <geom type='sphere' size='0.05' mass='0.1'/>",
                "      </body>",
                "    </body>",
                "  </worldbody>",
                "</mujoco>",
            ]
        )
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    motion_dir = tmp_path / "motion"
    motion_dir.mkdir()
    (motion_dir / "meta.json").write_text(
        json.dumps(
            {
                "fps": 50,
                "body_names": ["pelvis", "torso_link"],
                "joint_names": ["hip_joint"],
            }
        ),
        encoding="utf-8",
    )
    body_pos = np.zeros((4, 2, 3), dtype=np.float32)
    body_pos[:, 0] = [0.0, 0.0, 1.0]
    body_pos[:, 1] = [0.0, 0.0, 1.2]
    body_quat = np.zeros((4, 2, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    np.savez_compressed(
        motion_dir / "motion.npz",
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=np.zeros_like(body_pos),
        body_ang_vel_w=np.zeros_like(body_pos),
        joint_pos=np.asarray([[0.0], [0.2], [0.3], [0.4]], dtype=np.float32),
        joint_vel=np.zeros((4, 1), dtype=np.float32),
    )
    ref_cfg = {
        "motion_path": str(motion_dir),
        "future_steps": [1],
        "body_names": ["pelvis", "torso_link"],
        "joint_names": ["hip_joint"],
        "root_body_name": "pelvis",
    }
    policy_config = {
        "observation": {
            "command": {
                "ref_body_pos_future_local": dict(ref_cfg),
                "ref_joint_pos_future": dict(ref_cfg),
                "ref_motion_phase": dict(ref_cfg),
            },
            "policy": {
                "joint_pos_history": {"history_steps": [0], "joint_names": ["hip_joint"]},
            },
            "ref_joint_pos_": {"ref_joint_pos_action_policy": {}},
        },
        "policy_joint_names": ["hip_joint"],
        "isaac_joint_names": ["hip_joint"],
        "default_joint_pos": {"hip_joint": 0.0},
        "action_scale": {"hip_joint": 0.5},
    }
    bridge = module.WBCDirectObservationBridge(
        model=model,
        policy_config=policy_config,
        policy_joint_names=["hip_joint"],
        action_dim=1,
        object_names=[],
    )

    first = bridge.build(model, data, reference_step=0)
    third = bridge.build(model, data, reference_step=2)

    np.testing.assert_allclose(first["ref_joint_pos_"], [[0.0]], atol=1e-6)
    np.testing.assert_allclose(third["ref_joint_pos_"], [[0.6]], atol=1e-6)


def test_wbc_direct_action_adapter_matches_joint_position_delay_and_alpha():
    module = _load_runner_module()
    adapter = module.WBCDirectActionAdapter(
        default_joint_pos=np.asarray([0.5], dtype=np.float32),
        action_scale=np.asarray([0.25], dtype=np.float32),
        delay=4,
        alpha=0.5,
    )

    adapter.record(np.asarray([2.0], dtype=np.float32))
    first_tick_target = adapter.joint_position_target(substep=0, decimation=4)
    adapter.record(np.asarray([4.0], dtype=np.float32))
    second_tick_target = adapter.joint_position_target(substep=0, decimation=4)

    np.testing.assert_allclose(first_tick_target, [0.5], atol=1e-6)
    np.testing.assert_allclose(second_tick_target, [0.75], atol=1e-6)


def test_wbc_joint_position_target_updates_observation_applied_action():
    module = _load_runner_module()

    class FakeActionAdapter:
        def __init__(self):
            self.applied_action = np.asarray([0.25], dtype=np.float32)

        def joint_position_target(self, *, substep, decimation):
            assert substep == 0
            assert decimation == 4
            return np.asarray([0.75], dtype=np.float32)

    class FakeObservationBridge:
        def __init__(self):
            self.recorded_applied_action = None

        def record_applied_action(self, applied_action):
            self.recorded_applied_action = np.asarray(applied_action, dtype=np.float32).copy()

    policy = object.__new__(module.DirectSim2RealPolicy)
    policy.wbc_action_adapter = FakeActionAdapter()
    policy.wbc_observation_bridge = FakeObservationBridge()
    policy.default_dof_angles = np.asarray([0.0, -0.5], dtype=np.float32)
    policy.controlled_joint_indices = [1]

    q_target = policy.wbc_joint_position_target(substep=0, decimation=4)

    np.testing.assert_allclose(q_target, [0.0, 0.75], atol=1e-6)
    np.testing.assert_allclose(
        policy.wbc_observation_bridge.recorded_applied_action,
        [0.25],
        atol=1e-6,
    )


def test_materialize_mjcf_static_body_poses_updates_included_static_body(tmp_path):
    import mujoco

    module = _load_runner_module()
    object_xml = tmp_path / "object.xml"
    object_xml.write_text(
        "\n".join(
            [
                "<mujoco>",
                "  <worldbody>",
                "    <body name='static_support' pos='1 0 0'>",
                "      <geom type='box' size='0.1 0.1 0.1'/>",
                "    </body>",
                "    <body name='dynamic_box' pos='2 0 0'>",
                "      <freejoint name='dynamic_box_root'/>",
                "      <geom type='box' size='0.1 0.1 0.1' mass='1'/>",
                "    </body>",
                "  </worldbody>",
                "</mujoco>",
            ]
        ),
        encoding="utf-8",
    )
    scene_xml = tmp_path / "scene.xml"
    scene_xml.write_text(
        "\n".join(
            [
                "<mujoco>",
                "  <include file='object.xml'/>",
                "</mujoco>",
            ]
        ),
        encoding="utf-8",
    )

    materialized = module.materialize_mjcf_static_body_poses(
        scene_path=scene_xml,
        body_pose_by_name={
            "static_support": {
                "pos": [0.25, -0.5, 0.75],
                "quat": [0.9238795325, 0.0, 0.0, 0.3826834324],
            },
            "dynamic_box": {
                "pos": [0.5, 0.5, 0.5],
                "quat": [1.0, 0.0, 0.0, 0.0],
            },
        },
        cache_dir=tmp_path / "cache",
    )

    model = mujoco.MjModel.from_xml_path(str(materialized["scene_path"]))
    static_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "static_support")
    dynamic_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "dynamic_box")

    np.testing.assert_allclose(model.body_pos[static_body_id], [0.25, -0.5, 0.75], atol=1e-6)
    np.testing.assert_allclose(model.body_quat[static_body_id], [0.9238795325, 0.0, 0.0, 0.3826834324], atol=1e-6)
    np.testing.assert_allclose(model.body_pos[dynamic_body_id], [2.0, 0.0, 0.0], atol=1e-6)
    assert materialized["applied_static_body_names"] == ["static_support"]
    assert materialized["skipped_freejoint_body_names"] == ["dynamic_box"]


def test_normalize_video_stride_rejects_non_positive_values():
    module = _load_runner_module()

    assert module.normalize_video_stride(5) == 5
    with pytest.raises(ValueError, match="video_stride"):
        module.normalize_video_stride(0)


def test_elastic_band_force_matches_sim2real_formula():
    module = _load_runner_module()

    force = module.compute_elastic_band_force(
        point=np.array([0.0, 0.0, 3.0]),
        position=np.array([0.0, 0.0, 1.0]),
        linear_velocity=np.array([0.0, 0.0, 0.5]),
        stiffness=200.0,
        damping=100.0,
        length=0.0,
    )

    np.testing.assert_allclose(force, [0.0, 0.0, 350.0])


def test_official_sim2real_scene_config_disables_virtual_gantry_after_start_by_default():
    module = _load_runner_module()

    scene_config = module.official_sim2real_scene_config(
        {
            "ROBOT_SCENE": "data/robots/g1/g1_29dof_rubberhand-suitcase.xml",
            "SIMULATE_DT": 0.005,
            "ENABLE_ELASTIC_BAND": True,
        }
    )

    assert scene_config["ENABLE_ELASTIC_BAND"] is False


def test_official_sim2real_scene_config_can_keep_virtual_gantry_for_comparison():
    module = _load_runner_module()

    scene_config = module.official_sim2real_scene_config(
        {
            "ROBOT_SCENE": "data/robots/g1/g1_29dof_rubberhand-suitcase.xml",
            "SIMULATE_DT": 0.005,
            "ENABLE_ELASTIC_BAND": True,
        },
        disable_virtual_gantry_after_start=False,
    )

    assert scene_config["ENABLE_ELASTIC_BAND"] is True


def test_resolve_wbc_export_inputs_rewrites_motion_and_scene_paths(tmp_path):
    module = _load_runner_module()
    wbc_root = tmp_path / "HDMI"
    task_path = wbc_root / "cfg/task/G1/hdmi/carry_box.yaml"
    motion_dir = wbc_root / "data/motion/carry_box"
    export_dir = wbc_root / "scripts/exports/G1CarryBox"
    task_path.parent.mkdir(parents=True)
    motion_dir.mkdir(parents=True)
    export_dir.mkdir(parents=True)
    task_path.write_text(
        "\n".join(
            [
                "name: G1CarryBox",
                "command:",
                "  data_path: data/motion/carry_box",
                "  object_asset_name: bread_box",
                "  object_body_name: bread_box",
                "  root_body_name: pelvis",
                "  extra_object_names: [support0]",
            ]
        ),
        encoding="utf-8",
    )
    policy_path = export_dir / "policy-test-final.onnx"
    policy_yaml = policy_path.with_suffix(".yaml")
    policy_json = policy_path.with_suffix(".json")
    policy_path.write_bytes(b"onnx")
    policy_json.write_text("{}", encoding="utf-8")
    policy_yaml.write_text(
        "\n".join(
            [
                "observation:",
                "  command:",
                "    ref_motion_phase:",
                "      motion_path: data/motion/carry_box",
                "      motion_duration_second: 1.0",
                "      future_steps: [1]",
                "      body_names: [pelvis]",
                "      joint_names: []",
                "      root_body_name: pelvis",
                "policy_joint_names: []",
                "isaac_joint_names: []",
                "default_joint_pos: 0.0",
                "action_scale: 1.0",
            ]
        ),
        encoding="utf-8",
    )

    resolved = module.resolve_wbc_export_inputs(
        wbc_root=wbc_root,
        task_yaml=task_path,
        policy_model=policy_path,
    )

    assert resolved["scenario"].name == "G1CarryBox"
    assert resolved["policy_config"]["observation"]["command"]["ref_motion_phase"]["motion_path"] == str(motion_dir)
    assert resolved["scene_config"]["ROBOT_SCENE"].endswith("g1_29dof_nohand-bread_box.xml")
    assert resolved["scene_config"]["SIMULATE_DT"] == 0.002
    assert resolved["scene_config"]["ENABLE_ELASTIC_BAND"] is False
    assert resolved["publish_object_names"] == ("bread_box", "support0", "pelvis")
    assert resolved["policy_model"] == policy_path
    assert resolved["policy_metadata"] == policy_json


def test_resolve_wbc_export_inputs_uses_hdmi_default_contact_eefs_when_task_omits_override(tmp_path):
    module = _load_runner_module()
    wbc_root = tmp_path / "HDMI"
    task_path = wbc_root / "cfg/task/G1/hdmi/carry_box.yaml"
    export_dir = wbc_root / "scripts/exports/G1CarryBox"
    task_path.parent.mkdir(parents=True)
    export_dir.mkdir(parents=True)
    task_path.write_text(
        "\n".join(
            [
                "name: G1CarryBox",
                "command:",
                "  data_path: data/motion/carry_box",
                "  object_asset_name: bread_box",
                "  object_body_name: bread_box",
                "  root_body_name: pelvis",
                "  contact_target_pos_offset:",
                "    - [0.0, 0.2, 0.1]",
                "    - [0.0, -0.2, 0.1]",
                "  contact_eef_pos_offset:",
                "    - [0.05, 0.0, 0.0]",
                "    - [0.05, 0.0, 0.0]",
            ]
        ),
        encoding="utf-8",
    )
    policy_path = export_dir / "policy-test-final.onnx"
    policy_yaml = policy_path.with_suffix(".yaml")
    policy_json = policy_path.with_suffix(".json")
    policy_path.write_bytes(b"onnx")
    policy_json.write_text("{}", encoding="utf-8")
    policy_yaml.write_text(
        "\n".join(
            [
                "observation:",
                "  object:",
                "    ref_contact_pos_b:",
                "      object_name: bread_box",
                "      root_body_name: pelvis",
                "      contact_target_pos_offset:",
                "        - [0.0, 0.2, 0.1]",
                "        - [0.0, -0.2, 0.1]",
                "  priv:",
                "    diff_contact_pos_b: {}",
                "policy_joint_names: []",
                "isaac_joint_names: []",
                "default_joint_pos: 0.0",
                "action_scale: 1.0",
            ]
        ),
        encoding="utf-8",
    )

    resolved = module.resolve_wbc_export_inputs(
        wbc_root=wbc_root,
        task_yaml=task_path,
        policy_model=policy_path,
    )

    ref_cfg = resolved["policy_config"]["observation"]["object"]["ref_contact_pos_b"]
    diff_cfg = resolved["policy_config"]["observation"]["priv"]["diff_contact_pos_b"]
    assert ref_cfg["contact_eef_body_name"] == ["left_wrist_yaw_link", "right_wrist_yaw_link"]
    assert diff_cfg["contact_eef_body_name"] == ["left_wrist_yaw_link", "right_wrist_yaw_link"]
    assert ref_cfg["contact_eef_pos_offset"] == [[0.05, 0.0, 0.0], [0.05, 0.0, 0.0]]
    assert diff_cfg["contact_eef_pos_offset"] == [[0.05, 0.0, 0.0], [0.05, 0.0, 0.0]]


def test_resolve_wbc_export_inputs_publishes_asset_and_contact_body_names_for_door(tmp_path):
    module = _load_runner_module()
    wbc_root = tmp_path / "HDMI"
    task_path = wbc_root / "cfg/task/G1/hdmi/open_door-hand.yaml"
    motion_dir = wbc_root / "data/motion/open_door"
    export_dir = wbc_root / "scripts/exports/G1PushDoorHand"
    task_path.parent.mkdir(parents=True)
    motion_dir.mkdir(parents=True)
    export_dir.mkdir(parents=True)
    task_path.write_text(
        "\n".join(
            [
                "name: G1PushDoorHand",
                "robot:",
                "  robot_type: g1_29dof_rubberhand",
                "command:",
                "  data_path: data/motion/open_door",
                "  object_asset_name: door",
                "  object_body_name: door_panel",
                "  root_body_name: pelvis",
                "  object_joint_name: door_joint",
            ]
        ),
        encoding="utf-8",
    )
    policy_path = export_dir / "policy-test-final.onnx"
    policy_path.write_bytes(b"onnx")
    policy_path.with_suffix(".json").write_text("{}", encoding="utf-8")
    policy_path.with_suffix(".yaml").write_text(
        "\n".join(
            [
                "observation:",
                "  command:",
                "    ref_motion_phase:",
                "      motion_path: data/motion/open_door",
                "      motion_duration_second: 1.0",
                "      future_steps: [1]",
                "      body_names: [pelvis, door]",
                "      joint_names: []",
                "      root_body_name: pelvis",
                "policy_joint_names: []",
                "isaac_joint_names: []",
                "default_joint_pos: 0.0",
                "action_scale: 1.0",
            ]
        ),
        encoding="utf-8",
    )

    resolved = module.resolve_wbc_export_inputs(
        wbc_root=wbc_root,
        task_yaml=task_path,
        policy_model=policy_path,
    )

    assert resolved["publish_object_names"] == ("door", "door_panel", "pelvis")


def test_resolve_wbc_export_inputs_enables_elastic_band_for_rubberhand_fallback(tmp_path):
    module = _load_runner_module()
    wbc_root = tmp_path / "HDMI"
    task_path = wbc_root / "cfg/task/G1/hdmi/carry_box.yaml"
    motion_dir = wbc_root / "data/motion/carry_box"
    export_dir = wbc_root / "scripts/exports/G1CarryBox"
    task_path.parent.mkdir(parents=True)
    motion_dir.mkdir(parents=True)
    export_dir.mkdir(parents=True)
    task_path.write_text(
        "\n".join(
            [
                "name: G1CarryBox",
                "robot:",
                "  robot_type: g1_29dof_rubberhand-feet_sphere-eef_box-body_capsule",
                "command:",
                "  data_path: data/motion/carry_box",
                "  object_asset_name: bread_box",
                "  object_body_name: bread_box",
                "  root_body_name: pelvis",
            ]
        ),
        encoding="utf-8",
    )
    policy_path = export_dir / "policy-test-final.onnx"
    policy_path.write_bytes(b"onnx")
    policy_path.with_suffix(".json").write_text("{}", encoding="utf-8")
    policy_path.with_suffix(".yaml").write_text(
        "\n".join(
            [
                "observation:",
                "  command:",
                "    ref_motion_phase:",
                "      motion_path: data/motion/carry_box",
                "      motion_duration_second: 1.0",
                "      future_steps: [1]",
                "      body_names: [pelvis]",
                "      joint_names: []",
                "      root_body_name: pelvis",
                "policy_joint_names: []",
                "isaac_joint_names: []",
                "default_joint_pos: 0.0",
                "action_scale: 1.0",
            ]
        ),
        encoding="utf-8",
    )

    resolved = module.resolve_wbc_export_inputs(
        wbc_root=wbc_root,
        sim2real_root=tmp_path / "missing-sim2real",
        task_yaml=task_path,
        policy_model=policy_path,
    )

    assert resolved["scene_asset_source"] == "wbc_assets_mjcf"
    assert resolved["scene_config"]["ENABLE_ELASTIC_BAND"] is True


def test_resolve_wbc_export_inputs_prefers_loadable_sim2real_rubberhand_scene(tmp_path):
    module = _load_runner_module()
    wbc_root = tmp_path / "HDMI"
    sim2real_root = tmp_path / "sim2real"
    task_path = wbc_root / "cfg/task/G1/hdmi/move_suitcase.yaml"
    base_task_path = wbc_root / "cfg/task/base/hdmi-base.yaml"
    motion_dir = wbc_root / "data/motion/move_suitcase"
    export_dir = wbc_root / "scripts/exports/G1MoveSuitcase"
    scene_path = sim2real_root / "data/robots/g1/g1_29dof_rubberhand-suitcase.xml"
    task_path.parent.mkdir(parents=True)
    base_task_path.parent.mkdir(parents=True)
    motion_dir.mkdir(parents=True)
    export_dir.mkdir(parents=True)
    scene_path.parent.mkdir(parents=True)
    scene_path.write_text(
        "\n".join(
            [
                "<mujoco>",
                "  <worldbody>",
                "    <body name='pelvis' pos='0 0 1'>",
                "      <freejoint name='pelvis_root'/>",
                "      <geom type='sphere' size='0.1' mass='1'/>",
                "    </body>",
                "  </worldbody>",
                "</mujoco>",
            ]
        ),
        encoding="utf-8",
    )
    task_path.write_text(
        "\n".join(
            [
                "name: G1MoveSuitcase",
                "defaults:",
                "  - base/hdmi-base",
                "  - _self_",
                "robot:",
                "  robot_type: g1_29dof_rubberhand-feet_sphere-eef_box-body_capsule",
                "command:",
                "  data_path: data/motion/move_suitcase",
                "  object_asset_name: suitcase",
                "  object_body_name: suitcase",
                "  root_body_name: pelvis",
            ]
        ),
        encoding="utf-8",
    )
    base_task_path.write_text(
        "\n".join(
            [
                "sim:",
                "  step_dt: 0.02",
                "  mujoco_physics_dt: 0.002",
            ]
        ),
        encoding="utf-8",
    )
    policy_path = export_dir / "policy-test-final.onnx"
    policy_path.write_bytes(b"onnx")
    policy_path.with_suffix(".json").write_text("{}", encoding="utf-8")
    policy_path.with_suffix(".yaml").write_text(
        "\n".join(
            [
                "observation:",
                "  command:",
                "    ref_motion_phase:",
                "      motion_path: data/motion/move_suitcase",
                "      motion_duration_second: 1.0",
                "      future_steps: [1]",
                "      body_names: [pelvis]",
                "      joint_names: []",
                "      root_body_name: pelvis",
                "policy_joint_names: []",
                "isaac_joint_names: []",
                "default_joint_pos: 0.0",
                "action_scale: 1.0",
            ]
        ),
        encoding="utf-8",
    )

    resolved = module.resolve_wbc_export_inputs(
        wbc_root=wbc_root,
        sim2real_root=sim2real_root,
        task_yaml=task_path,
        policy_model=policy_path,
    )

    assert resolved["scene_config"]["ROBOT_SCENE"] == str(scene_path)
    assert resolved["scene_config"]["SIMULATE_DT"] == 0.002
    assert resolved["scene_asset_source"] == "sim2real_hdmi"


def test_resolve_wbc_export_inputs_infers_foam_support_scene_from_extra_object(tmp_path):
    module = _load_runner_module()
    wbc_root = tmp_path / "HDMI"
    sim2real_root = tmp_path / "sim2real"
    task_path = wbc_root / "cfg/task/G1/hdmi/move_foam.yaml"
    motion_dir = wbc_root / "data/motion/move_foam"
    export_dir = wbc_root / "scripts/exports/G1TrackFoam"
    scene_path = sim2real_root / "data/robots/g1/g1_29dof_rubberhand-foam_with_support.xml"
    task_path.parent.mkdir(parents=True)
    motion_dir.mkdir(parents=True)
    export_dir.mkdir(parents=True)
    scene_path.parent.mkdir(parents=True)
    scene_path.write_text(
        "\n".join(
            [
                "<mujoco>",
                "  <worldbody>",
                "    <body name='pelvis' pos='0 0 1'>",
                "      <freejoint name='pelvis_root'/>",
                "      <geom type='sphere' size='0.1' mass='1'/>",
                "    </body>",
                "    <body name='foam' pos='0 0 0.1'><geom type='box' size='0.1 0.1 0.1' mass='1'/></body>",
                "    <body name='stool_support' pos='0 0 0.1'><geom type='box' size='0.1 0.1 0.1' mass='1'/></body>",
                "  </worldbody>",
                "</mujoco>",
            ]
        ),
        encoding="utf-8",
    )
    task_path.write_text(
        "\n".join(
            [
                "name: G1TrackFoam",
                "robot:",
                "  robot_type: g1_29dof_rubberhand-feet_sphere-eef_box-body_capsule",
                "command:",
                "  data_path: data/motion/move_foam",
                "  object_asset_name: foam",
                "  object_body_name: foam",
                "  root_body_name: pelvis",
                "  extra_object_names: [stool_support]",
            ]
        ),
        encoding="utf-8",
    )
    policy_path = export_dir / "policy-test-final.onnx"
    policy_path.write_bytes(b"onnx")
    policy_path.with_suffix(".json").write_text("{}", encoding="utf-8")
    policy_path.with_suffix(".yaml").write_text(
        "\n".join(
            [
                "observation:",
                "  command:",
                "    ref_motion_phase:",
                "      motion_path: data/motion/move_foam",
                "      motion_duration_second: 1.0",
                "      future_steps: [1]",
                "      body_names: [pelvis]",
                "      joint_names: []",
                "      root_body_name: pelvis",
                "policy_joint_names: []",
                "isaac_joint_names: []",
                "default_joint_pos: 0.0",
                "action_scale: 1.0",
            ]
        ),
        encoding="utf-8",
    )

    resolved = module.resolve_wbc_export_inputs(
        wbc_root=wbc_root,
        sim2real_root=sim2real_root,
        task_yaml=task_path,
        policy_model=policy_path,
    )

    assert resolved["object_type"] == "foam_with_support"
    assert resolved["scene_config"]["ROBOT_SCENE"] == str(scene_path)
    assert resolved["publish_object_names"] == ("foam", "stool_support", "pelvis")


def test_load_wbc_reference_initial_state_uses_named_motion_frame(tmp_path):
    module = _load_runner_module()
    motion_dir = tmp_path / "motion"
    motion_dir.mkdir()
    (motion_dir / "meta.json").write_text(
        json.dumps(
            {
                "fps": 50,
                "body_names": ["pelvis", "bread_box", "support0"],
                "joint_names": ["left_hip_pitch_joint", "right_wrist_yaw_joint"],
            }
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        motion_dir / "motion.npz",
        body_pos_w=np.asarray(
            [
                [[0.0, 0.0, 0.7], [1.0, 0.0, 0.2], [2.0, 0.0, 0.1]],
                [[0.1, 0.2, 0.8], [1.1, 0.2, 0.3], [2.1, 0.2, 0.2]],
            ],
            dtype=np.float64,
        ),
        body_quat_w=np.asarray(
            [
                [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
                [[0.5, 0.5, 0.5, 0.5], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            ],
            dtype=np.float64,
        ),
        body_lin_vel_w=np.asarray(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.3, 0.4, 0.5], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=np.float64,
        ),
        body_ang_vel_w=np.asarray(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.6, 0.7, 0.8], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=np.float64,
        ),
        joint_pos=np.asarray([[0.0, 0.0], [0.11, -0.22]], dtype=np.float64),
        joint_vel=np.asarray([[0.0, 0.0], [1.1, -2.2]], dtype=np.float64),
    )
    policy_config = {"observation": {"command": {"ref_motion_phase": {"motion_path": str(motion_dir)}}}}

    state = module.load_wbc_reference_initial_state(
        policy_config=policy_config,
        root_body_name="pelvis",
        object_body_names=["bread_box", "support0", "missing_body"],
        initial_step=1,
    )

    assert state["motion_step"] == 1
    assert state["root_body_name"] == "pelvis"
    np.testing.assert_allclose(state["root_pos"], [0.1, 0.2, 0.8])
    np.testing.assert_allclose(state["root_quat"], [0.5, 0.5, 0.5, 0.5])
    np.testing.assert_allclose(state["root_lin_vel"], [0.3, 0.4, 0.5])
    np.testing.assert_allclose(state["root_ang_vel"], [0.6, 0.7, 0.8])
    assert state["joint_pos_by_name"]["left_hip_pitch_joint"] == 0.11
    assert state["joint_vel_by_name"]["right_wrist_yaw_joint"] == -2.2
    np.testing.assert_allclose(state["body_pose_by_name"]["bread_box"]["pos"], [1.1, 0.2, 0.3])
    np.testing.assert_allclose(state["body_pose_by_name"]["support0"]["quat"], [0.0, 0.0, 1.0, 0.0])
    assert "missing_body" not in state["body_pose_by_name"]


def test_reference_tracker_matches_motion_object_name_to_mujoco_body_alias(tmp_path):
    import mujoco

    module = _load_runner_module()
    model = mujoco.MjModel.from_xml_string(
        "\n".join(
            [
                "<mujoco>",
                "  <worldbody>",
                "    <body name='pelvis' pos='0 0 1'>",
                "      <freejoint name='pelvis_root'/>",
                "      <geom type='sphere' size='0.1' mass='1'/>",
                "    </body>",
                "    <body name='suitcase_body' pos='0 0 0.1'>",
                "      <freejoint name='suitcase_root'/>",
                "      <geom type='box' size='0.1 0.1 0.1' mass='1'/>",
                "    </body>",
                "  </worldbody>",
                "</mujoco>",
            ]
        )
    )
    motion_dir = tmp_path / "motion"
    motion_dir.mkdir()
    (motion_dir / "meta.json").write_text(
        json.dumps(
            {
                "fps": 50,
                "body_names": ["pelvis", "suitcase"],
                "joint_names": [],
            }
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        motion_dir / "motion.npz",
        body_pos_w=np.zeros((2, 2, 3), dtype=np.float64),
        body_quat_w=np.zeros((2, 2, 4), dtype=np.float64),
        body_lin_vel_w=np.zeros((2, 2, 3), dtype=np.float64),
        body_ang_vel_w=np.zeros((2, 2, 3), dtype=np.float64),
        joint_pos=np.zeros((2, 0), dtype=np.float64),
        joint_vel=np.zeros((2, 0), dtype=np.float64),
    )
    policy_config = {"observation": {"command": {"ref_motion_phase": {"motion_path": str(motion_dir)}}}}

    tracker = module.build_wbc_reference_tracker(
        model=model,
        policy_config=policy_config,
        joint_names=[],
        object_body_names=["suitcase"],
    )

    assert "suitcase" in tracker["tracked_body_names"]
    assert tracker["tracked_object_body_names"] == ["suitcase"]
    assert tracker["missing_object_body_names"] == []
    assert len(tracker["object_body_ids"]) == 1
