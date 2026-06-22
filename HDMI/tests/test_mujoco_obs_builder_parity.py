import importlib.util
import json
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_cli_module():
    script_path = ROOT / "scripts/mujoco_obs_builder_parity.py"
    spec = importlib.util.spec_from_file_location("mujoco_obs_builder_parity_script", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_trace(path: Path, steps):
    path.write_text(json.dumps({"steps": steps}))
    return path


def test_obs_builder_parity_passes_matching_component_traces():
    module = _load_cli_module()
    trace = [
        {
            "step": 0,
            "groups": {
                "policy": {
                    "joint_pos_history": [[0.0, 1.0]],
                    "prev_actions": [[0.25, -0.25]],
                },
                "command": {"ref_motion_phase": [[0.1]]},
            },
        },
        {
            "step": 1,
            "groups": {
                "policy": {
                    "joint_pos_history": [[0.5, 1.5]],
                    "prev_actions": [[0.0, 0.0]],
                },
                "command": {"ref_motion_phase": [[0.2]]},
            },
        },
    ]

    report = module.compare_observation_component_traces(trace, trace, max_abs=1e-6)

    assert report["gate_passed"] is True
    assert report["failure_count"] == 0
    assert report["step_count"] == 2
    assert report["compared_component_count"] == 6
    assert report["max_abs"] == 0.0
    assert report["groups"]["policy"]["components"]["joint_pos_history"]["steps_compared"] == 2


def test_obs_builder_parity_reports_component_delta():
    module = _load_cli_module()
    env_trace = [
        {
            "step": 0,
            "groups": {
                "policy": {
                    "joint_pos_history": [[0.0, 1.0]],
                    "prev_actions": [[0.0]],
                }
            },
        }
    ]
    builder_trace = [
        {
            "step": 0,
            "groups": {
                "policy": {
                    "joint_pos_history": [[0.0, 1.2]],
                    "prev_actions": [[0.0]],
                }
            },
        }
    ]

    report = module.compare_observation_component_traces(env_trace, builder_trace, max_abs=0.05)

    assert report["gate_passed"] is False
    assert report["failure_count"] == 1
    assert report["max_abs"] == pytest.approx(0.2)
    failure = report["failures"][0]
    assert failure["step"] == 0
    assert failure["group"] == "policy"
    assert failure["component"] == "joint_pos_history"
    assert failure["reason"] == "max_abs_exceeded"
    assert report["groups"]["policy"]["components"]["joint_pos_history"]["mean_abs"] == pytest.approx(0.1)


def test_obs_builder_parity_cli_writes_report_and_fails_required_gate(tmp_path, capsys):
    module = _load_cli_module()
    env_path = _write_trace(
        tmp_path / "env.json",
        [{"step": 0, "groups": {"object": {"object_xy_b": [[1.0, 2.0]]}}}],
    )
    builder_path = _write_trace(
        tmp_path / "builder.json",
        [{"step": 0, "groups": {"object": {"object_xy_b": [[1.0, 2.5]]}}}],
    )
    output_path = tmp_path / "report.json"

    exit_code = module.main(
        [
            "--env-trace-json",
            str(env_path),
            "--builder-trace-json",
            str(builder_path),
            "--max-abs",
            "0.01",
            "--require-pass",
            "--output",
            str(output_path),
        ]
    )
    stdout_report = json.loads(capsys.readouterr().out)
    file_report = json.loads(output_path.read_text())

    assert exit_code == 1
    assert stdout_report == file_report
    assert stdout_report["mode"] == "offline_trace"
    assert stdout_report["gate_passed"] is False
    assert stdout_report["failures"][0]["component"] == "object_xy_b"


def test_live_policy_state_uses_env_joint_history_buffer_when_order_matches():
    module = _load_cli_module()
    joint_history = SimpleNamespace(
        joint_names=["j0", "j1"],
        joint_ids=torch.tensor([0, 1]),
        buffer=torch.tensor([[[1.5, 2.5], [1.0, 2.0]]]),
        joint_pos_offset=torch.tensor([[0.1, 0.2]]),
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=torch.tensor([[10.0, 20.0]]),
            root_ang_vel_b=torch.zeros(1, 3),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
            root_link_pos_w=torch.zeros(1, 3),
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        )
    )
    base_env = SimpleNamespace(
        scene={"robot": robot},
        observation_funcs={"policy": SimpleNamespace(funcs={"joint_pos_history": joint_history})},
        command_manager=SimpleNamespace(),
        action_manager=SimpleNamespace(applied_action=None, action_buf=None),
    )
    bundle = SimpleNamespace(
        observation_builder=SimpleNamespace(joint_pos_names=["j0", "j1"]),
        observation_default_joint_pos=torch.tensor([9.0, 9.0]),
    )

    state = module._policy_state_from_env(
        base_env=base_env,
        bundle=bundle,
        observation_joint_ids=[0, 1],
    )

    assert torch.allclose(state.joint_pos, torch.tensor([[1.5, 2.5]]))
    assert torch.allclose(state.joint_pos_offset, torch.tensor([[0.1, 0.2]]))


def test_live_policy_state_reorders_env_joint_history_buffer_by_builder_names():
    module = _load_cli_module()
    joint_history = SimpleNamespace(
        joint_names=["j0", "j1"],
        joint_ids=torch.tensor([0, 1]),
        buffer=torch.tensor([[[1.5, 2.5], [1.0, 2.0]]]),
        joint_pos_offset=torch.tensor([[0.1, 0.2]]),
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=torch.tensor([[10.0, 20.0]]),
            root_ang_vel_b=torch.zeros(1, 3),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
            root_link_pos_w=torch.zeros(1, 3),
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        )
    )
    base_env = SimpleNamespace(
        scene={"robot": robot},
        observation_funcs={"policy": SimpleNamespace(funcs={"joint_pos_history": joint_history})},
        command_manager=SimpleNamespace(),
        action_manager=SimpleNamespace(applied_action=None, action_buf=None),
    )
    bundle = SimpleNamespace(
        observation_builder=SimpleNamespace(joint_pos_names=["j1", "j0"]),
        observation_default_joint_pos=torch.tensor([9.0, 9.0]),
    )

    state = module._policy_state_from_env(
        base_env=base_env,
        bundle=bundle,
        observation_joint_ids=[1, 0],
    )

    assert torch.allclose(state.joint_pos, torch.tensor([[2.5, 1.5]]))
    assert torch.allclose(state.joint_pos_offset, torch.tensor([[0.2, 0.1]]))


def test_live_policy_state_carries_reference_future_root_fields():
    module = _load_cli_module()
    ref_root_pos_future_w = torch.tensor([[[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]]])
    ref_root_quat_future_w = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [0.99, 0.0, 0.0, 0.01]]]
    )
    tracking_body_pos_w = torch.ones(1, 2, 3)
    tracking_body_quat_w = torch.ones(1, 2, 4)
    ref_body_quat_future_w = torch.ones(1, 2, 2, 4)
    ref_object_pos_future_w = torch.ones(1, 2, 3)
    applied_torque = torch.tensor([[0.5, 0.6]])
    ref_joint_pos_action = torch.tensor([[1.0, 1.0]])
    object_joint_pos = torch.tensor([0.1])
    object_joint_vel = torch.tensor([0.2])
    object_joint_torque = torch.tensor([[0.3]])
    object_view = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.tensor([[0.0, 0.0, 1.0]]),
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            joint_pos=object_joint_pos,
            joint_vel=object_joint_vel,
            applied_torque=object_joint_torque,
        )
    )
    robot = SimpleNamespace(
        body_names=["b0", "b1"],
        data=SimpleNamespace(
            joint_pos=torch.tensor([[10.0, 20.0]]),
            root_ang_vel_b=torch.zeros(1, 3),
            root_lin_vel_b=torch.zeros(1, 3),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
            root_link_pos_w=torch.zeros(1, 3),
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            applied_torque=applied_torque,
            body_pos_w=torch.zeros(1, 2, 3),
            body_quat_w=torch.zeros(1, 2, 4),
            body_lin_vel_w=torch.zeros(1, 2, 3),
            body_ang_vel_w=torch.zeros(1, 2, 3),
        )
    )
    base_env = SimpleNamespace(
        scene={"robot": robot},
        observation_funcs={"policy": SimpleNamespace(funcs={})},
        command_manager=SimpleNamespace(
            object=object_view,
            object_joint_pos=object_joint_pos,
            object_joint_vel=object_joint_vel,
            robot_body_pos_w=tracking_body_pos_w,
            robot_body_quat_w=tracking_body_quat_w,
            ref_root_pos_future_w=ref_root_pos_future_w,
            ref_root_quat_future_w=ref_root_quat_future_w,
            ref_body_quat_future_w=ref_body_quat_future_w,
            ref_object_pos_future_w=ref_object_pos_future_w,
            dataset=SimpleNamespace(joint_names=["unused_joint", "j1", "j0"]),
            current_ref_motion=SimpleNamespace(
                joint_pos=torch.tensor([[99.0, 2.4, 0.7]])
            ),
        ),
        action_manager=SimpleNamespace(applied_action=None, action_buf=None),
    )
    bundle = SimpleNamespace(
        policy_joint_names=["j0", "j1"],
        observation_builder=SimpleNamespace(joint_pos_names=["j0", "j1"]),
        action_scale=torch.tensor([0.5, 2.0]),
        default_joint_pos=torch.tensor([0.2, 0.4]),
        observation_default_joint_pos=torch.tensor([9.0, 9.0]),
    )

    state = module._policy_state_from_env(
        base_env=base_env,
        bundle=bundle,
        observation_joint_ids=[0, 1],
    )

    assert state.ref_root_pos_future_w is ref_root_pos_future_w
    assert state.ref_root_quat_future_w is ref_root_quat_future_w
    assert state.tracking_body_pos_w is tracking_body_pos_w
    assert state.tracking_body_quat_w is tracking_body_quat_w
    assert state.ref_body_quat_future_w is ref_body_quat_future_w
    assert state.ref_object_pos_future_w is ref_object_pos_future_w
    assert state.body_names == ["b0", "b1"]
    assert state.applied_torque is applied_torque
    assert torch.allclose(state.ref_joint_pos_action, ref_joint_pos_action)
    assert torch.allclose(state.object_joint_pos, object_joint_pos.unsqueeze(1))
    assert torch.allclose(state.object_joint_vel, object_joint_vel.unsqueeze(1))
    assert state.object_joint_torque is object_joint_torque


def test_env_component_groups_reorders_joint_pos_history_by_builder_names():
    module = _load_cli_module()
    joint_history = SimpleNamespace(
        joint_names=["j0", "j1"],
        history_steps=[0, 1],
        compute=lambda: torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
    )
    base_env = SimpleNamespace(
        observation_funcs={
            "policy": SimpleNamespace(funcs={"joint_pos_history": joint_history})
        }
    )
    bundle = SimpleNamespace(
        observation_builder=SimpleNamespace(joint_pos_names=["j1", "j0"])
    )

    groups = module._env_component_groups(
        base_env=base_env,
        groups=["policy"],
        bundle=bundle,
    )
