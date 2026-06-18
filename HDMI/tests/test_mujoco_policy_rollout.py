import json
import importlib
from pathlib import Path

import numpy as np
import torch
import yaml
from tensordict.nn import TensorDictModule

from active_adaptation.mujoco import MujocoMotionReference
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
    joint_pos = np.array([[0.10], [0.30]], dtype=np.float32)
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
