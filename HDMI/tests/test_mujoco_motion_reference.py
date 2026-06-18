import json

import numpy as np
import pytest
import torch

from active_adaptation.mujoco.motion_reference import MujocoMotionReference


def _write_motion(tmp_path):
    body_names = ["hand_link", "pelvis", "foot_link"]
    joint_names = ["knee_joint", "hip_joint", "ankle_joint"]
    steps = 5
    body_ids = np.arange(len(body_names))[None, :, None]
    joint_ids = np.arange(len(joint_names))[None, :]
    time = np.arange(steps)[:, None, None]
    joint_time = np.arange(steps)[:, None]

    body_pos_w = time * 100.0 + body_ids * 10.0 + np.array([1.0, 2.0, 3.0])
    body_quat_w = np.zeros((steps, len(body_names), 4), dtype=np.float32)
    body_quat_w[..., 0] = 1.0
    body_lin_vel_w = time * 10.0 + body_ids + np.array([0.1, 0.2, 0.3])
    body_ang_vel_w = time * 20.0 + body_ids + np.array([0.4, 0.5, 0.6])
    joint_pos = joint_time * 10.0 + joint_ids + 0.01
    joint_vel = joint_time * 20.0 + joint_ids + 0.02

    np.savez_compressed(
        tmp_path / "motion.npz",
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
    )
    (tmp_path / "meta.json").write_text(
        json.dumps({"body_names": body_names, "joint_names": joint_names, "fps": 50.0})
    )
    return body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w, joint_pos, joint_vel


def test_motion_reference_slices_future_frames_in_exported_name_order(tmp_path):
    body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w, joint_pos, joint_vel = _write_motion(tmp_path)
    reference = MujocoMotionReference.from_motion_dir(
        tmp_path,
        body_names=["pelvis", "hand_link"],
        joint_names=["hip_joint", "knee_joint"],
        root_body_name="pelvis",
        future_steps=[1, 3],
    )

    fields = reference.observation_fields_at(step=2)

    expected_indices = [3, 4]
    assert torch.allclose(
        fields.ref_body_pos_future_w,
        torch.as_tensor(body_pos_w[expected_indices][:, [1, 0]][None], dtype=torch.float32),
    )
    assert torch.allclose(
        fields.ref_root_pos_w,
        torch.as_tensor(body_pos_w[3, 1][None], dtype=torch.float32),
    )
    assert torch.allclose(
        fields.ref_root_quat_w,
        torch.as_tensor(body_quat_w[3, 1][None], dtype=torch.float32),
    )
    assert torch.allclose(
        fields.ref_joint_pos_future,
        torch.as_tensor(joint_pos[expected_indices][:, [1, 0]][None], dtype=torch.float32),
    )
    assert torch.allclose(
        reference.body_lin_vel_w[expected_indices][:, reference.body_indices],
        torch.as_tensor(body_lin_vel_w[expected_indices][:, [1, 0]], dtype=torch.float32),
    )
    assert torch.allclose(
        reference.body_ang_vel_w[expected_indices][:, reference.body_indices],
        torch.as_tensor(body_ang_vel_w[expected_indices][:, [1, 0]], dtype=torch.float32),
    )
    assert torch.allclose(
        reference.joint_vel[expected_indices][:, reference.joint_indices],
        torch.as_tensor(joint_vel[expected_indices][:, [1, 0]], dtype=torch.float32),
    )
    assert torch.equal(fields.motion_t, torch.tensor([2]))
    assert torch.equal(fields.motion_len, torch.tensor([5]))


def test_motion_reference_reports_missing_names(tmp_path):
    _write_motion(tmp_path)

    with pytest.raises(ValueError, match="missing body names.*torso_link"):
        MujocoMotionReference.from_motion_dir(
            tmp_path,
            body_names=["pelvis", "torso_link"],
            joint_names=["hip_joint"],
            root_body_name="pelvis",
            future_steps=[1],
        )
