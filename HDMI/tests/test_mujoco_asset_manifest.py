import json
from pathlib import Path

import pytest

from active_adaptation.assets_mjcf.manifest import (
    build_name_index,
    load_mujoco_asset_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
DOOR_SCENE = ROOT / "active_adaptation/assets_mjcf/g1_29dof_nohand/g1_29dof_nohand-door.xml"
DOOR_MOTION_META = ROOT / "data/motion/data_for_sim/traverse_door/meta.json"


def _load_door_motion_meta():
    with DOOR_MOTION_META.open("r") as f:
        return json.load(f)


def test_door_scene_manifest_covers_motion_body_and_joint_names():
    motion_meta = _load_door_motion_meta()
    manifest = load_mujoco_asset_manifest(DOOR_SCENE)

    body_index = build_name_index(
        required_names=motion_meta["body_names"],
        available_names=manifest.body_names,
        label="body",
    )
    joint_index = build_name_index(
        required_names=motion_meta["joint_names"],
        available_names=manifest.tracking_joint_names,
        label="joint",
    )

    assert [manifest.body_names[i] for i in body_index] == motion_meta["body_names"]
    assert [manifest.tracking_joint_names[i] for i in joint_index] == motion_meta["joint_names"]


def test_door_scene_manifest_exposes_ordered_qpos_and_qvel_addresses():
    motion_meta = _load_door_motion_meta()
    manifest = load_mujoco_asset_manifest(DOOR_SCENE)

    qposadr = manifest.qpos_addresses_for(motion_meta["joint_names"])
    qveladr = manifest.qvel_addresses_for(motion_meta["joint_names"])

    assert len(qposadr) == len(motion_meta["joint_names"])
    assert len(qveladr) == len(motion_meta["joint_names"])
    assert qposadr[0] == manifest.joint_qposadr["left_hip_pitch_joint"]
    assert qveladr[-1] == manifest.joint_qveladr["door_joint"]


def test_door_scene_separates_tracking_joints_from_actuated_policy_joints():
    manifest = load_mujoco_asset_manifest(DOOR_SCENE)

    assert "door_joint" in manifest.tracking_joint_names
    assert "door_joint" not in manifest.actuated_joint_names
    assert len(manifest.actuated_joint_names) == manifest.nu


def test_build_name_index_rejects_missing_and_duplicate_names():
    with pytest.raises(ValueError, match="missing body names"):
        build_name_index(["pelvis", "missing_link"], ["pelvis"], label="body")

    with pytest.raises(ValueError, match="duplicate joint names"):
        build_name_index(
            required_names=["left_hip_pitch_joint"],
            available_names=["left_hip_pitch_joint", "left_hip_pitch_joint"],
            label="joint",
        )
