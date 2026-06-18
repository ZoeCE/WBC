import os
from dataclasses import replace

import mujoco
import active_adaptation.utils.symmetry as symmetry_utils
from active_adaptation.assets_mjcf.types import MJArticulationCfg, MJObjectSpec


PATH = os.path.dirname(__file__)


class RobotRegistry(dict):
    def __init__(self):
        super().__init__()
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return

        from active_adaptation.assets_mjcf.manifest import load_mujoco_asset_manifest

        mjcf_path = os.path.join(PATH, "g1_29dof_nohand", "g1_29dof_nohand.xml")
        manifest = load_mujoco_asset_manifest(mjcf_path)
        self["g1_29dof"] = MJArticulationCfg(
            mjcf_path=mjcf_path,
            init_state={
                "pos": [0.0, 0.0, 0.76],
                "joint_pos": {".*": 0.0},
            },
            actuators={
                "legs": {
                    "stiffness": {
                        ".*_hip_pitch_joint": 100.0,
                        ".*_hip_roll_joint": 100.0,
                        ".*_hip_yaw_joint": 100.0,
                        ".*_knee_joint": 150.0,
                        ".*_ankle_pitch_joint": 40.0,
                        ".*_ankle_roll_joint": 40.0,
                    },
                    "damping": {
                        ".*_hip_pitch_joint": 2.5,
                        ".*_hip_roll_joint": 2.5,
                        ".*_hip_yaw_joint": 2.5,
                        ".*_knee_joint": 4.0,
                        ".*_ankle_pitch_joint": 2.0,
                        ".*_ankle_roll_joint": 2.0,
                    },
                },
                "waist": {
                    "stiffness": {
                        "waist_yaw_joint": 80.0,
                        "waist_roll_joint": 40.0,
                        "waist_pitch_joint": 40.0,
                    },
                    "damping": {
                        "waist_yaw_joint": 2.0,
                        "waist_roll_joint": 2.0,
                        "waist_pitch_joint": 2.0,
                    },
                },
                "arms": {
                    "stiffness": {
                        ".*_shoulder_.*_joint": 40.0,
                        ".*_elbow_joint": 40.0,
                        ".*_wrist_roll_joint": 20.0,
                        ".*_wrist_pitch_joint": 20.0,
                        ".*_wrist_yaw_joint": 20.0,
                    },
                    "damping": {
                        ".*_shoulder_.*_joint": 1.5,
                        ".*_elbow_joint": 1.5,
                        ".*_wrist_roll_joint": 1.0,
                        ".*_wrist_pitch_joint": 1.0,
                        ".*_wrist_yaw_joint": 1.0,
                    },
                },
            },
            body_names_isaac=manifest.body_names,
            joint_names_isaac=manifest.actuated_joint_names,
            joint_symmetry_mapping=symmetry_utils.mirrored({
                "left_hip_pitch_joint": (1, "right_hip_pitch_joint"),
                "left_hip_roll_joint": (-1, "right_hip_roll_joint"),
                "left_hip_yaw_joint": (-1, "right_hip_yaw_joint"),
                "left_knee_joint": (1, "right_knee_joint"),
                "left_ankle_pitch_joint": (1, "right_ankle_pitch_joint"),
                "left_ankle_roll_joint": (-1, "right_ankle_roll_joint"),
                "waist_yaw_joint": (-1, "waist_yaw_joint"),
                "waist_roll_joint": (-1, "waist_roll_joint"),
                "waist_pitch_joint": (1, "waist_pitch_joint"),
                "left_shoulder_pitch_joint": (1, "right_shoulder_pitch_joint"),
                "left_shoulder_roll_joint": (-1, "right_shoulder_roll_joint"),
                "left_shoulder_yaw_joint": (-1, "right_shoulder_yaw_joint"),
                "left_elbow_joint": (1, "right_elbow_joint"),
                "left_wrist_yaw_joint": (-1, "right_wrist_yaw_joint"),
                "left_wrist_roll_joint": (-1, "right_wrist_roll_joint"),
                "left_wrist_pitch_joint": (1, "right_wrist_pitch_joint"),
            }),
            spatial_symmetry_mapping={},
        )
        self._loaded = True

    def with_object(self, robot_name: str, object_asset_name: str, object_type: str | None = None):
        self._ensure_loaded()
        object_type = object_type or object_asset_name
        cache_key = f"{robot_name}-{object_type}"
        if cache_key in self:
            return super().__getitem__(cache_key)

        base_cfg = super().__getitem__(robot_name)
        from active_adaptation.assets_mjcf.manifest import load_mujoco_asset_manifest

        mjcf_path = os.path.join(PATH, "g1_29dof_nohand", f"g1_29dof_nohand-{object_type}.xml")
        if not os.path.exists(mjcf_path):
            raise KeyError(f"No MuJoCo object scene for {robot_name=} and {object_type=}: {mjcf_path}")

        base_manifest = load_mujoco_asset_manifest(base_cfg.mjcf_path)
        model = mujoco.MjModel.from_xml_path(mjcf_path)
        object_specs = _build_object_specs(
            model=model,
            robot_body_names=base_manifest.body_names,
            robot_joint_names=base_manifest.tracking_joint_names,
        )
        if object_asset_name not in object_specs:
            raise KeyError(
                f"Object asset {object_asset_name!r} is not a top-level object body in {mjcf_path}. "
                f"Available objects: {sorted(object_specs)}"
            )

        cfg = replace(base_cfg, mjcf_path=mjcf_path, object_specs=object_specs)
        super().__setitem__(cache_key, cfg)
        return cfg

    def __getitem__(self, key):
        self._ensure_loaded()
        return super().__getitem__(key)

    def keys(self):
        self._ensure_loaded()
        return super().keys()


def _build_object_specs(model, robot_body_names, robot_joint_names):
    robot_body_names = set(robot_body_names)
    robot_joint_names = set(robot_joint_names)
    specs: dict[str, MJObjectSpec] = {}

    for body_id in range(1, model.nbody):
        body_name = model.body(body_id).name
        if body_name in robot_body_names:
            continue

        parent_id = int(model.body_parentid[body_id])
        parent_name = model.body(parent_id).name if parent_id > 0 else ""
        if parent_id != 0 and parent_name not in robot_body_names:
            continue

        subtree_ids = _body_subtree_ids(model, body_id)
        body_names = tuple(model.body(i).name for i in subtree_ids)
        subtree_id_set = set(subtree_ids)
        joint_names = []
        for joint_id in range(model.njnt):
            joint = model.joint(joint_id)
            joint_name = joint.name
            if joint_name in robot_joint_names or joint.type == mujoco.mjtJoint.mjJNT_FREE:
                continue
            if int(model.jnt_bodyid[joint_id]) in subtree_id_set:
                joint_names.append(joint_name)

        specs[body_name] = MJObjectSpec(
            asset_name=body_name,
            body_names=body_names,
            joint_names=tuple(joint_names),
        )

    return specs


def _body_subtree_ids(model, root_body_id: int):
    ids = []
    stack = [root_body_id]
    while stack:
        body_id = stack.pop()
        ids.append(body_id)
        children = [
            i for i in range(1, model.nbody)
            if int(model.body_parentid[i]) == body_id
        ]
        stack.extend(reversed(children))
    return ids


ROBOTS = RobotRegistry()
