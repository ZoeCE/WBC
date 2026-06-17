import os

import active_adaptation.utils.symmetry as symmetry_utils
from active_adaptation.assets_mjcf.types import MJArticulationCfg


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

    def __getitem__(self, key):
        self._ensure_loaded()
        return super().__getitem__(key)

    def keys(self):
        self._ensure_loaded()
        return super().keys()


ROBOTS = RobotRegistry()
