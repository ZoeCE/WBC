from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import mujoco


@dataclass(frozen=True)
class MujocoAssetManifest:
    mjcf_path: Path
    nq: int
    nv: int
    nu: int
    body_names: list[str]
    tracking_joint_names: list[str]
    actuated_joint_names: list[str]
    joint_qposadr: dict[str, int]
    joint_qveladr: dict[str, int]
    actuator_id: dict[str, int]

    def qpos_addresses_for(self, joint_names: Sequence[str]) -> list[int]:
        joint_index = build_name_index(joint_names, self.tracking_joint_names, label="joint")
        return [self.joint_qposadr[self.tracking_joint_names[i]] for i in joint_index]

    def qvel_addresses_for(self, joint_names: Sequence[str]) -> list[int]:
        joint_index = build_name_index(joint_names, self.tracking_joint_names, label="joint")
        return [self.joint_qveladr[self.tracking_joint_names[i]] for i in joint_index]


def load_mujoco_asset_manifest(mjcf_path: str | Path) -> MujocoAssetManifest:
    path = Path(mjcf_path)
    model = mujoco.MjModel.from_xml_path(str(path))

    body_names = [model.body(i).name for i in range(1, model.nbody)]
    tracking_joint_names: list[str] = []
    joint_qposadr: dict[str, int] = {}
    joint_qveladr: dict[str, int] = {}

    for joint_id in range(model.njnt):
        joint = model.joint(joint_id)
        if joint.type == mujoco.mjtJoint.mjJNT_FREE:
            continue
        tracking_joint_names.append(joint.name)
        joint_qposadr[joint.name] = int(model.jnt_qposadr[joint_id])
        joint_qveladr[joint.name] = int(model.jnt_dofadr[joint_id])

    actuated_joint_names: list[str] = []
    actuator_id: dict[str, int] = {}
    for act_id in range(model.nu):
        actuator = model.actuator(act_id)
        if actuator.trntype != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        joint_id = int(actuator.trnid[0])
        joint_name = model.joint(joint_id).name
        actuated_joint_names.append(joint_name)
        actuator_id[joint_name] = act_id

    return MujocoAssetManifest(
        mjcf_path=path,
        nq=model.nq,
        nv=model.nv,
        nu=model.nu,
        body_names=body_names,
        tracking_joint_names=tracking_joint_names,
        actuated_joint_names=actuated_joint_names,
        joint_qposadr=joint_qposadr,
        joint_qveladr=joint_qveladr,
        actuator_id=actuator_id,
    )


def build_name_index(required_names: Sequence[str], available_names: Sequence[str], label: str) -> list[int]:
    duplicate_required = _duplicates(required_names)
    duplicate_available = _duplicates(available_names)
    if duplicate_required or duplicate_available:
        duplicates = duplicate_required or duplicate_available
        raise ValueError(f"duplicate {label} names: {duplicates}")

    available_index = {name: i for i, name in enumerate(available_names)}
    missing = [name for name in required_names if name not in available_index]
    if missing:
        raise ValueError(f"missing {label} names: {missing}")

    return [available_index[name] for name in required_names]


def _duplicates(names: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in names:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    return duplicates
