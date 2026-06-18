import torch
import numpy as np
import mujoco
import mujoco.viewer
import time
import warnings
from pathlib import Path
from typing import Sequence, Union, Any, Dict
from dataclasses import dataclass, replace
from types import SimpleNamespace

from isaaclab.utils import string as string_utils
from scipy.spatial.transform import Rotation as sRot

from active_adaptation.assets_mjcf.types import MJArticulationCfg, MJObjectSpec
from tensordict import TensorClass


ArrayType = Union[np.ndarray, torch.Tensor]


def quat_rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate xyz vectors by wxyz quaternions without importing Isaac/Omni."""
    xyz = quat[..., 1:]
    t = torch.linalg.cross(xyz, vec, dim=-1) * 2
    return vec + quat[..., 0:1] * t + torch.linalg.cross(xyz, t, dim=-1)


def quat_rotate_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate xyz vectors by inverse wxyz quaternions without importing Isaac/Omni."""
    xyz = quat[..., 1:]
    t = torch.linalg.cross(xyz, vec, dim=-1) * 2
    return vec - quat[..., 0:1] * t + torch.linalg.cross(xyz, t, dim=-1)


@dataclass
class MJArticulationData:
    default_joint_pos: ArrayType
    default_joint_vel: ArrayType
    default_root_state: ArrayType
    default_mass: ArrayType
    default_inertia: ArrayType
    
    joint_stiffness: ArrayType = None
    joint_damping: ArrayType = None

    body_pos_w: ArrayType = None
    body_quat_w: ArrayType = None
    
    joint_pos: ArrayType = None
    joint_pos_target: ArrayType = None
    
    joint_vel: ArrayType = None
    joint_vel_target: ArrayType = None

    applied_torque: ArrayType = None
    joint_effort_limits: ArrayType = None
    projected_gravity_b: ArrayType = None
    
    body_vel_w: ArrayType = None
    # body_lin_vel_w: ArrayType = None
    # body_ang_vel_w: ArrayType = None
    root_lin_vel_w: ArrayType = None
    root_ang_vel_w: ArrayType = None
    root_ang_vel_b: ArrayType = None
    root_lin_vel_b: ArrayType = None
    heading_w: ArrayType = None
    soft_joint_pos_limits: ArrayType = None
    joint_pos_limits: ArrayType = None
    joint_vel_limits: ArrayType = None
    soft_joint_vel_limits: ArrayType = None

    @property
    def body_lin_vel_w(self):
        return self.body_vel_w[..., 3:]
    
    @property
    def body_ang_vel_w(self):
        return self.body_vel_w[..., :3]

    @property
    def root_pos_w(self):
        return self.body_pos_w[..., 0, :]
    
    @property
    def root_quat_w(self):
        return self.body_quat_w[..., 0, :]

    @property
    def root_link_pos_w(self):
        return self.root_pos_w

    @property
    def root_link_quat_w(self):
        return self.root_quat_w

    @property
    def body_link_pos_w(self):
        return self.body_pos_w

    @property
    def body_link_quat_w(self):
        return self.body_quat_w

    @property
    def body_com_lin_vel_w(self):
        return self.body_lin_vel_w

    @property
    def body_com_ang_vel_w(self):
        return self.body_ang_vel_w
    
    # @property
    # def root_lin_vel_w(self):
    #     return self.body_vel_w[..., 0, :3]
    
    # @property
    # def root_ang_vel_w(self):
    #     return self.body_vel_w[..., 0, 3:]
    
    @property
    def root_state_w(self):
        return torch.cat([self.body_pos_w[:, 0, :], self.body_quat_w[:, 0, :]], dim=-1)


class MJPhysicsView:
    def __init__(self, articulation: "MJArticulation"):
        self.articulation = articulation


class MJArticulation:

    is_fixed_base = False

    def __init__(self, cfg: MJArticulationCfg, num_envs: int = 1):
        if num_envs < 1:
            raise ValueError(f"num_envs must be positive, got {num_envs}.")

        self.cfg = cfg
        self.num_instances = int(num_envs)
        self.mj_models = [mujoco.MjModel.from_xml_path(cfg.mjcf_path) for _ in range(self.num_instances)]
        self.mj_model = self.mj_models[0]
        self.mj_datas = [mujoco.MjData(model) for model in self.mj_models]
        self.mj_data = self.mj_datas[0]
        
        self.body_names_isaac = list(cfg.body_names_isaac)
        self.body_names_mjc = []
        body_adrs = []
        for i in range(1, self.mj_model.nbody): # skip the world body
            body = self.mj_model.body(i)
            self.body_names_mjc.append(body.name)
            body_adrs.append(i)
        
        object_body_names = {
            body_name
            for object_spec in self.cfg.object_specs.values()
            for body_name in object_spec.body_names
        }
        missing_isaac_bodies = set(self.body_names_isaac) - set(self.body_names_mjc)
        unexpected_mjc_bodies = set(self.body_names_mjc) - set(self.body_names_isaac) - object_body_names
        if missing_isaac_bodies or unexpected_mjc_bodies:
            warnings.warn(
                f"Isaac body names do not match mujoco body names:\n"
                f"Isaac - Mujoco: {missing_isaac_bodies}\n"
                f"Mujoco - Isaac: {unexpected_mjc_bodies}\n",
                category=UserWarning
            )

        # find only the actuated joints
        self.joint_names_isaac = list(cfg.joint_names_isaac)
        self.joint_names_mjc = []

        joint_mj_ids = []
        joint_qposadr = []
        joint_qveladr = []
        for i in range(self.mj_model.nu):
            actuator = self.mj_model.actuator(i)
            if actuator.trntype == mujoco.mjtTrn.mjTRN_JOINT:
                joint_id = actuator.trnid[0]
                joint = self.mj_model.joint(actuator.trnid[0])
                self.joint_names_mjc.append(joint.name)
                joint_mj_ids.append(joint_id)
                joint_qposadr.append(self.mj_model.jnt_qposadr[joint_id])
                joint_qveladr.append(self.mj_model.jnt_dofadr[joint_id])
        
        object_joint_names = {
            joint_name
            for object_spec in self.cfg.object_specs.values()
            for joint_name in object_spec.joint_names
        }
        missing_isaac_joints = set(self.joint_names_isaac) - set(self.joint_names_mjc)
        unexpected_mjc_joints = set(self.joint_names_mjc) - set(self.joint_names_isaac) - object_joint_names
        if missing_isaac_joints or unexpected_mjc_joints:
            warnings.warn(
                f"Isaac joint names do not match mujoco joint names:\n"
                f"Isaac - Mujoco: {missing_isaac_joints}\n"
                f"Mujoco - Isaac: {unexpected_mjc_joints}\n",
                category=UserWarning
            )
        
        # Isaac assets may have less joints/bodies due to asset simplification
        self._jnt_isaac2mjc = [self.joint_names_isaac.index(joint_name) for joint_name in self.joint_names_mjc if joint_name in self.joint_names_isaac]
        self._jnt_mjc2isaac = [self.joint_names_mjc.index(joint_name) for joint_name in self.joint_names_isaac]
        self._body_isaac2mjc = [self.body_names_isaac.index(body_name) for body_name in self.body_names_mjc if body_name in self.body_names_isaac]
        self._body_mjc2isaac = [self.body_names_mjc.index(body_name) for body_name in self.body_names_isaac]
        
        self.body_adrs = np.array(body_adrs)
        self.joint_mj_ids = np.array(joint_mj_ids)
        self.joint_qposadr = np.array(joint_qposadr)
        self.joint_qveladr = np.array(joint_qveladr)
        
        # read/write mujoco data in isaac order
        self.body_adrs_read = self.body_adrs[self._body_mjc2isaac]
        self.body_adrs_write = self.body_adrs[self._body_isaac2mjc]
        self.joint_qposadr_read = self.joint_qposadr[self._jnt_mjc2isaac]
        self.joint_qveladr_read = self.joint_qveladr[self._jnt_mjc2isaac]
        self.joint_mj_ids_read = self.joint_mj_ids[self._jnt_mjc2isaac]
        self.joint_qposadr_write = self.joint_qposadr[self._jnt_isaac2mjc]
        self.joint_qveladr_write = self.joint_qveladr[self._jnt_isaac2mjc]
        self.root_qposadr, self.root_qveladr = self._resolve_root_free_joint_addresses()
        self.geom_adrs_by_body = []
        geom_adrs = []
        for body_adr in self.body_adrs_read:
            body_geom_adrs = np.array([
                geom_id for geom_id in range(self.mj_model.ngeom)
                if int(self.mj_model.geom_bodyid[geom_id]) == int(body_adr)
            ], dtype=np.int64)
            geom_adrs.extend(body_geom_adrs.tolist())
            self.geom_adrs_by_body.append(body_geom_adrs)
        self.geom_adrs = np.array(geom_adrs, dtype=np.int64)

        joint_ids, joint_names, joint_pos = string_utils.resolve_matching_names_values(self.cfg.init_state["joint_pos"], self.joint_names_isaac)
        if len(joint_names) < len(self.joint_names_isaac):
            print(f"Missing joint names: {set(self.joint_names_isaac) - set(joint_names)}")
        default_joint_pos = torch.zeros(self.num_joints)
        default_joint_pos[joint_ids] = torch.as_tensor(joint_pos)
        for jname, jpos in zip(self.joint_names, default_joint_pos, strict=True):
            print(jname, jpos)
        default_joint_vel = torch.zeros(self.num_joints)

        joint_stiffness = torch.zeros(self.num_joints)
        joint_damping = torch.zeros(self.num_joints)
        joint_pos_limits = _joint_pos_limits(self.mj_model, self.joint_mj_ids_read)
        joint_vel_limits = torch.full((self.num_joints,), float("inf"), dtype=torch.float32)
        joint_effort_limits = _joint_effort_limits(self.mj_model, self.joint_mj_ids_read)
        
        for actuator_name, actuator_cfg in self.cfg.actuators.items():
            ids, _, values = string_utils.resolve_matching_names_values(actuator_cfg["stiffness"], self.joint_names_isaac)
            joint_stiffness[ids] = torch.as_tensor(values)
            ids, _, values = string_utils.resolve_matching_names_values(actuator_cfg["damping"], self.joint_names_isaac)
            joint_damping[ids] = torch.as_tensor(values)

        diag_inertia = torch.as_tensor(self.mj_model.body_inertia[self.body_adrs], dtype=torch.float32)
        default_root_state = torch.tensor(
            [[*cfg.init_state["pos"], 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
            dtype=torch.float32,
        )
        self._data = MJArticulationData(
            default_joint_pos=default_joint_pos.expand(self.num_instances, -1).clone(),
            default_joint_vel=default_joint_vel.expand(self.num_instances, -1).clone(),
            default_root_state=default_root_state.expand(self.num_instances, -1).clone(),
            default_mass=torch.as_tensor(self.mj_model.body_mass[self.body_adrs], dtype=torch.float32).expand(self.num_instances, -1).clone(),
            default_inertia=diag_inertia.expand(self.num_instances, -1, -1).clone(),
            joint_stiffness=joint_stiffness.expand(self.num_instances, -1).clone(),
            joint_damping=joint_damping.expand(self.num_instances, -1).clone(),
            applied_torque=torch.zeros(self.num_instances, self.num_joints),
            joint_effort_limits=joint_effort_limits.expand(self.num_instances, -1).clone(),
            soft_joint_pos_limits=joint_pos_limits.expand(self.num_instances, -1, -1).clone(),
            joint_pos_limits=joint_pos_limits.expand(self.num_instances, -1, -1).clone(),
            joint_vel_limits=joint_vel_limits.expand(self.num_instances, -1).clone(),
            soft_joint_vel_limits=joint_vel_limits.expand(self.num_instances, -1).clone(),
        )
        self._data.joint_pos_target = self._data.default_joint_pos.clone()
        self._data.joint_vel_target = self._data.default_joint_vel.clone()

        self._external_force_b = torch.zeros(self.num_instances, self.num_bodies, 3)
        self._external_torque_b = torch.zeros(self.num_instances, self.num_bodies, 3)
        self.has_external_wrench = False

        self.timestamp = 0.
        self.root_physx_view = MJRootPhysxView(self)

        for model, data in zip(self.mj_models, self.mj_datas, strict=True):
            mujoco.mj_forward(model, data)
        self.update(0.0)
    
    @property
    def joint_names(self):
        return self.joint_names_isaac
    
    @property
    def body_names(self):
        return self.body_names_isaac

    @property
    def num_joints(self):
        return len(self.joint_names)
    
    @property
    def num_bodies(self):
        return len(self.body_names)
    
    @property
    def data(self):
        return self._data

    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        """Find bodies in the articulation based on the name keys.

        Please check the :meth:`omni.isaac.lab.utils.string_utils.resolve_matching_names` function for more
        information on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the body indices and names.
        """
        return string_utils.resolve_matching_names(name_keys, self.body_names_isaac, preserve_order)

    def find_joints(
        self, name_keys: str | Sequence[str], joint_subset: list[str] | None = None, preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        """Find joints in the articulation based on the name keys.

        Please see the :func:`omni.isaac.lab.utils.string.resolve_matching_names` function for more information
        on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the joint names.
            joint_subset: A subset of joints to search for. Defaults to None, which means all joints
                in the articulation are searched.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the joint indices and names.
        """
        if joint_subset is None:
            joint_subset = self.joint_names_isaac
        # find joints
        return string_utils.resolve_matching_names(name_keys, joint_subset, preserve_order)

    def update(self, dt: float):
        jpos = []
        jvel = []
        body_pos_w = []
        body_vel_w = []
        body_quat_w = []
        projected_gravity_b = []
        heading_w = []

        for data in self.mj_datas:
            jpos.append(np.asarray(data.qpos[self.joint_qposadr_read], dtype=np.float32).copy())
            jvel.append(np.asarray(data.qvel[self.joint_qveladr_read], dtype=np.float32).copy())
            body_pos_w.append(np.asarray(data.xpos[self.body_adrs_read], dtype=np.float32).copy())
            body_vel_w.append(np.asarray(data.cvel[self.body_adrs_read], dtype=np.float32).copy())
            body_quat_w_env = np.asarray(data.xquat[self.body_adrs_read], dtype=np.float32).copy()
            body_quat_w.append(body_quat_w_env)

            rot = sRot.from_quat(body_quat_w_env[0], scalar_first=True)
            projected_gravity_b.append(rot.inv().apply(np.array([0.0, 0.0, -1.0], dtype=np.float32)))
            heading_w.append(rot.as_euler("xyz", degrees=False)[2])

        self._data = replace(
            self._data,
            body_pos_w=torch.as_tensor(np.stack(body_pos_w), dtype=torch.float32),
            body_quat_w=torch.as_tensor(np.stack(body_quat_w), dtype=torch.float32),
            body_vel_w=torch.as_tensor(np.stack(body_vel_w), dtype=torch.float32),
            joint_pos=torch.as_tensor(np.stack(jpos), dtype=torch.float32),
            joint_pos_target=self._data.joint_pos_target.clone(),
            joint_vel=torch.as_tensor(np.stack(jvel), dtype=torch.float32),
            joint_vel_target=self._data.joint_vel_target.clone(),
            projected_gravity_b=torch.as_tensor(np.stack(projected_gravity_b), dtype=torch.float32),
            heading_w=torch.as_tensor(heading_w, dtype=torch.float32),
        )
        self._data.root_lin_vel_w = self._data.body_lin_vel_w[:, 0]
        self._data.root_ang_vel_w = self._data.body_ang_vel_w[:, 0]
        self._data.root_ang_vel_b = quat_rotate_inverse(self._data.root_quat_w, self._data.root_ang_vel_w)
        self._data.root_lin_vel_b = quat_rotate_inverse(self._data.root_quat_w, self._data.root_lin_vel_w)

        if hasattr(self, "_log_path"):
            self._log_states.append(self._data)

    def write_root_state_to_sim(self, root_state: ArrayType, env_ids: ArrayType = None):
        env_ids_t = self._normalize_env_ids(env_ids)
        root_state_t = self._rows_for_envs(root_state, env_ids_t)
        root_qposadr, root_qveladr = self._require_root_free_joint_addresses()

        for row, env_id in enumerate(env_ids_t.tolist()):
            data = self.mj_datas[env_id]
            root_np = root_state_t[row].detach().cpu().numpy()
            data.qpos[root_qposadr:root_qposadr + 3] = root_np[:3]
            data.qpos[root_qposadr + 3:root_qposadr + 7] = root_np[3:7]
            data.qvel[root_qveladr:root_qveladr + 6] = 0.0
            self._write_joint_state_to_data(
                data,
                joint_pos=self._data.default_joint_pos[env_id],
                joint_vel=self._data.default_joint_vel[env_id],
                joint_ids=slice(None),
            )
            mujoco.mj_forward(self.mj_models[env_id], data)
        self.update(0.0)

    def write_root_link_pose_to_sim(self, root_pose: ArrayType, env_ids: ArrayType = None):
        env_ids_t = self._normalize_env_ids(env_ids)
        root_pose_t = self._rows_for_envs(root_pose, env_ids_t)
        root_qposadr, _ = self._require_root_free_joint_addresses()

        for row, env_id in enumerate(env_ids_t.tolist()):
            data = self.mj_datas[env_id]
            root_np = root_pose_t[row].detach().cpu().numpy()
            data.qpos[root_qposadr:root_qposadr + 3] = root_np[:3]
            data.qpos[root_qposadr + 3:root_qposadr + 7] = root_np[3:7]
            mujoco.mj_forward(self.mj_models[env_id], data)
        self.update(0.0)

    def write_root_com_velocity_to_sim(self, root_velocity: ArrayType, env_ids: ArrayType = None):
        env_ids_t = self._normalize_env_ids(env_ids)
        root_velocity_t = self._rows_for_envs(root_velocity, env_ids_t)
        _, root_qveladr = self._require_root_free_joint_addresses()

        for row, env_id in enumerate(env_ids_t.tolist()):
            data = self.mj_datas[env_id]
            data.qvel[root_qveladr:root_qveladr + 6] = root_velocity_t[row].detach().cpu().numpy()
            mujoco.mj_forward(self.mj_models[env_id], data)
        self.update(0.0)

    def set_joint_position_target(self, target: ArrayType, joint_ids: ArrayType = None, env_ids: ArrayType = None):
        env_ids_t = self._normalize_env_ids(env_ids)
        target_t = self._rows_for_envs(target, env_ids_t)
        joint_ids_n = self._normalize_joint_ids(joint_ids)
        if isinstance(joint_ids_n, slice):
            self._data.joint_pos_target[env_ids_t] = target_t
        else:
            self._data.joint_pos_target[env_ids_t[:, None], joint_ids_n] = target_t

    def set_joint_velocity_target(self, target: ArrayType, joint_ids: ArrayType = None, env_ids: ArrayType = None):
        env_ids_t = self._normalize_env_ids(env_ids)
        target_t = self._rows_for_envs(target, env_ids_t)
        joint_ids_n = self._normalize_joint_ids(joint_ids)
        if isinstance(joint_ids_n, slice):
            self._data.joint_vel_target[env_ids_t] = target_t
        else:
            self._data.joint_vel_target[env_ids_t[:, None], joint_ids_n] = target_t

    def write_data_to_sim(self):
        current_pos = torch.stack(
            [torch.as_tensor(data.qpos[self.joint_qposadr_read].copy(), dtype=torch.float32) for data in self.mj_datas]
        )
        current_vel = torch.stack(
            [torch.as_tensor(data.qvel[self.joint_qveladr_read].copy(), dtype=torch.float32) for data in self.mj_datas]
        )
        pos_error = self._data.joint_pos_target - current_pos
        vel_error = self._data.joint_vel_target - current_vel

        torque = self._data.joint_stiffness * pos_error + self._data.joint_damping * vel_error
        self._data.applied_torque = torque.float()

        torque_np = torque.detach().cpu().numpy()
        for env_id, data in enumerate(self.mj_datas):
            data.ctrl[self._jnt_mjc2isaac] = torque_np[env_id]

        if self.has_external_wrench:
            force_w = quat_rotate(self._data.root_quat_w[:, None, :], self._external_force_b).detach().cpu().numpy()
            torque_w = quat_rotate(self._data.root_quat_w[:, None, :], self._external_torque_b).detach().cpu().numpy()
            for env_id, data in enumerate(self.mj_datas):
                data.xfrc_applied[self.body_adrs_write, :3] = force_w[env_id]
                data.xfrc_applied[self.body_adrs_write, 3:] = torque_w[env_id]

    def write_joint_position_to_sim(self, joint_pos: ArrayType, joint_ids: ArrayType = None, env_ids: ArrayType = None):
        self.write_joint_state_to_sim(joint_pos, None, joint_ids=joint_ids, env_ids=env_ids)

    def write_joint_velocity_to_sim(self, joint_vel: ArrayType, joint_ids: ArrayType = None, env_ids: ArrayType = None):
        self.write_joint_state_to_sim(None, joint_vel, joint_ids=joint_ids, env_ids=env_ids)

    def write_joint_state_to_sim(
        self,
        joint_pos: ArrayType,
        joint_vel: ArrayType,
        joint_ids: ArrayType = None,
        env_ids: ArrayType = None,
    ):
        env_ids_t = self._normalize_env_ids(env_ids)
        joint_ids_n = self._normalize_joint_ids(joint_ids)
        joint_pos_t = None if joint_pos is None else self._rows_for_envs(joint_pos, env_ids_t)
        joint_vel_t = None if joint_vel is None else self._rows_for_envs(joint_vel, env_ids_t)

        for row, env_id in enumerate(env_ids_t.tolist()):
            data = self.mj_datas[env_id]
            self._write_joint_state_to_data(
                data,
                joint_pos=None if joint_pos_t is None else joint_pos_t[row],
                joint_vel=None if joint_vel_t is None else joint_vel_t[row],
                joint_ids=joint_ids_n,
            )
            mujoco.mj_forward(self.mj_models[env_id], data)
        self.update(0.0)

    def _write_joint_state_to_data(self, data, joint_pos: torch.Tensor | None, joint_vel: torch.Tensor | None, joint_ids):
        if joint_pos is not None:
            joint_pos_all = torch.as_tensor(data.qpos[self.joint_qposadr_read].copy(), dtype=torch.float32)
            joint_pos_all[joint_ids] = joint_pos.detach().cpu().to(dtype=torch.float32)
            data.qpos[self.joint_qposadr_read] = joint_pos_all.numpy()
        if joint_vel is not None:
            joint_vel_all = torch.as_tensor(data.qvel[self.joint_qveladr_read].copy(), dtype=torch.float32)
            joint_vel_all[joint_ids] = joint_vel.detach().cpu().to(dtype=torch.float32)
            data.qvel[self.joint_qveladr_read] = joint_vel_all.numpy()

    def _normalize_env_ids(self, env_ids: ArrayType | int | slice | None) -> torch.Tensor:
        if env_ids is None or (isinstance(env_ids, slice) and env_ids == slice(None)):
            return torch.arange(self.num_instances, dtype=torch.long)
        if isinstance(env_ids, int):
            env_ids_t = torch.tensor([env_ids], dtype=torch.long)
        else:
            env_ids_t = torch.as_tensor(env_ids, dtype=torch.long).reshape(-1)
        if env_ids_t.numel() and (torch.any(env_ids_t < 0) or torch.any(env_ids_t >= self.num_instances)):
            raise ValueError(f"env_ids must be in [0, {self.num_instances}), got {env_ids_t.tolist()}.")
        return env_ids_t

    def _normalize_joint_ids(self, joint_ids: ArrayType | int | slice | None):
        if joint_ids is None:
            return slice(None)
        if isinstance(joint_ids, slice):
            return joint_ids
        if isinstance(joint_ids, int):
            return torch.tensor([joint_ids], dtype=torch.long)
        joint_ids_t = torch.as_tensor(joint_ids, dtype=torch.long).reshape(-1)
        if joint_ids_t.numel() and (torch.any(joint_ids_t < 0) or torch.any(joint_ids_t >= self.num_joints)):
            raise ValueError(f"joint_ids must be in [0, {self.num_joints}), got {joint_ids_t.tolist()}.")
        return joint_ids_t

    def _resolve_root_free_joint_addresses(self) -> tuple[int | None, int | None]:
        root_body_id = int(self.body_adrs_read[0])
        joint_adr = int(self.mj_model.body_jntadr[root_body_id])
        joint_num = int(self.mj_model.body_jntnum[root_body_id])
        if joint_adr < 0 or joint_num == 0:
            return None, None
        for joint_id in range(joint_adr, joint_adr + joint_num):
            if self.mj_model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                return int(self.mj_model.jnt_qposadr[joint_id]), int(self.mj_model.jnt_dofadr[joint_id])
        return None, None

    def _require_root_free_joint_addresses(self) -> tuple[int, int]:
        if self.root_qposadr is None or self.root_qveladr is None:
            raise RuntimeError(f"Articulation root body {self.body_names[0]!r} does not have a free joint.")
        return self.root_qposadr, self.root_qveladr

    def _rows_for_envs(self, value: ArrayType, env_ids: torch.Tensor) -> torch.Tensor:
        value_t = torch.as_tensor(value, dtype=torch.float32)
        if value_t.ndim == 1:
            return value_t.unsqueeze(0).expand(env_ids.numel(), -1).clone()
        if value_t.shape[0] == env_ids.numel():
            return value_t.clone()
        if value_t.shape[0] == self.num_instances:
            return value_t.index_select(0, env_ids).clone()
        raise ValueError(
            f"Expected first dimension to be len(env_ids)={env_ids.numel()} or num_envs={self.num_instances}, "
            f"got {tuple(value_t.shape)}."
        )

    def setup_logger(self, name: str):
        self._log_path = Path.cwd() / f"{name}.pt"
        self._log_states = []


def _joint_pos_limits(model: mujoco.MjModel, joint_ids: Sequence[int]) -> torch.Tensor:
    limits = torch.empty(len(joint_ids), 2, dtype=torch.float32)
    for out_id, joint_id in enumerate(joint_ids):
        if bool(model.jnt_limited[joint_id]):
            limits[out_id] = torch.as_tensor(model.jnt_range[joint_id], dtype=torch.float32)
        else:
            limits[out_id, 0] = -float("inf")
            limits[out_id, 1] = float("inf")
    return limits


def _joint_effort_limits(model: mujoco.MjModel, joint_ids: Sequence[int]) -> torch.Tensor:
    limits = torch.full((len(joint_ids),), float("inf"), dtype=torch.float32)
    joint_id_to_out = {int(joint_id): out_id for out_id, joint_id in enumerate(joint_ids)}
    for actuator_id in range(model.nu):
        actuator = model.actuator(actuator_id)
        if actuator.trntype != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        out_id = joint_id_to_out.get(int(actuator.trnid[0]))
        if out_id is None or not bool(model.actuator_forcelimited[actuator_id]):
            continue
        force_range = torch.as_tensor(model.actuator_forcerange[actuator_id], dtype=torch.float32).abs()
        limits[out_id] = force_range.max()
    return limits


class MJRootPhysxView:
    def __init__(self, asset):
        self.asset = asset
        self.max_shapes = len(asset.geom_adrs)
        self.geom_adrs = asset.geom_adrs
        self._material_properties = self._read_material_properties()

    def get_masses(self):
        masses = [
            torch.as_tensor(model.body_mass[self.asset.body_adrs_read].copy(), dtype=torch.float32)
            for model in self.asset.mj_models
        ]
        return torch.stack(masses)

    def set_masses(self, masses: ArrayType, indices: ArrayType):
        indices_t = self._normalize_indices(indices)
        masses_t = self._rows_for_indices(masses, indices_t, trailing_shape=(self.asset.num_bodies,))
        for row, env_id in enumerate(indices_t.tolist()):
            self.asset.mj_models[env_id].body_mass[self.asset.body_adrs_read] = masses_t[row].detach().cpu().numpy()
        self.asset.data.default_mass[indices_t] = masses_t

    def get_inertias(self):
        inertias = [
            torch.as_tensor(model.body_inertia[self.asset.body_adrs_read].copy(), dtype=torch.float32)
            for model in self.asset.mj_models
        ]
        return torch.stack(inertias)

    def set_inertias(self, inertias: ArrayType, indices: ArrayType):
        indices_t = self._normalize_indices(indices)
        inertias_t = self._rows_for_indices(inertias, indices_t, trailing_shape=(self.asset.num_bodies, 3))
        for row, env_id in enumerate(indices_t.tolist()):
            self.asset.mj_models[env_id].body_inertia[self.asset.body_adrs_read] = inertias_t[row].detach().cpu().numpy()
        self.asset.data.default_inertia[indices_t] = inertias_t

    def get_coms(self):
        coms = [
            torch.as_tensor(model.body_ipos[self.asset.body_adrs_read].copy(), dtype=torch.float32)
            for model in self.asset.mj_models
        ]
        return torch.stack(coms)

    def set_coms(self, coms: ArrayType, indices: ArrayType):
        indices_t = self._normalize_indices(indices)
        coms_t = self._rows_for_indices(coms, indices_t, trailing_shape=(self.asset.num_bodies, 3))
        for row, env_id in enumerate(indices_t.tolist()):
            self.asset.mj_models[env_id].body_ipos[self.asset.body_adrs_read] = coms_t[row].detach().cpu().numpy()
            mujoco.mj_forward(self.asset.mj_models[env_id], self.asset.mj_datas[env_id])

    def shape_ids_for_bodies(self, body_ids: Sequence[int]) -> torch.Tensor:
        shape_ids = []
        for body_id in body_ids:
            geom_adrs = self.asset.geom_adrs_by_body[int(body_id)]
            shape_ids.extend(np.where(np.isin(self.geom_adrs, geom_adrs))[0].tolist())
        return torch.as_tensor(shape_ids, dtype=torch.long)

    def get_material_properties(self):
        return self._material_properties.clone()

    def set_material_properties(self, materials: ArrayType, indices: ArrayType):
        indices_t = self._normalize_indices(indices)
        materials_t = torch.as_tensor(materials, dtype=torch.float32)
        if materials_t.ndim == 1:
            materials_t = materials_t.reshape(indices_t.numel(), self.max_shapes, 3)
        elif materials_t.ndim == 2 and materials_t.shape[-1] == 3:
            materials_t = materials_t.reshape(indices_t.numel(), self.max_shapes, 3)
        elif materials_t.ndim != 3:
            raise ValueError(f"materials must flatten or have shape (num_envs, max_shapes, 3), got {tuple(materials_t.shape)}.")
        if materials_t.shape != (indices_t.numel(), self.max_shapes, 3):
            if materials_t.shape[0] == self.asset.num_instances:
                materials_t = materials_t.index_select(0, indices_t)
            else:
                raise ValueError(
                    f"materials shape must be {(indices_t.numel(), self.max_shapes, 3)} or "
                    f"({self.asset.num_instances}, {self.max_shapes}, 3), got {tuple(materials_t.shape)}."
                )

        for row, env_id in enumerate(indices_t.tolist()):
            self._material_properties[env_id] = materials_t[row]
            if self.max_shapes:
                self.asset.mj_models[env_id].geom_friction[self.asset.geom_adrs, 0] = materials_t[row, :, 1].detach().cpu().numpy()

    def _read_material_properties(self):
        materials = torch.zeros(self.asset.num_instances, self.max_shapes, 3)
        for env_id, model in enumerate(self.asset.mj_models):
            if self.max_shapes:
                dynamic_friction = torch.as_tensor(model.geom_friction[self.asset.geom_adrs, 0].copy(), dtype=torch.float32)
                materials[env_id, :, 0] = dynamic_friction
                materials[env_id, :, 1] = dynamic_friction
        return materials

    def _normalize_indices(self, indices: ArrayType | int | slice | None):
        if indices is None or (isinstance(indices, slice) and indices == slice(None)):
            return torch.arange(self.asset.num_instances, dtype=torch.long)
        if isinstance(indices, int):
            indices_t = torch.tensor([indices], dtype=torch.long)
        else:
            indices_t = torch.as_tensor(indices, dtype=torch.long).reshape(-1)
        if indices_t.numel() and (torch.any(indices_t < 0) or torch.any(indices_t >= self.asset.num_instances)):
            raise ValueError(f"indices must be in [0, {self.asset.num_instances}), got {indices_t.tolist()}.")
        return indices_t

    def _rows_for_indices(self, value: ArrayType, indices: torch.Tensor, trailing_shape: tuple[int, ...]):
        value_t = torch.as_tensor(value, dtype=torch.float32)
        expected_subset = (indices.numel(), *trailing_shape)
        expected_all = (self.asset.num_instances, *trailing_shape)
        if tuple(value_t.shape) == expected_subset:
            return value_t.clone()
        if tuple(value_t.shape) == expected_all:
            return value_t.index_select(0, indices).clone()
        raise ValueError(f"Expected value shape {expected_subset} or {expected_all}, got {tuple(value_t.shape)}.")


@dataclass
class MJObjectViewData:
    default_root_state: ArrayType
    default_mass: ArrayType
    default_inertia: ArrayType
    default_joint_pos: ArrayType
    default_joint_vel: ArrayType
    soft_joint_pos_limits: ArrayType
    soft_joint_vel_limits: ArrayType
    body_link_pos_w: ArrayType = None
    body_link_quat_w: ArrayType = None
    body_com_lin_vel_w: ArrayType = None
    body_com_ang_vel_w: ArrayType = None
    joint_pos: ArrayType = None
    joint_vel: ArrayType = None

    @property
    def root_link_pos_w(self):
        return self.body_link_pos_w[:, 0]

    @property
    def root_link_quat_w(self):
        return self.body_link_quat_w[:, 0]

    @property
    def body_pos_w(self):
        return self.body_link_pos_w

    @property
    def body_quat_w(self):
        return self.body_link_quat_w

    @property
    def root_pos_w(self):
        return self.root_link_pos_w

    @property
    def root_quat_w(self):
        return self.root_link_quat_w


class MJObjectView:
    is_fixed_base = False

    def __init__(self, articulation: MJArticulation, spec: MJObjectSpec):
        self.articulation = articulation
        self.spec = spec
        self.cfg = SimpleNamespace(spawn=SimpleNamespace(scale=None))
        self.num_instances = articulation.num_instances
        self.body_names = list(spec.body_names)
        self.joint_names = list(spec.joint_names)
        self.mj_models = articulation.mj_models
        self.mj_model = articulation.mj_model
        self.mj_datas = articulation.mj_datas
        self.mj_data = articulation.mj_data

        self.body_adrs_read = np.array([
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in self.body_names
        ], dtype=np.int32)
        if np.any(self.body_adrs_read < 0):
            missing = [name for name, body_id in zip(self.body_names, self.body_adrs_read) if body_id < 0]
            raise ValueError(f"Missing MuJoCo object body names: {missing}")
        body_adrs_set = set(int(body_id) for body_id in self.body_adrs_read)
        self.geom_adrs = np.array([
            geom_id for geom_id in range(self.mj_model.ngeom)
            if int(self.mj_model.geom_bodyid[geom_id]) in body_adrs_set
        ], dtype=np.int32)

        self.joint_mj_ids = np.array([
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.joint_names
        ], dtype=np.int32)
        if np.any(self.joint_mj_ids < 0):
            missing = [name for name, joint_id in zip(self.joint_names, self.joint_mj_ids) if joint_id < 0]
            raise ValueError(f"Missing MuJoCo object joint names: {missing}")

        self.joint_qposadr_read = self.mj_model.jnt_qposadr[self.joint_mj_ids] if self.num_joints else np.array([], dtype=np.int32)
        self.joint_qveladr_read = self.mj_model.jnt_dofadr[self.joint_mj_ids] if self.num_joints else np.array([], dtype=np.int32)
        self.root_free_joint_id = self._find_root_free_joint()
        self.root_qposadr = None if self.root_free_joint_id is None else int(self.mj_model.jnt_qposadr[self.root_free_joint_id])
        self.root_qveladr = None if self.root_free_joint_id is None else int(self.mj_model.jnt_dofadr[self.root_free_joint_id])

        default_joint_pos = torch.zeros(self.num_instances, self.num_joints)
        default_joint_vel = torch.zeros(self.num_instances, self.num_joints)
        if self.num_joints:
            default_joint_pos = torch.stack([
                torch.as_tensor(data.qpos[self.joint_qposadr_read].copy(), dtype=torch.float32)
                for data in self.mj_datas
            ])
            default_joint_vel = torch.stack([
                torch.as_tensor(data.qvel[self.joint_qveladr_read].copy(), dtype=torch.float32)
                for data in self.mj_datas
            ])

        joint_pos_limits = _joint_pos_limits(self.mj_model, self.joint_mj_ids)
        joint_vel_limits = torch.full((self.num_joints,), float("inf"), dtype=torch.float32)
        default_mass = torch.stack([
            torch.as_tensor(model.body_mass[self.body_adrs_read].copy(), dtype=torch.float32)
            for model in self.mj_models
        ])
        default_inertia = torch.stack([
            torch.as_tensor(model.body_inertia[self.body_adrs_read].copy(), dtype=torch.float32)
            for model in self.mj_models
        ])
        default_root_state = torch.zeros(self.num_instances, 13)
        self._data = MJObjectViewData(
            default_root_state=default_root_state,
            default_mass=default_mass,
            default_inertia=default_inertia,
            default_joint_pos=default_joint_pos,
            default_joint_vel=default_joint_vel,
            soft_joint_pos_limits=joint_pos_limits.expand(self.num_instances, -1, -1).clone(),
            soft_joint_vel_limits=joint_vel_limits.expand(self.num_instances, -1).clone(),
        )
        self._custom_friction = torch.zeros(self.num_instances, dtype=torch.float32)
        self._custom_damping = torch.zeros(self.num_instances, dtype=torch.float32)
        self.root_physx_view = MJRootPhysxView(self)
        self.update(0.0)
        self._data.default_root_state[:, 0:3] = self._data.root_link_pos_w
        self._data.default_root_state[:, 3:7] = self._data.root_link_quat_w

    @property
    def num_bodies(self):
        return len(self.body_names)

    @property
    def num_joints(self):
        return len(self.joint_names)

    @property
    def data(self):
        return self._data

    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False):
        return string_utils.resolve_matching_names(name_keys, self.body_names, preserve_order)

    def find_joints(self, name_keys: str | Sequence[str], preserve_order: bool = False):
        return string_utils.resolve_matching_names(name_keys, self.joint_names, preserve_order)

    def update(self, dt: float):
        body_pos_w = []
        body_quat_w = []
        body_vel_w = []
        joint_pos = []
        joint_vel = []

        for data in self.mj_datas:
            body_pos_w.append(np.asarray(data.xpos[self.body_adrs_read], dtype=np.float32).copy())
            body_quat_w.append(np.asarray(data.xquat[self.body_adrs_read], dtype=np.float32).copy())
            body_vel_w.append(np.asarray(data.cvel[self.body_adrs_read], dtype=np.float32).copy())
            if self.num_joints:
                joint_pos.append(np.asarray(data.qpos[self.joint_qposadr_read], dtype=np.float32).copy())
                joint_vel.append(np.asarray(data.qvel[self.joint_qveladr_read], dtype=np.float32).copy())

        body_vel_w = torch.as_tensor(np.stack(body_vel_w), dtype=torch.float32)
        self._data.body_link_pos_w = torch.as_tensor(np.stack(body_pos_w), dtype=torch.float32)
        self._data.body_link_quat_w = torch.as_tensor(np.stack(body_quat_w), dtype=torch.float32)
        self._data.body_com_ang_vel_w = body_vel_w[..., :3]
        self._data.body_com_lin_vel_w = body_vel_w[..., 3:]
        if self.num_joints:
            self._data.joint_pos = torch.as_tensor(np.stack(joint_pos), dtype=torch.float32)
            self._data.joint_vel = torch.as_tensor(np.stack(joint_vel), dtype=torch.float32)
        else:
            self._data.joint_pos = torch.zeros(self.num_instances, 0)
            self._data.joint_vel = torch.zeros(self.num_instances, 0)

    def write_root_link_pose_to_sim(self, root_pose: ArrayType, env_ids: ArrayType = None):
        if self.root_qposadr is None:
            raise RuntimeError(f"Object {self.spec.asset_name!r} does not have a root free joint.")
        env_ids_t = self.articulation._normalize_env_ids(env_ids)
        root_pose_t = self.articulation._rows_for_envs(root_pose, env_ids_t)

        for row, env_id in enumerate(env_ids_t.tolist()):
            data = self.mj_datas[env_id]
            root_np = root_pose_t[row].detach().cpu().numpy()
            data.qpos[self.root_qposadr:self.root_qposadr + 3] = root_np[:3]
            data.qpos[self.root_qposadr + 3:self.root_qposadr + 7] = root_np[3:7]
            mujoco.mj_forward(self.mj_models[env_id], data)
        self.articulation.update(0.0)
        self.update(0.0)

    def write_root_com_velocity_to_sim(self, root_velocity: ArrayType, env_ids: ArrayType = None):
        if self.root_qveladr is None:
            raise RuntimeError(f"Object {self.spec.asset_name!r} does not have a root free joint.")
        env_ids_t = self.articulation._normalize_env_ids(env_ids)
        root_velocity_t = self.articulation._rows_for_envs(root_velocity, env_ids_t)

        for row, env_id in enumerate(env_ids_t.tolist()):
            data = self.mj_datas[env_id]
            data.qvel[self.root_qveladr:self.root_qveladr + 6] = root_velocity_t[row].detach().cpu().numpy()
            mujoco.mj_forward(self.mj_models[env_id], data)
        self.articulation.update(0.0)
        self.update(0.0)

    def write_joint_state_to_sim(
        self,
        joint_pos: ArrayType,
        joint_vel: ArrayType,
        joint_ids: ArrayType = None,
        env_ids: ArrayType = None,
    ):
        env_ids_t = self.articulation._normalize_env_ids(env_ids)
        joint_ids_n = self._normalize_joint_ids(joint_ids)
        joint_pos_t = None if joint_pos is None else self.articulation._rows_for_envs(joint_pos, env_ids_t)
        joint_vel_t = None if joint_vel is None else self.articulation._rows_for_envs(joint_vel, env_ids_t)

        joint_ids_idx = joint_ids_n.detach().cpu().numpy() if isinstance(joint_ids_n, torch.Tensor) else joint_ids_n
        qposadr = self.joint_qposadr_read[joint_ids_idx]
        qveladr = self.joint_qveladr_read[joint_ids_idx]
        for row, env_id in enumerate(env_ids_t.tolist()):
            data = self.mj_datas[env_id]
            if joint_pos_t is not None:
                data.qpos[qposadr] = joint_pos_t[row].detach().cpu().numpy()
            if joint_vel_t is not None:
                data.qvel[qveladr] = joint_vel_t[row].detach().cpu().numpy()
            mujoco.mj_forward(self.mj_models[env_id], data)
        self.articulation.update(0.0)
        self.update(0.0)

    def write_joint_armature_to_sim(self, armature: ArrayType, joint_ids: ArrayType = None, env_ids: ArrayType = None):
        env_ids_t = self.articulation._normalize_env_ids(env_ids)
        joint_ids_n = self._normalize_joint_ids(joint_ids)
        joint_ids_idx = joint_ids_n.detach().cpu().numpy() if isinstance(joint_ids_n, torch.Tensor) else joint_ids_n
        dof_adrs = self.joint_qveladr_read[joint_ids_idx]
        armature_t = self.articulation._rows_for_envs(armature, env_ids_t)

        for row, env_id in enumerate(env_ids_t.tolist()):
            values = armature_t[row].detach().cpu().numpy()
            if np.ndim(values) == 0:
                values = np.asarray([values])
            self.mj_models[env_id].dof_armature[dof_adrs] = values

    def write_data_to_sim(self):
        if not self.num_joints:
            return

        joint_vel = self.data.joint_vel
        if joint_vel is None:
            joint_vel = torch.stack([
                torch.as_tensor(data.qvel[self.joint_qveladr_read].copy(), dtype=torch.float32)
                for data in self.mj_datas
            ])

        joint_friction = -torch.sign(joint_vel) * (joint_vel.abs() > 0.01) * self._custom_friction[:, None]
        joint_damping = -joint_vel * self._custom_damping[:, None]
        torque = joint_friction + joint_damping
        torque_np = torque.detach().cpu().numpy()

        for env_id, data in enumerate(self.mj_datas):
            data.qfrc_applied[self.joint_qveladr_read] = torque_np[env_id]

    def _find_root_free_joint(self):
        root_body_id = int(self.body_adrs_read[0])
        for joint_id in range(self.mj_model.njnt):
            if (
                int(self.mj_model.jnt_bodyid[joint_id]) == root_body_id
                and self.mj_model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
            ):
                return joint_id
        return None

    def _normalize_joint_ids(self, joint_ids: ArrayType | int | slice | None):
        if joint_ids is None:
            return slice(None)
        if isinstance(joint_ids, slice):
            return joint_ids
        if isinstance(joint_ids, int):
            joint_ids_t = torch.tensor([joint_ids], dtype=torch.long)
        else:
            joint_ids_t = torch.as_tensor(joint_ids, dtype=torch.long).reshape(-1)
        if joint_ids_t.numel() and (torch.any(joint_ids_t < 0) or torch.any(joint_ids_t >= self.num_joints)):
            raise ValueError(f"joint_ids must be in [0, {self.num_joints}), got {joint_ids_t.tolist()}.")
        return joint_ids_t


@dataclass(frozen=True)
class MJContactSensorCfg:
    target: str
    body_names: Sequence[str] | None = None
    filter_body_names: Sequence[str] | None = None
    history_length: int = 1


@dataclass
class MjContactData:
    net_forces_w: ArrayType = None
    force_matrix_w: ArrayType = None
    net_forces_w_history: ArrayType = None
    current_contact_time: ArrayType = None
    current_air_time: ArrayType = None
    last_contact_time: ArrayType = None
    last_air_time: ArrayType = None


class MjContactSensor:
    contact_force_threshold = 1e-5

    def __init__(
        self,
        articulation: MJArticulation,
        body_names: Sequence[str] | None = None,
        filter_body_names: Sequence[str] | None = None,
        history_length: int = 1,
    ):
        self.articulation = articulation
        self.body_indices, self.body_names = self._resolve_body_names(body_names)
        self.body_adrs_read = self.articulation.body_adrs_read[self.body_indices]
        self.filter_body_names = list(filter_body_names or [])
        self.history_length = max(1, int(history_length))
        self.filter_body_adrs = np.array([
            mujoco.mj_name2id(self.articulation.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in self.filter_body_names
        ], dtype=np.int32)
        if np.any(self.filter_body_adrs < 0):
            missing = [name for name, body_id in zip(self.filter_body_names, self.filter_body_adrs) if body_id < 0]
            raise ValueError(f"Missing MuJoCo contact filter body names: {missing}")
        self._data = MjContactData(
            net_forces_w=torch.zeros(self.articulation.num_instances, len(self.body_names), 3),
            force_matrix_w=torch.zeros(self.articulation.num_instances, len(self.body_names), len(self.filter_body_names), 3),
            net_forces_w_history=torch.zeros(
                self.articulation.num_instances,
                self.history_length,
                len(self.body_names),
                3,
            ),
            current_contact_time=torch.zeros(self.articulation.num_instances, len(self.body_names)),
            current_air_time=torch.zeros(self.articulation.num_instances, len(self.body_names)),
            last_contact_time=torch.zeros(self.articulation.num_instances, len(self.body_names)),
            last_air_time=torch.zeros(self.articulation.num_instances, len(self.body_names)),
        )
    
    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False):
        return string_utils.resolve_matching_names(name_keys, self.body_names, preserve_order)

    def update(self, dt: float):
        cfrc_ext = np.stack([data.cfrc_ext[self.body_adrs_read, :3].copy() for data in self.articulation.mj_datas])
        net_forces_w = torch.as_tensor(cfrc_ext, dtype=torch.float32)
        force_matrix_w = self._compute_force_matrix_w()
        self._data.net_forces_w = net_forces_w
        self._data.force_matrix_w = force_matrix_w
        self._update_contact_state(net_forces_w, force_matrix_w, float(dt))

    @property
    def data(self):
        return self._data

    def compute_first_contact(self, dt: float) -> torch.Tensor:
        return (self._data.current_contact_time > 0.0) & (self._data.current_contact_time <= float(dt) + 1e-8)

    def _resolve_body_names(self, body_names):
        if body_names is None:
            return list(range(self.articulation.num_bodies)), list(self.articulation.body_names)
        indices, names = self.articulation.find_bodies(body_names, preserve_order=True)
        return indices, names

    def _compute_force_matrix_w(self):
        matrix = torch.zeros(
            self.articulation.num_instances,
            len(self.body_names),
            len(self.filter_body_names),
            3,
            dtype=torch.float32,
        )
        if not self.filter_body_names:
            return matrix

        sensor_body_to_idx = {int(body_id): idx for idx, body_id in enumerate(self.body_adrs_read)}
        filter_body_to_idx = {int(body_id): idx for idx, body_id in enumerate(self.filter_body_adrs)}
        force_contact = np.zeros(6, dtype=np.float64)
        for env_id, data in enumerate(self.articulation.mj_datas):
            model = self.articulation.mj_models[env_id]
            for contact_id in range(data.ncon):
                contact = data.contact[contact_id]
                body1 = int(model.geom_bodyid[contact.geom1])
                body2 = int(model.geom_bodyid[contact.geom2])
                body_idx = sensor_body_to_idx.get(body1)
                filter_idx = filter_body_to_idx.get(body2)
                sign = 1.0
                if body_idx is None or filter_idx is None:
                    body_idx = sensor_body_to_idx.get(body2)
                    filter_idx = filter_body_to_idx.get(body1)
                    sign = -1.0
                if body_idx is None or filter_idx is None:
                    continue

                mujoco.mj_contactForce(model, data, contact_id, force_contact)
                frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
                force_w = frame.T @ force_contact[:3] * sign
                matrix[env_id, body_idx, filter_idx] += torch.as_tensor(force_w, dtype=torch.float32)

        return matrix

    def _update_contact_state(self, net_forces_w: torch.Tensor, force_matrix_w: torch.Tensor, dt: float) -> None:
        self._data.net_forces_w_history = self._data.net_forces_w_history.roll(1, dims=1)
        self._data.net_forces_w_history[:, 0] = net_forces_w

        if self.filter_body_names:
            contact_signal = force_matrix_w.norm(dim=-1).amax(dim=-1)
        else:
            contact_signal = net_forces_w.norm(dim=-1)
        in_contact = contact_signal > self.contact_force_threshold
        was_in_contact = self._data.current_contact_time > 0.0
        previous_contact_time = self._data.current_contact_time.clone()
        previous_air_time = self._data.current_air_time.clone()

        dt_t = torch.full_like(self._data.current_contact_time, dt)
        self._data.current_contact_time = torch.where(
            in_contact,
            self._data.current_contact_time + dt_t,
            torch.zeros_like(self._data.current_contact_time),
        )
        self._data.current_air_time = torch.where(
            in_contact,
            torch.zeros_like(self._data.current_air_time),
            self._data.current_air_time + dt_t,
        )
        first_contact = in_contact & ~was_in_contact
        first_air = ~in_contact & was_in_contact
        self._data.last_air_time = torch.where(first_contact, previous_air_time, self._data.last_air_time)
        self._data.last_contact_time = torch.where(first_air, previous_contact_time, self._data.last_contact_time)


class MJScene:
    def __init__(self, cfg, num_envs: int = 1, launch_viewer: bool = True, env_spacing: float = 0.0):
        if num_envs < 1:
            raise ValueError(f"num_envs must be positive, got {num_envs}.")

        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.articulations = {}
        self.rigid_objects = {}
        self.sensors = {}
        self._sim_articulations = []
        self._object_views = []

        cfg_items = list(self._iter_cfg_items())
        for asset_name, asset_cfg in cfg_items:
            print(asset_name, asset_cfg)
            if isinstance(asset_cfg, MJArticulationCfg):
                articulation = MJArticulation(asset_cfg, num_envs=self.num_envs)
                articulation.setup_logger(asset_name)
                self.articulations[asset_name] = articulation
                self._sim_articulations.append(articulation)

        for articulation in list(self._sim_articulations):
            for object_spec in articulation.cfg.object_specs.values():
                object_view = MJObjectView(articulation, object_spec)
                self._object_views.append(object_view)
                if object_view.num_joints:
                    self.articulations[object_spec.asset_name] = object_view
                else:
                    self.rigid_objects[object_spec.asset_name] = object_view

        for asset_name, asset_cfg in cfg_items:
            if isinstance(asset_cfg, str):
                target = self.articulations[asset_cfg]
                self.sensors[asset_name] = MjContactSensor(target, history_length=3)
            elif isinstance(asset_cfg, MJContactSensorCfg):
                target = self.articulations[asset_cfg.target]
                self.sensors[asset_name] = MjContactSensor(
                    target,
                    body_names=asset_cfg.body_names,
                    filter_body_names=asset_cfg.filter_body_names,
                    history_length=asset_cfg.history_length,
                )

        self.viewer = None
        if launch_viewer:
            self.viewer = mujoco.viewer.launch_passive(
                self.articulations["robot"].mj_model,
                self.articulations["robot"].mj_data,
                show_left_ui=False,
                show_right_ui=False,
            )
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.viewer.cam.trackbodyid = 1

        self.mj_model = self.articulations["robot"].mj_model
        self.mj_data = self.articulations["robot"].mj_data
        self.env_origins = self._make_env_origins(self.num_envs, env_spacing)

    def reset(self, env_ids: torch.Tensor):
        for articulation in self._sim_articulations:
            continue
            articulation.reset(env_ids)

    def update(self, dt: float):
        for articulation in self._sim_articulations:
            articulation.update(dt)
        for object_view in self._object_views:
            object_view.update(dt)
        for sensor in self.sensors.values():
            sensor.update(dt)
    
    def write_data_to_sim(self):
        for articulation in self._sim_articulations:
            articulation.write_data_to_sim()
        for object_view in self._object_views:
            object_view.write_data_to_sim()

    def __getitem__(self, key: str):
        result = self.articulations.get(key)
        result = result or self.rigid_objects.get(key)
        result = result or self.sensors.get(key)
        return result

    def _iter_cfg_items(self):
        seen: set[str] = set()
        for source in (vars(type(self.cfg)), vars(self.cfg)):
            for name in source:
                if name in seen or name.startswith("_"):
                    continue
                seen.add(name)
                value = getattr(self.cfg, name)
                if isinstance(value, (MJArticulationCfg, MJContactSensorCfg, str)):
                    yield name, value

    def __contains__(self, key: str) -> bool:
        return key in self.articulations or key in self.rigid_objects or key in self.sensors

    @staticmethod
    def _make_env_origins(num_envs: int, env_spacing: float) -> torch.Tensor:
        origins = torch.zeros(num_envs, 3)
        if env_spacing:
            cols = int(np.ceil(np.sqrt(num_envs)))
            ids = torch.arange(num_envs)
            origins[:, 0] = (ids % cols).float() * env_spacing
            origins[:, 1] = (ids // cols).float() * env_spacing
        return origins

    def create_arrow_marker(self, radius: float, rgba):
        if self.viewer is None:
            raise RuntimeError("MJScene was created without a viewer.")
        scene = self.viewer.user_scn
        scene.ngeom += 1
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom - 1],
            mujoco.mjtGeom.mjGEOM_ARROW,
            size=np.array([radius, radius, 1.0], dtype=np.float64),
            pos=np.zeros(3),
            mat=sRot.random().as_matrix().reshape(-1),
            rgba=np.array(rgba, dtype=np.float64),
        )
        return MjvGeom(scene.geoms[scene.ngeom - 1])
    
    def create_sphere_marker(self, radius: float, rgba):
        if self.viewer is None:
            raise RuntimeError("MJScene was created without a viewer.")
        scene = self.viewer.user_scn
        scene.ngeom += 1
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom - 1],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([radius, radius, radius], dtype=np.float64),
            pos=np.zeros(3),
            mat=sRot.random().as_matrix().reshape(-1),
            rgba=np.array(rgba, dtype=np.float64),
        )
        return MjvGeom(scene.geoms[scene.ngeom - 1])


    def close(self):
        if self.viewer is not None:
            self.viewer.close()
        for articulation in self._sim_articulations:
            path = articulation._log_path
            if articulation._log_states:
                states = torch.stack(articulation._log_states)
                torch.save(states.__dict__, path)


class MJSim:

    device = "cpu"

    def __init__(self, scene: MJScene, realtime: bool = True):
        self.scene = scene
        self.mj_model = scene.mj_model
        self.mj_data = scene.mj_data
        self.realtime = realtime

    def render(self):
        if self.scene.viewer is not None:
            self.scene.viewer.sync()

    def get_physics_dt(self):
        return self.mj_model.opt.timestep

    def has_gui(self):
        return self.scene.viewer is not None

    def step(self, render: bool = False):
        for articulation in self.scene._sim_articulations:
            for model, data in zip(articulation.mj_models, articulation.mj_datas, strict=True):
                mujoco.mj_step(model, data)
                mujoco.mj_rnePostConstraint(model, data)
        if self.realtime:
            time.sleep(self.get_physics_dt())


class MjvGeom:
    def __init__(self, geom):
        self.geom: mujoco.MjvGeom = geom

    def from_to(self, from_, to):
        mujoco.mjv_connector(
            self.geom,
            self.geom.type,
            width=0.05,
            from_=np.array(from_.reshape(3)).astype(np.float64),
            to=np.array(to.reshape(3)).astype(np.float64),
        )
