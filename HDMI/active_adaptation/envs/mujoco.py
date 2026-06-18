import torch
import numpy as np
import mujoco
import mujoco.viewer
import time
import warnings
from pathlib import Path
from typing import Sequence, Union, Any, Dict
from dataclasses import dataclass, replace

from isaaclab.utils import string as string_utils
from scipy.spatial.transform import Rotation as sRot

from active_adaptation.assets_mjcf.types import MJArticulationCfg
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
    projected_gravity_b: ArrayType = None
    
    body_vel_w: ArrayType = None
    # body_lin_vel_w: ArrayType = None
    # body_ang_vel_w: ArrayType = None
    root_lin_vel_w: ArrayType = None
    root_ang_vel_w: ArrayType = None
    root_ang_vel_b: ArrayType = None
    root_lin_vel_b: ArrayType = None
    heading_w: ArrayType = None

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
        self.mj_model = mujoco.MjModel.from_xml_path(cfg.mjcf_path)
        self.mj_datas = [mujoco.MjData(self.mj_model) for _ in range(self.num_instances)]
        self.mj_data = self.mj_datas[0]
        
        self.body_names_isaac = list(cfg.body_names_isaac)
        self.body_names_mjc = []
        body_adrs = []
        for i in range(1, self.mj_model.nbody): # skip the world body
            body = self.mj_model.body(i)
            self.body_names_mjc.append(body.name)
            body_adrs.append(i)
        
        if not set(self.body_names_isaac) == set(self.body_names_mjc):
            warnings.warn(
                f"Isaac body names do not match mujoco body names:\n"
                f"Isaac - Mujoco: {set(self.body_names_isaac) - set(self.body_names_mjc)}\n"
                f"Mujoco - Isaac: {set(self.body_names_mjc) - set(self.body_names_isaac)}\n",
                category=UserWarning
            )

        # find only the actuated joints
        self.joint_names_isaac = list(cfg.joint_names_isaac)
        self.joint_names_mjc = []

        joint_qposadr = []
        joint_qveladr = []
        for i in range(self.mj_model.nu):
            actuator = self.mj_model.actuator(i)
            if actuator.trntype == mujoco.mjtTrn.mjTRN_JOINT:
                joint_id = actuator.trnid[0]
                joint = self.mj_model.joint(actuator.trnid[0])
                self.joint_names_mjc.append(joint.name)
                joint_qposadr.append(self.mj_model.jnt_qposadr[joint_id])
                joint_qveladr.append(self.mj_model.jnt_dofadr[joint_id])
        
        if not set(self.joint_names_isaac) == set(self.joint_names_mjc):
            warnings.warn(
                f"Isaac joint names do not match mujoco joint names:\n"
                f"Isaac - Mujoco: {set(self.joint_names_isaac) - set(self.joint_names_mjc)}\n"
                f"Mujoco - Isaac: {set(self.joint_names_mjc) - set(self.joint_names_isaac)}\n",
                category=UserWarning
            )
        
        # Isaac assets may have less joints/bodies due to asset simplification
        self._jnt_isaac2mjc = [self.joint_names_isaac.index(joint_name) for joint_name in self.joint_names_mjc if joint_name in self.joint_names_isaac]
        self._jnt_mjc2isaac = [self.joint_names_mjc.index(joint_name) for joint_name in self.joint_names_isaac]
        self._body_isaac2mjc = [self.body_names_isaac.index(body_name) for body_name in self.body_names_mjc if body_name in self.body_names_isaac]
        self._body_mjc2isaac = [self.body_names_mjc.index(body_name) for body_name in self.body_names_isaac]
        
        self.body_adrs = np.array(body_adrs)
        self.joint_qposadr = np.array(joint_qposadr)
        self.joint_qveladr = np.array(joint_qveladr)
        
        # read/write mujoco data in isaac order
        self.body_adrs_read = self.body_adrs[self._body_mjc2isaac]
        self.body_adrs_write = self.body_adrs[self._body_isaac2mjc]
        self.joint_qposadr_read = self.joint_qposadr[self._jnt_mjc2isaac]
        self.joint_qveladr_read = self.joint_qveladr[self._jnt_mjc2isaac]
        self.joint_qposadr_write = self.joint_qposadr[self._jnt_isaac2mjc]
        self.joint_qveladr_write = self.joint_qveladr[self._jnt_isaac2mjc]

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
            default_inertia=diag_inertia.diag_embed().flatten()[None].expand(self.num_instances, -1).clone(),
            joint_stiffness=joint_stiffness.expand(self.num_instances, -1).clone(),
            joint_damping=joint_damping.expand(self.num_instances, -1).clone(),
            applied_torque=torch.zeros(self.num_instances, self.num_joints),
        )
        self._data.joint_pos_target = self._data.default_joint_pos.clone()
        self._data.joint_vel_target = self._data.default_joint_vel.clone()

        self._external_force_b = torch.zeros(self.num_instances, self.num_bodies, 3)
        self._external_torque_b = torch.zeros(self.num_instances, self.num_bodies, 3)
        self.has_external_wrench = False

        self.timestamp = 0.

        for data in self.mj_datas:
            mujoco.mj_forward(self.mj_model, data)
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

        for row, env_id in enumerate(env_ids_t.tolist()):
            data = self.mj_datas[env_id]
            root_np = root_state_t[row].detach().cpu().numpy()
            data.qpos[:3] = root_np[:3]
            data.qpos[3:7] = root_np[3:7]
            data.qvel[:6] = 0.0
            self._write_joint_state_to_data(
                data,
                joint_pos=self._data.default_joint_pos[env_id],
                joint_vel=self._data.default_joint_vel[env_id],
                joint_ids=slice(None),
            )
            mujoco.mj_forward(self.mj_model, data)
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
            mujoco.mj_forward(self.mj_model, data)
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


@dataclass
class MjContactData:
    net_forces_w: ArrayType = None


class MjContactSensor:
    def __init__(self, articulation: MJArticulation):
        self.articulation = articulation
        self.body_names = self.articulation.body_names
        self.body_adrs_read = self.articulation.body_adrs_read
        self._data = MjContactData(
            net_forces_w=torch.zeros(1, self.articulation.num_bodies, 3)
        )
    
    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False):
        return self.articulation.find_bodies(name_keys, preserve_order)

    def update(self, dt: float):
        cfrc_ext = np.stack([data.cfrc_ext[self.body_adrs_read, :3].copy() for data in self.articulation.mj_datas])
        self._data.net_forces_w = torch.as_tensor(cfrc_ext, dtype=torch.float32)

    @property
    def data(self):
        return self._data


class MJScene:
    def __init__(self, cfg, num_envs: int = 1, launch_viewer: bool = True, env_spacing: float = 0.0):
        if num_envs < 1:
            raise ValueError(f"num_envs must be positive, got {num_envs}.")

        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.articulations = {}
        self.sensors = {}

        cfg_items = list(self._iter_cfg_items())
        for asset_name, asset_cfg in cfg_items:
            print(asset_name, asset_cfg)
            if isinstance(asset_cfg, MJArticulationCfg):
                articulation = MJArticulation(asset_cfg, num_envs=self.num_envs)
                articulation.setup_logger(asset_name)
                self.articulations[asset_name] = articulation

        for asset_name, asset_cfg in cfg_items:
            if isinstance(asset_cfg, str):
                target = self.articulations[asset_cfg]
                self.sensors[asset_name] = MjContactSensor(target)

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
        for articulation in self.articulations.values():
            continue
            articulation.reset(env_ids)

    def update(self, dt: float):
        for articulation in self.articulations.values():
            articulation.update(dt)
        for sensor in self.sensors.values():
            sensor.update(dt)
    
    def write_data_to_sim(self):
        for articulation in self.articulations.values():
            articulation.write_data_to_sim()

    def __getitem__(self, key: str):
        result = self.articulations.get(key)
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
                if isinstance(value, (MJArticulationCfg, str)):
                    yield name, value

    def __contains__(self, key: str) -> bool:
        return key in self.articulations or key in self.sensors

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
        for articulation in self.articulations.values():
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
        for articulation in self.scene.articulations.values():
            for data in articulation.mj_datas:
                mujoco.mj_step(articulation.mj_model, data)
                mujoco.mj_rnePostConstraint(articulation.mj_model, data)
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
