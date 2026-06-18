from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch


@dataclass
class MujocoPolicyState:
    root_ang_vel_b: torch.Tensor
    projected_gravity_b: torch.Tensor
    joint_pos: torch.Tensor
    joint_pos_offset: torch.Tensor | None = None
    applied_action: torch.Tensor | None = None
    action_history: torch.Tensor | None = None
    ref_body_pos_future_w: torch.Tensor | None = None
    ref_root_pos_w: torch.Tensor | None = None
    ref_root_quat_w: torch.Tensor | None = None
    ref_joint_pos_future: torch.Tensor | None = None
    motion_t: torch.Tensor | None = None
    motion_len: torch.Tensor | None = None
    robot_root_pos_w: torch.Tensor | None = None
    robot_root_quat_w: torch.Tensor | None = None
    contact_target_pos_w: torch.Tensor | None = None
    contact_eef_pos_w: torch.Tensor | None = None
    object_pos_w: torch.Tensor | None = None
    object_quat_w: torch.Tensor | None = None


class MujocoObservationBuilder:
    """Build policy observation tensors from MuJoCo state using exported HDMI obs config."""

    def __init__(
        self,
        observation_cfg: Mapping,
        policy_joint_names: list[str],
        observation_joint_names: Sequence[str] | None = None,
    ):
        self.observation_cfg = observation_cfg
        self.policy_joint_names = list(policy_joint_names)
        fallback_joint_names = observation_joint_names if observation_joint_names is not None else policy_joint_names
        self.joint_pos_names = _joint_pos_names(observation_cfg, fallback_joint_names)
        self.action_dim = len(self.policy_joint_names)
        self._history: dict[str, torch.Tensor] = {}
        self._initialized = False
        self._last_group_dims: dict[str, int] = {}

    def reset(self, state: MujocoPolicyState) -> None:
        self._history.clear()
        for obs_key, params in self._history_specs():
            history_steps = _history_steps(params)
            size = max(history_steps) + 1
            value = self._state_value(obs_key, state)
            self._history[obs_key] = value.unsqueeze(1).expand(-1, size, -1).clone()
        self._initialized = True

    def update(self, state: MujocoPolicyState) -> None:
        if not self._initialized:
            self.reset(state)
            return
        for obs_key, _ in self._history_specs():
            value = self._state_value(obs_key, state)
            self._history[obs_key] = self._history[obs_key].roll(1, dims=1)
            self._history[obs_key][:, 0] = value

    def build_group(
        self,
        group_name: str,
        state: MujocoPolicyState,
        return_components: bool = False,
    ) -> torch.Tensor | OrderedDict[str, torch.Tensor]:
        group_cfg = self.observation_cfg[group_name]
        if not self._initialized and any(key.endswith("_history") for key in group_cfg):
            self.reset(state)

        components: OrderedDict[str, torch.Tensor] = OrderedDict()
        for obs_key, params in group_cfg.items():
            params = params or {}
            components[obs_key] = self._build_component(obs_key, params, state)

        self._last_group_dims[group_name] = sum(tensor.shape[-1] for tensor in components.values())
        if return_components:
            return components
        return torch.cat(list(components.values()), dim=-1)

    def group_dim(self, group_name: str) -> int:
        if group_name not in self._last_group_dims:
            raise RuntimeError(f"Group '{group_name}' has not been built yet.")
        return self._last_group_dims[group_name]

    def _history_specs(self):
        seen: set[str] = set()
        for group_cfg in self.observation_cfg.values():
            for obs_key, params in group_cfg.items():
                if obs_key.endswith("_history") and obs_key not in seen:
                    seen.add(obs_key)
                    yield obs_key, params or {}

    def _state_value(self, obs_key: str, state: MujocoPolicyState) -> torch.Tensor:
        if obs_key == "root_ang_vel_history":
            return state.root_ang_vel_b
        if obs_key == "projected_gravity_history":
            return state.projected_gravity_b
        if obs_key == "joint_pos_history":
            return _require_last_dim(state.joint_pos, len(self.joint_pos_names), "joint_pos")
        raise NotImplementedError(f"Unsupported MuJoCo history observation '{obs_key}'.")

    def _build_component(self, obs_key: str, params: Mapping, state: MujocoPolicyState) -> torch.Tensor:
        if obs_key in ("root_ang_vel_history", "projected_gravity_history"):
            return self._select_history(obs_key, params)
        if obs_key == "joint_pos_history":
            joint_pos = self._select_history(obs_key, params)
            steps = len(_history_steps(params))
            offset = _joint_offset(state, len(self.joint_pos_names))
            offset = offset.unsqueeze(1).expand(-1, steps, -1).reshape(joint_pos.shape)
            return joint_pos - offset
        if obs_key == "ref_body_pos_future_local":
            return _ref_body_pos_future_local(state)
        if obs_key == "ref_joint_pos_future":
            ref_joint_pos = _required_tensor(state, "ref_joint_pos_future")
            return ref_joint_pos.reshape(ref_joint_pos.shape[0], -1)
        if obs_key == "ref_motion_phase":
            return _ref_motion_phase(state)
        if obs_key == "ref_contact_pos_b":
            return _ref_contact_pos_b(state, yaw_only=bool(params.get("yaw_only", False)))
        if obs_key == "diff_contact_pos_b":
            return _diff_contact_pos_b(state)
        if obs_key == "object_xy_b":
            return _object_xy_b(state)
        if obs_key == "object_heading_b":
            return _object_heading_b(state)
        if obs_key == "object_pos_b":
            return _object_pos_b(state)
        if obs_key == "object_ori_b":
            return _object_ori_b(state)
        if obs_key == "prev_actions":
            return self._prev_actions(params, state)
        if obs_key == "applied_action":
            return _applied_action(state, self.action_dim)
        raise NotImplementedError(f"Unsupported MuJoCo observation '{obs_key}'.")

    def _select_history(self, obs_key: str, params: Mapping) -> torch.Tensor:
        if obs_key not in self._history:
            raise RuntimeError(f"History observation '{obs_key}' has not been initialized.")
        steps = torch.as_tensor(_history_steps(params), device=self._history[obs_key].device)
        return self._history[obs_key].index_select(1, steps).reshape(self._history[obs_key].shape[0], -1)

    def _prev_actions(self, params: Mapping, state: MujocoPolicyState) -> torch.Tensor:
        steps = int(params.get("steps", 1))
        flatten = bool(params.get("flatten", True))
        permute = bool(params.get("permute", False))
        action_history = state.action_history
        if action_history is None:
            batch = state.joint_pos.shape[0]
            action_history = torch.zeros(batch, self.action_dim, steps, dtype=state.joint_pos.dtype, device=state.joint_pos.device)
        if action_history.shape[-1] < steps:
            raise ValueError(f"action_history has {action_history.shape[-1]} steps, need {steps}.")
        action_history = action_history[:, :, :steps]
        if permute:
            action_history = action_history.permute(0, 2, 1)
        if flatten:
            return action_history.reshape(action_history.shape[0], -1)
        return action_history


def _history_steps(params: Mapping) -> list[int]:
    return list(params.get("history_steps", [0]))


def _joint_pos_names(observation_cfg: Mapping, fallback_joint_names: Sequence[str]) -> list[str]:
    fallback = [str(name) for name in fallback_joint_names]
    resolved: list[str] | None = None
    for group_cfg in observation_cfg.values():
        for obs_key, params in group_cfg.items():
            if obs_key != "joint_pos_history":
                continue
            params = params or {}
            names = params.get("joint_names")
            if names is None:
                names_t = fallback
            elif isinstance(names, str):
                names_t = [names]
            else:
                names_t = [str(name) for name in names]
            if resolved is None:
                resolved = names_t
            elif resolved != names_t:
                raise ValueError(
                    "MuJoCo policy export uses inconsistent joint_pos_history joint_names "
                    f"{resolved} and {names_t}."
                )
    return resolved if resolved is not None else fallback


def _joint_offset(state: MujocoPolicyState, joint_pos_dim: int) -> torch.Tensor:
    if state.joint_pos_offset is None:
        return torch.zeros_like(state.joint_pos)
    return _require_last_dim(state.joint_pos_offset, joint_pos_dim, "joint_pos_offset")


def _require_last_dim(value: torch.Tensor, expected_dim: int, name: str) -> torch.Tensor:
    if value.shape[-1] != expected_dim:
        raise ValueError(f"MujocoPolicyState.{name} dim {value.shape[-1]} != joint_pos_history dim {expected_dim}.")
    return value


def _applied_action(state: MujocoPolicyState, action_dim: int) -> torch.Tensor:
    if state.applied_action is None:
        return torch.zeros(state.joint_pos.shape[0], action_dim, dtype=state.joint_pos.dtype, device=state.joint_pos.device)
    return state.applied_action


def _required_tensor(state: MujocoPolicyState, attr: str) -> torch.Tensor:
    value = getattr(state, attr)
    if value is None:
        raise ValueError(f"MujocoPolicyState.{attr} is required for this observation.")
    return value


def _ref_body_pos_future_local(state: MujocoPolicyState) -> torch.Tensor:
    ref_body_pos_future_w = _required_tensor(state, "ref_body_pos_future_w")
    ref_root_pos_w = _required_tensor(state, "ref_root_pos_w")[:, None, None, :].clone()
    ref_root_quat_w = _required_tensor(state, "ref_root_quat_w")[:, None, None, :]

    ref_root_pos_w[..., 2] = 0.0
    ref_root_quat_w = _yaw_quat(ref_root_quat_w)
    ref_body_pos_future_local = _quat_rotate_inverse(ref_root_quat_w, ref_body_pos_future_w - ref_root_pos_w)
    return ref_body_pos_future_local.reshape(ref_body_pos_future_local.shape[0], -1)


def _ref_contact_pos_b(state: MujocoPolicyState, yaw_only: bool = False) -> torch.Tensor:
    contact_target_pos_w = _required_tensor(state, "contact_target_pos_w")
    robot_root_pos_w = _required_tensor(state, "robot_root_pos_w")[:, None, :]
    robot_root_quat_w = _required_tensor(state, "robot_root_quat_w")[:, None, :]
    if yaw_only:
        robot_root_quat_w = _yaw_quat(robot_root_quat_w)
    ref_contact_pos_b = _quat_rotate_inverse(robot_root_quat_w, contact_target_pos_w - robot_root_pos_w)
    return ref_contact_pos_b.reshape(ref_contact_pos_b.shape[0], -1)


def _diff_contact_pos_b(state: MujocoPolicyState) -> torch.Tensor:
    contact_target_pos_w = _required_tensor(state, "contact_target_pos_w")
    contact_eef_pos_w = _required_tensor(state, "contact_eef_pos_w")
    robot_root_quat_w = _required_tensor(state, "robot_root_quat_w")[:, None, :]
    if contact_target_pos_w.shape != contact_eef_pos_w.shape:
        raise ValueError(
            "contact_target_pos_w shape "
            f"{tuple(contact_target_pos_w.shape)} != contact_eef_pos_w shape {tuple(contact_eef_pos_w.shape)}."
        )
    diff_contact_pos_b = _quat_rotate_inverse(robot_root_quat_w, contact_target_pos_w - contact_eef_pos_w)
    return diff_contact_pos_b.reshape(diff_contact_pos_b.shape[0], -1)


def _object_xy_b(state: MujocoPolicyState) -> torch.Tensor:
    object_pos_w = _required_tensor(state, "object_pos_w")
    robot_root_pos_w = _required_tensor(state, "robot_root_pos_w")
    robot_root_quat_w = _yaw_quat(_required_tensor(state, "robot_root_quat_w"))
    object_pos_b = _quat_rotate_inverse(robot_root_quat_w, object_pos_w - robot_root_pos_w)
    return object_pos_b[:, :2]


def _object_heading_b(state: MujocoPolicyState) -> torch.Tensor:
    object_quat_w = _required_tensor(state, "object_quat_w")
    robot_root_quat_w = _required_tensor(state, "robot_root_quat_w")
    object_yaw_b = _wrap_to_pi(_yaw_from_quat(object_quat_w) - _yaw_from_quat(robot_root_quat_w))
    return torch.stack((torch.cos(object_yaw_b), torch.sin(object_yaw_b)), dim=-1)


def _object_pos_b(state: MujocoPolicyState) -> torch.Tensor:
    object_pos_w = _required_tensor(state, "object_pos_w")
    robot_root_pos_w = _required_tensor(state, "robot_root_pos_w")
    robot_root_quat_w = _required_tensor(state, "robot_root_quat_w")
    return _quat_rotate_inverse(robot_root_quat_w, object_pos_w - robot_root_pos_w)


def _object_ori_b(state: MujocoPolicyState) -> torch.Tensor:
    object_quat_w = _required_tensor(state, "object_quat_w")
    robot_root_quat_w = _required_tensor(state, "robot_root_quat_w")
    object_quat_b = _quat_mul(
        _quat_conjugate(robot_root_quat_w),
        object_quat_w,
    )
    object_ori_b = _matrix_from_quat(object_quat_b)
    return object_ori_b.reshape(object_ori_b.shape[0], -1)


def _ref_motion_phase(state: MujocoPolicyState) -> torch.Tensor:
    motion_t = _required_tensor(state, "motion_t")
    motion_len = _required_tensor(state, "motion_len")
    if motion_t.ndim == 1:
        motion_t = motion_t.unsqueeze(-1)
    if motion_len.ndim == 1:
        motion_len = motion_len.unsqueeze(-1)
    return motion_t.to(dtype=state.joint_pos.dtype) / motion_len.to(dtype=state.joint_pos.dtype).clamp_min(1)


def _yaw_quat(quat: torch.Tensor) -> torch.Tensor:
    qw, qx, qy, qz = torch.unbind(quat, dim=-1)
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    zeros = torch.zeros_like(yaw)
    return torch.stack((torch.cos(yaw / 2), zeros, zeros, torch.sin(yaw / 2)), dim=-1)


def _yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    qw, qx, qy, qz = torch.unbind(quat, dim=-1)
    return torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _quat_rotate_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    quat = quat.expand(*vec.shape[:-1], 4)
    xyz = quat[..., 1:]
    t = torch.linalg.cross(xyz, vec, dim=-1) * 2
    return vec - quat[..., 0:1] * t + torch.linalg.cross(xyz, t, dim=-1)


def _quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat((quat[..., :1], -quat[..., 1:]), dim=-1)


def _quat_mul(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = torch.unbind(lhs, dim=-1)
    w2, x2, y2, z2 = torch.unbind(rhs, dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _matrix_from_quat(quat: torch.Tensor) -> torch.Tensor:
    qw, qx, qy, qz = torch.unbind(quat, dim=-1)
    two = 2.0
    row0 = torch.stack(
        (
            1 - two * (qy * qy + qz * qz),
            two * (qx * qy - qz * qw),
            two * (qx * qz + qy * qw),
        ),
        dim=-1,
    )
    row1 = torch.stack(
        (
            two * (qx * qy + qz * qw),
            1 - two * (qx * qx + qz * qz),
            two * (qy * qz - qx * qw),
        ),
        dim=-1,
    )
    row2 = torch.stack(
        (
            two * (qx * qz - qy * qw),
            two * (qy * qz + qx * qw),
            1 - two * (qx * qx + qy * qy),
        ),
        dim=-1,
    )
    return torch.stack((row0, row1, row2), dim=-2)
