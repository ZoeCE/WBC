from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import re

import torch
import yaml
from tensordict import TensorDict, TensorDictBase

from .observation_builder import MujocoObservationBuilder, MujocoPolicyState


@dataclass(frozen=True)
class MujocoPolicyAction:
    raw_action: torch.Tensor
    scaled_action: torch.Tensor
    joint_position_target: torch.Tensor
    tensordict: TensorDictBase


@dataclass
class MujocoPolicyBundle:
    policy: torch.nn.Module
    config: Mapping[str, Any]
    observation_builder: MujocoObservationBuilder
    policy_joint_names: list[str]
    observation_joint_names: list[str]
    action_scale: torch.Tensor
    default_joint_pos: torch.Tensor
    observation_default_joint_pos: torch.Tensor
    policy_path: Path
    config_path: Path

    @classmethod
    def load(
        cls,
        policy_path: str | Path,
        config_path: str | Path | None = None,
        *,
        map_location: str | torch.device = "cpu",
    ) -> "MujocoPolicyBundle":
        policy_path = Path(policy_path)
        config_path = Path(config_path) if config_path is not None else _default_config_path(policy_path)
        config = _load_yaml_mapping(config_path)
        observation_cfg = _required_mapping(config, "observation", config_path)
        policy_joint_names = _required_string_list(config, "policy_joint_names", config_path)
        fallback_observation_joint_names = _optional_string_list(config, "isaac_joint_names") or policy_joint_names
        observation_builder = MujocoObservationBuilder(
            observation_cfg,
            policy_joint_names=policy_joint_names,
            observation_joint_names=fallback_observation_joint_names,
        )

        policy = _torch_load_policy(policy_path, map_location=map_location)
        policy.eval()
        action_scale = resolve_named_values(
            config.get("action_scale", config.get("action_scaling", 1.0)),
            policy_joint_names,
            field_name="action_scale",
            require_all=True,
        )
        default_joint_pos = resolve_named_values(
            config.get("default_joint_pos", 0.0),
            policy_joint_names,
            field_name="default_joint_pos",
            require_all=False,
            default=0.0,
        )
        observation_default_joint_pos = resolve_named_values(
            config.get("default_joint_pos", 0.0),
            observation_builder.joint_pos_names,
            field_name="default_joint_pos",
            require_all=False,
            default=0.0,
        )
        return cls(
            policy=policy,
            config=config,
            observation_builder=observation_builder,
            policy_joint_names=policy_joint_names,
            observation_joint_names=observation_builder.joint_pos_names,
            action_scale=action_scale,
            default_joint_pos=default_joint_pos,
            observation_default_joint_pos=observation_default_joint_pos,
            policy_path=policy_path,
            config_path=config_path,
        )

    @property
    def action_dim(self) -> int:
        return len(self.policy_joint_names)

    def reset(self, state: MujocoPolicyState) -> None:
        self.observation_builder.reset(state)

    def update(self, state: MujocoPolicyState) -> None:
        self.observation_builder.update(state)

    def build_tensordict(
        self,
        state: MujocoPolicyState,
        *,
        is_init: torch.Tensor | bool | None = None,
        extra_inputs: Mapping[str, Any] | None = None,
    ) -> TensorDict:
        data = {
            group_name: self.observation_builder.build_group(group_name, state)
            for group_name in self.observation_builder.observation_cfg
        }
        if not data:
            raise ValueError(f"Policy config {self.config_path} does not contain observation groups.")

        first_tensor = next(iter(data.values()))
        batch = first_tensor.shape[0]
        data["is_init"] = _is_init_tensor(is_init, batch, first_tensor)
        if extra_inputs:
            data.update(extra_inputs)
        return TensorDict(data, batch_size=[batch])

    def act(
        self,
        state: MujocoPolicyState,
        *,
        is_init: torch.Tensor | bool | None = None,
        extra_inputs: Mapping[str, Any] | None = None,
    ) -> MujocoPolicyAction:
        td = self.build_tensordict(state, is_init=is_init, extra_inputs=extra_inputs)
        with torch.inference_mode():
            out = self.policy(td)
        raw_action = out["action"]
        scaled_action = self.scale_action(raw_action)
        return MujocoPolicyAction(
            raw_action=raw_action,
            scaled_action=scaled_action,
            joint_position_target=self.joint_position_target(raw_action),
            tensordict=out,
        )

    def scale_action(self, raw_action: torch.Tensor) -> torch.Tensor:
        _require_action_dim(raw_action, self.action_dim)
        scale = self.action_scale.to(device=raw_action.device, dtype=raw_action.dtype)
        return raw_action * scale

    def joint_position_target(self, raw_action: torch.Tensor) -> torch.Tensor:
        default = self.default_joint_pos.to(device=raw_action.device, dtype=raw_action.dtype)
        return default + self.scale_action(raw_action)


def resolve_named_values(
    spec: Any,
    names: Sequence[str],
    *,
    field_name: str,
    require_all: bool,
    default: float | None = None,
) -> torch.Tensor:
    if isinstance(spec, (int, float)):
        return torch.full((len(names),), float(spec), dtype=torch.float32)
    if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)) and not isinstance(spec, Mapping):
        if len(spec) != len(names):
            raise ValueError(f"{field_name} length {len(spec)} does not match {len(names)} policy joints.")
        return torch.as_tensor(spec, dtype=torch.float32)
    if not isinstance(spec, Mapping):
        raise TypeError(f"{field_name} must be a scalar, sequence, or mapping, got {type(spec).__name__}.")

    values: list[float] = []
    unmatched: list[str] = []
    for name in names:
        matches = [(pattern, value) for pattern, value in spec.items() if _matches_name(str(pattern), name)]
        if not matches:
            if require_all:
                unmatched.append(name)
                values.append(0.0)
            else:
                values.append(float(default if default is not None else 0.0))
            continue
        if len(matches) > 1:
            patterns = ", ".join(pattern for pattern, _ in matches)
            raise ValueError(f"{field_name} has multiple matches for joint {name!r}: {patterns}.")
        values.append(float(matches[0][1]))

    if unmatched:
        raise ValueError(f"{field_name} does not define values for policy joints: {unmatched}.")
    return torch.tensor(values, dtype=torch.float32)


def _matches_name(pattern: str, name: str) -> bool:
    if pattern == name:
        return True
    try:
        return re.fullmatch(pattern, name) is not None
    except re.error:
        return False


def _torch_load_policy(policy_path: Path, *, map_location: str | torch.device) -> torch.nn.Module:
    try:
        return torch.load(policy_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(policy_path, map_location=map_location)


def _default_config_path(policy_path: Path) -> Path:
    for suffix in (".yaml", ".yml"):
        candidate = policy_path.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    return policy_path.with_suffix(".yaml")


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing exported policy config: {path}")
    cfg = yaml.safe_load(path.read_text()) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Exported policy config must be a mapping: {path}")
    return cfg


def _required_mapping(config: Mapping[str, Any], key: str, path: Path) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Exported policy config {path} must contain mapping key {key!r}.")
    return value


def _required_string_list(config: Mapping[str, Any], key: str, path: Path) -> list[str]:
    value = config.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"Exported policy config {path} must contain list key {key!r}.")
    return [str(item) for item in value]


def _optional_string_list(config: Mapping[str, Any], key: str) -> list[str] | None:
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"Exported policy config key {key!r} must be a list when provided.")
    return [str(item) for item in value]


def _is_init_tensor(is_init: torch.Tensor | bool | None, batch: int, like: torch.Tensor) -> torch.Tensor:
    if is_init is None:
        return torch.zeros(batch, 1, dtype=torch.bool, device=like.device)
    if isinstance(is_init, bool):
        return torch.full((batch, 1), is_init, dtype=torch.bool, device=like.device)
    if is_init.ndim == 0:
        is_init = is_init.reshape(1, 1).expand(batch, 1)
    elif is_init.ndim == 1:
        is_init = is_init.reshape(batch, 1)
    return is_init.to(device=like.device, dtype=torch.bool)


def _require_action_dim(action: torch.Tensor, action_dim: int) -> None:
    if action.shape[-1] != action_dim:
        raise ValueError(f"Policy action dim {action.shape[-1]} != exported action dim {action_dim}.")
