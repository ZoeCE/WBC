from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class MujocoObjectBodyRandomizationSample:
    mass: torch.Tensor
    inertia: torch.Tensor
    materials: torch.Tensor


@dataclass(frozen=True)
class MujocoObjectJointRandomizationSample:
    friction: torch.Tensor
    damping: torch.Tensor
    armature: torch.Tensor


def sample_object_body_randomization(
    *,
    default_mass: torch.Tensor,
    default_inertia: torch.Tensor,
    default_materials: torch.Tensor,
    mass_range: tuple[float, float],
    dynamic_friction_range: tuple[float, float],
    restitution_range: tuple[float, float],
    static_friction_range: tuple[float, float] | None = None,
    static_dynamic_friction_ratio_range: tuple[float, float] | None = None,
    shape_ids: torch.Tensor | Sequence[int] | None = None,
    generator: torch.Generator | None = None,
) -> MujocoObjectBodyRandomizationSample:
    """Sample MuJoCo object body randomization terms with HDMI Isaac parity."""
    _validate_static_friction_mode(static_friction_range, static_dynamic_friction_ratio_range)
    _validate_body_inputs(default_mass, default_inertia, default_materials)

    mass = _sample_uniform_like(default_mass, mass_range, generator=generator)
    inertia = _scale_inertia(default_inertia, mass / default_mass)
    materials = default_materials.clone()

    selected_shape_ids = _shape_ids(shape_ids, num_shapes=default_materials.shape[1], device=default_materials.device)
    if selected_shape_ids.numel() == 0:
        return MujocoObjectBodyRandomizationSample(mass=mass, inertia=inertia, materials=materials)

    num_envs = default_materials.shape[0]
    material_shape = (num_envs, 1)
    dynamic_friction = _sample_uniform(
        dynamic_friction_range,
        material_shape,
        dtype=default_materials.dtype,
        device=default_materials.device,
        generator=generator,
    )
    restitution = _sample_uniform(
        restitution_range,
        material_shape,
        dtype=default_materials.dtype,
        device=default_materials.device,
        generator=generator,
    )
    if static_friction_range is not None:
        static_friction = _sample_uniform(
            static_friction_range,
            material_shape,
            dtype=default_materials.dtype,
            device=default_materials.device,
            generator=generator,
        )
    else:
        static_friction_ratio = _sample_uniform(
            static_dynamic_friction_ratio_range,
            material_shape,
            dtype=default_materials.dtype,
            device=default_materials.device,
            generator=generator,
        )
        static_friction = dynamic_friction * static_friction_ratio

    shape_count = selected_shape_ids.numel()
    materials[:, selected_shape_ids, 0] = static_friction.expand(num_envs, shape_count)
    materials[:, selected_shape_ids, 1] = dynamic_friction.expand(num_envs, shape_count)
    materials[:, selected_shape_ids, 2] = restitution.expand(num_envs, shape_count)
    return MujocoObjectBodyRandomizationSample(mass=mass, inertia=inertia, materials=materials)


def sample_object_joint_randomization(
    *,
    num_envs: int,
    friction_range: tuple[float, float],
    damping_range: tuple[float, float],
    armature_range: tuple[float, float],
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> MujocoObjectJointRandomizationSample:
    """Sample per-env MuJoCo object joint randomization terms with HDMI Isaac parity."""
    if num_envs < 0:
        raise ValueError(f"num_envs must be non-negative, got {num_envs}.")
    target_device = torch.device("cpu") if device is None else torch.device(device)
    shape = (num_envs,)
    return MujocoObjectJointRandomizationSample(
        friction=_sample_uniform(friction_range, shape, dtype=dtype, device=target_device, generator=generator),
        damping=_sample_uniform(damping_range, shape, dtype=dtype, device=target_device, generator=generator),
        armature=_sample_uniform(armature_range, shape, dtype=dtype, device=target_device, generator=generator),
    )


def _validate_static_friction_mode(
    static_friction_range: tuple[float, float] | None,
    static_dynamic_friction_ratio_range: tuple[float, float] | None,
) -> None:
    has_static = static_friction_range is not None
    has_ratio = static_dynamic_friction_ratio_range is not None
    if has_static == has_ratio:
        raise ValueError(
            "Specify exactly one of static_friction_range or static_dynamic_friction_ratio_range."
        )


def _validate_body_inputs(
    default_mass: torch.Tensor,
    default_inertia: torch.Tensor,
    default_materials: torch.Tensor,
) -> None:
    if default_mass.ndim == 0:
        raise ValueError("default_mass must have at least one dimension.")
    if torch.any(default_mass == 0):
        raise ValueError("default_mass must not contain zeros.")
    if default_inertia.ndim not in (default_mass.ndim, default_mass.ndim + 1):
        raise ValueError(
            "default_inertia rank must match default_mass rank or add one trailing inertia dimension, "
            f"got mass {tuple(default_mass.shape)} and inertia {tuple(default_inertia.shape)}."
        )
    if default_inertia.shape[: default_mass.ndim] != default_mass.shape:
        raise ValueError(
            f"default_inertia leading shape {tuple(default_inertia.shape[: default_mass.ndim])} "
            f"!= default_mass shape {tuple(default_mass.shape)}."
        )
    if default_materials.ndim != 3 or default_materials.shape[-1] != 3:
        raise ValueError(
            "default_materials must have shape (num_envs, num_shapes, 3), "
            f"got {tuple(default_materials.shape)}."
        )
    if default_materials.shape[0] != default_mass.shape[0]:
        raise ValueError(
            f"default_materials env dim {default_materials.shape[0]} != default_mass env dim {default_mass.shape[0]}."
        )


def _scale_inertia(default_inertia: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if default_inertia.ndim == scale.ndim:
        return default_inertia * scale
    return default_inertia * scale.unsqueeze(-1)


def _shape_ids(
    shape_ids: torch.Tensor | Sequence[int] | None,
    *,
    num_shapes: int,
    device: torch.device,
) -> torch.Tensor:
    if shape_ids is None:
        ids = torch.arange(num_shapes, dtype=torch.long, device=device)
    else:
        ids = torch.as_tensor(shape_ids, dtype=torch.long, device=device)
    if ids.ndim != 1:
        raise ValueError(f"shape_ids must be a 1D tensor or sequence, got shape {tuple(ids.shape)}.")
    if ids.numel() and (torch.any(ids < 0) or torch.any(ids >= num_shapes)):
        raise ValueError(f"shape_ids must be in [0, {num_shapes}), got {ids.tolist()}.")
    return ids


def _sample_uniform_like(
    reference: torch.Tensor,
    value_range: tuple[float, float],
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    return _sample_uniform(value_range, reference.shape, dtype=reference.dtype, device=reference.device, generator=generator)


def _sample_uniform(
    value_range: tuple[float, float],
    shape: tuple[int, ...] | torch.Size,
    *,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    low, high = _range_bounds(value_range)
    if low == high:
        return torch.full(tuple(shape), low, dtype=dtype, device=device)
    return torch.empty(tuple(shape), dtype=dtype, device=device).uniform_(low, high, generator=generator)


def _range_bounds(value_range: tuple[float, float]) -> tuple[float, float]:
    if len(value_range) != 2:
        raise ValueError(f"range must contain exactly two values, got {value_range}.")
    low, high = float(value_range[0]), float(value_range[1])
    if low > high:
        raise ValueError(f"range lower bound {low} exceeds upper bound {high}.")
    return low, high
