import pytest
import torch

from active_adaptation.mujoco.domain_randomization import (
    sample_object_body_randomization,
    sample_object_joint_randomization,
)


def test_object_body_randomization_scales_mass_inertia_and_selected_materials():
    default_mass = torch.tensor(
        [
            [2.0, 4.0],
            [1.0, 3.0],
        ]
    )
    default_inertia = torch.tensor(
        [
            [[2.0, 4.0, 6.0], [4.0, 8.0, 12.0]],
            [[1.0, 2.0, 3.0], [3.0, 6.0, 9.0]],
        ]
    )
    default_materials = torch.zeros(2, 4, 3)

    sampled = sample_object_body_randomization(
        default_mass=default_mass,
        default_inertia=default_inertia,
        default_materials=default_materials,
        mass_range=(3.0, 3.0),
        dynamic_friction_range=(0.7, 0.7),
        restitution_range=(0.2, 0.2),
        static_dynamic_friction_ratio_range=(1.5, 1.5),
        shape_ids=torch.tensor([1, 3]),
    )

    assert torch.allclose(sampled.mass, torch.full_like(default_mass, 3.0))
    expected_scale = torch.full_like(default_mass, 3.0) / default_mass
    assert torch.allclose(sampled.inertia, default_inertia * expected_scale.unsqueeze(-1))

    assert torch.allclose(sampled.materials[:, [1, 3], 0], torch.full((2, 2), 1.05))
    assert torch.allclose(sampled.materials[:, [1, 3], 1], torch.full((2, 2), 0.7))
    assert torch.allclose(sampled.materials[:, [1, 3], 2], torch.full((2, 2), 0.2))
    assert torch.allclose(sampled.materials[:, [0, 2]], torch.zeros(2, 2, 3))


def test_object_body_randomization_accepts_static_friction_range_directly():
    sampled = sample_object_body_randomization(
        default_mass=torch.ones(1, 1),
        default_inertia=torch.ones(1, 1, 3),
        default_materials=torch.zeros(1, 2, 3),
        mass_range=(2.0, 2.0),
        static_friction_range=(0.8, 0.8),
        dynamic_friction_range=(0.5, 0.5),
        restitution_range=(0.1, 0.1),
    )

    assert torch.allclose(sampled.materials[..., 0], torch.full((1, 2), 0.8))
    assert torch.allclose(sampled.materials[..., 1], torch.full((1, 2), 0.5))
    assert torch.allclose(sampled.materials[..., 2], torch.full((1, 2), 0.1))


def test_object_body_randomization_requires_one_static_friction_mode():
    kwargs = dict(
        default_mass=torch.ones(1, 1),
        default_inertia=torch.ones(1, 1, 3),
        default_materials=torch.zeros(1, 1, 3),
        mass_range=(1.0, 1.0),
        dynamic_friction_range=(0.5, 0.5),
        restitution_range=(0.0, 0.0),
    )

    with pytest.raises(ValueError, match="exactly one"):
        sample_object_body_randomization(**kwargs)

    with pytest.raises(ValueError, match="exactly one"):
        sample_object_body_randomization(
            **kwargs,
            static_friction_range=(0.5, 0.5),
            static_dynamic_friction_ratio_range=(1.0, 1.0),
        )


def test_object_joint_randomization_samples_per_env_terms():
    sampled = sample_object_joint_randomization(
        num_envs=3,
        friction_range=(0.1, 0.1),
        damping_range=(2.0, 2.0),
        armature_range=(0.03, 0.03),
    )

    assert torch.allclose(sampled.friction, torch.full((3,), 0.1))
    assert torch.allclose(sampled.damping, torch.full((3,), 2.0))
    assert torch.allclose(sampled.armature, torch.full((3,), 0.03))
