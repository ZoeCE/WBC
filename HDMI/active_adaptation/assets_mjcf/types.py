from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass(frozen=True)
class MJObjectSpec:
    asset_name: str
    body_names: Sequence[str]
    joint_names: Sequence[str] = ()


@dataclass
class MJArticulationCfg:
    mjcf_path: str
    init_state: Any
    actuators: dict
    body_names_isaac: Sequence[str]
    joint_names_isaac: Sequence[str]
    joint_symmetry_mapping: dict | None = None
    spatial_symmetry_mapping: dict | None = None
    object_specs: dict[str, MJObjectSpec] = field(default_factory=dict)
