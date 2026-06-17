from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class MJArticulationCfg:
    mjcf_path: str
    init_state: Any
    actuators: dict
    body_names_isaac: Sequence[str]
    joint_names_isaac: Sequence[str]
    joint_symmetry_mapping: dict | None = None
    spatial_symmetry_mapping: dict | None = None
