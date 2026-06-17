import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class MujocoReferenceObservationFields:
    ref_body_pos_future_w: torch.Tensor
    ref_root_pos_w: torch.Tensor
    ref_root_quat_w: torch.Tensor
    ref_joint_pos_future: torch.Tensor
    motion_t: torch.Tensor
    motion_len: torch.Tensor


@dataclass(frozen=True)
class MujocoMotionReference:
    body_names: list[str]
    joint_names: list[str]
    requested_body_names: list[str]
    requested_joint_names: list[str]
    root_body_name: str
    future_steps: torch.Tensor
    body_indices: torch.Tensor
    joint_indices: torch.Tensor
    root_body_index: int
    body_pos_w: torch.Tensor
    body_quat_w: torch.Tensor
    joint_pos: torch.Tensor
    fps: float

    @classmethod
    def from_motion_dir(
        cls,
        motion_dir: str | Path,
        body_names: Sequence[str],
        joint_names: Sequence[str],
        root_body_name: str,
        future_steps: Sequence[int],
    ) -> "MujocoMotionReference":
        motion_dir = Path(motion_dir)
        meta = json.loads((motion_dir / "meta.json").read_text())
        motion = np.load(motion_dir / "motion.npz")

        available_body_names = list(meta["body_names"])
        available_joint_names = list(meta["joint_names"])
        requested_body_names = list(body_names)
        requested_joint_names = list(joint_names)

        _check_unique(available_body_names, "body")
        _check_unique(available_joint_names, "joint")
        body_indices = _indices_for(requested_body_names, available_body_names, "body")
        joint_indices = _indices_for(requested_joint_names, available_joint_names, "joint")
        root_body_index = _indices_for([root_body_name], available_body_names, "body")[0]

        return cls(
            body_names=available_body_names,
            joint_names=available_joint_names,
            requested_body_names=requested_body_names,
            requested_joint_names=requested_joint_names,
            root_body_name=root_body_name,
            future_steps=torch.as_tensor(list(future_steps), dtype=torch.long),
            body_indices=torch.as_tensor(body_indices, dtype=torch.long),
            joint_indices=torch.as_tensor(joint_indices, dtype=torch.long),
            root_body_index=root_body_index,
            body_pos_w=torch.as_tensor(motion["body_pos_w"], dtype=torch.float32),
            body_quat_w=torch.as_tensor(motion["body_quat_w"], dtype=torch.float32),
            joint_pos=torch.as_tensor(motion["joint_pos"], dtype=torch.float32),
            fps=float(meta["fps"]),
        )

    @property
    def num_steps(self) -> int:
        return int(self.body_pos_w.shape[0])

    def observation_fields_at(self, step: int | torch.Tensor) -> MujocoReferenceObservationFields:
        step = torch.as_tensor(step, dtype=torch.long, device=self.body_pos_w.device)
        if step.ndim == 0:
            step = step.unsqueeze(0)
        future_indices = step[:, None] + self.future_steps.to(step.device)[None]
        future_indices = future_indices.clamp_max(self.num_steps - 1)
        root_indices = future_indices[:, 0]

        return MujocoReferenceObservationFields(
            ref_body_pos_future_w=self.body_pos_w[future_indices][:, :, self.body_indices],
            ref_root_pos_w=self.body_pos_w[root_indices, self.root_body_index],
            ref_root_quat_w=self.body_quat_w[root_indices, self.root_body_index],
            ref_joint_pos_future=self.joint_pos[future_indices][:, :, self.joint_indices],
            motion_t=step,
            motion_len=torch.full_like(step, self.num_steps),
        )


def _check_unique(names: Sequence[str], kind: str) -> None:
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate {kind} names in motion metadata: {duplicates}")


def _indices_for(requested_names: Sequence[str], available_names: Sequence[str], kind: str) -> list[int]:
    missing = [name for name in requested_names if name not in available_names]
    if missing:
        raise ValueError(f"missing {kind} names in motion metadata: {missing}")
    return [available_names.index(name) for name in requested_names]
