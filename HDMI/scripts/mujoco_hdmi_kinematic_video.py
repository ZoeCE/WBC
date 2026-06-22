from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import yaml


os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco


SCRIPT_DIR = Path(__file__).resolve().parent
HDMI_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, HDMI_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from active_adaptation.assets_mjcf import ROBOTS
from active_adaptation.mujoco.task_mapping import validate_task_motion_mapping


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_render_report(
        task_yaml=Path(args.task_yaml),
        output_path=Path(args.output),
        robot_name=args.robot_name,
        max_frames=args.max_frames,
        fps=args.fps,
        width=args.width,
        height=args.height,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


def build_render_report(
    *,
    task_yaml: Path,
    output_path: Path,
    robot_name: str = "g1_29dof",
    max_frames: int = 120,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
    dry_run: bool = False,
) -> dict[str, Any]:
    task_cfg = _load_yaml_mapping(task_yaml)
    task_name = str(task_cfg.get("name") or task_yaml.stem)
    command_cfg = task_cfg.get("command") if isinstance(task_cfg.get("command"), dict) else {}
    object_asset_name = str(command_cfg.get("object_asset_name") or "")
    object_type = str(command_cfg.get("object_type") or object_asset_name)
    if not object_asset_name:
        raise ValueError(f"{task_yaml}: command.object_asset_name is required for video rendering.")

    mapping = validate_task_motion_mapping(task_yaml, robot_name=robot_name)
    robot_cfg = ROBOTS.with_object(robot_name, object_asset_name=object_asset_name, object_type=object_type)
    motion = _load_motion(mapping.motion_dir)
    model = mujoco.MjModel.from_xml_path(str(robot_cfg.mjcf_path))
    render_frame_indices = _frame_indices(motion["frame_count"], max_frames=max_frames)
    qpos_mapping = _build_qpos_mapping(model, motion["body_names"], motion["joint_names"])

    report = {
        "task_name": task_name,
        "task_stem": task_yaml.stem,
        "task_yaml": str(task_yaml),
        "motion_dir": str(mapping.motion_dir),
        "mjcf_path": str(robot_cfg.mjcf_path),
        "output_path": str(output_path),
        "dry_run": bool(dry_run),
        "fps": int(fps),
        "width": int(width),
        "height": int(height),
        "max_frames": int(max_frames),
        "motion_frame_count": int(motion["frame_count"]),
        "render_frame_count": int(len(render_frame_indices)),
        "free_joint_mappings": qpos_mapping["free_joint_mappings"],
        "hinge_joint_count": len(qpos_mapping["hinge_joint_mappings"]),
        "mapped_hinge_joint_names": [entry["joint_name"] for entry in qpos_mapping["hinge_joint_mappings"]],
        "output_exists": output_path.is_file(),
        "output_size_bytes": output_path.stat().st_size if output_path.is_file() else 0,
    }
    if dry_run:
        return report

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _render_video(
        model=model,
        motion=motion,
        qpos_mapping=qpos_mapping,
        frame_indices=render_frame_indices,
        output_path=output_path,
        fps=fps,
        width=width,
        height=height,
    )
    report["output_exists"] = output_path.is_file()
    report["output_size_bytes"] = output_path.stat().st_size if output_path.is_file() else 0
    return report


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a headless MuJoCo kinematic playback video for one HDMI task.")
    parser.add_argument("--task-yaml", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--robot-name", default="g1_29dof")
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected YAML mapping, got {type(data).__name__}.")
    return data


def _load_motion(motion_dir: Path) -> dict[str, Any]:
    meta = json.loads((motion_dir / "meta.json").read_text(encoding="utf-8"))
    data = np.load(motion_dir / "motion.npz")
    body_pos_w = data["body_pos_w"]
    return {
        "body_names": list(meta["body_names"]),
        "joint_names": list(meta["joint_names"]),
        "body_pos_w": body_pos_w,
        "body_quat_w": data["body_quat_w"],
        "joint_pos": data["joint_pos"],
        "frame_count": int(body_pos_w.shape[0]),
    }


def _frame_indices(frame_count: int, *, max_frames: int) -> np.ndarray:
    if frame_count <= 0:
        raise ValueError("motion contains no frames.")
    count = min(int(max_frames), int(frame_count))
    if count <= 0:
        raise ValueError("max_frames must be positive.")
    return np.linspace(0, frame_count - 1, count, dtype=int)


def _build_qpos_mapping(model: mujoco.MjModel, body_names: Sequence[str], joint_names: Sequence[str]) -> dict[str, Any]:
    body_index = {name: index for index, name in enumerate(body_names)}
    joint_index = {name: index for index, name in enumerate(joint_names)}
    free_joint_mappings = []
    hinge_joint_mappings = []
    for joint_id in range(model.njnt):
        joint = model.joint(joint_id)
        joint_name = joint.name
        qpos_address = int(model.jnt_qposadr[joint_id])
        if joint.type == mujoco.mjtJoint.mjJNT_FREE:
            body_name = joint_name[:-5] if joint_name.endswith("_root") else joint_name
            if body_name in body_index:
                free_joint_mappings.append(
                    {
                        "joint_name": joint_name,
                        "body_name": body_name,
                        "qpos_address": qpos_address,
                        "motion_body_index": int(body_index[body_name]),
                    }
                )
            continue
        if joint_name in joint_index:
            hinge_joint_mappings.append(
                {
                    "joint_name": joint_name,
                    "qpos_address": qpos_address,
                    "motion_joint_index": int(joint_index[joint_name]),
                }
            )
    return {
        "free_joint_mappings": free_joint_mappings,
        "hinge_joint_mappings": hinge_joint_mappings,
    }


def _render_video(
    *,
    model: mujoco.MjModel,
    motion: dict[str, Any],
    qpos_mapping: dict[str, Any],
    frame_indices: Sequence[int],
    output_path: Path,
    fps: int,
    width: int,
    height: int,
) -> None:
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=int(height), width=int(width))
    frames = []
    try:
        for step in frame_indices:
            data.qpos[:] = 0.0
            for entry in qpos_mapping["free_joint_mappings"]:
                qpos_address = int(entry["qpos_address"])
                body_index = int(entry["motion_body_index"])
                data.qpos[qpos_address : qpos_address + 3] = motion["body_pos_w"][step, body_index]
                data.qpos[qpos_address + 3 : qpos_address + 7] = motion["body_quat_w"][step, body_index]
            for entry in qpos_mapping["hinge_joint_mappings"]:
                data.qpos[int(entry["qpos_address"])] = motion["joint_pos"][step, int(entry["motion_joint_index"])]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data)
            frames.append(renderer.render())
    finally:
        renderer.close()
    imageio.mimsave(output_path, frames, fps=int(fps))


if __name__ == "__main__":
    raise SystemExit(main())
