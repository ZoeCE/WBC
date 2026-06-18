import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


HDMI_ROOT = Path(__file__).resolve().parents[1]
if str(HDMI_ROOT) not in sys.path:
    sys.path.insert(0, str(HDMI_ROOT))

from active_adaptation.assets_mjcf import ROBOTS
from active_adaptation.mujoco import (
    MujocoMotionReference,
    PlaybackParityMetrics,
    compute_kinematic_motion_playback_parity,
)


def run_parity(
    *,
    motion_dir: str | Path,
    robot_name: str = "g1_29dof",
    object_name: str | None = None,
    object_type: str | None = None,
    object_body_name: str | None = None,
    object_joint_name: str | None = None,
    root_body_name: str | None = None,
    steps: Sequence[int] | None = None,
    num_envs: int = 1,
    reward_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    motion_dir = Path(motion_dir)
    meta = _load_motion_meta(motion_dir)
    body_names = list(meta["body_names"])
    joint_names = list(meta["joint_names"])
    root_body_name = root_body_name or body_names[0]

    scene = _build_scene(
        robot_name=robot_name,
        object_name=object_name,
        object_type=object_type,
        num_envs=num_envs,
    )
    reference = MujocoMotionReference.from_motion_dir(
        motion_dir=motion_dir,
        body_names=body_names,
        joint_names=joint_names,
        root_body_name=root_body_name,
        future_steps=[0],
    )
    metrics = compute_kinematic_motion_playback_parity(
        scene=scene,
        reference=reference,
        steps=steps,
        reward_cfg=reward_config,
        object_name=object_name,
        object_body_name=object_body_name,
        object_joint_name=object_joint_name,
    )
    return summarize_metrics(metrics)


def summarize_metrics(metrics: PlaybackParityMetrics) -> dict[str, Any]:
    reward = metrics.reward
    summary = {
        "steps": int(metrics.q_l2.shape[0]),
        "envs": int(metrics.q_l2.shape[1]) if metrics.q_l2.ndim > 1 else 1,
        "q_l2_max": float(metrics.q_l2.max().item()),
        "q_l2_mean": float(metrics.q_l2.mean().item()),
        "body_pos_l2_max": float(metrics.body_pos_l2.max().item()),
        "body_pos_l2_mean": float(metrics.body_pos_l2.mean().item()),
        "reward_shape": None,
        "reward_mean": None,
    }
    if reward is not None:
        summary["reward_shape"] = list(reward.shape)
        summary["reward_mean"] = float(reward.mean().item())
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    with contextlib.redirect_stdout(sys.stderr):
        reward_config = _load_json(args.reward_config_json) if args.reward_config_json else None
        summary = run_parity(
            motion_dir=args.motion_dir,
            robot_name=args.robot_name,
            object_name=args.object_name,
            object_type=args.object_type,
            object_body_name=args.object_body_name,
            object_joint_name=args.object_joint_name,
            root_body_name=args.root_body_name,
            steps=_parse_steps(args.steps),
            num_envs=args.num_envs,
            reward_config=reward_config,
        )
    print(json.dumps(summary, sort_keys=True))
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MuJoCo kinematic playback parity on a motion.npz + meta.json directory."
    )
    parser.add_argument("--motion-dir", required=True, help="Directory containing motion.npz and meta.json.")
    parser.add_argument("--robot-name", default="g1_29dof")
    parser.add_argument("--object-name", default=None)
    parser.add_argument("--object-type", default=None)
    parser.add_argument("--object-body-name", default=None)
    parser.add_argument("--object-joint-name", default=None)
    parser.add_argument("--root-body-name", default=None)
    parser.add_argument("--steps", default=None, help="Comma-separated playback frame indices. Defaults to all frames.")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--reward-config-json", default=None)
    return parser.parse_args(argv)


def _parse_steps(raw_steps: str | None) -> list[int] | None:
    if raw_steps is None or raw_steps == "":
        return None
    steps = []
    for raw_step in raw_steps.split(","):
        raw_step = raw_step.strip()
        if not raw_step:
            continue
        steps.append(int(raw_step))
    return steps


def _load_motion_meta(motion_dir: Path) -> dict[str, Any]:
    meta_path = motion_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing motion metadata: {meta_path}")
    return json.loads(meta_path.read_text())


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _build_scene(
    *,
    robot_name: str,
    object_name: str | None,
    object_type: str | None,
    num_envs: int,
):
    from active_adaptation.envs import mujoco as mujoco_env

    class SceneCfg:
        pass

    if object_name is None:
        SceneCfg.robot = ROBOTS[robot_name]
    else:
        SceneCfg.robot = ROBOTS.with_object(
            robot_name,
            object_asset_name=object_name,
            object_type=object_type,
        )
    return mujoco_env.MJScene(SceneCfg(), num_envs=num_envs, launch_viewer=False)


if __name__ == "__main__":
    raise SystemExit(main())
