import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


HDMI_ROOT = Path(__file__).resolve().parents[1]
if str(HDMI_ROOT) not in sys.path:
    sys.path.insert(0, str(HDMI_ROOT))

from active_adaptation.assets_mjcf import ROBOTS
from active_adaptation.mujoco import (
    MujocoMotionReference,
    PlaybackParityMetrics,
    compute_kinematic_motion_playback_parity,
)


KINEMATIC_REWARD_TERMS = {
    "keypoint_pos_tracking_product",
    "keypoint_position_tracking_product",
    "keypoint_pos_tracking_local_product",
    "keypoint_position_tracking_local_product",
    "joint_pos_tracking_product",
    "joint_position_tracking_product",
    "object_pos_tracking",
    "object_ori_tracking",
    "object_joint_pos_tracking",
}


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
    reward_config, reward_terms_used, reward_terms_skipped = filter_kinematic_reward_config(reward_config)
    metrics = compute_kinematic_motion_playback_parity(
        scene=scene,
        reference=reference,
        steps=steps,
        reward_cfg=reward_config,
        object_name=object_name,
        object_body_name=object_body_name,
        object_joint_name=object_joint_name,
    )
    return summarize_metrics(
        metrics,
        reward_terms_used=reward_terms_used,
        reward_terms_skipped=reward_terms_skipped,
    )


def summarize_metrics(
    metrics: PlaybackParityMetrics,
    *,
    reward_terms_used: Sequence[str] = (),
    reward_terms_skipped: Sequence[str] = (),
) -> dict[str, Any]:
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
        "reward_terms_used": list(reward_terms_used),
        "reward_terms_skipped": list(reward_terms_skipped),
    }
    if reward is not None:
        summary["reward_shape"] = list(reward.shape)
        summary["reward_mean"] = float(reward.mean().item())
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    with contextlib.redirect_stdout(sys.stderr):
        reward_config = _load_reward_config_from_args(args)
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
    parser.add_argument("--task-yaml", default=None, help="HDMI task YAML. Its reward section is used for playback.")
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


def _load_reward_config_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.reward_config_json and args.task_yaml:
        raise ValueError("--reward-config-json and --task-yaml are mutually exclusive.")
    if args.reward_config_json:
        return _load_json(args.reward_config_json)
    if args.task_yaml:
        return load_task_reward_config(args.task_yaml)
    return None


def load_task_reward_config(task_yaml: str | Path) -> dict[str, Any]:
    cfg = _load_task_yaml_with_defaults(Path(task_yaml))
    reward_cfg = cfg.get("reward", {})
    if not isinstance(reward_cfg, dict):
        raise ValueError(f"Task YAML reward section must be a mapping, got {type(reward_cfg).__name__}.")
    return reward_cfg


def filter_kinematic_reward_config(
    reward_config: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, list[str], list[str]]:
    if reward_config is None:
        return None, [], []

    filtered: dict[str, Any] = {}
    used: list[str] = []
    skipped: list[str] = []
    for group_name, group_cfg in reward_config.items():
        if group_name == "_mult_dt_":
            continue
        if not isinstance(group_cfg, Mapping):
            continue

        group_out: dict[str, Any] = {}
        if bool(group_cfg.get("_multiplicative", False)):
            group_out["_multiplicative"] = True

        for term_name, term_cfg in group_cfg.items():
            if term_name == "_multiplicative" or term_cfg is None:
                continue
            if not isinstance(term_cfg, Mapping):
                term_cfg = {}
            if not bool(term_cfg.get("enabled", True)):
                continue

            formula_name = _reward_formula_name(term_name)
            qualified_name = f"{group_name}.{term_name}"
            if formula_name not in KINEMATIC_REWARD_TERMS:
                skipped.append(qualified_name)
                continue
            group_out[term_name] = dict(term_cfg)
            used.append(qualified_name)

        if any(key != "_multiplicative" for key in group_out):
            filtered[group_name] = group_out

    return (filtered or None), used, skipped


def _load_task_yaml_with_defaults(path: Path) -> dict[str, Any]:
    raw_cfg = _load_yaml_mapping(path)
    defaults = raw_cfg.pop("defaults", None)
    self_cfg = raw_cfg
    if defaults is None:
        return self_cfg

    merged: dict[str, Any] = {}
    inserted_self = False
    for entry in defaults:
        if entry == "_self_":
            merged = _deep_merge_dicts(merged, self_cfg)
            inserted_self = True
            continue
        default_path = _resolve_default_yaml(path, entry)
        merged = _deep_merge_dicts(merged, _load_task_yaml_with_defaults(default_path))
    if not inserted_self:
        merged = _deep_merge_dicts(merged, self_cfg)
    return merged


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text()) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return cfg


def _resolve_default_yaml(path: Path, entry: Any) -> Path:
    if isinstance(entry, Mapping):
        if len(entry) != 1:
            raise ValueError(f"Unsupported Hydra default entry in {path}: {entry!r}")
        entry = next(iter(entry.values()))
    if not isinstance(entry, str):
        raise ValueError(f"Unsupported Hydra default entry in {path}: {entry!r}")
    if entry == "_self_":
        return path

    rel = Path(entry.lstrip("/")).with_suffix(".yaml")
    for root in (path.parent, *path.parents):
        candidate = root / rel
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not resolve Hydra default {entry!r} from {path}.")


def _deep_merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _reward_formula_name(term_name: str) -> str:
    if "(" not in term_name:
        return term_name
    if not term_name.endswith(")"):
        raise ValueError(f"Invalid reward term alias syntax: {term_name!r}.")
    return term_name.rsplit("(", 1)[1][:-1]


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
