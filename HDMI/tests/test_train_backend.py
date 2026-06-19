import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from hydra import compose, initialize_config_dir
import active_adaptation as aa
from omegaconf import OmegaConf
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load_train_module():
    script_path = ROOT / "scripts/train.py"
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("train_script_for_backend_test", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_train_sequential_module():
    script_path = ROOT / "scripts/train_sequential.py"
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("train_sequential_script_for_backend_test", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_play_module():
    script_path = ROOT / "scripts/play.py"
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("play_script_for_backend_test", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_eval_module():
    script_path = ROOT / "scripts/eval.py"
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("eval_script_for_backend_test", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_render_module():
    script_path = ROOT / "scripts/render.py"
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("render_script_for_backend_test", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_eval_multiple_module():
    script_path = ROOT / "scripts/eval_multiple.py"
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("eval_multiple_script_for_backend_test", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_eval_run_module():
    script_path = ROOT / "scripts/eval_run.py"
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("eval_run_script_for_backend_test", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compose_mujoco_train_smoke_cfg():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    with initialize_config_dir(config_dir=str((ROOT / "cfg").resolve()), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "backend=mujoco",
                "task=G1/tracking/walk",
                "task.num_envs=1",
                "task.max_episode_length=4",
                "task.randomization={}",
                "algo.train_every=2",
                "wandb.mode=disabled",
                "total_frames=2",
            ],
        )
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    return cfg


def test_train_config_declares_backend_override_key():
    cfg = yaml.safe_load((ROOT / "cfg/train.yaml").read_text())

    assert cfg["backend"] == "isaac"


def test_play_config_declares_export_policy_benchmark_iters():
    cfg = yaml.safe_load((ROOT / "cfg/play.yaml").read_text())
    assert cfg["export_policy_benchmark_iters"] == 1000


def test_train_sequential_config_declares_backend_override_key():
    cfg = yaml.safe_load((ROOT / "cfg/train_sequential.yaml").read_text())
    assert cfg["backend"] == "isaac"


def test_play_config_declares_backend_override_key():
    cfg = yaml.safe_load((ROOT / "cfg/play.yaml").read_text())
    assert cfg["backend"] == "isaac"


def test_eval_config_declares_backend_override_key():
    cfg = yaml.safe_load((ROOT / "cfg/eval.yaml").read_text())
    assert cfg["backend"] == "isaac"


def test_render_config_declares_backend_override_key():
    cfg = yaml.safe_load((ROOT / "cfg/render.yaml").read_text())
    assert cfg["backend"] == "isaac"


def test_scripts_do_not_import_isaac_app_at_module_import_time():
    offenders = []
    for script_path in sorted((ROOT / "scripts").glob("*.py")):
        tree = ast.parse(script_path.read_text(), filename=str(script_path))
        offenders.extend(
            f"{script_path.name}:{node.lineno}"
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module == "isaaclab.app"
            for alias in node.names
            if alias.name == "AppLauncher"
        )

    assert offenders == []


def test_mujoco_backend_sets_backend_without_launching_isaac_app():
    aa.set_backend("isaac")
    script = _load_train_module()
    cfg = OmegaConf.create({"backend": "mujoco", "app": {"headless": True}})

    try:
        simulation_app = script._configure_backend_and_app(cfg)

        assert simulation_app is None
        assert aa.get_backend() == "mujoco"
    finally:
        aa.set_backend("isaac")


def test_train_sequential_mujoco_backend_sets_backend_without_launching_isaac_app():
    aa.set_backend("isaac")
    script = _load_train_sequential_module()
    cfg = OmegaConf.create({"backend": "mujoco", "app": {"headless": True, "enable_cameras": False}})

    try:
        simulation_app = script._configure_backend_and_app(cfg)

        assert simulation_app is None
        assert aa.get_backend() == "mujoco"
    finally:
        aa.set_backend("isaac")


def test_play_mujoco_backend_sets_backend_without_launching_isaac_app():
    aa.set_backend("isaac")
    script = _load_play_module()
    cfg = OmegaConf.create({"backend": "mujoco", "app": {"headless": True, "enable_cameras": False}})

    try:
        simulation_app = script._configure_backend_and_app(cfg)

        assert simulation_app is None
        assert aa.get_backend() == "mujoco"
    finally:
        aa.set_backend("isaac")


def test_eval_mujoco_backend_sets_backend_without_launching_isaac_app():
    aa.set_backend("isaac")
    script = _load_eval_module()
    cfg = OmegaConf.create({"backend": "mujoco", "app": {"headless": True, "enable_cameras": False}})

    try:
        simulation_app = script._configure_backend_and_app(cfg)

        assert simulation_app is None
        assert aa.get_backend() == "mujoco"
    finally:
        aa.set_backend("isaac")


def test_render_mujoco_backend_sets_backend_without_launching_isaac_app():
    aa.set_backend("isaac")
    script = _load_render_module()
    cfg = OmegaConf.create({"backend": "mujoco", "app": {"headless": True, "enable_cameras": False}})

    try:
        simulation_app = script._configure_backend_and_app(cfg)

        assert simulation_app is None
        assert aa.get_backend() == "mujoco"
    finally:
        aa.set_backend("isaac")


def test_eval_multiple_mujoco_backend_sets_backend_without_launching_isaac_app():
    aa.set_backend("isaac")
    script = _load_eval_multiple_module()
    cfg = OmegaConf.create({"backend": "mujoco", "app": {"headless": True, "enable_cameras": False}})

    try:
        simulation_app = script._configure_backend_and_app(cfg)

        assert simulation_app is None
        assert aa.get_backend() == "mujoco"
    finally:
        aa.set_backend("isaac")


def test_eval_run_parser_accepts_mujoco_playback_flag():
    script = _load_eval_run_module()

    args = script._parse_args(["--run_path", "entity/project/run", "--play-mujoco"])

    assert args.run_path == "entity/project/run"
    assert args.play_mujoco is True


def test_play_mujoco_asset_meta_uses_mjcf_cfg_without_isaac_assets_import():
    script = _load_play_module()
    robot_cfg = SimpleNamespace(
        joint_names_isaac=["joint_a", "joint_b"],
        body_names_isaac=["body_a"],
        actuators={
            "group": {
                "stiffness": {"joint_a": 10.0, "joint_b": 20.0},
                "damping": {"joint_a": 1.0, "joint_b": 2.0},
            },
        },
        init_state={"joint_pos": {"joint_a": 0.1, "joint_b": -0.2}},
    )
    env = SimpleNamespace(scene={"robot": SimpleNamespace(cfg=robot_cfg)})

    meta = script._get_policy_asset_meta(env)

    assert meta == {
        "joint_names_isaac": ["joint_a", "joint_b"],
        "body_names_isaac": ["body_a"],
        "actuators": robot_cfg.actuators,
        "init_state": robot_cfg.init_state,
    }


def test_play_export_policy_asset_metadata_records_body_and_joint_order():
    script = _load_play_module()
    policy_config = {}
    asset_meta = {
        "joint_names_isaac": ["joint_a", "joint_b"],
        "body_names_isaac": ["pelvis", "left_foot"],
    }

    script._annotate_policy_asset_metadata(policy_config, asset_meta)

    assert policy_config["isaac_joint_names"] == ["joint_a", "joint_b"]
    assert policy_config["isaac_body_names"] == ["pelvis", "left_foot"]


def test_play_export_allows_policy_without_command_observation_group():
    script = _load_play_module()

    assert script._get_exported_command_observation_group({"policy": {"joint_pos_history": {}}}) is None


def test_play_export_finds_command_observation_group_when_present():
    script = _load_play_module()
    command_group = {"ref_body_pos_future_local": {}}

    assert script._get_exported_command_observation_group({"command": command_group}) is command_group


def test_play_export_policy_benchmark_can_be_skipped():
    script = _load_play_module()
    calls = []

    class Policy:
        def __call__(self, td):
            calls.append(td)
            return td

    elapsed = script._benchmark_policy(
        Policy(),
        {"policy": torch.zeros(1, 2)},
        iters=0,
    )

    assert elapsed is None
    assert calls == []


def test_mujoco_backend_can_import_simple_env_without_isaac_app():
    aa.set_backend("mujoco")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    try:
        from active_adaptation.envs import SimpleEnv

        assert SimpleEnv.__name__ == "SimpleEnv"
    finally:
        aa.set_backend("isaac")


def test_mujoco_train_smoke_builds_env_policy_and_steps_once():
    aa.set_backend("mujoco")
    env = None
    try:
        cfg = _compose_mujoco_train_smoke_cfg()
        from scripts.helpers import make_env_policy

        env, policy, _vecnorm = make_env_policy(cfg)
        carry = env.reset()
        rollout_policy = policy.get_rollout_policy("train")
        carry = rollout_policy(carry.clone(False))
        tensordict, next_carry = env.step_and_maybe_reset(carry.clone(False))

        assert env.num_envs == 1
        assert tensordict["next", "reward"].shape[0] == 1
        assert tensordict["next", "done"].shape == (1, 1)
        assert next_carry.batch_size == carry.batch_size
    finally:
        if env is not None:
            env.close()
        aa.set_backend("isaac")


def _run_train_script_smoke(tmp_path, task: str, run_name: str):
    run_dir = tmp_path / run_name
    command = [
        sys.executable,
        str(ROOT / "scripts/train.py"),
        "backend=mujoco",
        f"task={task}",
        "task.num_envs=1",
        "task.max_episode_length=4",
        "task.viewer.env_spacing=0",
        "algo.train_every=2",
        "algo.ppo_epochs=1",
        "algo.num_minibatches=1",
        "algo.compile=false",
        "total_frames=2",
        "save_interval=-1",
        "wandb.mode=disabled",
        "eval_render=false",
        f"hydra.run.dir={run_dir}",
    ]
    env = {
        **os.environ,
        "WANDB_SILENT": "true",
    }

    result = subprocess.run(
        command,
        cwd=ROOT.parent,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=240,
    )

    return result


def _run_train_sequential_script_smoke(tmp_path, task: str, run_name: str):
    run_dir = tmp_path / run_name
    command = [
        sys.executable,
        str(ROOT / "scripts/train_sequential.py"),
        "backend=mujoco",
        "stages=[ppo]",
        f"task={task}",
        "task.num_envs=1",
        "task.max_episode_length=4",
        "task.viewer.env_spacing=0",
        "algo.train_every=2",
        "algo.ppo_epochs=1",
        "algo.num_minibatches=1",
        "total_frames=2",
        "save_interval=-1",
        "wandb.mode=disabled",
        "eval_render=false",
        f"hydra.run.dir={run_dir}",
    ]
    env = {
        **os.environ,
        "WANDB_SILENT": "true",
    }

    return subprocess.run(
        command,
        cwd=ROOT.parent,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=240,
    )


def test_mujoco_train_script_runs_minimal_ppo_loop(tmp_path):
    result = _run_train_script_smoke(tmp_path, "G1/tracking/walk", "mujoco-train-smoke")

    assert result.returncode == 0, result.stdout[-4000:]
    assert "Average inference time" in result.stdout


def test_mujoco_object_train_script_runs_minimal_ppo_loop(tmp_path):
    result = _run_train_script_smoke(tmp_path, "G1/hdmi/push_box", "mujoco-object-train-smoke")

    assert result.returncode == 0, result.stdout[-4000:]
    assert "Average inference time" in result.stdout


def test_train_sequential_script_propagates_child_failure(tmp_path):
    run_dir = tmp_path / "mujoco-train-sequential-child-failure"
    command = [
        sys.executable,
        str(ROOT / "scripts/train_sequential.py"),
        "backend=not-a-backend",
        "stages=[ppo]",
        "task=G1/tracking/walk",
        "task.num_envs=1",
        "total_frames=2",
        "wandb.mode=disabled",
        f"hydra.run.dir={run_dir}",
    ]

    result = subprocess.run(
        command,
        cwd=ROOT.parent,
        env={**os.environ, "WANDB_SILENT": "true"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
    )

    assert result.returncode != 0, result.stdout[-4000:]
    assert "exited with code" in result.stdout
    assert "All training stages completed successfully" not in result.stdout


def test_mujoco_train_sequential_script_runs_single_stage(tmp_path):
    result = _run_train_sequential_script_smoke(tmp_path, "G1/hdmi/push_box", "mujoco-train-sequential-smoke")

    assert result.returncode == 0, result.stdout[-4000:]
    assert "COMPLETED STAGE 1/1" in result.stdout
