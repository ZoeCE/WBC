import importlib.util
import sys
from pathlib import Path

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
