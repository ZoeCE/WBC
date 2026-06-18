import importlib.util
import sys
from pathlib import Path

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
