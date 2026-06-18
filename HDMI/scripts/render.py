import torch
import hydra
import numpy as np
import einops
import time
import sys
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig

import active_adaptation as aa

import wandb
import logging
from tqdm import tqdm
from scripts.helpers import make_env_policy, evaluate

import os
import datetime
import termcolor

@hydra.main(config_path="../cfg", config_name="render", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    
    simulation_app = _configure_backend_and_app(cfg)

    # from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
    # print("isaac dir:", ISAAC_NUCLEUS_DIR)
    # breakpoint()

    env = None
    try:
        env, agent, vecnorm = make_env_policy(cfg)
    
        policy_eval = agent.get_rollout_policy("eval")
        evaluate(env, policy_eval, render=cfg.eval_render, render_mode=cfg.render_mode, seed=cfg.seed)
    finally:
        if env is not None:
            env.close()
        if simulation_app is not None:
            simulation_app.close()
    if simulation_app is not None:
        os._exit(0)
    return None


def _configure_backend_and_app(cfg: DictConfig):
    backend = cfg.get("backend", aa.get_backend())
    aa.set_backend(backend)
    if backend == "mujoco":
        return None

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(OmegaConf.to_container(cfg.app))
    return app_launcher.app


if __name__ == "__main__":
    main()

