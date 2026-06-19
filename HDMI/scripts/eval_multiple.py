import torch
import hydra
import numpy as np
import einops
import time
import sys
import os
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(FILE_PATH)
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

import active_adaptation as aa

import wandb
import logging
from tqdm import tqdm
from scripts.helpers import make_env_policy, evaluate

import datetime
import termcolor
from pathlib import Path


DEFAULT_CHECKPOINT_STEPS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]


def _checkpoint_file_name(step) -> str:
    return "checkpoint_final.pt" if str(step) == "final" else f"checkpoint_{step}.pt"


def _build_checkpoint_specs(base_checkpoint_path, checkpoint_steps):
    if base_checkpoint_path is None:
        raise ValueError("checkpoint_path is required for eval_multiple.py")

    base_checkpoint_path = str(base_checkpoint_path)
    if base_checkpoint_path.startswith("run:"):
        run_path = base_checkpoint_path.replace("run:", "", 1)
        return [
            {
                "label": f"checkpoint_{step}",
                "source": "wandb",
                "run_path": run_path,
                "step": step,
            }
            for step in checkpoint_steps
        ]

    path = Path(base_checkpoint_path).expanduser()
    if path.is_file():
        return [
            {
                "label": path.stem,
                "source": "local",
                "path": str(path),
            }
        ]

    if path.is_dir():
        specs = []
        for step in checkpoint_steps:
            candidate = path / _checkpoint_file_name(step)
            if candidate.is_file():
                specs.append(
                    {
                        "label": candidate.stem,
                        "source": "local",
                        "path": str(candidate),
                    }
                )
        final_checkpoint = path / "checkpoint_final.pt"
        if final_checkpoint.is_file() and all(spec["path"] != str(final_checkpoint) for spec in specs):
            specs.append(
                {
                    "label": final_checkpoint.stem,
                    "source": "local",
                    "path": str(final_checkpoint),
                }
            )
        if specs:
            return specs

    raise FileNotFoundError(f"No local checkpoint files found for {base_checkpoint_path}")


def _load_checkpoint_state(checkpoint_spec, *, device):
    if checkpoint_spec["source"] == "local":
        return torch.load(checkpoint_spec["path"], map_location=device, weights_only=False)

    wandb_run = wandb.Api().run(checkpoint_spec["run_path"])
    file = wandb_run.file(_checkpoint_file_name(checkpoint_spec["step"]))
    temp_dir = "temp_checkpoints"
    os.makedirs(temp_dir, exist_ok=True)
    checkpoint_file = file.download(root=temp_dir, replace=True)
    try:
        return torch.load(checkpoint_file.name, map_location=device, weights_only=False)
    finally:
        checkpoint_file.close()

@hydra.main(config_path="../cfg", config_name="eval", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    
    # --- 1. Parse Checkpoint Path ---
    base_checkpoint_path = cfg.checkpoint_path
    cfg.checkpoint_path = None
    checkpoint_steps = list(cfg.get("checkpoint_steps", DEFAULT_CHECKPOINT_STEPS))
    checkpoint_specs = _build_checkpoint_specs(base_checkpoint_path, checkpoint_steps)
    # checkpoint_steps = [200, 600, 1000]
    
    # --- 2. Initialize Env and Policy Shell ---
    # This is done only once to save time
    simulation_app = _configure_backend_and_app(cfg)

    env, agent, vecnorm = make_env_policy(cfg)
    
    # --- 3. Evaluation Loop for Each Checkpoint ---
    all_results = {}

    for checkpoint_spec in tqdm(checkpoint_specs, desc="Evaluating Checkpoints"):
        checkpoint_label = checkpoint_spec["label"]
        print(termcolor.colored(f"\n===== Evaluating Checkpoint: {checkpoint_label} =====", "cyan"))

        try:
            state_dict = _load_checkpoint_state(checkpoint_spec, device=env.device)
            print(termcolor.colored(f"Successfully loaded {checkpoint_spec['source']} checkpoint {checkpoint_label}.", "green"))

            # Load the state dict into the policy
            agent.load_state_dict(state_dict["policy"])
            if "vecnorm" in state_dict:
                vecnorm.load_state_dict(state_dict["vecnorm"])
                new_observation_norms = vecnorm.to_observation_norm().transforms

                from torchrl.envs.transforms import Compose, ObservationNorm
                new_transforms_list = []
                for transform in env.transform:
                    if not isinstance(transform, ObservationNorm):
                        new_transforms_list.append(transform.clone())
                new_transforms_list.extend(new_observation_norms)
                env.transform = Compose(*new_transforms_list)

        except Exception as e:
            print(termcolor.colored(f"Failed to load checkpoint {checkpoint_label}. Error: {e}", "red"))
            continue

        # Define keys for data collection during rollout
        keys = [
            ("next", "stats"),
            ("next", "done"), 
            ("next", "reward"),
            "value_obs",
            "value_priv",
            "value_adapt",
            "context_expert",
            "context_scale",
            "context_adapt",
            "context_adapt_scale",
            "action_kl",
        ]
        policy_keys = ["dr_", "dr_pred"]
    
        # Get the evaluation policy and run the evaluation
        policy_eval = agent.get_rollout_policy("eval")
        render_mode = cfg.get("render_mode", "rgb_array")
        
        # We can disable rendering for multiple evaluations to speed it up
        info, trajs, stats, policy_trajs = evaluate(
            env, 
            policy_eval, 
            render=cfg.eval_render, 
            render_mode=render_mode, 
            seed=cfg.seed, 
            keys=keys, 
            policy_keys=policy_keys
        )
        
        # Store info for this policy
        info["task"] = cfg.task.name
        info["algo"] = cfg.algo.name
        info["checkpoint_label"] = checkpoint_label
        info["checkpoint_source"] = checkpoint_spec["source"]
        
        all_results[checkpoint_label] = info
        print(termcolor.colored(f"--- Results for {checkpoint_label} ---", "yellow"))
        print(OmegaConf.to_yaml(info))

    # --- 4. Print and Save All Collected Info ---
    print(termcolor.colored("\n\n===== All Evaluation Results =====", "magenta"))
    # Convert to a dict for clean YAML output
    final_output = OmegaConf.create(all_results)
    print(OmegaConf.to_yaml(final_output))

    time_str = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
    dir_path = os.path.join(os.path.dirname(__file__), "eval_multiple", cfg.task.name)
    os.makedirs(dir_path, exist_ok=True)
    
    # Extract run ID for a more descriptive filename
    run_id = Path(str(base_checkpoint_path).replace("run:", "")).name
    path = os.path.join(dir_path, f"{cfg.task.name}-{run_id}-{time_str}.yaml")
    
    with open(path, "w") as f:
        OmegaConf.save(config=final_output, f=f)
    print(termcolor.colored(f"Saved all results to: {path}", "green"))

    # --- 5. Cleanup ---
    env.close()
    if simulation_app is not None:
        simulation_app.close()
        os._exit(0) # Use os._exit to force exit in Isaac Sim
    return final_output


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
