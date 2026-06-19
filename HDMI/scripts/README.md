# Training & Evaluation Scripts

Entry points that wire Hydra configs, Isaac Sim, and PPO policies.

## Training Execution Flow (`train.py`)

1. **Hydra + W&B setup**
   - Hydra loads `cfg/train.yaml` and constructs the Isaac Sim `AppLauncher`.
   - W&B is initialized from `cfg.wandb` (project, mode, tags).

2. **Create environment and policy**
   - `make_env_policy(cfg)` builds:
     - the vectorized Isaac env (Active Adaptation `_Env` subclass),
     - the PPO policy (`ppo` / `ppo_roa` / `ppo_amp`),
     - optional VecNorm transform.

3. **Rollout and training loop**
   - The env is reset once and a rollout policy is created: `rollout_policy = policy.get_rollout_policy("train")`.
   - A TensorDict buffer `data_buf` of shape `[num_envs, train_every, ...]` is allocated based on a one-step probe.
   - For each iteration:
     - With `ExplorationType.RANDOM`, the rollout policy is applied for `train_every` steps, and `env.step_and_maybe_reset` fills `data_buf` (including `next`-fields and `is_init` masks).
     - The critic is run once on `data_buf` and the bootstrapped `next["state_value"]` is computed.
     - `policy.train_op(data_buf)` is called to perform PPO updates (and any adaptation/estimation steps), returning a metrics dict.
     - Episode stats and env performance metrics are aggregated and logged to W&B; checkpoints are written when `should_save(i)` is true.
   - After training, a final checkpoint is saved and `evaluate(...)` runs an eval rollout with `policy.get_rollout_policy("eval")`, logging results to W&B before clean shutdown.

## Evaluation & visualization
- `play.py` — loads a checkpoint (local path or `run:<wandb-run>`) and runs rollouts; can export ONNX when `export_policy=true`.
- `eval.py` / `eval_multiple.py` / `eval_run.py` — batch evaluation helpers; `eval_run.py` can fetch and visualize remote W&B runs.
- `mujoco_policy_export_audit.py` — checks that `play.py export_policy=true` produced a loadable `.pt + .yaml` bundle and that its exported body/joint metadata maps to the task motion and MuJoCo MJCF names.
- `vis/` — MuJoCo visualization utilities (e.g., `mujoco_mocap_viewer.py`, `motion_data_publisher.py`).

## Typical commands
```bash
# Train (teacher policy)
python scripts/train.py algo=ppo_roa_train task=G1/hdmi/move_suitcase

# Finetune student
python scripts/train.py algo=ppo_roa_finetune task=G1/hdmi/move_suitcase checkpoint_path=run:<teacher-wandb_run_path>

# Evaluate Student
python scripts/play.py algo=ppo_roa_finetune task=G1/hdmi/move_suitcase checkpoint_path=run:<student-wandb_run_path>

# Export policy for MuJoCo parity
python scripts/play.py algo=ppo_roa_finetune task=G1/hdmi/push_box checkpoint_path=run:<student-wandb_run_path> export_policy=true export_policy_exit=true export_policy_benchmark_iters=0 headless=true backend=isaac

# Gate the exported policy before MuJoCo playback/rollout parity
PYTHONPATH=. python scripts/mujoco_policy_export_audit.py --task-yaml cfg/task/G1/hdmi/push_box.yaml --checkpoint-path run:<student-wandb_run_path> --require-policy --require-reference-observation
PYTHONPATH=. python scripts/mujoco_playback_parity.py --task-yaml cfg/task/G1/hdmi/push_box.yaml --policy-path scripts/exports/G1PushBox/policy-<run>-<checkpoint>.pt --policy-rollout --steps 0,1 --max-q-l2 1e-5 --max-body-pos-l2 0.05 --max-policy-rollout-q-l2 1.0 --max-policy-rollout-body-pos-l2 0.2 --min-policy-rollout-reward-mean 0.0

# Gate a MuJoCo PPO training smoke or longer run without W&B
PYTHONPATH=. python scripts/train.py backend=mujoco task=G1/hdmi/push_box task.num_envs=8 algo.train_every=8 total_frames=128 wandb.mode=disabled eval_render=false train_summary_path=/tmp/wbc_mujoco_summary_gate.json
PYTHONPATH=. python scripts/mujoco_train_summary_gate.py /tmp/wbc_mujoco_summary_gate.json --require-backend mujoco --require-checkpoint --min-env-frames 128 --min-eval-metric performance/inference_time 0.0 --max-eval-metric performance/inference_time 0.05

# Aggregate MuJoCo migration evidence across payloads, task mapping, and training
PYTHONPATH=. python scripts/mujoco_policy_export_audit.py --task-yaml cfg/task/G1/hdmi/push_box.yaml --policy-path scripts/exports/G1PushBox/policy-<run>-<checkpoint>.pt --require-policy --require-reference-observation > /tmp/wbc_policy_audit.json
PYTHONPATH=. python scripts/mujoco_playback_parity.py --task-yaml cfg/task/G1/hdmi/push_box.yaml --policy-path scripts/exports/G1PushBox/policy-<run>-<checkpoint>.pt --policy-rollout --steps 0,1 --max-q-l2 1e-5 --max-body-pos-l2 0.05 --max-policy-rollout-q-l2 1.0 --max-policy-rollout-body-pos-l2 0.2 --min-policy-rollout-reward-mean 0.0 > /tmp/wbc_playback_parity.json
PYTHONPATH=. python scripts/mujoco_migration_audit.py --require-payloads --task-dir cfg/task/G1/hdmi --require-task-mappings --min-task-mappings 1 --training-summary /tmp/wbc_mujoco_summary_gate.json --require-training-summaries --min-training-summaries 1 --min-training-env-frames 128 --min-training-eval-metric eval/object_tracking/return 0.05 --max-training-eval-metric performance/inference_time 0.05 --policy-export-report /tmp/wbc_policy_audit.json --require-policy-export --playback-parity-report /tmp/wbc_playback_parity.json --require-playback-parity

# Diagnose closed-loop policy rollout drift over longer horizons from one trace
PYTHONPATH=. python scripts/mujoco_playback_parity.py --task-yaml cfg/task/G1/hdmi/push_box.yaml --policy-path scripts/exports/G1PushBox/policy-<run>-<checkpoint>.pt --policy-rollout --trace-json /tmp/wbc_push_box_rollout_trace.json > /tmp/wbc_push_box_rollout_summary.json
# The sweep report includes first_crossings with the first per-step threshold violation for each metric.
PYTHONPATH=. python scripts/mujoco_horizon_sweep.py /tmp/wbc_push_box_rollout_trace.json --horizon 2,8,32,128,all --max-policy-rollout-q-l2 1.0 --max-policy-rollout-body-pos-l2 0.2 --min-policy-rollout-reward-mean 0.0
```
