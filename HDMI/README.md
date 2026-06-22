# HDMI: Learning Interactive Humanoid Whole-Body Control from Human Videos

<div align="center">
<a href="https://hdmi-humanoid.github.io/">
  <img alt="Website" src="https://img.shields.io/badge/Website-Visit-blue?style=flat&logo=google-chrome"/>
</a>

<a href="https://www.youtube.com/watch?v=GvIBzM7ieaA&list=PL0WMh2z6WXob0roqIb-AG6w7nQpCHyR0Z&index=12">
  <img alt="Video" src="https://img.shields.io/badge/Video-YouTube-red?style=flat&logo=youtube"/>
</a>

<a href="https://arxiv.org/pdf/2509.16757">
  <img alt="Arxiv" src="https://img.shields.io/badge/Paper-Arxiv-b31b1b?style=flat&logo=arxiv"/>
</a>

<a href="https://github.com/LeCAR-Lab/HDMI/stargazers">
    <img alt="GitHub stars" src="https://img.shields.io/github/stars/LeCAR-Lab/HDMI?style=social"/>
</a>


</div>

HDMI is a framework that enables humanoid robots to acquire diverse whole-body interaction skills directly from monocular RGB videos of human demonstrations. This repository contains the official training code for **HDMI: Learning Interactive Humanoid Whole-Body Control from Human Videos**.


## 🚀 Quick Start

Set up the environment, then install IsaacSim, IsaacLab, and HDMI:

```bash
# 1) Conda env
conda create -n hdmi python=3.10 -y
conda activate hdmi

# 2) IsaacSim
pip install "isaacsim[all,extscache]==4.5.0" --extra-index-url https://pypi.nvidia.com
isaacsim # test isaacsim

# 3) IsaacLab
cd ..
git clone git@github.com:isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.2.0
./isaaclab.sh -i none

# 4) HDMI
cd ..
git clone https://github.com/LeCAR-Lab/HDMI
cd HDMI
pip install -e .
```

## Repository Structure
This codebase is designed to be a flexible, high-performance RL framework for Isaac Sim, built from composable MDP components, modular RL algorithms, and Hydra-driven configs. It relies on tensordict/torchrl for efficient data flow.

- `active_adaptation/`
  - `envs/` — unified base env with composable modular MDP components: [Documentation →](active_adaptation/envs/README.md).
  - `learning/` — single-file PPO implementations: [Documentation →](active_adaptation/learning/README.md).
- `scripts/` — training, evaluation, visualization entry points: [Documentation →](scripts/README.md).
- `cfg/` — Hydra configs for tasks, algorithms, and app launch settings
- `data/` — motion assets and samples referenced by configs

HDMI-specific code is primarily in `active_adaptation/envs/mdp/commands/hdmi/` (commands, observations, rewards) and `active_adaptation/learning/ppo_roa.py` (PPO with residual action distillation).

## Data Preparation

### Desired Data Format
The training scripts load motion data from `motion.npz` (see `active_adaptation/utils/motion.py`). The desired data format is as follows:
- Body states: `pos`, `quat`, `lin_vel`, `ang_vel` → `[T, B, 3/4]`
- Joint states: `pos`, `vel` → `[T, J]`

`T` = time steps, `B` = bodies (including appended objects), `J` = joints. Body/joint ordering is defined in the accompanying `meta.json`.

### Processing Steps
To turn HOI/video data into this format:
1) Convert human motion to robot motion via GVHMR → GMR/LocoMujoco to obtain robot body/joint states.
2) Extract the object trajectory (position, orientation, velocities).
3) Append the object name to `meta.json`, then concatenate the object body states (`pos`, `quat`, `lin_vel`, `ang_vel`) to the robot body states so shapes become `[T, B_robot + B_object, 3/4]`.

### Verify Your Data
Visualize motions in Isaac Sim with `+task.command.replay_motion=true`:

```bash
python scripts/play.py algo=ppo_roa_train task=G1/hdmi/move_suitcase +task.command.replay_motion=true
```

Or visualize a `motion.npz` in MuJoCo:

```bash
# one terminal
python scripts/vis/mujoco_mocap_viewer.py
# another terminal
python scripts/vis/motion_data_publisher.py <path-to-motion-folder>
```

### External Payload Manifest
Large local payloads for MuJoCo migration are tracked by checksum in
`mujoco_external_payloads.yaml` instead of being added to regular Git history.
This covers task-owned `motion.npz` files and local Isaac USD robot assets under
`active_adaptation/assets/g1/`.

Audit the current workspace:

```bash
python scripts/mujoco_external_payloads.py --include-task-motion
python scripts/mujoco_external_payloads.py --verify-sha256 --require-present
```

`required_missing` must be empty before running MuJoCo training or playback.

## Train and Evaluate

Teacher policy
```bash
# train teacher
python scripts/train.py algo=ppo_roa_train task=G1/hdmi/move_suitcase
# evaluate teacher
python scripts/play.py algo=ppo_roa_train task=G1/hdmi/move_suitcase checkpoint_path=run:<teacher-wandb_run_path>
```

Student policy
```bash
# train student
python scripts/train.py algo=ppo_roa_finetune task=G1/hdmi/move_suitcase checkpoint_path=run:<teacher-wandb_run_path>
# evaluate student
python scripts/play.py algo=ppo_roa_finetune task=G1/hdmi/move_suitcase checkpoint_path=run:<student-wandb_run_path>
```

To export trained policies, add `export_policy=true` to the play script.

## Sim2Real

Please see [github.com/EGalahad/sim2real](https://github.com/EGalahad/sim2real) for details.

## MuJoCo Sim2Sim: 4 Checkpoint Reproduction

This project vendors the HDMI sim2real deployment payload under `third_party/sim2real_hdmi/`. It is not a submodule and does not require a separate `sim2real-hdmi` checkout for the 4 checkpoint reproduction below.

### 1. Core Definition

The delivered sim2sim target is strict closed-loop replay of the 4 trained ONNX policies that exist in `third_party/sim2real_hdmi/checkpoints`, using MuJoCo dynamics and the same policy observation/action contract:

$$
a_t=\pi(o_t),\quad q^{target}_t=q^{default}+a_t\odot s,\quad \tau_t=k_p(q^{target}_t-q_t)-k_d\dot q_t
$$

### 2. System Position

`checkpoint(.onnx/.yaml/.json) -> DirectSim2RealPolicy -> grouped observation {command, policy, object?} -> ONNX action -> PD target -> MuJoCo step -> success metric`

### 3. Scope

| Scenario | Policy | Success metric |
| --- | --- | --- |
| `G1Dance1Subject2` | `policy-1781wsjf-final.onnx` | `not_fallen == 1` |
| `G1TrackSuitcase` | `policy-v55m8a23-final.onnx` | `suitcase_xy_displacement >= 0.5` |
| `G1PushDoorHand` | `policy-xg6644nr-final.onnx` | `door_joint_abs >= 0.2` |
| `G1RollBall` | `policy-yte3rr8b-final.onnx` | `ball_xy_displacement >= 0.3` |

The 13 HDMI/WBC training tasks are a larger migration target and are not required for this 4-checkpoint sim2sim gate.

### 4. Run the Gate

```bash
cd /home/zoe/Workspace/wbc-HDMI/HDMI

/home/zoe/miniconda3/envs/wbc/bin/python \
  scripts/mujoco_hdmi_sim2real_runner.py \
  --scenario all \
  --duration-sec 6 \
  --output-dir /tmp/wbc_hdmi_goal_full_parity_v2/sim2real_official_4ckpt_6s_bundled_current \
  --no-video
```

Expected current result:

| Scenario | Pass | Measured value |
| --- | --- | --- |
| `G1Dance1Subject2` | yes | `not_fallen = 1.0` |
| `G1TrackSuitcase` | yes | `suitcase_xy_displacement = 1.90876592` |
| `G1PushDoorHand` | yes | `door_joint_abs = 0.81166974` |
| `G1RollBall` | yes | `ball_xy_displacement = 0.89243166` |

Read the machine-checkable summary:

```bash
cat /tmp/wbc_hdmi_goal_full_parity_v2/sim2real_official_4ckpt_6s_bundled_current/summary.json
```

### 5. Export Videos

```bash
cd /home/zoe/Workspace/wbc-HDMI/HDMI

/home/zoe/miniconda3/envs/wbc/bin/python \
  scripts/mujoco_hdmi_sim2real_runner.py \
  --scenario all \
  --duration-sec 8 \
  --output-dir /tmp/wbc_hdmi_goal_full_parity_v2/sim2real_official_4ckpt_8s_video_current
```

Copy videos to Mac:

```bash
mkdir -p ~/wbc_outputs/sim2real_official_4ckpt_8s_video_current
scp 'a5000-wsl:/tmp/wbc_hdmi_goal_full_parity_v2/sim2real_official_4ckpt_8s_video_current/*.mp4' \
  ~/wbc_outputs/sim2real_official_4ckpt_8s_video_current/
```

### 6. Debugging Checklist

| Symptom | Check |
| --- | --- |
| Cannot import MuJoCo or Torch | Use `/home/zoe/miniconda3/envs/wbc/bin/python` |
| Missing checkpoint or MJCF | Confirm `third_party/sim2real_hdmi/checkpoints` and `third_party/sim2real_hdmi/data/robots/g1` exist |
| Observation shape matches but behavior is wrong | Check `isaac_joint_names`, `policy_joint_names`, body names, and MJCF joint names one by one |
| Policy action is near zero or unstable | Inspect `.yaml` action scale, default joint pose, and object/reference motion path |
| Video render fails | First run with `--no-video`; then check EGL/MuJoCo rendering setup |

## Citation

If you find our work useful for your research, please consider cite us:

```
@misc{weng2025hdmilearninginteractivehumanoid,
      title={HDMI: Learning Interactive Humanoid Whole-Body Control from Human Videos},
      author={Haoyang Weng and Yitang Li and Nikhil Sobanbabu and Zihan Wang and Zhengyi Luo and Tairan He and Deva Ramanan and Guanya Shi},
      year={2025},
      eprint={2509.16757},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2509.16757},
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=lecar-lab/hdmi&type=date&legend=top-left)](https://www.star-history.com/#lecar-lab/hdmi&type=date&legend=top-left)
