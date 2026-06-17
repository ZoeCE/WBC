# Active Adaptation Environments

This folder holds the environment scaffolding that wires Isaac Sim/MuJoCo scenes to modular MDP components.

## Files at a glance
- `base.py` — top-level env class (`_Env`) that instantiates and orchestrates MDP pieces. It registers callbacks for `reset`, `pre_step`, `post_step`, `update`, and `debug_draw`.
- `mdp/base.py` — base interfaces for MDP components (`Action`, `Command`, `Observation`, `Reward`, `Termination`). Components share a common lifecycle (`reset`, `step`, `update`) so the env can call them uniformly.
- `mdp/` — concrete MDP modules for task specializations.

## Core Execution Flow
- Commands: `command_manager.reset` runs on env reset; `command_manager.step` runs every step (registered as `pre_step`); `command_manager.debug_draw` runs when GUI is enabled.
- Actions: `action_manager.reset` runs on reset; actions are applied during `step` after policy outputs are clipped/scaled.
- Observations: each observation group is an `ObsGroup` that concatenates per-sensor builders; groups are cached and exposed via TorchRL specs.
- Rewards/terminations: reward terms accumulate during `step`; terminations (done/terminated/truncated) are computed and stored in the TensorDict.
- Randomizations/addons: optional hooks registered as callbacks to perturb environments or add symmetry transforms.

### Callback lifecycle in code
Registration (excerpt):
```python
# base.py
self._reset_callbacks = []
self._pre_step_callbacks = []
self._post_step_callbacks = []
self._update_callbacks = []
self._debug_draw_callbacks = []

self._reset_callbacks.append(self.command_manager.reset)
self._debug_draw_callbacks.append(self.command_manager.debug_draw)

self._reset_callbacks.append(self.action_manager.reset)
self._debug_draw_callbacks.append(self.action_manager.debug_draw)

# observations / rewards / terminations also append their
# startup/reset/update/post_step hooks here
```

Execution at reset:
```python
def _reset(...):
    ...
    if len(env_ids):
        self._reset_idx(env_ids)
        self.scene.reset(env_ids)
    self.episode_length_buf[env_ids] = 0
    for callback in self._reset_callbacks:
        callback(env_ids)
    tensordict.update(self.observation_spec.zero())
    return tensordict
```

Execution each step (pre/post/compute/update):
```python
def _step(tensordict):
    for substep in range(self.decimation):
        self.apply_action(tensordict, substep)
        for callback in self._pre_step_callbacks:
            callback(substep)
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(self.physics_dt)
        for callback in self._post_step_callbacks:
            callback(substep)
    self._update()                       # calls all update callbacks
    tensordict.update(self._compute_reward())
    self.command_manager.update()        # runs after rewards
    self._compute_observation(tensordict)
    tensordict.set("terminated", self._compute_termination())
    ...
```

## HDMI task components
HDMI tasks are assembled from modules in `active_adaptation/envs/mdp/commands/hdmi`:
- **Command (`command.py`)**  
  - `RobotTracking` implements robot tracking with reference state initialization in `sample_init`.
  - `RobotObjectTracking` implements robot-object co-tracking.
- **Observations (`observations.py`)**  
  - Reference Motion: future ref joints (`ref_joint_pos_future`, `ref_joint_vel_future`), ref roots in robot frame, local/body deltas (`diff_body_*`).  
  - Action-aligned refs: `ref_joint_pos_action` / `ref_joint_pos_action_policy` expose targets normalized by action scaling, used for residual action space policies.
  - Object features: contact targets vs. EEF (`ref_contact_pos_b`, `diff_contact_pos_b`), object pose in robot frame (`object_xy_b`, `object_heading_b`, `object_pos_b`, `object_ori_b`), object joint states, and future deltas (`diff_object_*_future`). 
- **Rewards (`rewards.py`)**  
  - Keypoint/body tracking (pos/ori/vel) with global or root-aligned frames, exponential or error forms.  
  - Joint tracking (pos/vel) with tolerances.  
  - Object tracking: position/orientation/joint targets.  
  - Contact reward: EEF-to-target distance and force terms (`eef_contact_exp`, `eef_contact_exp_max`, `eef_contact_all`) conditioned on reference contact labels.
