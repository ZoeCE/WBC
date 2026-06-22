from types import SimpleNamespace

import active_adaptation as aa

aa.set_backend("mujoco")

from active_adaptation.envs.base import _Env


class _FakeEnv(_Env):
    def _reset_idx(self, env_ids):
        raise NotImplementedError


class _FakeObservation:
    def __init__(self):
        self.update_count = 0

    def update(self):
        self.update_count += 1


class _FakeHdmiCommandObservation(_FakeObservation):
    __module__ = "active_adaptation.envs.mdp.commands.hdmi.observations"


def test_post_command_observation_refresh_updates_command_object_and_hdmi_priv_without_rerolling_histories():
    env = _FakeEnv.__new__(_FakeEnv)
    policy_history = _FakeObservation()
    priv_history = _FakeObservation()
    priv_command_future = _FakeHdmiCommandObservation()
    command_future = _FakeObservation()
    object_xy = _FakeObservation()
    object_contact = _FakeObservation()
    env.observation_funcs = {
        "policy": SimpleNamespace(funcs={"joint_pos_history": policy_history}),
        "priv": SimpleNamespace(
            funcs={
                "joint_pos_history": priv_history,
                "diff_body_lin_vel_future_local": priv_command_future,
            }
        ),
        "command": SimpleNamespace(funcs={"ref_body_pos_future_local": command_future}),
        "object": SimpleNamespace(funcs={"object_xy_b": object_xy, "ref_contact_pos_b": object_contact}),
    }

    env._refresh_post_command_observation_cache()

    assert policy_history.update_count == 0
    assert priv_history.update_count == 0
    assert priv_command_future.update_count == 1
    assert command_future.update_count == 1
    assert object_xy.update_count == 1
    assert object_contact.update_count == 1
