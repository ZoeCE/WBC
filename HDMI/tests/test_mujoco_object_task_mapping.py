from pathlib import Path

import pytest
import yaml


def _object_task_commands():
    task_dir = Path("HDMI/cfg/task/G1/hdmi")
    commands = []
    for path in sorted(task_dir.glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text()) or {}
        command = cfg.get("command") or {}
        if "object_asset_name" in command:
            commands.append((path, command))
    return commands


@pytest.mark.parametrize(
    "task_path, command",
    _object_task_commands(),
    ids=lambda item: item.name if isinstance(item, Path) else item.get("object_asset_name"),
)
def test_mujoco_object_scene_matches_hdmi_task_object_names(task_path, command):
    from active_adaptation.assets_mjcf import ROBOTS

    object_asset_name = command["object_asset_name"]
    object_body_name = command["object_body_name"]
    object_joint_name = command.get("object_joint_name")
    extra_object_names = command.get("extra_object_names", [])

    cfg = ROBOTS.with_object("g1_29dof", object_asset_name=object_asset_name)
    specs = cfg.object_specs

    assert object_asset_name in specs, (
        f"{task_path.name}: missing object spec {object_asset_name!r}; available={sorted(specs)}"
    )
    object_spec = specs[object_asset_name]
    assert object_body_name in object_spec.body_names, (
        f"{task_path.name}: object_body_name={object_body_name!r} not in MuJoCo bodies "
        f"{tuple(object_spec.body_names)!r}"
    )
    if object_joint_name is not None:
        assert object_joint_name in object_spec.joint_names, (
            f"{task_path.name}: object_joint_name={object_joint_name!r} not in MuJoCo joints "
            f"{tuple(object_spec.joint_names)!r}"
        )
    for extra_object_name in extra_object_names:
        assert extra_object_name in specs, f"{task_path.name}: missing extra object {extra_object_name!r}"
