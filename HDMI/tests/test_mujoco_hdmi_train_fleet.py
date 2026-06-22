import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_train_fleet_module():
    script_path = ROOT / "scripts/mujoco_hdmi_train_fleet.py"
    spec = importlib.util.spec_from_file_location("mujoco_hdmi_train_fleet", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_task(root: Path, stem: str = "push_box", name: str = "G1PushBox") -> Path:
    task_path = root / "cfg/task/G1/hdmi" / f"{stem}.yaml"
    task_path.parent.mkdir(parents=True, exist_ok=True)
    task_path.write_text(
        "\n".join(
            [
                f"name: {name}",
                "robot:",
                "  robot_type: g1_29dof_nohand-feet_sphere-eef_L-body_capsule",
                "command:",
                "  data_path: data/motion/g1/push_box/example",
                "  object_asset_name: 'box'",
                "  object_body_name: 'box'",
            ]
        ),
        encoding="utf-8",
    )
    return task_path


def test_train_fleet_dry_run_writes_manifest_scripts_and_closed_loop_audit_command(tmp_path, capsys):
    module = _load_train_fleet_module()
    local_root = tmp_path / "HDMI"
    task_path = _write_task(local_root)
    output_root = tmp_path / "fleet"

    exit_code = module.main(
        [
            "--local-root",
            str(local_root),
            "--task-yaml",
            str(task_path),
            "--output-root",
            str(output_root),
            "--python",
            "/envs/wbc/bin/python",
            "--total-frames",
            "64",
            "--task-num-envs",
            "8",
            "--train-every",
            "8",
            "--num-minibatches",
            "1",
            "--ppo-epochs",
            "1",
            "--wandb-mode",
            "disabled",
            "--session-prefix",
            "wbc-test",
            "--min-success",
            "0.9",
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["launch_requested"] is False
    assert report["task_count"] == 1
    assert report["launched_task_count"] == 0
    assert Path(report["manifest_path"]).is_file()
    job = report["jobs"][0]
    assert job["task_name"] == "G1PushBox"
    assert job["task_override"] == "G1/hdmi/push_box"
    assert job["session_name"] == "wbc-test-G1PushBox"
    assert job["summary_path"].endswith("summaries/G1PushBox_summary.json")
    assert job["log_path"].endswith("logs/G1PushBox.log")
    assert job["command"] == [
        "/envs/wbc/bin/python",
        "scripts/train.py",
        "backend=mujoco",
        "algo=ppo_roa_finetune",
        "task=G1/hdmi/push_box",
        "checkpoint_path=null",
        "task.num_envs=8",
        "algo.train_every=8",
        "algo.num_minibatches=1",
        "algo.ppo_epochs=1",
        "total_frames=64",
        "wandb.mode=disabled",
        "eval_render=false",
        "headless=true",
        f"train_summary_path={job['summary_path']}",
    ]
    runner_script = Path(job["runner_script"])
    assert runner_script.is_file()
    runner_text = runner_script.read_text(encoding="utf-8")
    assert "set -euo pipefail" in runner_text
    assert f"cd {local_root}" in runner_text
    assert "2>&1 | tee -a" in runner_text
    assert report["closed_loop_success_audit_command"] == [
        "/envs/wbc/bin/python",
        "scripts/mujoco_closed_loop_success_audit.py",
        "--task-yaml",
        str(task_path),
        "--summary",
        job["summary_path"],
        "--min-success",
        "0.9",
        "--require-backend",
        "mujoco",
        "--require-checkpoint",
    ]


def test_train_fleet_uses_checkpoint_manifest_and_launch_runner(tmp_path):
    module = _load_train_fleet_module()
    local_root = tmp_path / "HDMI"
    task_path = _write_task(local_root)
    checkpoint_manifest = tmp_path / "checkpoints.yaml"
    checkpoint_manifest.write_text("G1/hdmi/push_box: /ckpts/push_box_teacher.pt\n", encoding="utf-8")
    launched = []

    report = module.build_train_fleet(
        local_root=local_root,
        task_yamls=[task_path],
        output_root=tmp_path / "fleet",
        python="/envs/wbc/bin/python",
        checkpoint_manifest=checkpoint_manifest,
        total_frames=128,
        task_num_envs=16,
        train_every=8,
        num_minibatches=2,
        ppo_epochs=1,
        wandb_mode="online",
        session_prefix="wbc-test",
        launch=True,
        command_runner=lambda command: launched.append(command),
    )

    assert report["launch_requested"] is True
    assert report["launched_task_count"] == 1
    job = report["jobs"][0]
    assert "checkpoint_path=/ckpts/push_box_teacher.pt" in job["command"]
    assert launched == [
        ["tmux", "new-session", "-d", "-s", "wbc-test-G1PushBox", f"bash {job['runner_script']}"]
    ]
