import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_smoke_module():
    script_path = ROOT / "scripts/mujoco_hdmi_isaac_smoke_fleet.py"
    spec = importlib.util.spec_from_file_location("mujoco_hdmi_isaac_smoke_fleet", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_isaac_smoke_plan_builds_timeout_bounded_missing_checkpoint_jobs(tmp_path):
    module = _load_smoke_module()
    golden_plan = {
        "source_reference": {"golden_reference": "original HDMI/Isaac trained checkpoint"},
        "task_count": 13,
        "external_policy_ready_count": 3,
        "missing_checkpoint_task_count": 10,
        "jobs": [
            {
                "task_name": "G1PushBox",
                "task_override": "G1/hdmi/push_box",
                "task_path": "/repo/cfg/task/G1/hdmi/push_box.yaml",
                "motion_path": "data/motion/g1/push_box/example",
            }
        ],
    }
    report = module.build_isaac_smoke_plan(
        golden_plan=golden_plan,
        local_root=tmp_path / "HDMI",
        output_root=tmp_path / "smoke",
        python="/envs/wbc/bin/python",
        train_algo="ppo_roa_train",
        task_num_envs=2,
        train_every=8,
        num_minibatches=1,
        ppo_epochs=1,
        timeout_sec=123,
        kill_after_sec=9,
        episode_length=32,
        wandb_mode="disabled",
    )

    assert report["task_count"] == 13
    assert report["external_policy_ready_count"] == 3
    assert report["missing_checkpoint_task_count"] == 10
    assert report["smoke_task_count"] == 1
    assert report["total_frames_per_smoke"] == 16
    assert report["kill_after_sec"] == 9
    assert report["episode_length"] == 32
    assert Path(report["manifest_path"]).is_file()
    assert Path(report["run_all_script"]).is_file()
    job = report["jobs"][0]
    assert job["task_name"] == "G1PushBox"
    assert job["command"] == [
        "timeout",
        "--kill-after=9s",
        "123",
        "/envs/wbc/bin/python",
        "scripts/train.py",
        "backend=isaac",
        "algo=ppo_roa_train",
        "task=G1/hdmi/push_box",
        "task.num_envs=2",
        "algo.train_every=8",
        "algo.num_minibatches=1",
        "algo.ppo_epochs=1",
        "total_frames=16",
        "task.max_episode_length=32",
        "wandb.mode=disabled",
        "eval_render=false",
        "headless=true",
        f"train_summary_path={job['summary_path']}",
    ]
    runner_text = Path(job["runner_script"]).read_text(encoding="utf-8")
    assert "SMOKE_EXIT:$status" in runner_text
    assert "set -uo pipefail" in runner_text
    assert "export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}" in runner_text
    run_all_text = Path(report["run_all_script"]).read_text(encoding="utf-8")
    assert "SMOKE_FAILURES:$failures" in run_all_text


def test_isaac_smoke_plan_can_launch_one_sequential_tmux_session(tmp_path):
    module = _load_smoke_module()
    launched = []
    golden_plan = {
        "jobs": [
            {
                "task_name": "G1PushBox",
                "task_override": "G1/hdmi/push_box",
            }
        ]
    }
