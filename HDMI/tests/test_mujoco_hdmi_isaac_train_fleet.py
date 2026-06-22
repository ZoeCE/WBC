import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_train_module():
    script_path = ROOT / "scripts/mujoco_hdmi_isaac_train_fleet.py"
    spec = importlib.util.spec_from_file_location("mujoco_hdmi_isaac_train_fleet", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _golden_plan():
    return {
        "source_reference": {"golden_reference": "original HDMI/Isaac trained checkpoint"},
        "task_count": 13,
        "external_policy_ready_count": 3,
        "missing_checkpoint_task_count": 10,
        "jobs": [
            {
                "task_name": "G1CarryAndPlaceBreadBox",
                "task_override": "G1/hdmi/carry_and_place_bread_box",
                "task_path": "/repo/cfg/task/G1/hdmi/carry_and_place_bread_box.yaml",
                "motion_path": "data/motion/data_for_sim/carry_and_place_bread_box-0829",
            },
            {
                "task_name": "G1PushBox",
                "task_override": "G1/hdmi/push_box",
                "task_path": "/repo/cfg/task/G1/hdmi/push_box.yaml",
                "motion_path": "data/motion/g1/push_box/example",
            },
        ],
    }


def test_isaac_train_plan_builds_sequential_golden_checkpoint_jobs(tmp_path):
    module = _load_train_module()
    report = module.build_isaac_train_plan(
        golden_plan=_golden_plan(),
        local_root=tmp_path / "HDMI",
        output_root=tmp_path / "isaac_train",
        python="/envs/wbc/bin/python",
        train_algo="ppo_roa_train",
        total_frames=150_000_000,
        task_num_envs=4096,
        train_every=32,
        wandb_mode="online",
        max_tasks=1,
    )

    assert report["task_count"] == 13
    assert report["external_policy_ready_count"] == 3
    assert report["missing_checkpoint_task_count"] == 10
    assert report["selected_task_count"] == 1
    assert report["pending_task_count"] == 1
    assert report["skipped_task_count"] == 0
    assert report["frames_per_batch"] == 131072
    assert report["total_frames"] == 150_000_000
    assert Path(report["manifest_path"]).is_file()
    assert Path(report["run_all_script"]).is_file()
    job = report["jobs"][0]
    assert job["task_name"] == "G1CarryAndPlaceBreadBox"
    assert job["command"] == [
        "/envs/wbc/bin/python",
        "scripts/train.py",
        "backend=isaac",
        "algo=ppo_roa_train",
        "task=G1/hdmi/carry_and_place_bread_box",
        "task.num_envs=4096",
        "algo.train_every=32",
        "total_frames=150000000",
        "wandb.mode=online",
        "eval_render=false",
        "headless=true",
        f"train_summary_path={job['summary_path']}",
    ]
    runner_text = Path(job["runner_script"]).read_text(encoding="utf-8")
    assert "export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}" in runner_text
    assert "ISAAC_TRAIN_EXIT:$status" in runner_text
    run_all_text = Path(report["run_all_script"]).read_text(encoding="utf-8")
    assert f"bash {job['runner_script']}" in run_all_text
    assert "tmux new-session" not in run_all_text
    assert "ISAAC_TRAIN_FAILURES:$failures" in run_all_text

def test_isaac_train_plan_launches_one_tmux_session_and_filters_tasks(tmp_path):
    module = _load_train_module()
    launched = []

    report = module.build_isaac_train_plan(
        golden_plan=_golden_plan(),
        local_root=tmp_path / "HDMI",
        output_root=tmp_path / "isaac_train",
        python="/envs/wbc/bin/python",
        include_tasks=["G1/hdmi/push_box"],
        session_name="wbc-hdmi-isaac-train-test",
        launch=True,
        command_runner=lambda command: launched.append(command),
    )

    assert report["launch_requested"] is True
    assert report["launched"] is True
    assert report["selected_task_count"] == 1
    assert report["pending_task_count"] == 1
    assert report["jobs"][0]["task_name"] == "G1PushBox"
    assert launched == [
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            "wbc-hdmi-isaac-train-test",
            f"bash {report['run_all_script']}",
        ]
    ]


def test_isaac_train_plan_skips_completed_summary_with_checkpoint(tmp_path):
    module = _load_train_module()
    output_root = tmp_path / "isaac_train"
    summary_dir = output_root / "summaries"
    checkpoint = output_root / "checkpoint_final.pt"
    summary_dir.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    (summary_dir / "G1CarryAndPlaceBreadBox_summary.json").write_text(
        json.dumps({"checkpoint_final": str(checkpoint)}),
        encoding="utf-8",
    )

    report = module.build_isaac_train_plan(
        golden_plan=_golden_plan(),
        local_root=tmp_path / "HDMI",
        output_root=output_root,
        max_tasks=1,
        skip_completed=True,
    )

    assert report["selected_task_count"] == 1
    assert report["pending_task_count"] == 0
    assert report["skipped_task_count"] == 1
    assert report["jobs"] == []
    assert report["skipped_jobs"][0]["task_name"] == "G1CarryAndPlaceBreadBox"
    assert report["skipped_jobs"][0]["skip_reason"] == "completed_summary_with_checkpoint"


def test_isaac_train_plan_can_continue_after_failed_job_when_requested(tmp_path):
    module = _load_train_module()
    report = module.build_isaac_train_plan(
        golden_plan=_golden_plan(),
        local_root=tmp_path / "HDMI",
        output_root=tmp_path / "isaac_train",
        continue_on_failure=True,
    )

    run_all_text = Path(report["run_all_script"]).read_text(encoding="utf-8")
    assert '  exit "$failures"\nfi' not in run_all_text
    assert "ISAAC_TRAIN_FAILURES:$failures" in run_all_text
