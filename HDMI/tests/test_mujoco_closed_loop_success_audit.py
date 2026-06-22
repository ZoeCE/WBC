import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_success_audit_module():
    script_path = ROOT / "scripts/mujoco_closed_loop_success_audit.py"
    spec = importlib.util.spec_from_file_location("mujoco_closed_loop_success_audit_script", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_summary(path: Path, checkpoint: Path, *, task: str, success: float, env_frames: int = 1000) -> Path:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"checkpoint")
    path.write_text(
        json.dumps(
            {
                "backend": "mujoco",
                "task": task,
                "checkpoint_final": str(checkpoint),
                "env_frames": env_frames,
                "eval": {"eval/success": success, "eval/episode_len": 100.0},
                "train": {},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_closed_loop_success_audit_passes_single_task_with_success_summary(tmp_path, capsys):
    module = _load_success_audit_module()
    task_path = ROOT / "cfg/task/G1/hdmi/push_box.yaml"
    summary_path = _write_summary(
        tmp_path / "push_box_success.json",
        tmp_path / "checkpoint_final.pt",
        task="G1PushBox",
        success=0.92,
        env_frames=2000,
    )

    exit_code = module.main(
        [
            "--task-yaml",
            str(task_path),
            "--summary",
            str(summary_path),
            "--min-success",
            "0.9",
            "--min-env-frames",
            "1000",
            "--require-checkpoint",
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["closed_loop_success_gate_passed"] is True
    assert report["task_count"] == 1
    assert report["passed_task_count"] == 1
    assert report["tasks"][0]["task_name"] == "G1PushBox"
    assert report["tasks"][0]["best_success"] == 0.92
    assert report["tasks"][0]["gate_passed"] is True
    assert report["failures"] == []


def test_closed_loop_success_audit_reports_missing_and_unsuccessful_fleet_tasks(tmp_path, capsys):
    module = _load_success_audit_module()
    summary_path = _write_summary(
        tmp_path / "push_box_failed.json",
        tmp_path / "checkpoint_final.pt",
        task="G1PushBox",
        success=0.0,
        env_frames=2000,
    )

    exit_code = module.main(
        [
            "--task-dir",
            str(ROOT / "cfg/task/G1/hdmi"),
            "--summary",
            str(summary_path),
            "--expected-task-count",
            "13",
            "--min-success",
            "0.9",
            "--min-env-frames",
            "1000",
            "--require-checkpoint",
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert report["closed_loop_success_gate_passed"] is False
    assert report["task_count"] == 13
    assert report["passed_task_count"] == 0
    assert report["missing_summary_task_count"] == 12
    failures_by_task = {failure["task_name"]: failure["reason"] for failure in report["failures"]}
    assert failures_by_task["G1PushBox"] == "success_below_min"
    assert failures_by_task["G1TrackSuitcase"] == "summary_missing"
