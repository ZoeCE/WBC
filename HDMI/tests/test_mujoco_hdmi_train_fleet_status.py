import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_status_module():
    script_path = ROOT / "scripts/mujoco_hdmi_train_fleet_status.py"
    spec = importlib.util.spec_from_file_location("mujoco_hdmi_train_fleet_status", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_manifest(tmp_path: Path, *, with_summary: bool) -> Path:
    output_root = tmp_path / "fleet"
    summary_path = output_root / "summaries" / "G1PushBox_summary.json"
    log_path = output_root / "logs" / "G1PushBox.log"
    checkpoint_path = output_root / "checkpoint_final.pt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("0%| | 0/256\n 12%|# | 31/256 [09:56<1:11:27, 19.06s/it]\n", encoding="utf-8")
    if with_summary:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(b"checkpoint")
        summary_path.write_text(
            json.dumps(
                {
                    "backend": "mujoco",
                    "task": "G1PushBox",
                    "env_frames": 1048576,
                    "total_frames": 1048576,
                    "total_iters": 256,
                    "checkpoint_final": str(checkpoint_path),
                    "eval": {"eval/success": 0.75, "episode_cnt": 8},
                }
            ),
            encoding="utf-8",
        )
    manifest = {
        "task_count": 1,
        "output_root": str(output_root),
        "jobs": [
            {
                "task_name": "G1PushBox",
                "task_override": "G1/hdmi/push_box",
                "session_name": "wbc-hdmi-pilot-G1PushBox",
                "summary_path": str(summary_path),
                "log_path": str(log_path),
                "checkpoint_path": None,
                "command": ["python", "scripts/train.py"],
            }
        ],
    }
    manifest_path = output_root / "fleet_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_train_fleet_status_reports_running_job_without_summary(tmp_path, capsys):
    module = _load_status_module()
    manifest_path = _write_manifest(tmp_path, with_summary=False)

    exit_code = module.main(
        [
            "--manifest",
            str(manifest_path),
            "--running-session",
            "wbc-hdmi-pilot-G1PushBox",
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["task_count"] == 1
    assert report["completed_task_count"] == 0
    assert report["running_task_count"] == 1
    assert report["failed_task_count"] == 0
    job = report["jobs"][0]
    assert job["status"] == "running"
    assert job["summary_exists"] is False
    assert job["session_running"] is True
    assert job["progress"] == {"iteration": 31, "total_iterations": 256, "percent": 12.0}


def test_train_fleet_status_does_not_count_pending_missing_summary_as_failed(tmp_path):
    module = _load_status_module()
    manifest_path = _write_manifest(tmp_path, with_summary=False)

    report = module.build_train_fleet_status(
        manifest_path=manifest_path,
        running_sessions=set(),
        min_success=0.9,
    )

    assert report["task_count"] == 1
    assert report["completed_task_count"] == 0
    assert report["running_task_count"] == 0
    assert report["pending_task_count"] == 1
    assert report["failed_task_count"] == 0
    job = report["jobs"][0]
    assert job["status"] == "pending"
    assert job["failure_reasons"] == ["summary_missing"]


def test_train_fleet_status_reports_completed_summary_success_and_checkpoint(tmp_path):
    module = _load_status_module()
    manifest_path = _write_manifest(tmp_path, with_summary=True)

    report = module.build_train_fleet_status(
        manifest_path=manifest_path,
        running_sessions=set(),
        min_success=0.9,
    )

    assert report["task_count"] == 1
    assert report["completed_task_count"] == 1
    assert report["running_task_count"] == 0
    assert report["failed_task_count"] == 1
    job = report["jobs"][0]
    assert job["status"] == "completed"
    assert job["checkpoint_exists"] is True
    assert job["success"] == 0.75
    assert job["success_passed"] is False
    assert job["failure_reasons"] == ["success_below_min"]
