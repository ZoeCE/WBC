import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_deliverable_module():
    script_path = ROOT / "scripts/mujoco_hdmi_deliverable_audit.py"
    spec = importlib.util.spec_from_file_location("mujoco_hdmi_deliverable_audit", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_deliverable_audit_reports_video_metrics_success_and_difference_reports(tmp_path, capsys):
    module = _load_deliverable_module()
    goal_report = tmp_path / "goal_report.json"
    goal_report.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_name": "G1PushBox",
                        "task_override": "G1/hdmi/push_box",
                        "task_stem": "push_box",
                        "gates": {
                            "source_reference": True,
                            "kinematic": True,
                            "policy_export": True,
                            "open_loop_dynamics": True,
                            "closed_loop_success": True,
                        },
                        "missing": [],
                        "success": 0.95,
                        "open_loop_metrics": {"policy_rollout_q_l2_max": 0.1},
                    },
                    {
                        "task_name": "G1MoveSuitcase",
                        "task_override": "G1/hdmi/move_suitcase",
                        "task_stem": "move_suitcase",
                        "gates": {
                            "source_reference": True,
                            "kinematic": True,
                            "policy_export": False,
                            "open_loop_dynamics": False,
                            "closed_loop_success": False,
                        },
                        "missing": ["policy_export", "open_loop_dynamics", "closed_loop_success"],
                        "success": None,
                        "open_loop_metrics": {},
                    },
                    {
                        "task_name": "G1AlmostThere",
                        "task_override": "G1/hdmi/almost_there",
                        "task_stem": "almost_there",
                        "gates": {
                            "source_reference": True,
                            "kinematic": True,
                            "policy_export": False,
                            "open_loop_dynamics": True,
                            "closed_loop_success": True,
                        },
                        "missing": ["policy_export"],
                        "success": 0.91,
                        "open_loop_metrics": {"policy_rollout_q_l2_max": 0.2},
                    },
                    {
                        "task_name": "G1ClosedLoopOnly",
                        "task_override": "G1/hdmi/closed_loop_only",
                        "task_stem": "closed_loop_only",
                        "gates": {
                            "source_reference": True,
                            "kinematic": True,
                            "policy_export": True,
                            "open_loop_dynamics": False,
                            "closed_loop_success": True,
                        },
                        "missing": ["open_loop_dynamics"],
                        "success": 0.88,
                        "open_loop_metrics": {},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    video_path = video_dir / "push_box_closed_loop.mp4"
    video_path.write_bytes(b"video")
    almost_video_path = video_dir / "almost_there_kinematic.mp4"
    almost_video_path.write_bytes(b"video")
    closed_loop_only_video_path = video_dir / "closed_loop_only_policy.mp4"
    closed_loop_only_video_path.write_bytes(b"video")
    report_dir = tmp_path / "task_reports"

    exit_code = module.main(
        [
            "--goal-report",
            str(goal_report),
            "--video-root",
            str(video_dir),
            "--report-dir",
            str(report_dir),
            "--write-task-reports",
            "--output",
            str(tmp_path / "deliverables.json"),
        ]
    )
    report = json.loads(capsys.readouterr().out)
    saved = json.loads((tmp_path / "deliverables.json").read_text())

    assert exit_code == 0
    assert saved == report
    assert report["task_count"] == 4
    assert report["video_task_count"] == 3
    assert report["metrics_task_count"] == 4
    assert report["success_task_count"] == 3
    assert report["difference_report_task_count"] == 4
    assert report["runnable_policy_deliverable_count"] == 2
    assert report["complete_task_count"] == 1
    assert report["gate_passed"] is False
    by_task = {task["task_name"]: task for task in report["tasks"]}
    assert by_task["G1PushBox"]["deliverables_complete"] is True
    assert by_task["G1PushBox"]["runnable_policy_deliverable"] is True
    assert by_task["G1PushBox"]["runnable_policy_blocking_reasons"] == []
    assert by_task["G1PushBox"]["video_artifacts"] == [str(video_path)]
    assert by_task["G1PushBox"]["metric_sources"] == ["kinematic", "open_loop_dynamics", "closed_loop_success"]
    assert by_task["G1PushBox"]["difference_report_available"] is True
    assert Path(by_task["G1PushBox"]["difference_report_path"]).is_file()
    assert by_task["G1MoveSuitcase"]["deliverables_complete"] is False
    assert by_task["G1MoveSuitcase"]["blocking_reasons"] == [
        "missing_video",
        "missing_success",
        "missing_policy_export",
        "missing_open_loop_dynamics",
        "missing_closed_loop_success",
    ]
    assert by_task["G1MoveSuitcase"]["runnable_policy_deliverable"] is False
    assert by_task["G1MoveSuitcase"]["runnable_policy_blocking_reasons"] == [
        "missing_video",
        "missing_success",
        "missing_policy_export",
        "missing_closed_loop_success",
    ]
    assert "missing_policy_export" in Path(by_task["G1MoveSuitcase"]["difference_report_path"]).read_text()
    assert by_task["G1AlmostThere"]["video_artifacts"] == [str(almost_video_path)]
    assert by_task["G1AlmostThere"]["success_available"] is True
    assert by_task["G1AlmostThere"]["deliverables_complete"] is False
    assert by_task["G1AlmostThere"]["blocking_reasons"] == ["missing_policy_export"]
    assert by_task["G1AlmostThere"]["runnable_policy_deliverable"] is False
    assert by_task["G1AlmostThere"]["runnable_policy_blocking_reasons"] == ["missing_policy_export"]
    assert by_task["G1ClosedLoopOnly"]["video_artifacts"] == [str(closed_loop_only_video_path)]
    assert by_task["G1ClosedLoopOnly"]["deliverables_complete"] is False
    assert by_task["G1ClosedLoopOnly"]["blocking_reasons"] == ["missing_open_loop_dynamics"]
    assert by_task["G1ClosedLoopOnly"]["runnable_policy_deliverable"] is True
    assert by_task["G1ClosedLoopOnly"]["runnable_policy_blocking_reasons"] == []
