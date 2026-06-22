import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_fleet_module():
    script_path = ROOT / "scripts/mujoco_hdmi_obs_action_fleet.py"
    spec = importlib.util.spec_from_file_location("mujoco_hdmi_obs_action_fleet", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_task(root: Path, stem: str, name: str) -> Path:
    path = root / "cfg/task/G1/hdmi" / f"{stem}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"name: {name}\n", encoding="utf-8")
    return path


def test_obs_action_fleet_classifies_existing_external_runnable_and_missing_tasks(tmp_path, capsys):
    module = _load_fleet_module()
    push_box = _write_task(tmp_path, "push_box", "G1PushBox")
    suitcase = _write_task(tmp_path, "move_suitcase", "G1TrackSuitcase")
    foam = _write_task(tmp_path, "move_foam", "G1TrackFoam")

    existing_report = tmp_path / "push_box_obs_action.json"
    existing_report.write_text(
        json.dumps(
            {
                "task_override": "G1/hdmi/push_box",
                "task_name": "G1PushBox",
                "policy_path": "/exports/G1PushBox/policy-good-final.pt",
                "gate_passed": True,
                "max_abs": 0.0,
                "mean_abs": 0.0,
                "failure_count": 0,
                "compared_component_count": 12,
                "groups_compared": ["policy", "command"],
            }
        ),
        encoding="utf-8",
    )
    external_report = tmp_path / "external_policy_source.json"
    external_policy_path = tmp_path / "external/G1TrackSuitcase/policy-v55m8a23-final.pt"
    external_policy_path.parent.mkdir(parents=True)
    external_policy_path.write_bytes(b"placeholder")
    external_report.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_name": "G1TrackSuitcase",
                        "task_override": "G1/hdmi/move_suitcase",
                        "task_stem": "move_suitcase",
                        "policy_source": "external",
                        "policy_source_ready": True,
                        "policy_path": str(external_policy_path),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    exit_code = module.main(
        [
            "--task-yaml",
            str(push_box),
            "--task-yaml",
            str(suitcase),
            "--task-yaml",
            str(foam),
            "--existing-obs-action-report",
            str(existing_report),
            "--external-policy-source-report",
            str(external_report),
            "--output-dir",
            str(tmp_path / "obs_action"),
            "--python",
            "/envs/wbc/bin/python",
            "--expected-task-count",
            "3",
            "--require-task-count",
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["task_count"] == 3
    assert report["task_count_gate_passed"] is True
    assert report["passed_task_count"] == 1
    assert report["runnable_task_count"] == 1
    assert report["missing_policy_task_count"] == 1
    assert report["failed_task_count"] == 0
    assert report["all_passed"] is False
    assert report["gate_passed"] is False
    assert report["requested_gate_passed"] is True
    by_task = {task["task_name"]: task for task in report["tasks"]}
    assert by_task["G1PushBox"]["status"] == "passed_existing_obs_action"
    assert by_task["G1PushBox"]["gate_passed"] is True
    assert by_task["G1PushBox"]["obs_action_metrics"] == {
        "max_abs": 0.0,
        "mean_abs": 0.0,
        "failure_count": 0,
        "compared_component_count": 12,
    }
    assert by_task["G1TrackSuitcase"]["status"] == "needs_obs_action_run"
    assert by_task["G1TrackSuitcase"]["policy_source"] == "external"
    assert by_task["G1TrackSuitcase"]["selected_policy_path"] == str(external_policy_path)
    assert by_task["G1TrackSuitcase"]["missing_reasons"] == ["missing_obs_action_report"]
    assert by_task["G1TrackSuitcase"]["obs_action_command"] == [
        "/envs/wbc/bin/python",
        "scripts/mujoco_obs_builder_parity.py",
        "--task-yaml",
        str(suitcase),
        "--policy-path",
        str(external_policy_path),
        "--steps",
        "4",
        "--action-source",
        "zero",
        "--max-abs",
        "1e-06",
        "--output",
        str(tmp_path / "obs_action/move_suitcase_obs_action_parity.json"),
    ]
    assert by_task["G1TrackFoam"]["status"] == "missing_policy_export"
    assert by_task["G1TrackFoam"]["missing_reasons"] == ["missing_policy_export"]
