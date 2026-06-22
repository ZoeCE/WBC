import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_fleet_module():
    script_path = ROOT / "scripts/mujoco_hdmi_open_loop_fleet.py"
    spec = importlib.util.spec_from_file_location("mujoco_hdmi_open_loop_fleet", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_task(root: Path, stem: str, name: str) -> Path:
    path = root / "cfg/task/G1/hdmi" / f"{stem}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"name: {name}\n", encoding="utf-8")
    return path


def test_open_loop_fleet_classifies_existing_runnable_and_missing_tasks(tmp_path, capsys):
    module = _load_fleet_module()
    push_box = _write_task(tmp_path, "push_box", "G1PushBox")
    suitcase = _write_task(tmp_path, "move_suitcase", "G1MoveSuitcase")
    foam = _write_task(tmp_path, "move_foam", "G1MoveFoam")

    existing_report = tmp_path / "open_loop_aggregate.json"
    existing_report.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task": "push_box",
                        "task_name": "G1PushBox",
                        "gate_passed": True,
                        "policy_rollout_q_l2_max": 0.13,
                        "policy_rollout_body_pos_l2_max": 0.003,
                        "policy_rollout_reward_mean": 1.27,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    export_dir = tmp_path / "exports/G1MoveSuitcase"
    export_dir.mkdir(parents=True)
    policy_path = export_dir / "policy-run-final.pt"
    policy_path.write_bytes(b"placeholder")

    exit_code = module.main(
        [
            "--task-yaml",
            str(push_box),
            "--task-yaml",
            str(suitcase),
            "--task-yaml",
            str(foam),
            "--exports-dir",
            str(tmp_path / "exports"),
            "--existing-open-loop-report",
            str(existing_report),
            "--output-dir",
            str(tmp_path / "open_loop"),
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
    assert report["all_passed"] is False
    by_task = {task["task_name"]: task for task in report["tasks"]}
    assert by_task["G1PushBox"]["status"] == "passed_existing_open_loop"
    assert by_task["G1PushBox"]["gate_passed"] is True
    assert by_task["G1PushBox"]["open_loop_metrics"] == {
        "policy_rollout_q_l2_max": 0.13,
        "policy_rollout_body_pos_l2_max": 0.003,
        "policy_rollout_reward_mean": 1.27,
    }
    assert by_task["G1PushBox"]["policy_rollout_q_l2_max"] == 0.13
    assert by_task["G1PushBox"]["policy_rollout_body_pos_l2_max"] == 0.003
    assert by_task["G1PushBox"]["policy_rollout_reward_mean"] == 1.27
    assert by_task["G1MoveSuitcase"]["status"] == "needs_open_loop_run"
    assert by_task["G1MoveSuitcase"]["selected_policy_path"] == str(policy_path)
    assert by_task["G1MoveSuitcase"]["missing_reasons"] == ["missing_open_loop_report"]
    assert by_task["G1MoveSuitcase"]["playback_command"] == [
        "/envs/wbc/bin/python",
        "scripts/mujoco_playback_parity.py",
        "--task-yaml",
        str(suitcase),
        "--policy-path",
        str(policy_path),
        "--policy-rollout",
        "--require-reference-observation",
        "--trace-json",
        str(tmp_path / "open_loop/move_suitcase_rollout_trace.json"),
        "--steps",
        "0,1",
        "--max-q-l2",
        "1e-6",
        "--max-body-pos-l2",
        "1e-5",
        "--min-reward-mean",
        "0.0",
        "--max-policy-rollout-q-l2",
        "1.0",
        "--max-policy-rollout-body-pos-l2",
        "0.05",
        "--min-policy-rollout-reward-mean",
        "0.0",
    ]
    assert by_task["G1MoveFoam"]["status"] == "missing_policy_export"
    assert by_task["G1MoveFoam"]["missing_reasons"] == ["missing_policy_export"]


def test_open_loop_fleet_uses_external_policy_source_when_no_golden_export_exists(tmp_path, capsys):
    module = _load_fleet_module()
    suitcase = _write_task(tmp_path, "move_suitcase", "G1MoveSuitcase")
    external_policy = tmp_path / "sim2real/checkpoints/G1TrackSuitcase/policy-v55m8a23-final.pt"
    external_policy.parent.mkdir(parents=True)
    external_policy.write_bytes(b"placeholder")
    external_report = tmp_path / "external_policy_source.json"
    external_report.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_name": "G1MoveSuitcase",
                        "task_override": "G1/hdmi/move_suitcase",
                        "task_stem": "move_suitcase",
                        "task_path": str(suitcase),
                        "policy_source_ready": True,
                        "policy_path": str(external_policy),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    exit_code = module.main(
        [
            "--task-yaml",
            str(suitcase),
            "--exports-dir",
            str(tmp_path / "exports"),
            "--external-policy-source-report",
            str(external_report),
            "--output-dir",
            str(tmp_path / "open_loop"),
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    task = report["tasks"][0]
    assert task["status"] == "needs_open_loop_run"
    assert task["policy_source"] == "external"
