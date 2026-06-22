import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_goal_report_module():
    script_path = ROOT / "scripts/mujoco_hdmi_goal_report.py"
    spec = importlib.util.spec_from_file_location("mujoco_hdmi_goal_report", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_task(root: Path, stem: str, name: str) -> Path:
    path = root / "cfg/task/G1/hdmi" / f"{stem}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"name: {name}\n", encoding="utf-8")
    return path


def test_goal_report_aggregates_four_layer_evidence_and_missing_items(tmp_path, capsys):
    module = _load_goal_report_module()
    task_a = _write_task(tmp_path, "push_box", "G1PushBox")
    task_b = _write_task(tmp_path, "move_suitcase", "G1MoveSuitcase")

    source_report = tmp_path / "source.json"
    source_report.write_text(json.dumps({"gate_passed": True}), encoding="utf-8")

    kinematic_report = tmp_path / "kinematic.json"
    kinematic_report.write_text(
        json.dumps(
            {
                "tasks": [
                    {"task_name": "G1PushBox", "task_override": "G1/hdmi/push_box", "parity_passed": True},
                    {"task_name": "G1MoveSuitcase", "task_override": "G1/hdmi/move_suitcase", "parity_passed": True},
                ]
            }
        ),
        encoding="utf-8",
    )

    policy_report = tmp_path / "policy.json"
    policy_report.write_text(
        json.dumps(
            {
                "tasks": [
                    {"task_name": "G1PushBox", "task_override": "G1/hdmi/push_box", "gate_passed": True},
                    {"task_name": "G1MoveSuitcase", "task_override": "G1/hdmi/move_suitcase", "gate_passed": False},
                ]
            }
        ),
        encoding="utf-8",
    )

    closed_loop_report = tmp_path / "closed_loop.json"
    closed_loop_report.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_name": "G1PushBox",
                        "task_override": "G1/hdmi/push_box",
                        "passed": False,
                        "best_success": 0.25,
                        "failures": [{"reason": "success_below_min"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    exit_code = module.main(
        [
            "--task-yaml",
            str(task_a),
            "--task-yaml",
            str(task_b),
            "--source-reference-report",
            str(source_report),
            "--kinematic-report",
            str(kinematic_report),
            "--policy-report",
            str(policy_report),
            "--closed-loop-report",
            str(closed_loop_report),
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["task_count"] == 2
    assert report["all_tasks_complete"] is False
    assert report["source_reference_passed"] is True
    by_task = {task["task_name"]: task for task in report["tasks"]}
    assert by_task["G1PushBox"]["gates"] == {
        "source_reference": True,
        "kinematic": True,
        "policy_export": True,
        "closed_loop_success": False,
    }
    assert by_task["G1PushBox"]["success"] == 0.25
    assert by_task["G1PushBox"]["missing"] == ["closed_loop_success"]
    assert by_task["G1MoveSuitcase"]["missing"] == ["policy_export", "closed_loop_success"]
    assert report["missing_by_gate"] == {
        "source_reference": [],
        "kinematic": [],
        "policy_export": ["G1MoveSuitcase"],
        "closed_loop_success": ["G1PushBox", "G1MoveSuitcase"],
    }


def test_goal_report_accepts_kinematic_aggregate_rows_keyed_by_task_stem(tmp_path):
    module = _load_goal_report_module()
    task_path = _write_task(tmp_path, "push_box", "G1PushBox")

    source_report = tmp_path / "source.json"
    source_report.write_text(json.dumps({"gate_passed": True}), encoding="utf-8")
    kinematic_report = tmp_path / "aggregate.json"
    kinematic_report.write_text(
        json.dumps(
            {
                "all_thresholds_passed": True,
                "rows": [
                    {
                        "task": "push_box",
                        "parity_passed": True,
                        "q_l2_max": 0.0,
                        "body_pos_l2_max": 0.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = module.build_goal_report(
        task_yamls=[task_path],
        source_reference_report=source_report,
        kinematic_report=kinematic_report,
    )

    task = report["tasks"][0]
    assert task["gates"]["kinematic"] is True
    assert "kinematic" not in task["missing"]


def test_goal_report_accepts_kinematic_video_fleet_records(tmp_path):
    module = _load_goal_report_module()
    task_path = _write_task(tmp_path, "move_suitcase", "G1TrackSuitcase")

    kinematic_report = tmp_path / "kinematic_video_fleet_summary.json"
    kinematic_report.write_text(
        json.dumps(
            {
                "task_count": 1,
                "success_task_count": 1,
                "records": [
                    {
                        "task_name": "G1TrackSuitcase",
                        "task_stem": "move_suitcase",
                        "task_yaml": str(task_path),
                        "returncode": 0,
                        "output_exists": True,
                        "output_size_bytes": 182809,
                        "render_frame_count": 60,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = module.build_goal_report(task_yamls=[task_path], kinematic_report=kinematic_report)

    task = report["tasks"][0]
    assert task["gates"]["kinematic"] is True
    assert "kinematic" not in task["missing"]


def test_goal_report_includes_open_loop_dynamics_when_report_is_provided(tmp_path, capsys):
    module = _load_goal_report_module()
    task_a = _write_task(tmp_path, "push_box", "G1PushBox")
    task_b = _write_task(tmp_path, "move_suitcase", "G1MoveSuitcase")

    source_report = tmp_path / "source.json"
    source_report.write_text(json.dumps({"gate_passed": True}), encoding="utf-8")
    kinematic_report = tmp_path / "kinematic.json"
    kinematic_report.write_text(
        json.dumps(
            {
                "tasks": [
                    {"task_name": "G1PushBox", "parity_passed": True},
                    {"task_name": "G1MoveSuitcase", "parity_passed": True},
                ]
            }
        ),
        encoding="utf-8",
    )
    open_loop_report = tmp_path / "open_loop.json"
    open_loop_report.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task": "push_box",
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

    exit_code = module.main(
        [
            "--task-yaml",
            str(task_a),
            "--task-yaml",
            str(task_b),
            "--source-reference-report",
            str(source_report),
            "--kinematic-report",
            str(kinematic_report),
            "--open-loop-report",
            str(open_loop_report),
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    by_task = {task["task_name"]: task for task in report["tasks"]}
    assert by_task["G1PushBox"]["gates"]["open_loop_dynamics"] is True
    assert by_task["G1PushBox"]["open_loop_metrics"] == {
        "policy_rollout_q_l2_max": 0.13,
        "policy_rollout_body_pos_l2_max": 0.003,
        "policy_rollout_reward_mean": 1.27,
    }
    assert by_task["G1MoveSuitcase"]["gates"]["open_loop_dynamics"] is False
    assert "open_loop_dynamics" in by_task["G1MoveSuitcase"]["missing"]
    assert report["missing_by_gate"]["open_loop_dynamics"] == ["G1MoveSuitcase"]


def test_goal_report_includes_obs_action_parity_when_report_is_provided(tmp_path, capsys):
    module = _load_goal_report_module()
    task_a = _write_task(tmp_path, "push_box", "G1PushBox")
    task_b = _write_task(tmp_path, "move_suitcase", "G1MoveSuitcase")

    obs_action_report = tmp_path / "obs_action.json"
    obs_action_report.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_name": "G1PushBox",
                        "task_override": "G1/hdmi/push_box",
                        "gate_passed": True,
                        "max_abs": 0.0,
                        "mean_abs": 0.0,
                        "failure_count": 0,
                        "compared_component_count": 40,
                    },
                    {
                        "task_name": "G1MoveSuitcase",
                        "task_override": "G1/hdmi/move_suitcase",
                        "gate_passed": False,
                        "max_abs": 1.8,
                        "mean_abs": 0.1,
                        "failure_count": 16,
                        "compared_component_count": 40,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    exit_code = module.main(
        [
            "--task-yaml",
            str(task_a),
            "--task-yaml",
            str(task_b),
            "--obs-action-report",
            str(obs_action_report),
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    by_task = {task["task_name"]: task for task in report["tasks"]}
    assert by_task["G1PushBox"]["gates"]["obs_action_parity"] is True
    assert by_task["G1PushBox"]["obs_action_metrics"] == {
        "max_abs": 0.0,
        "mean_abs": 0.0,
        "failure_count": 0,
        "compared_component_count": 40,
    }
    assert "obs_action_parity" not in by_task["G1PushBox"]["missing"]
    assert by_task["G1MoveSuitcase"]["gates"]["obs_action_parity"] is False
    assert "obs_action_parity" in by_task["G1MoveSuitcase"]["missing"]
    assert report["missing_by_gate"]["obs_action_parity"] == ["G1MoveSuitcase"]


def test_goal_report_includes_checkpoint_inventory_and_policy_export_plan_readiness(tmp_path, capsys):
    module = _load_goal_report_module()
    task_path = _write_task(tmp_path, "push_box", "G1PushBox")

    checkpoint_inventory = tmp_path / "checkpoint_inventory.json"
    checkpoint_inventory.write_text(
        json.dumps(
            {
                "checkpoint_count": 2,
                "golden_candidate_count": 2,
                "validated_golden_reference_count": 0,
                "weak_mujoco_seed_count": 2,
                "export_count": 5,
                "missing_golden_reference": False,
                "missing_validated_golden_reference": True,
            }
        ),
        encoding="utf-8",
    )
    policy_export_plan = tmp_path / "policy_export_plan.json"
    policy_export_plan.write_text(
        json.dumps(
            {
                "task_count": 1,
                "task_count_gate_passed": True,
                "provided_checkpoint_task_count": 0,
                "missing_checkpoint_task_count": 1,
                "ready_to_export": False,
                "gate_passed": False,
                "checkpoint_manifest_template": {"G1/hdmi/push_box": "<checkpoint_path>"},
            }
        ),
        encoding="utf-8",
    )
    checkpoint_source = tmp_path / "checkpoint_source.json"
    checkpoint_source.write_text(
        json.dumps(
            {
                "gate_passed": False,
                "actionable_golden_source_available": False,
                "checkpoint_source": {
                    "placeholder_run_reference_count": 2,
                    "concrete_run_reference_count": 0,
                    "direct_checkpoint_link_count": 0,
                    "checkpoint_file_count": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    exit_code = module.main(
        [
            "--task-yaml",
            str(task_path),
            "--checkpoint-inventory-report",
            str(checkpoint_inventory),
            "--policy-export-plan-report",
            str(policy_export_plan),
            "--checkpoint-source-report",
            str(checkpoint_source),
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["artifact_readiness"] == {
        "checkpoint_inventory": {
            "provided": True,
            "report_path": str(checkpoint_inventory),
            "checkpoint_count": 2,
            "golden_candidate_count": 2,
            "validated_golden_reference_count": 0,
            "weak_mujoco_seed_count": 2,
            "export_count": 5,
            "missing_golden_reference": False,
            "missing_validated_golden_reference": True,
            "gate_passed": False,
        },
        "policy_export_plan": {
            "provided": True,
            "report_path": str(policy_export_plan),
            "task_count": 1,
            "task_count_gate_passed": True,
            "provided_checkpoint_task_count": 0,
            "missing_checkpoint_task_count": 1,
            "ready_to_export": False,
            "gate_passed": False,
        },
        "checkpoint_source": {
            "provided": True,
            "report_path": str(checkpoint_source),
            "actionable_golden_source_available": False,
            "placeholder_run_reference_count": 2,
            "concrete_run_reference_count": 0,
            "direct_checkpoint_link_count": 0,
            "checkpoint_file_count": 0,
            "gate_passed": False,
        },
        "external_policy_source": {
            "provided": False,
            "report_path": None,
            "policy_count": None,
            "matched_policy_count": None,
            "unmatched_policy_count": None,
            "load_mapping_ready_task_count": None,
            "golden_ready_task_count": None,
            "training_provenance_policy_count": None,
            "missing_training_provenance_policy_count": None,
            "require_training_provenance": None,
            "gate_passed": False,
        },
    }


def test_goal_report_counts_external_policy_as_runnable_but_not_golden_ready(tmp_path, capsys):
    module = _load_goal_report_module()
    task_path = _write_task(tmp_path, "move_suitcase", "G1TrackSuitcase")

    external_policy_as_policy_report = tmp_path / "external_policy_non_strict.json"
    external_policy_as_policy_report.write_text(
        json.dumps(
            {
                "task_count": 1,
                "policy_count": 1,
                "matched_policy_count": 1,
                "unmatched_policy_count": 0,
                "ready_task_count": 1,
                "require_training_provenance": False,
                "training_provenance_policy_count": 0,
                "missing_training_provenance_policy_count": 1,
                "gate_passed": True,
                "tasks": [
                    {
                        "task_name": "G1TrackSuitcase",
                        "task_override": "G1/hdmi/move_suitcase",
                        "policy_source": "external",
                        "policy_source_ready": True,
                        "training_provenance_ready": False,
                        "gate_passed": True,
                        "missing_requirements": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    external_policy_strict = tmp_path / "external_policy_strict.json"
    external_policy_strict.write_text(
        json.dumps(
            {
                "task_count": 1,
                "policy_count": 1,
                "matched_policy_count": 1,
                "unmatched_policy_count": 0,
                "ready_task_count": 0,
                "require_training_provenance": True,
                "training_provenance_policy_count": 0,
                "missing_training_provenance_policy_count": 1,
                "gate_passed": False,
                "tasks": [
                    {
                        "task_name": "G1TrackSuitcase",
                        "task_override": "G1/hdmi/move_suitcase",
                        "policy_source": "external",
                        "policy_source_ready": True,
                        "training_provenance_ready": False,
                        "gate_passed": False,
                        "missing_requirements": ["external_policy_training_provenance"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = module.main(
        [
            "--task-yaml",
            str(task_path),
            "--policy-report",
            str(external_policy_as_policy_report),
            "--external-policy-source-report",
            str(external_policy_strict),
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    task = report["tasks"][0]
    assert task["gates"]["policy_export"] is True
    assert task["missing"] == ["source_reference", "kinematic", "closed_loop_success"]
    assert report["artifact_readiness"]["external_policy_source"] == {
        "provided": True,
        "report_path": str(external_policy_strict),
        "policy_count": 1,
        "matched_policy_count": 1,
        "unmatched_policy_count": 0,
        "load_mapping_ready_task_count": 1,
        "golden_ready_task_count": 0,
        "training_provenance_policy_count": 0,
        "missing_training_provenance_policy_count": 1,
        "require_training_provenance": True,
        "gate_passed": False,
    }


def test_goal_report_accepts_official_sim2real_fleet_closed_loop_success(tmp_path, capsys):
    module = _load_goal_report_module()
    suitcase_task = _write_task(tmp_path, "move_suitcase", "G1TrackSuitcase")
    push_box_task = _write_task(tmp_path, "push_box", "G1PushBox")

    closed_loop_report = tmp_path / "official_sim2real_fleet.json"
    closed_loop_report.write_text(
        json.dumps(
            {
                "task_count": 2,
                "runnable_task_count": 1,
                "closed_loop_success_count": 1,
                "tasks": [
                    {
                        "task_name": "G1TrackSuitcase",
                        "task_override": "G1/hdmi/move_suitcase",
                        "status": "closed_loop_success",
                        "closed_loop_success": True,
                        "closed_loop": {
                            "heuristic_success": True,
                            "success_metric": "suitcase_xy_displacement",
                            "success_metric_value": 1.8831053,
                        },
                    },
                    {
                        "task_name": "G1PushBox",
                        "task_override": "G1/hdmi/push_box",
                        "status": "missing_external_policy",
                        "closed_loop_success": False,
                        "missing_requirements": ["external_policy_for_task_motion"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = module.main(
        [
            "--task-yaml",
            str(suitcase_task),
            "--task-yaml",
            str(push_box_task),
            "--closed-loop-report",
            str(closed_loop_report),
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    by_task = {task["task_name"]: task for task in report["tasks"]}
    assert by_task["G1TrackSuitcase"]["gates"]["closed_loop_success"] is True
    assert by_task["G1TrackSuitcase"]["success"] == 1.8831053
    assert "closed_loop_success" not in by_task["G1TrackSuitcase"]["missing"]
    assert by_task["G1PushBox"]["gates"]["closed_loop_success"] is False
    assert report["missing_by_gate"]["closed_loop_success"] == ["G1PushBox"]
