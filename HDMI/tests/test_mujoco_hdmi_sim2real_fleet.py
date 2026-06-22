import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_fleet_module():
    script_path = ROOT / "scripts/mujoco_hdmi_sim2real_fleet.py"
    spec = importlib.util.spec_from_file_location("mujoco_hdmi_sim2real_fleet", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_sim2real_root_uses_bundled_payload():
    module = _load_fleet_module()

    assert module.DEFAULT_SIM2REAL_ROOT == ROOT / "third_party/sim2real_hdmi"
    assert (module.DEFAULT_SIM2REAL_ROOT / "checkpoints/G1RollBall").is_dir()


def test_build_fleet_plan_marks_only_wbc_tasks_with_known_sim2real_policy_as_runnable():
    module = _load_fleet_module()
    audit_report = {
        "task_count": 5,
        "ready_task_count": 4,
        "tasks": [
            {
                "task_name": "G1Dance1Subject2",
                "task_override": "G1/hdmi/dance1_subject2",
                "motion_path": "data/motion/lafan/dance1_subject2-270_1000",
                "policy_set": "G1Dance1Subject2",
                "policy_path": "/sim2real/checkpoints/G1Dance1Subject2/policy.pt",
                "policy_onnx_path": "/sim2real/checkpoints/G1Dance1Subject2/policy.onnx",
                "gate_passed": True,
                "missing_requirements": [],
            },
            {
                "task_name": "G1TrackSuitcase",
                "task_override": "G1/hdmi/move_suitcase",
                "motion_path": "data/motion/g1/omomo/sub1_suitcase_011",
                "policy_set": "G1TrackSuitcase",
                "policy_path": "/sim2real/checkpoints/G1TrackSuitcase/policy.pt",
                "policy_onnx_path": "/sim2real/checkpoints/G1TrackSuitcase/policy.onnx",
                "gate_passed": True,
                "missing_requirements": [],
            },
            {
                "task_name": "G1PushDoorHand",
                "task_override": "G1/hdmi/push_door-hand",
                "motion_path": "data/motion/data_for_sim/push_door-hand-0828",
                "policy_set": "G1PushDoorHand",
                "policy_path": "/sim2real/checkpoints/G1PushDoorHand/policy.pt",
                "policy_onnx_path": "/sim2real/checkpoints/G1PushDoorHand/policy.onnx",
                "gate_passed": True,
                "missing_requirements": [],
            },
            {
                "task_name": "G1PushBox",
                "task_override": "G1/hdmi/push_box",
                "motion_path": "data/motion/g1/push_box/push_box-VID_20250423_220958-light-high-adjust_root_height",
                "policy_set": None,
                "policy_path": None,
                "policy_onnx_path": None,
                "gate_passed": False,
                "missing_requirements": ["external_policy_for_task_motion"],
            },
            {
                "task_name": "G1UnknownExternal",
                "task_override": "G1/hdmi/unknown",
                "motion_path": "data/motion/unknown",
                "policy_set": "G1UnsupportedPolicy",
                "policy_path": "/sim2real/checkpoints/G1UnsupportedPolicy/policy.pt",
                "policy_onnx_path": "/sim2real/checkpoints/G1UnsupportedPolicy/policy.onnx",
                "gate_passed": True,
                "missing_requirements": [],
            },
        ],
    }

    plan = module.build_fleet_plan(audit_report)

    assert plan["task_count"] == 5
    assert plan["runnable_task_count"] == 3
    assert plan["missing_checkpoint_count"] == 1
    assert plan["unsupported_policy_count"] == 1
    by_task = {task["task_name"]: task for task in plan["tasks"]}
    assert by_task["G1Dance1Subject2"]["runnable"] is True
    assert by_task["G1Dance1Subject2"]["scenario"] == "G1Dance1Subject2"
    assert by_task["G1TrackSuitcase"]["runnable"] is True
    assert by_task["G1TrackSuitcase"]["scenario"] == "G1TrackSuitcase"
    assert by_task["G1PushDoorHand"]["runnable"] is True
    assert by_task["G1PushBox"]["runnable"] is False
    assert by_task["G1PushBox"]["status"] == "missing_external_policy"
    assert by_task["G1UnknownExternal"]["status"] == "unsupported_external_policy"


def test_build_wbc_export_fleet_plan_discovers_only_complete_onnx_exports(tmp_path):
    module = _load_fleet_module()
    task_dir = tmp_path / "cfg/task/G1/hdmi"
    exports_dir = tmp_path / "scripts/exports"
    task_dir.mkdir(parents=True)

    _write_task_yaml(task_dir / "carry_and_place_bread_box.yaml", "G1CarryAndPlaceBreadBox")
    _write_task_yaml(task_dir / "carry_box_over_shoulder.yaml", "G1LiftBoxOverShoulder")
    _write_task_yaml(task_dir / "push_box.yaml", "G1PushBox")
    _write_export(exports_dir / "G1CarryAndPlaceBreadBox", "policy-a-final", complete=True)
    _write_export(exports_dir / "G1LiftBoxOverShoulder", "policy-b-final", complete=True)
    _write_export(exports_dir / "G1PushBox", "policy-c-final", complete=False)

    plan = module.build_wbc_export_fleet_plan(
        task_dir=task_dir,
        exports_dir=exports_dir,
    )

    assert plan["task_count"] == 3
    assert plan["ready_task_count"] == 2
    assert plan["not_ready_task_count"] == 1
    by_task = {task["task_name"]: task for task in plan["tasks"]}
    assert by_task["G1CarryAndPlaceBreadBox"]["ready"] is True
    assert by_task["G1CarryAndPlaceBreadBox"]["task_yaml"].endswith("carry_and_place_bread_box.yaml")
    assert by_task["G1CarryAndPlaceBreadBox"]["policy_model"].endswith("policy-a-final.onnx")
    assert by_task["G1LiftBoxOverShoulder"]["ready"] is True
    assert by_task["G1PushBox"]["ready"] is False
    assert by_task["G1PushBox"]["missing_requirements"] == ["exported_policy_onnx", "exported_policy_json"]


def test_wbc_export_closed_loop_summary_separates_stability_from_reference_parity():
    module = _load_fleet_module()
    task = {
        "task_name": "G1CarryAndPlaceBreadBox",
        "ready": True,
        "status": "ready",
    }
    closed_loop = {
        "heuristic_success": True,
        "not_fallen": True,
        "reference_tracking": {
            "available": True,
            "q_ref_l2_mean": 3.2,
            "body_pos_ref_l2_mean": 4.9,
            "object_pos_ref_l2_mean": 1.0,
        },
    }

    report = module.wbc_export_task_report_from_summary(
        task,
        closed_loop,
        parity_thresholds={"q_ref_l2_mean": 0.5, "body_pos_ref_l2_mean": 0.5, "object_pos_ref_l2_mean": 0.5},
    )

    assert report["closed_loop_attempted"] is True
    assert report["stability_success"] is True
    assert report["reference_parity_success"] is False
    assert report["status"] == "reference_parity_failed"
    assert report["parity_failures"] == {
        "q_ref_l2_mean": {"value": 3.2, "threshold": 0.5},
        "body_pos_ref_l2_mean": {"value": 4.9, "threshold": 0.5},
        "object_pos_ref_l2_mean": {"value": 1.0, "threshold": 0.5},
    }


def test_main_runs_wbc_export_fleet_mode_and_can_gate_reference_parity(monkeypatch, tmp_path, capsys):
    module = _load_fleet_module()
    calls = {}

    def fake_run_wbc_export_fleet(**kwargs):
        calls.update(kwargs)
        return {
            "task_count": 13,
            "ready_task_count": 2,
            "reference_parity_success_count": 0,
            "stability_success_count": 2,
            "closed_loop_attempted_count": 2,
            "output_path": str(tmp_path / "summary.json"),
            "tasks": [],
        }

    monkeypatch.setattr(module, "run_wbc_export_fleet", fake_run_wbc_export_fleet)

    exit_code = module.main(
        [
            "--wbc-export-fleet",
            "--sim2real-root",
            "/sim2real",
            "--wbc-root",
            "/wbc",
            "--task-dir",
            "/wbc/cfg/task/G1/hdmi",
            "--exports-dir",
            "/wbc/scripts/exports",
            "--output-dir",
            str(tmp_path / "fleet"),
            "--duration-sec",
            "1.5",
            "--require-all-reference-parity",
        ]
    )
    summary = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert summary["ready_task_count"] == 2
    assert calls["sim2real_root"] == "/sim2real"
    assert calls["wbc_root"] == "/wbc"
    assert calls["task_dir"] == "/wbc/cfg/task/G1/hdmi"
    assert calls["exports_dir"] == "/wbc/scripts/exports"
    assert calls["duration_sec"] == 1.5


def _write_task_yaml(path: Path, name: str) -> None:
    path.write_text(
        "\n".join(
            [
                f"name: {name}",
                "command:",
                "  data_path: data/motion/example",
                "  object_asset_name: box",
                "  object_body_name: box",
            ]
        ),
        encoding="utf-8",
    )


def _write_export(export_dir: Path, stem: str, *, complete: bool) -> None:
    export_dir.mkdir(parents=True)
    (export_dir / f"{stem}.pt").write_bytes(b"pt")
    (export_dir / f"{stem}.yaml").write_text("policy_joint_names: []\n", encoding="utf-8")
    if complete:
        (export_dir / f"{stem}.onnx").write_bytes(b"onnx")
        (export_dir / f"{stem}.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
