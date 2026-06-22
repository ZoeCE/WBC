import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_plan_module():
    script_path = ROOT / "scripts/mujoco_hdmi_golden_checkpoint_plan.py"
    spec = importlib.util.spec_from_file_location("mujoco_hdmi_golden_checkpoint_plan", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_golden_checkpoint_plan_builds_isaac_training_and_export_queue_for_missing_tasks():
    module = _load_plan_module()
    audit_report = {
        "tasks": [
            {
                "task_name": "G1TrackSuitcase",
                "task_override": "G1/hdmi/move_suitcase",
                "motion_path": "data/motion/g1/omomo/sub1_suitcase_011",
                "gate_passed": True,
                "policy_set": "G1TrackSuitcase",
                "missing_requirements": [],
            },
            {
                "task_name": "G1PushBox",
                "task_override": "G1/hdmi/push_box",
                "motion_path": "data/motion/g1/push_box/example",
                "gate_passed": False,
                "policy_set": None,
                "missing_requirements": ["external_policy_for_task_motion"],
            },
        ]
    }

    plan = module.build_golden_checkpoint_plan(
        audit_report=audit_report,
        python="/envs/wbc/bin/python",
        train_algo="ppo_roa_train",
        export_algo="ppo_roa_train",
        wandb_mode="online",
    )

    assert plan["task_count"] == 2
    assert plan["external_policy_ready_count"] == 1
    assert plan["missing_checkpoint_task_count"] == 1
    assert plan["checkpoint_manifest_template"] == {"G1/hdmi/push_box": "<teacher_or_student_checkpoint>"}
    job = plan["jobs"][0]
    assert job["task_name"] == "G1PushBox"
    assert job["status"] == "needs_isaac_golden_checkpoint"
    assert job["train_command"] == [
        "/envs/wbc/bin/python",
        "scripts/train.py",
        "backend=isaac",
        "algo=ppo_roa_train",
        "task=G1/hdmi/push_box",
        "wandb.mode=online",
    ]
    assert job["export_command_template"] == [
        "/envs/wbc/bin/python",
        "scripts/play.py",
        "backend=mujoco",
        "algo=ppo_roa_train",
        "task=G1/hdmi/push_box",
        "checkpoint_path=<teacher_or_student_checkpoint>",
        "export_policy=true",
        "export_policy_exit=true",
        "export_policy_benchmark_iters=0",
        "export_onnx_policy=false",
        "headless=true",
    ]
