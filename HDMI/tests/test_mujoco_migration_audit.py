import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/mujoco_migration_audit.py"


def _load_audit_module():
    spec = importlib.util.spec_from_file_location("mujoco_migration_audit", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_payload_manifest(tmp_path: Path, *, present: bool = True) -> Path:
    payload_path = tmp_path / "payloads" / "motion.npz"
    if present:
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload_path.write_bytes(b"motion")
    manifest_path = tmp_path / "mujoco_external_payloads.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                "version: 1",
                "payloads:",
                "  - path: payloads/motion.npz",
                "    kind: motion_npz",
                "    required: true",
                "    size_bytes: 6",
            ]
        ),
        encoding="utf-8",
    )
    return manifest_path


def _write_train_summary(tmp_path: Path, *, env_frames: int = 128) -> Path:
    checkpoint = tmp_path / "checkpoint_final.pt"
    checkpoint.write_bytes(b"checkpoint")
    summary_path = tmp_path / "train_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "backend": "mujoco",
                "task": "G1PushBox",
                "checkpoint_final": str(checkpoint),
                "num_envs": 8,
                "train_every": 8,
                "total_frames": env_frames,
                "total_iters": 2,
                "env_frames": env_frames,
                "non_finite_keys": {"train": [], "eval": []},
                "train": {"rollout_fps": 100.0},
                "eval": {
                    "performance/inference_time": 0.001,
                    "eval/object_tracking/return": 0.06,
                },
            }
        ),
        encoding="utf-8",
    )
    return summary_path


def _write_policy_export_report(tmp_path: Path, *, gate_passed: bool = True) -> Path:
    report_path = tmp_path / "policy_export_audit.json"
    report_path.write_text(
        json.dumps(
            {
                "gate_passed": gate_passed,
                "missing_requirements": [] if gate_passed else ["policy_task_motion_mjcf_mapping"],
                "policy_exists": True,
                "policy_config_exists": True,
                "policy_loadable": True,
                "policy_mapping_ok": gate_passed,
                "policy_path": "scripts/exports/G1PushBox/policy-test.pt",
                "policy_config_path": "scripts/exports/G1PushBox/policy-test.yaml",
                "policy_action_dim": 23,
                "num_policy_joint_mappings": 23,
            }
        ),
        encoding="utf-8",
    )
    return report_path


def _write_playback_parity_report(tmp_path: Path, *, parity_passed: bool = True) -> Path:
    report_path = tmp_path / "playback_parity.json"
    report_path.write_text(
        json.dumps(
            {
                "parity_passed": parity_passed,
                "threshold_failures": []
                if parity_passed
                else [
                    {
                        "metric": "policy_rollout_q_l2_max",
                        "actual": 2.0,
                        "limit": 1.0,
                        "comparison": "<=",
                    }
                ],
                "q_l2_max": 0.0,
                "body_pos_l2_max": 0.01,
                "reward_mean": 1.0,
                "policy_rollout_q_l2_max": 0.2 if parity_passed else 2.0,
                "policy_rollout_body_pos_l2_max": 0.05,
                "policy_rollout_reward_mean": 1.1,
            }
        ),
        encoding="utf-8",
    )
    return report_path


def test_migration_audit_passes_with_payloads_mapping_training_policy_and_playback(tmp_path, capsys):
    audit = _load_audit_module()
    manifest_path = _write_payload_manifest(tmp_path, present=True)
    summary_path = _write_train_summary(tmp_path, env_frames=128)
    policy_report_path = _write_policy_export_report(tmp_path, gate_passed=True)
    playback_report_path = _write_playback_parity_report(tmp_path, parity_passed=True)

    exit_code = audit.main(
        [
            "--payload-manifest",
            str(manifest_path),
            "--payload-root",
            str(tmp_path),
            "--require-payloads",
            "--task-dir",
            str(ROOT / "cfg/task/G1/hdmi"),
            "--require-task-mappings",
            "--min-task-mappings",
            "1",
            "--training-summary",
            str(summary_path),
            "--require-training-summaries",
            "--min-training-env-frames",
            "128",
            "--min-training-eval-metric",
            "eval/object_tracking/return",
            "0.05",
            "--max-training-eval-metric",
            "performance/inference_time",
            "0.01",
            "--policy-export-report",
            str(policy_report_path),
            "--require-policy-export",
            "--playback-parity-report",
            str(playback_report_path),
            "--require-playback-parity",
        ]
    )

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert report["migration_passed"] is True
    assert report["failures"] == []
    assert report["payloads"]["gate_passed"] is True
    assert report["task_mapping"]["gate_passed"] is True
    assert report["task_mapping"]["num_tasks"] >= 1
    assert report["training"]["gate_passed"] is True
    assert report["policy_export"]["gate_passed"] is True
    assert report["policy_export"]["num_reports"] == 1
    assert report["playback_parity"]["gate_passed"] is True
    assert report["playback_parity"]["num_reports"] == 1


def test_migration_audit_fails_when_required_components_are_unhealthy(tmp_path, capsys):
    audit = _load_audit_module()
    manifest_path = _write_payload_manifest(tmp_path, present=False)
    summary_path = _write_train_summary(tmp_path, env_frames=32)
    policy_report_path = _write_policy_export_report(tmp_path, gate_passed=False)
    playback_report_path = _write_playback_parity_report(tmp_path, parity_passed=False)

    exit_code = audit.main(
        [
            "--payload-manifest",
            str(manifest_path),
            "--payload-root",
            str(tmp_path),
            "--require-payloads",
            "--task-dir",
            str(tmp_path / "empty-task-dir"),
            "--require-task-mappings",
            "--min-task-mappings",
            "1",
            "--training-summary",
            str(summary_path),
            "--require-training-summaries",
            "--min-training-env-frames",
            "128",
            "--policy-export-report",
            str(policy_report_path),
            "--require-policy-export",
            "--playback-parity-report",
            str(playback_report_path),
            "--require-playback-parity",
        ]
    )

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert report["migration_passed"] is False
    failure_components = {failure["component"] for failure in report["failures"]}
    assert "payloads" in failure_components
    assert "task_mapping" in failure_components
    assert "training" in failure_components
    assert "policy_export" in failure_components
    assert "playback_parity" in failure_components
