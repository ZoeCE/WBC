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


def test_migration_audit_passes_with_payloads_task_mapping_and_training_summary(tmp_path, capsys):
    audit = _load_audit_module()
    manifest_path = _write_payload_manifest(tmp_path, present=True)
    summary_path = _write_train_summary(tmp_path, env_frames=128)

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


def test_migration_audit_fails_when_required_components_are_unhealthy(tmp_path, capsys):
    audit = _load_audit_module()
    manifest_path = _write_payload_manifest(tmp_path, present=False)
    summary_path = _write_train_summary(tmp_path, env_frames=32)

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
        ]
    )

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert report["migration_passed"] is False
    failure_components = {failure["component"] for failure in report["failures"]}
    assert "payloads" in failure_components
    assert "task_mapping" in failure_components
    assert "training" in failure_components
