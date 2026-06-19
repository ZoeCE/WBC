import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/mujoco_train_summary_gate.py"


def _load_gate_module():
    spec = importlib.util.spec_from_file_location("mujoco_train_summary_gate", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_summary(
    path: Path,
    checkpoint: Path,
    *,
    env_frames: int = 128,
    backend: str = "mujoco",
    non_finite_train=None,
    non_finite_eval=None,
    eval_metrics=None,
    train_metrics=None,
):
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"checkpoint")
    summary = {
        "backend": backend,
        "task": "G1PushBox",
        "checkpoint_final": str(checkpoint),
        "num_envs": 8,
        "train_every": 8,
        "total_frames": env_frames,
        "total_iters": 2,
        "env_frames": env_frames,
        "non_finite_keys": {
            "train": list(non_finite_train or []),
            "eval": list(non_finite_eval or []),
        },
        "train": {
            "rollout_fps": 1024.0,
            **(train_metrics or {}),
        },
        "eval": {
            "performance/inference_time": 0.001,
            "eval/object_tracking/return": 1.25,
            **(eval_metrics or {}),
        },
    }
    path.write_text(json.dumps(summary), encoding="utf-8")
    return path


def test_summary_gate_passes_multiple_mujoco_summaries(tmp_path, capsys):
    gate = _load_gate_module()
    summary_a = _write_summary(tmp_path / "seed0.json", tmp_path / "seed0.pt")
    summary_b = _write_summary(
        tmp_path / "seed1.json",
        tmp_path / "seed1.pt",
        eval_metrics={"eval/object_tracking/return": 1.4},
    )

    exit_code = gate.main(
        [
            str(summary_a),
            str(summary_b),
            "--require-backend",
            "mujoco",
            "--require-checkpoint",
            "--min-env-frames",
            "128",
            "--min-eval-metric",
            "eval/object_tracking/return",
            "1.0",
            "--max-eval-metric",
            "performance/inference_time",
            "0.01",
            "--min-num-summaries",
            "2",
        ]
    )

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert report["gate_passed"] is True
    assert report["num_summaries"] == 2
    assert report["failures"] == []
    assert report["metric_ranges"]["eval"]["eval/object_tracking/return"]["min"] == 1.25
    assert report["metric_ranges"]["eval"]["eval/object_tracking/return"]["max"] == 1.4


def test_summary_gate_fails_incomplete_or_unhealthy_summary(tmp_path, capsys):
    gate = _load_gate_module()
    healthy = _write_summary(tmp_path / "healthy.json", tmp_path / "healthy.pt")
    missing_checkpoint = _write_summary(
        tmp_path / "missing-checkpoint.json",
        tmp_path / "missing.pt",
        env_frames=32,
        non_finite_eval=["eval/object_tracking/return"],
        eval_metrics={"eval/object_tracking/return": 0.2},
    )
    Path(json.loads(missing_checkpoint.read_text())["checkpoint_final"]).unlink()

    exit_code = gate.main(
        [
            str(healthy),
            str(missing_checkpoint),
            "--require-backend",
            "mujoco",
            "--require-checkpoint",
            "--min-env-frames",
            "128",
            "--min-eval-metric",
            "eval/object_tracking/return",
            "1.0",
        ]
    )

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert report["gate_passed"] is False
