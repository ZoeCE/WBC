import importlib.util
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load_export_module():
    script_path = ROOT / "scripts/mujoco_hdmi_isaac_export_fleet.py"
    spec = importlib.util.spec_from_file_location("mujoco_hdmi_isaac_export_fleet", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_task(task_dir: Path, *, stem: str = "push_box", name: str = "G1PushBox") -> Path:
    task_path = task_dir / f"{stem}.yaml"
    task_path.parent.mkdir(parents=True, exist_ok=True)
    task_path.write_text(
        "\n".join(
            [
                f"name: {name}",
                "command:",
                "  data_path: data/motion/example",
            ]
        ),
        encoding="utf-8",
    )
    return task_path


def _write_manifest(tmp_path: Path, *, summary_path: Path, task_name: str = "G1PushBox") -> Path:
    manifest = {
        "jobs": [
            {
                "task_name": task_name,
                "task_override": "G1/hdmi/push_box",
                "task_path": str(tmp_path / "cfg/task/G1/hdmi/push_box.yaml"),
                "summary_path": str(summary_path),
            }
        ]
    }
    manifest_path = tmp_path / "isaac_train_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_isaac_export_plan_builds_ready_export_jobs_from_training_manifest(tmp_path):
    module = _load_export_module()
    local_root = tmp_path / "HDMI"
    task_dir = local_root / "cfg/task/G1/hdmi"
    task_path = _write_task(task_dir)
    checkpoint = tmp_path / "checkpoint_final.pt"
    checkpoint.write_bytes(b"checkpoint")
    summary = tmp_path / "G1PushBox_summary.json"
    summary.write_text(
        json.dumps(
            {
                "backend": "isaac",
                "task": "G1PushBox",
                "checkpoint_final": str(checkpoint),
                "env_frames": 150000000,
                "total_frames": 150000000,
            }
        ),
        encoding="utf-8",
    )
    train_manifest = _write_manifest(tmp_path, summary_path=summary)

    report = module.build_isaac_export_plan(
        local_root=local_root,
        task_dir=task_dir,
        train_manifests=[train_manifest],
        output_root=tmp_path / "export",
        exports_dir=tmp_path / "exports",
        python="/envs/wbc/bin/python",
        algo="ppo_roa_train",
    )

    assert report["summary_source_count"] == 1
    assert report["ready_task_count"] == 1
    assert report["pending_task_count"] == 0
    assert report["all_ready"] is True
    assert yaml.safe_load(Path(report["checkpoint_manifest_path"]).read_text()) == {
        "G1/hdmi/push_box": str(checkpoint)
    }
    job = report["jobs"][0]
    assert job["task_name"] == "G1PushBox"
    assert job["task_override"] == "G1/hdmi/push_box"
    assert job["checkpoint_path"] == str(checkpoint)
    assert job["command"] == [
        "/envs/wbc/bin/python",
        "scripts/play.py",
        "algo=ppo_roa_train",
        "task=G1/hdmi/push_box",
        "algo.phase=adapt",
        f"checkpoint_path={checkpoint}",
        "export_policy=true",
        "export_policy_exit=true",
        "export_policy_benchmark_iters=0",
        "export_onnx_policy=true",
        "export_onnx_required=true",
        "headless=true",
        "backend=mujoco",
    ]
    runner_text = Path(job["runner_script"]).read_text(encoding="utf-8")
    assert "ISAAC_EXPORT_EXIT:$status" in runner_text
    assert "export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}" in runner_text
    run_all_text = Path(report["run_all_script"]).read_text(encoding="utf-8")
    assert f"bash {job['runner_script']}" in run_all_text
    assert report["policy_fleet_audit_command"] == [
        "/envs/wbc/bin/python",
        "scripts/mujoco_hdmi_policy_fleet_audit.py",
        "--task-yaml",
        str(task_path),
        "--exports-dir",
        str(tmp_path / "exports"),
        "--checkpoint-manifest",
        report["checkpoint_manifest_path"],
        "--require-policy",
        "--require-reference-observation",
        "--require-obs-action-smoke",
        "--require-onnx-policy",
    ]


def test_isaac_export_plan_reports_pending_when_summary_or_checkpoint_is_missing(tmp_path):
    module = _load_export_module()
    local_root = tmp_path / "HDMI"
    task_dir = local_root / "cfg/task/G1/hdmi"
    _write_task(task_dir)
    missing_summary = tmp_path / "missing_summary.json"
    train_manifest = _write_manifest(tmp_path, summary_path=missing_summary)

    report = module.build_isaac_export_plan(
        local_root=local_root,
        task_dir=task_dir,
        train_manifests=[train_manifest],
        output_root=tmp_path / "export",
    )

    assert report["ready_task_count"] == 0
    assert report["pending_task_count"] == 1
    assert report["all_ready"] is False
    assert report["jobs"] == []
    assert report["pending"][0]["failure_reasons"] == ["summary_missing"]
    assert yaml.safe_load(Path(report["checkpoint_manifest_path"]).read_text()) == {}

    checkpointless_summary = tmp_path / "G1PushBox_summary.json"
    checkpointless_summary.write_text(json.dumps({"task": "G1PushBox", "checkpoint_final": str(tmp_path / "no.pt")}))
    report = module.build_isaac_export_plan(
        local_root=local_root,
        task_dir=task_dir,
        summary_paths=[checkpointless_summary],
        output_root=tmp_path / "export2",
    )
    assert report["pending"][0]["failure_reasons"] == ["checkpoint_final_file_missing"]

    named_missing_summary = tmp_path / "G1PushBox" / "summary.json"
    report = module.build_isaac_export_plan(
        local_root=local_root,
        task_dir=task_dir,
        summary_paths=[named_missing_summary],
        output_root=tmp_path / "export3",
    )
    assert report["ready_task_count"] == 0
    assert report["pending_task_count"] == 1
    assert report["pending"][0]["task_name"] == "G1PushBox"
    assert report["pending"][0]["task_override"] == "G1/hdmi/push_box"
    assert report["pending"][0]["failure_reasons"] == ["summary_missing"]


def test_isaac_export_plan_matches_explicit_summary_by_task_name_and_launches_tmux(tmp_path):
    module = _load_export_module()
    local_root = tmp_path / "HDMI"
    task_dir = local_root / "cfg/task/G1/hdmi"
    _write_task(task_dir, stem="carry_and_place_bread_box", name="G1CarryAndPlaceBreadBox")
    checkpoint = tmp_path / "bread_checkpoint_final.pt"
    checkpoint.write_bytes(b"checkpoint")
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps({"backend": "isaac", "task": "G1CarryAndPlaceBreadBox", "checkpoint_final": str(checkpoint)}),
        encoding="utf-8",
    )
    launched = []

    report = module.build_isaac_export_plan(
        local_root=local_root,
        task_dir=task_dir,
        summary_paths=[summary],
        output_root=tmp_path / "export",
        session_name="wbc-export-test",
        launch=True,
        command_runner=lambda command: launched.append(command),
    )

    assert report["ready_task_count"] == 1
    assert report["jobs"][0]["task_override"] == "G1/hdmi/carry_and_place_bread_box"
    assert report["launch_requested"] is True
