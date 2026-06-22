import importlib.util
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_inventory_module():
    script_path = ROOT / "scripts/mujoco_hdmi_checkpoint_inventory.py"
    spec = importlib.util.spec_from_file_location("mujoco_hdmi_checkpoint_inventory_script", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_checkpoint(
    path: Path,
    *,
    backend: str,
    total_frames: int,
    task_name: str = "G1PushBox",
    source_checkpoint_path: str | None = None,
    wandb_id: str = "runid",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "backend": backend,
        "total_frames": total_frames,
        "algo": {"name": "ppo_roa"},
        "task": {"name": task_name, "num_envs": 64},
    }
    if source_checkpoint_path is not None:
        cfg["checkpoint_path"] = source_checkpoint_path
    torch.save({"cfg": cfg, "wandb": {"id": wandb_id, "name": f"{task_name}-{wandb_id}"}}, path)
    return path


def test_checkpoint_inventory_classifies_golden_candidate_and_weak_mujoco_seed(tmp_path, capsys):
    module = _load_inventory_module()
    isaac_ckpt = _write_checkpoint(
        tmp_path / "outputs/teacher/files/checkpoint_final.pt",
        backend="isaac",
        total_frames=120_000_000,
        source_checkpoint_path="run:upstream/hdmi/teacher",
        wandb_id="teacher1",
    )
    mujoco_ckpt = _write_checkpoint(
        tmp_path / "outputs/mujoco/files/checkpoint_final.pt",
        backend="mujoco",
        total_frames=65_536,
        source_checkpoint_path="run:local/hdmi/smoke",
        wandb_id="mujoco1",
    )

    exit_code = module.main(
        [
            "--checkpoint-root",
            str(tmp_path / "outputs"),
            "--min-golden-total-frames",
            "100000000",
            "--output",
            str(tmp_path / "inventory.json"),
        ]
    )
    printed = json.loads(capsys.readouterr().out)
    saved = json.loads((tmp_path / "inventory.json").read_text())

    assert exit_code == 0
    assert saved == printed
    assert printed["checkpoint_count"] == 2
    assert printed["golden_candidate_count"] == 1
    assert printed["weak_mujoco_seed_count"] == 1
    by_path = {entry["path"]: entry for entry in printed["checkpoints"]}
    assert by_path[str(isaac_ckpt)]["provenance_class"] == "golden_reference_candidate"
    assert by_path[str(isaac_ckpt)]["golden_candidate"] is True
    assert by_path[str(mujoco_ckpt)]["provenance_class"] == "weak_mujoco_seed"
    assert by_path[str(mujoco_ckpt)]["golden_candidate"] is False
    assert "not_isaac_backend" in by_path[str(mujoco_ckpt)]["rejection_reasons"]
    assert "insufficient_total_frames" in by_path[str(mujoco_ckpt)]["rejection_reasons"]


def test_checkpoint_inventory_requires_closed_loop_success_to_validate_golden_reference(tmp_path, capsys):
    module = _load_inventory_module()
    success_ckpt = _write_checkpoint(
        tmp_path / "outputs/success/files/checkpoint_final.pt",
        backend="isaac",
        total_frames=120_000_000,
        task_name="G1SuccessTask",
        wandb_id="success1",
    )
    failed_ckpt = _write_checkpoint(
        tmp_path / "outputs/failed/files/checkpoint_final.pt",
        backend="isaac",
        total_frames=120_000_000,
        task_name="G1FailedTask",
        wandb_id="failed1",
    )
    closed_loop_report = tmp_path / "closed_loop.json"
    closed_loop_report.write_text(
        json.dumps(
            {
                "items": [
                    {"scenario": "G1SuccessTask", "heuristic_success": True},
                    {"scenario": "G1FailedTask", "heuristic_success": False},
                ]
            }
        ),
        encoding="utf-8",
    )

    exit_code = module.main(
        [
            "--checkpoint-root",
            str(tmp_path / "outputs"),
            "--closed-loop-report",
            str(closed_loop_report),
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["golden_candidate_count"] == 2
    assert report["validated_golden_reference_count"] == 1
    assert report["missing_validated_golden_reference"] is False
    by_path = {entry["path"]: entry for entry in report["checkpoints"]}
    assert by_path[str(success_ckpt)]["validated_golden_reference"] is True
    assert by_path[str(success_ckpt)]["validation_rejection_reasons"] == []
    assert by_path[str(failed_ckpt)]["validated_golden_reference"] is False
    assert by_path[str(failed_ckpt)]["validation_rejection_reasons"] == ["closed_loop_success_false"]


def test_checkpoint_inventory_includes_exported_policy_run_ids_without_trusting_them(tmp_path, capsys):
    module = _load_inventory_module()
    export_dir = tmp_path / "exports/G1PushBox"
    export_dir.mkdir(parents=True)
    (export_dir / "policy-jl7or24q-final.pt").write_bytes(b"not-a-real-policy")
    (export_dir / "policy-jl7or24q-final.yaml").write_text("observation: {}\n")

    exit_code = module.main(
        [
            "--checkpoint-root",
            str(tmp_path / "missing_outputs"),
            "--exports-dir",
            str(tmp_path / "exports"),
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["checkpoint_count"] == 0
    assert report["export_count"] == 1
    assert report["exports"][0]["task_name"] == "G1PushBox"
