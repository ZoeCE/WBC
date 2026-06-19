from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent


def test_external_payload_manifest_covers_concrete_task_motion_data_paths():
    manifest_path = ROOT / "mujoco_external_payloads.yaml"
    assert manifest_path.exists()

    from active_adaptation.mujoco.external_payloads import (
        discover_task_motion_payloads,
        load_external_payload_manifest,
    )

    manifest = load_external_payload_manifest(manifest_path)
    discovered = discover_task_motion_payloads(ROOT / "cfg/task")
    declared = {
        payload.path
        for payload in manifest.payloads
        if payload.kind == "motion_npz"
    }

    assert sorted(set(discovered) - declared) == []


def test_external_payload_manifest_uses_relative_unique_paths():
    from active_adaptation.mujoco.external_payloads import load_external_payload_manifest

    manifest = load_external_payload_manifest(ROOT / "mujoco_external_payloads.yaml")
    paths = [payload.path for payload in manifest.payloads]

    assert all(not path.is_absolute() for path in paths)
    assert len(paths) == len(set(paths))


def test_external_payload_audit_reports_missing_files_without_hashing_them(tmp_path):
    from active_adaptation.mujoco.external_payloads import (
        ExternalPayload,
        ExternalPayloadManifest,
        audit_external_payloads,
    )

    manifest = ExternalPayloadManifest(
        payloads=(
            ExternalPayload(path=Path("missing/motion.npz"), kind="motion_npz", required=True),
        )
    )

    report = audit_external_payloads(manifest, root=tmp_path, verify_sha256=True)

    assert report["missing"] == ["missing/motion.npz"]
    assert report["present"] == []


def test_gitignore_keeps_new_external_payloads_out_of_regular_git():
    gitignore = (REPO_ROOT / ".gitignore").read_text()

    assert "HDMI/active_adaptation/assets/g1/" in gitignore
    assert "HDMI/data/motion/**/motion.npz" in gitignore
