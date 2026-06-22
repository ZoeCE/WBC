import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_source_audit_module():
    script_path = ROOT / "scripts/mujoco_hdmi_checkpoint_source_audit.py"
    spec = importlib.util.spec_from_file_location("mujoco_hdmi_checkpoint_source_audit", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoint_source_audit_rejects_readme_placeholders_as_golden_sources(tmp_path, capsys):
    module = _load_source_audit_module()
    repo = tmp_path / "HDMI"
    repo.mkdir()
    (repo / "README.md").write_text(
        "\n".join(
            [
                "Load a trained policy with:",
                "python scripts/play.py checkpoint_path=run:<teacher-wandb_run_path> export_policy=true",
                "MuJoCo motion viewer:",
                "python scripts/vis/mujoco_mocap_viewer.py",
                "python scripts/vis/motion_data_publisher.py <path-to-motion-folder>",
            ]
        ),
        encoding="utf-8",
    )
    (repo / "scripts").mkdir()
    (repo / "scripts/play.py").write_text(
        "cfg.checkpoint_path = 'run:<teacher-wandb_run_path>'\n",
        encoding="utf-8",
    )

    exit_code = module.main(["--root", str(repo), "--require-actionable-source"])
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert report["gate_passed"] is False
    assert report["actionable_golden_source_available"] is False
    assert report["checkpoint_source"]["placeholder_run_reference_count"] == 2
    assert report["checkpoint_source"]["concrete_run_reference_count"] == 0
    assert report["checkpoint_source"]["direct_checkpoint_link_count"] == 0
    assert report["failures"] == [
        {
            "component": "checkpoint_source",
            "reason": "no_actionable_golden_checkpoint_source",
        }
    ]


def test_checkpoint_source_audit_accepts_concrete_run_or_download_link(tmp_path, capsys):
    module = _load_source_audit_module()
    repo = tmp_path / "HDMI"
    repo.mkdir()
    (repo / "README.md").write_text(
        "\n".join(
            [
                "Original checkpoint:",
                "checkpoint_path=run:lecar-lab/hdmi/G1PushBoxTeacher123",
                "Download mirror:",
                "https://huggingface.co/lecar-lab/hdmi/resolve/main/G1PushBox/checkpoint_final.pt",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = module.main(["--root", str(repo), "--require-actionable-source"])
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["gate_passed"] is True
    assert report["actionable_golden_source_available"] is True
    assert report["checkpoint_source"]["concrete_run_reference_count"] == 1
    assert report["checkpoint_source"]["direct_checkpoint_link_count"] == 1
    assert report["checkpoint_source"]["concrete_run_references"] == [
        {
            "path": "README.md",
            "line": 2,
            "reference": "run:lecar-lab/hdmi/G1PushBoxTeacher123",
        }
    ]


def test_checkpoint_source_audit_ignores_test_fixture_references(tmp_path, capsys):
    module = _load_source_audit_module()
    repo = tmp_path / "HDMI"
    repo.mkdir()
    tests = repo / "tests"
    tests.mkdir()
    (tests / "test_fake_checkpoint_source.py").write_text(
        "\n".join(
            [
                "checkpoint_path='run:entity/project/test_fixture'",
                "url='https://huggingface.co/example/fixture/resolve/main/checkpoint_final.pt'",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = module.main(["--root", str(repo), "--require-actionable-source"])
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert report["gate_passed"] is False
    assert report["actionable_golden_source_available"] is False
    assert report["checkpoint_source"]["concrete_run_reference_count"] == 0
    assert report["checkpoint_source"]["direct_checkpoint_link_count"] == 0
    assert report["failures"] == [
        {
            "component": "checkpoint_source",
            "reason": "no_actionable_golden_checkpoint_source",
        }
    ]
