import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/mujoco_component_audit.py"


def _load_component_audit_module():
    spec = importlib.util.spec_from_file_location("mujoco_component_audit", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_component_audit_reports_required_mujoco_components(capsys):
    audit = _load_component_audit_module()

    exit_code = audit.main(["--root", str(ROOT)])

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert report["component_gate_passed"] is True
    assert report["covered_components"] == [
        "batched_env",
        "object_contact",
        "domain_randomization",
        "reward_parity",
    ]
    assert report["component_failures"] == []
    assert report["components"]["batched_env"]["gate_passed"] is True
    assert report["components"]["object_contact"]["gate_passed"] is True
    assert report["components"]["domain_randomization"]["gate_passed"] is True
    assert report["components"]["reward_parity"]["gate_passed"] is True


def test_component_audit_fails_when_required_test_file_is_missing(tmp_path, capsys):
    audit = _load_component_audit_module()

    exit_code = audit.main(
        [
            "--root",
            str(tmp_path),
            "--require-component",
            "reward_parity",
        ]
    )

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert report["component_gate_passed"] is False
    assert report["covered_components"] == ["reward_parity"]
    assert report["component_failures"] == [
        {
            "component": "reward_parity",
            "reason": "test_path_missing",
            "path": "tests/test_mujoco_reward_parity.py",
        }
    ]
