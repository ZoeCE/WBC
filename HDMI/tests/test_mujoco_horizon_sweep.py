import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/mujoco_horizon_sweep.py"


def _load_sweep_module():
    spec = importlib.util.spec_from_file_location("mujoco_horizon_sweep", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_trace(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "playback": {
                    "q_l2": {"shape": [4, 1], "values": [[0.0], [0.0], [0.0], [0.0]]},
                    "body_pos_l2": {
                        "shape": [4, 1],
                        "values": [[0.0], [0.01], [0.02], [0.03]],
                    },
                    "reward": {
                        "shape": [4, 1, 1],
                        "values": [[[1.0]], [[1.0]], [[1.0]], [[1.0]]],
                    },
                },
                "policy_rollout": {
                    "q_l2": {
                        "shape": [4, 1],
                        "values": [[0.10], [0.20], [1.20], [2.00]],
                    },
                    "body_pos_l2": {
                        "shape": [4, 1],
                        "values": [[0.01], [0.02], [0.30], [0.50]],
                    },
                    "reward": {
                        "shape": [4, 1, 1],
                        "values": [[[1.0]], [[1.0]], [[0.2]], [[0.0]]],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_horizon_sweep_reports_first_policy_rollout_failure(tmp_path, capsys):
    sweep = _load_sweep_module()
    trace_path = _write_trace(tmp_path / "trace.json")

    exit_code = sweep.main(
        [
            str(trace_path),
            "--horizon",
            "2",
            "--horizon",
            "4",
            "--max-policy-rollout-q-l2",
            "1.0",
            "--max-policy-rollout-body-pos-l2",
            "0.2",
            "--min-policy-rollout-reward-mean",
            "0.75",
        ]
    )

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert report["gate_passed"] is False
    assert report["num_steps"] == 4
    assert report["horizons"][0]["horizon"] == 2
    assert report["horizons"][0]["passed"] is True
    assert report["horizons"][0]["policy_rollout_q_l2_max"] == 0.2
    assert report["horizons"][1]["horizon"] == 4
    assert report["horizons"][1]["passed"] is False
    assert report["first_failure"]["horizon"] == 4
    failure_metrics = {failure["metric"] for failure in report["first_failure"]["threshold_failures"]}
    assert failure_metrics == {
        "policy_rollout_q_l2_max",
        "policy_rollout_body_pos_l2_max",
        "policy_rollout_reward_mean",
    }


def test_horizon_sweep_passes_all_horizon_alias(tmp_path, capsys):
    sweep = _load_sweep_module()
    trace_path = _write_trace(tmp_path / "trace.json")

    exit_code = sweep.main(
        [
            str(trace_path),
            "--horizon",
            "all",
            "--max-q-l2",
            "0.0",
            "--max-body-pos-l2",
            "0.05",
            "--min-reward-mean",
            "0.99",
        ]
    )

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert report["gate_passed"] is True
    assert report["horizons"][0]["horizon"] == "all"
    assert report["horizons"][0]["steps"] == 4
    assert report["horizons"][0]["q_l2_max"] == 0.0
    assert report["horizons"][0]["body_pos_l2_max"] == 0.03
    assert report["horizons"][0]["reward_mean"] == 1.0
