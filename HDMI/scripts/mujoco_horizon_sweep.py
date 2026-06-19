from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence


_THRESHOLD_CHECKS = (
    ("max_q_l2", "q_l2_max", "playback", "q_l2", "max", "<="),
    ("max_body_pos_l2", "body_pos_l2_max", "playback", "body_pos_l2", "max", "<="),
    ("min_reward_mean", "reward_mean", "playback", "reward", "mean", ">="),
    ("max_policy_rollout_q_l2", "policy_rollout_q_l2_max", "policy_rollout", "q_l2", "max", "<="),
    (
        "max_policy_rollout_body_pos_l2",
        "policy_rollout_body_pos_l2_max",
        "policy_rollout",
        "body_pos_l2",
        "max",
        "<=",
    ),
    (
        "min_policy_rollout_reward_mean",
        "policy_rollout_reward_mean",
        "policy_rollout",
        "reward",
        "mean",
        ">=",
    ),
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_horizon_sweep_report(
        trace_path=Path(args.trace_json),
        horizons=_parse_horizon_args(args.horizon),
        thresholds={
            "max_q_l2": args.max_q_l2,
            "max_body_pos_l2": args.max_body_pos_l2,
            "min_reward_mean": args.min_reward_mean,
            "max_policy_rollout_q_l2": args.max_policy_rollout_q_l2,
            "max_policy_rollout_body_pos_l2": args.max_policy_rollout_body_pos_l2,
            "min_policy_rollout_reward_mean": args.min_policy_rollout_reward_mean,
        },
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["gate_passed"] else 1


def build_horizon_sweep_report(
    *,
    trace_path: Path,
    horizons: Sequence[str],
    thresholds: dict[str, float | None],
) -> dict[str, Any]:
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    num_steps = _trace_num_steps(trace)
    resolved_horizons = [_resolve_horizon(raw, num_steps) for raw in (horizons or ("all",))]
    threshold_report = {name: value for name, value in thresholds.items() if value is not None}

    horizon_reports = []
    for label, step_count in resolved_horizons:
        metrics = _horizon_metrics(trace, step_count)
        failures = _threshold_failures(metrics, threshold_report)
        horizon_reports.append(
            {
                "horizon": label,
                "steps": step_count,
                **metrics,
                "threshold_failures": failures,
                "passed": not failures,
            }
        )

    first_failure = next((record for record in horizon_reports if not record["passed"]), None)
    return {
        "trace_path": str(trace_path),
        "num_steps": num_steps,
        "thresholds": threshold_report,
        "horizons": horizon_reports,
        "first_failure": first_failure,
        "first_crossings": _first_crossings(trace, threshold_report),
        "gate_passed": first_failure is None,
    }


def _horizon_metrics(trace: dict[str, Any], step_count: int) -> dict[str, float | None]:
    return {
        "q_l2_max": _tensor_prefix_max(trace, "playback", "q_l2", step_count),
        "body_pos_l2_max": _tensor_prefix_max(trace, "playback", "body_pos_l2", step_count),
        "reward_mean": _tensor_prefix_mean(trace, "playback", "reward", step_count),
        "policy_rollout_q_l2_max": _tensor_prefix_max(trace, "policy_rollout", "q_l2", step_count),
        "policy_rollout_body_pos_l2_max": _tensor_prefix_max(
            trace,
            "policy_rollout",
            "body_pos_l2",
            step_count,
        ),
        "policy_rollout_reward_mean": _tensor_prefix_mean(trace, "policy_rollout", "reward", step_count),
    }


def _threshold_failures(metrics: dict[str, float | None], thresholds: dict[str, float]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for threshold_name, metric_name, _section, _key, _reducer, comparison in _THRESHOLD_CHECKS:
        if threshold_name not in thresholds:
            continue
        limit = float(thresholds[threshold_name])
        actual = metrics.get(metric_name)
        if not _threshold_passes(actual, limit, comparison):
            failures.append(
                {
                    "metric": metric_name,
                    "actual": actual,
                    "limit": limit,
                    "comparison": comparison,
                }
            )
    return failures


def _first_crossings(trace: dict[str, Any], thresholds: dict[str, float]) -> list[dict[str, Any]]:
    crossings: list[dict[str, Any]] = []
    for threshold_name, metric_name, section, key, reducer, comparison in _THRESHOLD_CHECKS:
        if threshold_name not in thresholds:
            continue
        limit = float(thresholds[threshold_name])
        step_values = _tensor_step_values(trace, section, key, reducer)
        if not step_values:
            crossings.append(
                {
                    "metric": metric_name,
                    "step_index": None,
                    "step": None,
                    "actual": None,
                    "limit": limit,
                    "comparison": comparison,
                }
            )
            continue
        for step_index, actual in enumerate(step_values):
            if _threshold_passes(actual, limit, comparison):
                continue
            crossings.append(
                {
                    "metric": metric_name,
                    "step_index": step_index,
                    "step": step_index + 1,
                    "actual": actual,
                    "limit": limit,
                    "comparison": comparison,
                }
            )
            break
    return crossings


def _threshold_passes(actual: float | None, limit: float, comparison: str) -> bool:
    if actual is None:
        return False
    if not math.isfinite(float(actual)):
        return False
    if comparison == "<=":
        return float(actual) <= limit
    if comparison == ">=":
        return float(actual) >= limit
    raise ValueError(f"Unsupported comparison {comparison!r}.")


def _tensor_prefix_max(trace: dict[str, Any], section: str, key: str, step_count: int) -> float | None:
    values = _tensor_prefix_values(trace, section, key, step_count)
    return max(values) if values else None


def _tensor_prefix_mean(trace: dict[str, Any], section: str, key: str, step_count: int) -> float | None:
    values = _tensor_prefix_values(trace, section, key, step_count)
    return sum(values) / len(values) if values else None


def _tensor_step_values(trace: dict[str, Any], section: str, key: str, reducer: str) -> list[float | None]:
    tensor = (trace.get(section) or {}).get(key)
    if tensor is None:
        return []
    values = tensor.get("values")
    if values is None:
        return []

    step_values: list[float | None] = []
    for step in values:
        flat = [float(value) for value in _flatten(step)]
        if not flat:
            step_values.append(None)
        elif reducer == "max":
            step_values.append(max(flat))
        elif reducer == "mean":
            step_values.append(sum(flat) / len(flat))
        else:
            raise ValueError(f"Unsupported reducer {reducer!r}.")
    return step_values


def _tensor_prefix_values(trace: dict[str, Any], section: str, key: str, step_count: int) -> list[float]:
    tensor = (trace.get(section) or {}).get(key)
    if tensor is None:
        return []
    values = tensor.get("values")
    if values is None:
        return []
    return [float(value) for value in _flatten(values[:step_count])]


def _flatten(value: Any):
    if isinstance(value, list):
        for item in value:
            yield from _flatten(item)
    else:
        yield value


def _trace_num_steps(trace: dict[str, Any]) -> int:
    for section in ("policy_rollout", "playback"):
        tensor = (trace.get(section) or {}).get("q_l2")
        shape = (tensor or {}).get("shape")
        if shape:
            return int(shape[0])
    raise ValueError("Trace does not contain playback or policy_rollout q_l2 shape.")


def _parse_horizon_args(raw_horizons: Sequence[str]) -> tuple[str, ...]:
    values: list[str] = []
    for raw in raw_horizons:
        values.extend(part.strip() for part in str(raw).split(",") if part.strip())
    return tuple(values)


def _resolve_horizon(raw: str, num_steps: int) -> tuple[str | int, int]:
    if raw == "all":
        return "all", num_steps
    step_count = int(raw)
    if step_count < 1 or step_count > num_steps:
        raise ValueError(f"horizon must be in [1, {num_steps}] or 'all', got {raw!r}.")
    return step_count, step_count


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize MuJoCo playback/policy-rollout trace metrics over prefix horizons."
    )
    parser.add_argument("trace_json", help="Trace JSON emitted by scripts/mujoco_playback_parity.py --trace-json.")
    parser.add_argument(
        "--horizon",
        action="append",
        default=[],
        help="Prefix horizon in steps, 'all', or a comma-separated list. Defaults to all.",
    )
    parser.add_argument("--max-q-l2", type=float, default=None)
    parser.add_argument("--max-body-pos-l2", type=float, default=None)
    parser.add_argument("--min-reward-mean", type=float, default=None)
    parser.add_argument("--max-policy-rollout-q-l2", type=float, default=None)
    parser.add_argument("--max-policy-rollout-body-pos-l2", type=float, default=None)
    parser.add_argument("--min-policy-rollout-reward-mean", type=float, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
