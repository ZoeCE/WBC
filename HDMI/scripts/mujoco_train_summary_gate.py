from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_gate_report(
        summary_paths=[Path(path) for path in args.summaries],
        require_backend=args.require_backend,
        require_checkpoint=args.require_checkpoint,
        min_env_frames=args.min_env_frames,
        min_num_summaries=args.min_num_summaries,
        min_eval_metrics=_parse_metric_thresholds(args.min_eval_metric),
        max_eval_metrics=_parse_metric_thresholds(args.max_eval_metric),
        min_train_metrics=_parse_metric_thresholds(args.min_train_metric),
        max_train_metrics=_parse_metric_thresholds(args.max_train_metric),
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["gate_passed"] else 1


def build_gate_report(
    *,
    summary_paths: Sequence[Path],
    require_backend: str | None = None,
    require_checkpoint: bool = False,
    min_env_frames: int | None = None,
    min_num_summaries: int | None = None,
    min_eval_metrics: dict[str, float] | None = None,
    max_eval_metrics: dict[str, float] | None = None,
    min_train_metrics: dict[str, float] | None = None,
    max_train_metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    summaries = []
    failures: list[dict[str, Any]] = []

    for path in summary_paths:
        summary, load_failure = _load_summary(path)
        if load_failure is not None:
            failures.append(load_failure)
            continue
        summaries.append(summary)
        failures.extend(
            _summary_failures(
                path=path,
                summary=summary,
                require_backend=require_backend,
                require_checkpoint=require_checkpoint,
                min_env_frames=min_env_frames,
                min_eval_metrics=min_eval_metrics or {},
                max_eval_metrics=max_eval_metrics or {},
                min_train_metrics=min_train_metrics or {},
                max_train_metrics=max_train_metrics or {},
            )
        )

    if min_num_summaries is not None and len(summaries) < min_num_summaries:
        failures.append(
            {
                "reason": "num_summaries_below_min",
                "actual": len(summaries),
                "limit": int(min_num_summaries),
            }
        )

    return {
        "gate_passed": not failures,
        "num_summaries": len(summaries),
        "summary_paths": [str(path) for path in summary_paths],
        "thresholds": _threshold_report(
            require_backend=require_backend,
            require_checkpoint=require_checkpoint,
            min_env_frames=min_env_frames,
            min_num_summaries=min_num_summaries,
            min_eval_metrics=min_eval_metrics or {},
            max_eval_metrics=max_eval_metrics or {},
            min_train_metrics=min_train_metrics or {},
            max_train_metrics=max_train_metrics or {},
        ),
        "metric_ranges": {
            "train": _metric_ranges(summary.get("train", {}) for summary in summaries),
            "eval": _metric_ranges(summary.get("eval", {}) for summary in summaries),
        },
        "summaries": [_summary_record(summary) for summary in summaries],
        "failures": failures,
    }


def _summary_failures(
    *,
    path: Path,
    summary: dict[str, Any],
    require_backend: str | None,
    require_checkpoint: bool,
    min_env_frames: int | None,
    min_eval_metrics: dict[str, float],
    max_eval_metrics: dict[str, float],
    min_train_metrics: dict[str, float],
    max_train_metrics: dict[str, float],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []

    if require_backend is not None and summary.get("backend") != require_backend:
        failures.append(
            {
                "summary_path": str(path),
                "reason": "backend_mismatch",
                "actual": summary.get("backend"),
                "expected": require_backend,
            }
        )

    checkpoint = summary.get("checkpoint_final")
    if require_checkpoint and (not checkpoint or not Path(str(checkpoint)).is_file()):
        failures.append(
            {
                "summary_path": str(path),
                "reason": "checkpoint_missing",
                "checkpoint_final": checkpoint,
            }
        )

    env_frames = summary.get("env_frames")
    if min_env_frames is not None and not _at_least(env_frames, float(min_env_frames)):
        failures.append(
            {
                "summary_path": str(path),
                "reason": "env_frames_below_min",
                "actual": env_frames,
                "limit": int(min_env_frames),
            }
        )

    non_finite = summary.get("non_finite_keys") or {}
    train_non_finite = list(non_finite.get("train") or [])
    eval_non_finite = list(non_finite.get("eval") or [])
    train_non_finite.extend(_non_finite_metric_keys(summary.get("train", {})))
    eval_non_finite.extend(_non_finite_metric_keys(summary.get("eval", {})))
    if train_non_finite or eval_non_finite:
        failures.append(
            {
                "summary_path": str(path),
                "reason": "non_finite_keys",
                "train": sorted(set(train_non_finite)),
                "eval": sorted(set(eval_non_finite)),
            }
        )

    failures.extend(_metric_failures(path, "train", summary.get("train", {}), min_train_metrics, ">="))
    failures.extend(_metric_failures(path, "train", summary.get("train", {}), max_train_metrics, "<="))
    failures.extend(_metric_failures(path, "eval", summary.get("eval", {}), min_eval_metrics, ">="))
    failures.extend(_metric_failures(path, "eval", summary.get("eval", {}), max_eval_metrics, "<="))
    return failures


def _metric_failures(
    path: Path,
    group: str,
    metrics: dict[str, Any],
    thresholds: dict[str, float],
    comparison: str,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for key, limit in thresholds.items():
        actual = metrics.get(key)
        if not _threshold_passes(actual, limit, comparison):
            direction = "below_min" if comparison == ">=" else "above_max"
            failures.append(
                {
                    "summary_path": str(path),
                    "reason": f"{group}_metric_{direction}",
                    "metric": key,
                    "actual": actual,
                    "limit": limit,
                    "comparison": comparison,
                }
            )
    return failures


def _metric_ranges(metric_groups: Iterable[dict[str, Any]]) -> dict[str, dict[str, float]]:
    values: dict[str, list[float]] = {}
    for metrics in metric_groups:
        for key, value in metrics.items():
            if isinstance(value, bool):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                values.setdefault(str(key), []).append(number)
    return {
        key: {
            "min": min(items),
            "max": max(items),
            "mean": sum(items) / len(items),
        }
        for key, items in sorted(values.items())
        if items
    }


def _summary_record(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "backend": summary.get("backend"),
        "task": summary.get("task"),
        "checkpoint_final": summary.get("checkpoint_final"),
        "num_envs": summary.get("num_envs"),
        "train_every": summary.get("train_every"),
        "env_frames": summary.get("env_frames"),
    }


def _load_summary(path: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return {}, {
            "summary_path": str(path),
            "reason": "summary_load_error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _threshold_passes(actual: Any, limit: float, comparison: str) -> bool:
    try:
        actual_value = float(actual)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(actual_value):
        return False
    if comparison == ">=":
        return actual_value >= limit
    if comparison == "<=":
        return actual_value <= limit
    raise ValueError(f"Unsupported comparison {comparison!r}.")


def _at_least(actual: Any, limit: float) -> bool:
    return _threshold_passes(actual, limit, ">=")


def _non_finite_metric_keys(metrics: dict[str, Any]) -> list[str]:
    keys = []
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(number):
            keys.append(str(key))
    return keys


def _threshold_report(**kwargs: Any) -> dict[str, Any]:
    return {
        key: value
        for key, value in kwargs.items()
        if value not in (None, False, {}, [])
    }


def _parse_metric_thresholds(raw: Sequence[Sequence[str]] | None) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for metric, value in raw or []:
        thresholds[metric] = float(value)
    return thresholds


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate one or more MuJoCo training summary JSON files emitted by scripts/train.py."
    )
    parser.add_argument("summaries", nargs="+", help="Training summary JSON path(s).")
    parser.add_argument("--require-backend", default=None, help="Require every summary to use this backend.")
    parser.add_argument(
        "--require-checkpoint",
        action="store_true",
        help="Require each checkpoint_final path to exist.",
    )
    parser.add_argument("--min-env-frames", type=int, default=None)
    parser.add_argument("--min-num-summaries", type=int, default=None)
    parser.add_argument("--min-eval-metric", nargs=2, action="append", default=[], metavar=("KEY", "VALUE"))
    parser.add_argument("--max-eval-metric", nargs=2, action="append", default=[], metavar=("KEY", "VALUE"))
    parser.add_argument("--min-train-metric", nargs=2, action="append", default=[], metavar=("KEY", "VALUE"))
    parser.add_argument("--max-train-metric", nargs=2, action="append", default=[], metavar=("KEY", "VALUE"))
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
