from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence


HDMI_ROOT = Path(__file__).resolve().parents[1]
if str(HDMI_ROOT) not in sys.path:
    sys.path.insert(0, str(HDMI_ROOT))


COMPONENT_SPECS: dict[str, dict[str, Any]] = {
    "batched_env": {
        "module": "active_adaptation.envs.mujoco",
        "attrs": ["MJArticulation", "MJScene", "MJSim"],
        "test_paths": ["tests/test_mujoco_batched_env.py"],
    },
    "object_contact": {
        "module": "active_adaptation.mujoco.observation_builder",
        "attrs": ["MujocoObservationBuilder", "MujocoPolicyState"],
        "source_terms": ["ref_contact_pos_b", "diff_contact_pos_b"],
        "test_paths": ["tests/test_mujoco_observation_builder.py"],
    },
    "domain_randomization": {
        "module": "active_adaptation.mujoco.domain_randomization",
        "attrs": ["sample_object_body_randomization", "sample_object_joint_randomization"],
        "test_paths": [
            "tests/test_mujoco_domain_randomization.py",
            "tests/test_mujoco_domain_randomization_apply.py",
        ],
    },
    "reward_parity": {
        "module": "active_adaptation.mujoco.reward_parity",
        "attrs": [
            "keypoint_position_tracking_product",
            "joint_position_tracking_product",
            "object_position_tracking",
            "object_orientation_tracking",
            "object_joint_position_tracking",
            "eef_contact_exp",
            "eef_contact_exp_max",
            "eef_contact_all",
        ],
        "test_paths": ["tests/test_mujoco_reward_parity.py"],
    },
}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_component_audit(
        root=Path(args.root),
        required_components=args.require_component,
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["component_gate_passed"] else 1


def build_component_audit(
    *,
    root: Path = HDMI_ROOT,
    required_components: Sequence[str] | None = None,
) -> dict[str, Any]:
    component_names = list(required_components or COMPONENT_SPECS)
    components: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []

    for component_name in component_names:
        component_report = _audit_component(component_name, root=root)
        components[component_name] = component_report
        failures.extend(component_report["failures"])

    return {
        "component_gate_passed": not failures,
        "covered_components": component_names,
        "components": components,
        "component_failures": failures,
        "root": str(root),
    }


def _audit_component(component_name: str, *, root: Path) -> dict[str, Any]:
    spec = COMPONENT_SPECS.get(component_name)
    if spec is None:
        failure = {"component": component_name, "reason": "unknown_component"}
        return {
            "gate_passed": False,
            "checks": [],
            "failures": [failure],
        }

    checks: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    module_check = _check_module_attrs(
        component_name=component_name,
        module_name=spec["module"],
        attrs=spec.get("attrs", []),
    )
    checks.append(module_check)
    failures.extend(module_check["failures"])

    for term in spec.get("source_terms", []):
        source_check = _check_source_term(
            component_name=component_name,
            module_name=spec["module"],
            term=term,
        )
        checks.append(source_check)
        failures.extend(source_check["failures"])

    for relative_path in spec.get("test_paths", []):
        test_check = _check_test_path(
            component_name=component_name,
            root=root,
            relative_path=relative_path,
        )
        checks.append(test_check)
        failures.extend(test_check["failures"])

    return {
        "gate_passed": not failures,
        "checks": checks,
        "failures": failures,
    }


def _check_module_attrs(*, component_name: str, module_name: str, attrs: Sequence[str]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return {
            "kind": "module_attrs",
            "module": module_name,
            "attrs": list(attrs),
            "passed": False,
            "failures": [
                {
                    "component": component_name,
                    "reason": "module_import_error",
                    "module": module_name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            ],
        }

    missing_attrs = [attr for attr in attrs if not hasattr(module, attr)]
    if missing_attrs:
        failures.append(
            {
                "component": component_name,
                "reason": "module_attr_missing",
                "module": module_name,
                "attrs": missing_attrs,
            }
        )

    return {
        "kind": "module_attrs",
        "module": module_name,
        "attrs": list(attrs),
        "passed": not failures,
        "failures": failures,
    }


def _check_source_term(*, component_name: str, module_name: str, term: str) -> dict[str, Any]:
    module = sys.modules.get(module_name)
    if module is None:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            return {
                "kind": "source_term",
                "module": module_name,
                "term": term,
                "passed": False,
                "failures": [
                    {
                        "component": component_name,
                        "reason": "module_import_error",
                        "module": module_name,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                ],
            }

    source_path = Path(module.__file__ or "")
    try:
        source_text = source_path.read_text(encoding="utf-8")
    except Exception as exc:
        return {
            "kind": "source_term",
            "module": module_name,
            "term": term,
            "passed": False,
            "failures": [
                {
                    "component": component_name,
                    "reason": "source_read_error",
                    "path": str(source_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            ],
        }

    failure = None
    if term not in source_text:
        failure = {
            "component": component_name,
            "reason": "source_term_missing",
            "module": module_name,
            "term": term,
        }
    return {
        "kind": "source_term",
        "module": module_name,
        "term": term,
        "passed": failure is None,
        "failures": [] if failure is None else [failure],
    }


def _check_test_path(*, component_name: str, root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    failure = None
    if not path.is_file():
        failure = {
            "component": component_name,
            "reason": "test_path_missing",
            "path": relative_path,
        }
    return {
        "kind": "test_path",
        "path": relative_path,
        "passed": failure is None,
        "failures": [] if failure is None else [failure],
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit MuJoCo migration component coverage for batched envs, contact, randomization, and rewards."
    )
    parser.add_argument("--root", default=str(HDMI_ROOT), help="HDMI repository root used for test file checks.")
    parser.add_argument(
        "--require-component",
        action="append",
        default=[],
        metavar="NAME",
        help="Only require the named component. Can be passed multiple times; defaults to all components.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
