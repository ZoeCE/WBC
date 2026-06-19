from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


HDMI_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = HDMI_ROOT / "mujoco_external_payloads.yaml"


@dataclass(frozen=True)
class ExternalPayload:
    path: Path
    kind: str
    required: bool = True
    task_configs: tuple[Path, ...] = ()
    size_bytes: int | None = None
    sha256: str | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "path": self.path.as_posix(),
            "kind": self.kind,
            "required": self.required,
        }
        if self.task_configs:
            data["task_configs"] = [path.as_posix() for path in self.task_configs]
        if self.size_bytes is not None:
            data["size_bytes"] = self.size_bytes
        if self.sha256 is not None:
            data["sha256"] = self.sha256
        if self.note is not None:
            data["note"] = self.note
        return data


@dataclass(frozen=True)
class ExternalPayloadManifest:
    payloads: tuple[ExternalPayload, ...]
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "payloads": [payload.to_dict() for payload in self.payloads],
        }


def load_external_payload_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> ExternalPayloadManifest:
    manifest_path = Path(path)
    data = _load_yaml_mapping(manifest_path)
    payloads = tuple(_payload_from_mapping(entry, manifest_path) for entry in data.get("payloads", ()))
    duplicates = _duplicates(payload.path for payload in payloads)
    if duplicates:
        raise ValueError(f"{manifest_path}: duplicate payload paths: {duplicates}")
    return ExternalPayloadManifest(version=int(data.get("version", 1)), payloads=payloads)


def discover_task_motion_payloads(task_root: str | Path) -> dict[Path, tuple[Path, ...]]:
    task_root = Path(task_root)
    hdmi_root = _find_hdmi_root(task_root)
    tasks_by_payload: dict[Path, list[Path]] = {}
    for task_path in sorted(task_root.rglob("*.yaml")):
        task_cfg = _load_yaml_mapping(task_path)
        command_cfg = task_cfg.get("command")
        if not isinstance(command_cfg, Mapping):
            continue
        data_path = command_cfg.get("data_path")
        if not _is_concrete_motion_data_path(data_path):
            continue
        payload_path = Path(str(data_path)) / "motion.npz"
        tasks_by_payload.setdefault(payload_path, []).append(_relative_to(task_path, hdmi_root))
    return {payload: tuple(tasks) for payload, tasks in sorted(tasks_by_payload.items())}


def audit_external_payloads(
    manifest: ExternalPayloadManifest,
    *,
    root: str | Path = HDMI_ROOT,
    verify_sha256: bool = False,
) -> dict[str, Any]:
    root = Path(root)
    present: list[str] = []
    missing: list[str] = []
    required_missing: list[str] = []
    size_mismatch: list[dict[str, Any]] = []
    sha256_mismatch: list[dict[str, Any]] = []

    for payload in manifest.payloads:
        payload_key = payload.path.as_posix()
        payload_path = root / payload.path
        if not payload_path.exists():
            missing.append(payload_key)
            if payload.required:
                required_missing.append(payload_key)
            continue

        present.append(payload_key)
        if payload.size_bytes is not None:
            actual_size = payload_path.stat().st_size
            if actual_size != payload.size_bytes:
                size_mismatch.append(
                    {
                        "path": payload_key,
                        "expected": payload.size_bytes,
                        "actual": actual_size,
                    }
                )
        if verify_sha256 and payload.sha256 is not None:
            actual_sha256 = _sha256(payload_path)
            if actual_sha256 != payload.sha256:
                sha256_mismatch.append(
                    {
                        "path": payload_key,
                        "expected": payload.sha256,
                        "actual": actual_sha256,
                    }
                )

    return {
        "total": len(manifest.payloads),
        "present": present,
        "missing": missing,
        "required_missing": required_missing,
        "size_mismatch": size_mismatch,
        "sha256_mismatch": sha256_mismatch,
    }


def _payload_from_mapping(data: Any, manifest_path: Path) -> ExternalPayload:
    if not isinstance(data, Mapping):
        raise ValueError(f"{manifest_path}: payload entries must be mappings.")
    path = _relative_path(data.get("path"), "path", manifest_path)
    task_configs = tuple(
        _relative_path(value, "task_configs", manifest_path)
        for value in data.get("task_configs", ())
    )
    return ExternalPayload(
        path=path,
        kind=str(data.get("kind", "unknown")),
        required=bool(data.get("required", True)),
        task_configs=task_configs,
        size_bytes=_optional_int(data.get("size_bytes")),
        sha256=_optional_str(data.get("sha256")),
        note=_optional_str(data.get("note")),
    )


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: expected a YAML mapping.")
    return data


def _relative_path(value: Any, field_name: str, manifest_path: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{manifest_path}: payload.{field_name} must be a non-empty string.")
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"{manifest_path}: payload.{field_name} must be relative, got {value!r}.")
    return path


def _is_concrete_motion_data_path(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    if value in {"", "???"}:
        return False
    if "${" in value:
        return False
    return value.startswith("data/")


def _find_hdmi_root(path: Path) -> Path:
    resolved = path.resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / "cfg").exists() and (candidate / "active_adaptation").exists():
            return candidate
    if resolved.name == "task" and resolved.parent.name == "cfg":
        return resolved.parent.parent
    return resolved


def _relative_to(path: Path, root: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return path


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _duplicates(values) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        key = value.as_posix() if isinstance(value, Path) else str(value)
        if key in seen and key not in duplicates:
            duplicates.append(key)
        seen.add(key)
    return duplicates


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
