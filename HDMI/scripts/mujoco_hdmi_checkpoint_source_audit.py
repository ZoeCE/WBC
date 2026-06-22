from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
HDMI_ROOT = SCRIPT_DIR.parent
SOURCE_REFERENCE = {
    "repository": "https://github.com/LeCAR-Lab/HDMI",
    "purpose": (
        "Audit whether a source tree contains actionable original HDMI/Isaac checkpoint "
        "sources, rather than README placeholders such as run:<teacher-wandb_run_path>."
    ),
}
TEXT_SUFFIXES = {
    ".md",
    ".rst",
    ".txt",
    ".py",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".cfg",
    ".ini",
    ".sh",
}
SKIP_DIR_NAMES = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "outputs",
    "wandb",
    "runs",
    "logs",
    "exports",
    "tests",
}
RUN_REFERENCE_RE = re.compile(
    r"run:(?P<ref><[^>\s`\"')]+>|[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/[A-Za-z0-9_.:-]+)"
)
URL_RE = re.compile(r"https?://[^\s`\"')]+")
CHECKPOINT_URL_RE = re.compile(r"\.(?:pt|pth|ckpt)(?:$|[?#])", re.IGNORECASE)
LOCAL_CHECKPOINT_PATTERNS = ("checkpoint*.pt", "checkpoint*.pth", "*.ckpt")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_checkpoint_source_audit(
        roots=[Path(root) for root in args.root],
        require_actionable_source=args.require_actionable_source,
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, sort_keys=True))
    return 0 if report["gate_passed"] else 1


def build_checkpoint_source_audit(
    *,
    roots: Sequence[Path],
    require_actionable_source: bool = False,
) -> dict[str, Any]:
    scan_roots = [root.resolve() for root in roots] if roots else [HDMI_ROOT.resolve()]
    text_files = list(_iter_text_files(scan_roots))
    placeholder_refs: list[dict[str, Any]] = []
    concrete_refs: list[dict[str, Any]] = []
    direct_links: list[dict[str, Any]] = []

    for text_file in text_files:
        for line_number, line in _read_lines(text_file):
            for match in RUN_REFERENCE_RE.finditer(line):
                entry = {
                    "path": _display_path(text_file, scan_roots),
                    "line": line_number,
                    "reference": f"run:{match.group('ref')}",
                }
                if _is_placeholder_run_reference(match.group("ref")):
                    placeholder_refs.append(entry)
                else:
                    concrete_refs.append(entry)
            for url in _checkpoint_urls(line):
                direct_links.append(
                    {
                        "path": _display_path(text_file, scan_roots),
                        "line": line_number,
                        "url": url,
                    }
                )

    checkpoint_files = [
        {"path": _display_path(path, scan_roots), "root": str(_owning_root(path, scan_roots) or path.parent)}
        for path in _discover_local_checkpoint_files(scan_roots)
    ]
    actionable_available = bool(concrete_refs or direct_links or checkpoint_files)
    failures = []
    if require_actionable_source and not actionable_available:
        failures.append(
            {
                "component": "checkpoint_source",
                "reason": "no_actionable_golden_checkpoint_source",
            }
        )

    return {
        "gate_passed": not failures,
        "source_reference": SOURCE_REFERENCE,
        "roots": [str(root) for root in scan_roots],
        "actionable_golden_source_available": actionable_available,
        "checkpoint_source": {
            "scan_file_count": len(text_files),
            "placeholder_run_reference_count": len(placeholder_refs),
            "placeholder_run_references": placeholder_refs,
            "concrete_run_reference_count": len(concrete_refs),
            "concrete_run_references": concrete_refs,
            "direct_checkpoint_link_count": len(direct_links),
            "direct_checkpoint_links": direct_links,
            "checkpoint_file_count": len(checkpoint_files),
            "checkpoint_files": checkpoint_files,
        },
        "failures": failures,
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan HDMI source trees for actionable original checkpoint sources. "
            "README placeholders are reported but do not pass the gate."
        )
    )
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        help="Source tree to scan. Can be passed multiple times. Defaults to the current HDMI root.",
    )
    parser.add_argument("--require-actionable-source", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


def _iter_text_files(roots: Sequence[Path]) -> Iterable[Path]:
    for root in roots:
        if root.is_file():
            if root.suffix.lower() in TEXT_SUFFIXES:
                yield root
            continue
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*"), key=lambda item: str(item)):
            if not path.is_file():
                continue
            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if _has_skipped_part(path, root):
                continue
            yield path


def _read_lines(path: Path) -> Iterable[tuple[int, str]]:
    try:
        with path.open(encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                yield line_number, line.rstrip("\n")
    except UnicodeDecodeError:
        return


def _checkpoint_urls(line: str) -> list[str]:
    urls = []
    for match in URL_RE.finditer(line):
        url = match.group(0).rstrip(".,;]")
        if CHECKPOINT_URL_RE.search(url):
            urls.append(url)
    return urls


def _is_placeholder_run_reference(reference: str) -> bool:
    return reference.startswith("<") or "wandb_run_path" in reference or "your_" in reference.lower()


def _discover_local_checkpoint_files(roots: Sequence[Path]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root.is_file():
            candidates = [root] if _looks_like_checkpoint_file(root) else []
        elif root.is_dir():
            candidates = []
            for pattern in LOCAL_CHECKPOINT_PATTERNS:
                candidates.extend(root.rglob(pattern))
        else:
            candidates = []
        for candidate in candidates:
            if _has_skipped_part(candidate, _owning_root(candidate, roots) or candidate.parent):
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(candidate)
    return sorted(paths, key=lambda item: str(item))


def _looks_like_checkpoint_file(path: Path) -> bool:
    return path.suffix.lower() in {".pt", ".pth", ".ckpt"} and "checkpoint" in path.name.lower()


def _has_skipped_part(path: Path, root: Path) -> bool:
    try:
        relative_parts = path.relative_to(root).parts
    except ValueError:
        relative_parts = path.parts
    return any(part in SKIP_DIR_NAMES for part in relative_parts[:-1])


def _display_path(path: Path, roots: Sequence[Path]) -> str:
    owner = _owning_root(path, roots)
    if owner is None:
        return str(path)
    try:
        return str(path.relative_to(owner))
    except ValueError:
        return str(path)


def _owning_root(path: Path, roots: Sequence[Path]) -> Path | None:
    resolved = path.resolve()
    for root in roots:
        root_resolved = root.resolve()
        if resolved == root_resolved:
            return root
        try:
            resolved.relative_to(root_resolved)
            return root
        except ValueError:
            continue
    return None


if __name__ == "__main__":
    raise SystemExit(main())
