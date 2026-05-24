"""Preflight healthcheck for immutable bronze inputs."""

from __future__ import annotations

import argparse
import hashlib
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BRONZE = PROJECT_ROOT / "data" / "bronze_raw"
CATALOG = PROJECT_ROOT / "data" / "bronze_catalog"
QUARANTINE = PROJECT_ROOT / "data" / "bronze_quarantine"
IGNORE_PARTS = {"venv", ".venv", "__pycache__", "node_modules", ".cache", ".ipynb_checkpoints", ".git"}
SUSPICIOUS_EXT = {".exe", ".dll", ".so", ".elf", ".apk", ".bin", ".sh", ".ps1", ".bat", ".cmd", ".sol", ".js"}


def ignored(path: Path) -> bool:
    """Return true when a path matches global ignore rules."""

    return any(part in IGNORE_PARTS or "broken" in part.lower() for part in path.parts) or path.name.endswith(".crdownload")


def hash_file(path: Path) -> tuple[str, str]:
    """Hash small files fully and large files with first/last MiB plus size."""

    size = path.stat().st_size
    h = hashlib.sha256()
    with path.open("rb") as fh:
        if size <= 500 * 1024 * 1024:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
            return h.hexdigest(), "full"
        h.update(fh.read(1024 * 1024))
        fh.seek(max(size - 1024 * 1024, 0))
        h.update(fh.read(1024 * 1024))
        h.update(str(size).encode())
    return h.hexdigest(), "partial"


def folder_size(path: Path) -> int:
    """Compute folder size without following symlinks."""

    total = 0
    for item in path.rglob("*"):
        if item.is_file() and not item.is_symlink():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total


def quarantine_link(path: Path, reason: str) -> None:
    """Create a symlink pointer for flagged material without moving bronze."""

    QUARANTINE.mkdir(parents=True, exist_ok=True)
    name = f"{reason}__{str(path.relative_to(BRONZE)).replace('/', '__')}"
    link = QUARANTINE / name
    if not link.exists():
        try:
            link.symlink_to(path.resolve())
        except OSError:
            pass


def build_healthcheck() -> dict[str, pd.DataFrame]:
    """Scan bronze and return all healthcheck tables."""

    CATALOG.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    non_dataset: list[dict[str, Any]] = []
    loose: list[dict[str, Any]] = []
    suspicious: list[dict[str, Any]] = []
    large: list[dict[str, Any]] = []

    for path in sorted(BRONZE.rglob("*")):
        rel = path.relative_to(BRONZE)
        if any(part in IGNORE_PARTS or "broken" in part.lower() for part in rel.parts):
            non_dataset.append({"path": str(rel), "reason": "ignored_path"})
            quarantine_link(path, "ignored_path")
            continue
        if path.name.endswith(".crdownload"):
            incomplete.append({"path": str(rel), "reason": "partial_download"})
            quarantine_link(path, "partial_download")
            continue
        if path.is_dir():
            continue
        if not path.is_file():
            continue
        size = path.stat().st_size
        digest, method = hash_file(path)
        records.append({"path": str(rel), "size_bytes": size, "hash": digest, "hash_method": method})
        if path.parent == BRONZE and path.name != ".gitkeep":
            loose.append({"path": str(rel), "size_bytes": size, "reason": "root_level_file"})
        if path.suffix.lower() in SUSPICIOUS_EXT:
            suspicious.append({"path": str(rel), "size_bytes": size, "extension": path.suffix.lower()})
            quarantine_link(path, "suspicious")
        if size > 1024**3:
            large.append({"path": str(rel), "kind": "file", "size_bytes": size})

    seen = defaultdict(list)
    for rec in records:
        seen[(rec["size_bytes"], rec["hash"])].append(rec)
    duplicate_rows: list[dict[str, Any]] = []
    for group_id, rows in enumerate((v for v in seen.values() if len(v) > 1), start=1):
        for row in rows:
            duplicate_rows.append({**row, "group_id": group_id})

    if (BRONZE / "output2.csv").exists():
        for companion in ("Output1.csv", "output1.csv", "Output3.csv", "output3.csv"):
            if not (BRONZE / companion).exists():
                incomplete.append({"path": companion, "reason": "missing_cicmalmem_companion"})

    for folder in sorted(p for p in BRONZE.iterdir() if p.is_dir() and not ignored(p)):
        size = folder_size(folder)
        if size > 5 * 1024**3:
            large.append({"path": str(folder.relative_to(BRONZE)), "kind": "folder", "size_bytes": size})

    return {
        "duplicate_candidates": pd.DataFrame(duplicate_rows, columns=["path", "size_bytes", "hash", "hash_method", "group_id"]),
        "incomplete_downloads": pd.DataFrame(incomplete, columns=["path", "reason"]),
        "non_dataset_items": pd.DataFrame(non_dataset, columns=["path", "reason"]),
        "loose_files_needing_review": pd.DataFrame(loose, columns=["path", "size_bytes", "reason"]),
        "suspicious_binaries": pd.DataFrame(suspicious, columns=["path", "size_bytes", "extension"]),
        "large_items": pd.DataFrame(large, columns=["path", "kind", "size_bytes"]),
    }


def write_outputs(tables: dict[str, pd.DataFrame]) -> None:
    """Write CSV outputs and markdown summary."""

    for name, df in tables.items():
        df.to_csv(CATALOG / f"{name}.csv", index=False)
    lines = ["# Bronze Preflight Healthcheck", ""]
    for name, df in tables.items():
        lines.append(f"- `{name}.csv`: {len(df)} rows")
    lines.extend(["", "## Recommendations", "- Review loose root-level files and quarantined symlinks before planning final benchmark release.", "- Do not execute or install any suspicious scripts or binaries flagged here."])
    (CATALOG / "preflight_healthcheck.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    write_outputs(build_healthcheck())


if __name__ == "__main__":
    main()
