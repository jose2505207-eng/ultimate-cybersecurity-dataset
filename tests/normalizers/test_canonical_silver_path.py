from __future__ import annotations

from pathlib import Path


def test_tester_facing_files_use_only_canonical_silver_path():
    roots = [
        Path("README.md"),
        Path("DATASET_CARD.md"),
        Path("Makefile"),
        Path("docs"),
        Path("notebooks"),
        Path("scripts"),
        Path("tests"),
        Path("src"),
    ]
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        elif root.exists():
            files.extend(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts)

    deprecated_path = "silver" + "_clean"
    offenders = []
    for path in files:
        if path.suffix in {".pyc", ".parquet", ".gz"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if deprecated_path in text:
            offenders.append(str(path))

    assert offenders == []
