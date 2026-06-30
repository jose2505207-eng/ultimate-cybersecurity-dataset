from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from cyberdataset.utils import DATA_DIR, dump_json


SUPPORTED_EXTENSIONS = {".csv", ".json", ".jsonl", ".parquet", ".db", ".sqlite", ".sqlite3"}
METADATA_FILES = {"README.txt", "LICENSE.txt", "TERMS.txt", "labels.txt"}


def _cheap_row_count(path: Path) -> int | None:
    try:
        if path.suffix == ".csv":
            return max(sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1, 0)
        if path.suffix == ".jsonl":
            return sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore"))
        if path.suffix == ".parquet":
            return int(pd.read_parquet(path, columns=[]).shape[0])
    except Exception:
        return None
    return None


def scan_bronze(bronze_dir: str | Path | None = None) -> list[dict[str, Any]]:
    root = Path(bronze_dir) if bronze_dir else DATA_DIR / "bronze_raw"
    if not root.exists():
        return []

    inventory: list[dict[str, Any]] = []
    for source_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        files = sorted(path for path in source_dir.rglob("*") if path.is_file())
        data_files = [path for path in files if path.suffix.lower() in SUPPORTED_EXTENSIONS]
        metadata_names = {path.name for path in files}
        inventory.append(
            {
                "source": source_dir.name,
                "path": str(source_dir),
                "file_count": len(data_files),
                "formats": sorted({path.suffix.lower().lstrip(".") for path in data_files}),
                "files": [
                    {
                        "path": str(path),
                        "size_bytes": path.stat().st_size,
                        "row_count": _cheap_row_count(path),
                    }
                    for path in data_files
                ],
                "missing_metadata": sorted(METADATA_FILES - metadata_names),
            }
        )
    return inventory


def write_inventory_report() -> Path:
    report = scan_bronze()
    output = DATA_DIR / "reports" / "bronze_inventory_report.json"
    dump_json({"sources": report}, output)
    return output


def main() -> None:
    output = write_inventory_report()
    print(f"Wrote bronze inventory report to {output}")


if __name__ == "__main__":
    main()

