from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from cyberdataset.ingest.tabular import data_files, load_tables, normalize_to_frame, write_silver_frame
from cyberdataset.normalize import make_record


SOURCE_DATASET = "NVD"
LICENSE_NOTE = "NVD terms apply; verify current upstream terms before redistribution."


def _description(cve: dict[str, Any]) -> str:
    descriptions = cve.get("descriptions") or cve.get("description", {}).get("description_data") or []
    if isinstance(descriptions, list):
        for item in descriptions:
            if item.get("lang") == "en" and item.get("value"):
                return item["value"]
        if descriptions and descriptions[0].get("value"):
            return descriptions[0]["value"]
    return ""


def _severity(item: dict[str, Any]) -> str:
    metrics = item.get("metrics") or item.get("impact") or {}
    text = json.dumps(metrics).lower()
    for value in ("critical", "high", "medium", "low"):
        if value in text:
            return value
    return "unknown"


def _cwe(cve: dict[str, Any]) -> str | None:
    weaknesses = cve.get("weaknesses") or cve.get("problemtype", {}).get("problemtype_data") or []
    text = json.dumps(weaknesses)
    if "CWE-" not in text:
        return None
    marker = text[text.find("CWE-") :]
    token = marker.split('"', 1)[0].split("'", 1)[0].split("\\", 1)[0].split(",", 1)[0]
    return token[:32]


def _records_from_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "vulnerabilities" in data:
        items = data["vulnerabilities"]
        return [
            {
                "source_file": path.name,
                "cve": item.get("cve", {}),
                "raw": item,
            }
            for item in items
        ]
    if "CVE_Items" in data:
        return [
            {
                "source_file": path.name,
                "cve": item.get("cve", {}),
                "raw": item,
            }
            for item in data["CVE_Items"]
        ]
    return [{"source_file": path.name, "cve": data.get("cve", data), "raw": data}]


def load_raw(input_path: str | Path, limit: int | None = None) -> pd.DataFrame:
    path = Path(input_path)
    files = data_files(path)
    if files and all(file.suffix.lower() == ".json" for file in files):
        rows: list[dict[str, Any]] = []
        for file in files:
            rows.extend(_records_from_json(file))
            if limit is not None and len(rows) >= limit:
                rows = rows[:limit]
                break
        return pd.DataFrame(rows)
    return load_tables(input_path, limit=limit)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for idx, row in df.iterrows():
        cve = row.get("cve") if isinstance(row.get("cve"), dict) else {}
        raw = row.get("raw") if isinstance(row.get("raw"), dict) else row.to_dict()
        cve_id = cve.get("id") or cve.get("CVE_data_meta", {}).get("ID") or row.get("cve_id") or row.get("id")
        description = _description(cve) or str(row.get("description", ""))
        records.append(
            make_record(
                source_dataset=SOURCE_DATASET,
                source_type="advisory",
                main_category="vulnerability_advisory",
                attack_name="Vulnerability Advisory",
                source_label="advisory",
                raw_text_or_features={
                    "source_file": row.get("source_file") or row.get("__source_file"),
                    "source_row": int(idx),
                    "cve_id": cve_id,
                    "description": description,
                    "published": raw.get("published") or raw.get("publishedDate"),
                },
                source_key=cve_id or idx,
                cve_id=cve_id,
                cwe_id=_cwe(cve),
                severity=_severity(raw),
                license_note=LICENSE_NOTE,
            )
        )
    return normalize_to_frame(records)


def write_silver(df: pd.DataFrame, output_path: str | Path) -> None:
    write_silver_frame(df, output_path)

