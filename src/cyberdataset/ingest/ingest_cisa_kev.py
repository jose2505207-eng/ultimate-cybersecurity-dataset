from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from cyberdataset.ingest.tabular import first_present, load_tables, normalize_to_frame, write_silver_frame
from cyberdataset.normalize import make_record
from cyberdataset.utils import read_table


SOURCE_DATASET = "CISA_KEV"
LICENSE_NOTE = "CISA Known Exploited Vulnerabilities catalog terms apply."


def _load_json_catalog(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("vulnerabilities", data if isinstance(data, list) else [data])
    out = pd.DataFrame(rows)
    out["__source_file"] = path.name
    return out


def load_raw(input_path: str | Path, limit: int | None = None) -> pd.DataFrame:
    path = Path(input_path)
    if path.is_file() and path.suffix.lower() == ".json":
        df = _load_json_catalog(path)
    elif path.is_dir():
        frames = []
        for file in sorted(path.rglob("*")):
            if not file.is_file():
                continue
            if file.suffix.lower() == ".json":
                frames.append(_load_json_catalog(file))
            elif file.suffix.lower() in {".csv", ".jsonl", ".parquet"}:
                df_part = read_table(file)
                df_part["__source_file"] = file.name
                frames.append(df_part)
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        df = load_tables(input_path, limit=limit)
    return df.head(limit) if limit is not None else df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for idx, row in df.iterrows():
        cve_id = first_present(row, ["cveID", "cve_id", "cve"], idx)
        vendor = first_present(row, ["vendorProject", "vendor"], None)
        product = first_present(row, ["product"], None)
        name = first_present(row, ["vulnerabilityName", "name"], "Known Exploited Vulnerability")
        description = first_present(row, ["shortDescription", "description"], "")
        records.append(
            make_record(
                source_dataset=SOURCE_DATASET,
                source_type="advisory",
                main_category="vulnerability_advisory",
                attack_name="Known Exploited Vulnerability",
                source_label="advisory",
                raw_text_or_features={
                    "source_file": row.get("__source_file"),
                    "source_row": int(idx),
                    "cve_id": cve_id,
                    "vendor": vendor,
                    "product": product,
                    "name": name,
                    "description": description,
                    "date_added": first_present(row, ["dateAdded"], None),
                    "due_date": first_present(row, ["dueDate"], None),
                },
                source_key=cve_id,
                cve_id=str(cve_id),
                severity="critical",
                license_note=LICENSE_NOTE,
            )
        )
    return normalize_to_frame(records)


def write_silver(df: pd.DataFrame, output_path: str | Path) -> None:
    write_silver_frame(df, output_path)

