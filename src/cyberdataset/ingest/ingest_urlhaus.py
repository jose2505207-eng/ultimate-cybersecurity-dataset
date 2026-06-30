from __future__ import annotations

from pathlib import Path

import pandas as pd

from cyberdataset.ingest.tabular import compact_features, first_present, load_tables, normalize_to_frame, write_silver_frame
from cyberdataset.normalize import make_record


SOURCE_DATASET = "URLhaus"
LICENSE_NOTE = "Verify abuse.ch URLhaus upstream terms before redistribution."


def load_raw(input_path: str | Path, limit: int | None = None) -> pd.DataFrame:
    return load_tables(input_path, limit=limit)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for idx, row in df.iterrows():
        url = first_present(row, ["url"], "")
        threat = str(first_present(row, ["threat", "tags"], "Malicious URL")).strip()
        source_id = first_present(row, ["id", "url_id"], idx)
        records.append(
            make_record(
                source_dataset=SOURCE_DATASET,
                source_type="url",
                main_category="malware",
                attack_name=threat or "Malicious URL",
                source_label="malware",
                raw_text_or_features={
                    "source_file": row.get("__source_file"),
                    "source_row": int(idx),
                    "url": str(url),
                    "features": compact_features(row, exclude={"url"}),
                },
                source_key=source_id,
                mitre_tactic="Command and Control",
                license_note=LICENSE_NOTE,
            )
        )
    return normalize_to_frame(records)


def write_silver(df: pd.DataFrame, output_path: str | Path) -> None:
    write_silver_frame(df, output_path)

