from __future__ import annotations

from pathlib import Path

import pandas as pd

from cyberdataset.ingest.tabular import compact_features, first_present, load_tables, normalize_to_frame, write_silver_frame
from cyberdataset.normalize import make_record


SOURCE_DATASET = "UNSW_NB15"
LICENSE_NOTE = "Verify UNSW-NB15 upstream terms before redistribution."


def load_raw(input_path: str | Path, limit: int | None = None) -> pd.DataFrame:
    return load_tables(input_path, limit=limit)


def _category(attack_cat: str) -> tuple[str, str | None]:
    lowered = attack_cat.lower()
    if attack_cat.lower() in {"normal", "benign"}:
        return "network_intrusion", None
    if "dos" in lowered:
        return "denial_of_service", "Impact"
    if "backdoor" in lowered or "shellcode" in lowered or "worms" in lowered:
        return "malware", "Execution"
    if "recon" in lowered or "analysis" in lowered:
        return "network_intrusion", "Discovery"
    if "fuzzers" in lowered or "exploits" in lowered:
        return "network_intrusion", "Initial Access"
    return "network_intrusion", "Initial Access"


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for idx, row in df.iterrows():
        attack_cat = str(first_present(row, ["attack_cat", "attack category", "category"], "unknown")).strip()
        label_value = first_present(row, ["label", "Label"], None)
        source_label = "benign" if str(label_value) in {"0", "0.0"} or attack_cat.lower() in {"normal", "benign"} else "malicious"
        attack_name = "Benign Flow" if source_label == "benign" else attack_cat
        main_category, mitre_tactic = _category(attack_name)
        records.append(
            make_record(
                source_dataset=SOURCE_DATASET,
                source_type="network_flow",
                main_category=main_category,
                attack_name=attack_name,
                source_label=source_label,
                raw_text_or_features={
                    "source_file": row.get("__source_file"),
                    "source_row": int(idx),
                    "source_label": attack_cat,
                    "features": compact_features(row, exclude={"label", "Label", "attack_cat", "attack category", "category"}),
                },
                source_key=f"{row.get('__source_file', 'file')}:{idx}",
                mitre_tactic=mitre_tactic,
                license_note=LICENSE_NOTE,
            )
        )
    return normalize_to_frame(records)


def write_silver(df: pd.DataFrame, output_path: str | Path) -> None:
    write_silver_frame(df, output_path)

