from __future__ import annotations

from pathlib import Path

import pandas as pd

from cyberdataset.ingest.tabular import compact_features, first_present, load_tables, normalize_to_frame, write_silver_frame
from cyberdataset.normalize import make_record


SOURCE_DATASET = "CICIDS2017"
LICENSE_NOTE = "Verify CICIDS2017 upstream terms before redistribution."


def load_raw(input_path: str | Path, limit: int | None = None) -> pd.DataFrame:
    return load_tables(input_path, limit=limit)


def _attack_metadata(label: str) -> tuple[str, str, str, str | None]:
    normalized = label.strip()
    lowered = normalized.lower()
    if lowered in {"benign", "normal"}:
        return "benign", "Benign Flow", "network_intrusion", None
    if "ddos" in lowered or lowered.startswith("dos "):
        return "malicious", normalized, "denial_of_service", "Impact"
    if "web attack" in lowered or "xss" in lowered or "sql" in lowered:
        return "malicious", normalized, "web_application_attack", "Initial Access"
    if "patator" in lowered or "brute" in lowered:
        return "malicious", normalized, "credential_attack", "Credential Access"
    return "malicious", normalized, "network_intrusion", "Initial Access"


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for idx, row in df.iterrows():
        label = str(first_present(row, ["Label", "label", "attack", "class"], "unknown")).strip()
        source_label, attack_name, main_category, mitre_tactic = _attack_metadata(label)
        features = compact_features(row, exclude={"Label", "label", "attack", "class"})
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
                    "source_label": label,
                    "features": features,
                },
                source_key=f"{row.get('__source_file', 'file')}:{idx}",
                mitre_tactic=mitre_tactic,
                license_note=LICENSE_NOTE,
            )
        )
    return normalize_to_frame(records)


def write_silver(df: pd.DataFrame, output_path: str | Path) -> None:
    write_silver_frame(df, output_path)

