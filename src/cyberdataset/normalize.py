from __future__ import annotations

from typing import Any

import pandas as pd

from cyberdataset.safety import ensure_safe
from cyberdataset.schema import align_columns, normalize_types
from cyberdataset.splitting import assign_split
from cyberdataset.utils import config_path, load_yaml, stable_record_id


def load_label_mapping() -> dict[str, Any]:
    return load_yaml(config_path("label_mapping.yaml"))


def map_label(source_label: str | None) -> dict[str, Any]:
    mapping = load_label_mapping()
    key = (source_label or "unknown").strip().lower()
    return mapping["mappings"].get(key, mapping["default"])


def make_record(
    *,
    source_dataset: str,
    source_type: str,
    main_category: str,
    attack_name: str,
    source_label: str,
    raw_text_or_features: Any,
    source_key: str | int,
    severity: str = "unknown",
    mitre_tactic: str | None = None,
    mitre_technique_id: str | None = None,
    capec_id: str | None = None,
    cwe_id: str | None = None,
    cve_id: str | None = None,
    license_note: str = "Verify upstream terms before redistribution.",
    is_synthetic: bool = False,
) -> dict[str, Any]:
    mapped = map_label(source_label)
    record_id = stable_record_id(source_dataset, source_key)
    safe_text = ensure_safe(raw_text_or_features)
    return {
        "record_id": record_id,
        "source_dataset": source_dataset,
        "source_type": source_type,
        "main_category": main_category,
        "attack_name": attack_name,
        "attack_family": mapped["attack_family"],
        "label": mapped["label"],
        "binary_label": mapped["binary_label"],
        "mitre_tactic": mitre_tactic,
        "mitre_technique_id": mitre_technique_id,
        "capec_id": capec_id,
        "cwe_id": cwe_id,
        "cve_id": cve_id,
        "severity": severity,
        "raw_text_or_features": safe_text,
        "is_synthetic": is_synthetic,
        "is_safe_representation": True,
        "license": license_note,
        "split": assign_split(record_id),
    }


def finalize_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    return normalize_types(align_columns(pd.DataFrame.from_records(records)))

