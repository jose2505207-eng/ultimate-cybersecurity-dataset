from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from cyberdataset.utils import config_path, load_yaml


CANONICAL_COLUMNS = [
    "record_id",
    "source_dataset",
    "source_type",
    "main_category",
    "attack_name",
    "attack_family",
    "label",
    "binary_label",
    "mitre_tactic",
    "mitre_technique_id",
    "capec_id",
    "cwe_id",
    "cve_id",
    "severity",
    "raw_text_or_features",
    "is_synthetic",
    "is_safe_representation",
    "license",
    "split",
]


class DatasetValidationError(ValueError):
    """Raised when a normalized dataset violates the canonical contract."""


@dataclass(frozen=True)
class SchemaConfig:
    columns: dict[str, dict[str, Any]]

    @classmethod
    def load(cls) -> "SchemaConfig":
        return cls(columns=load_yaml(config_path("schema.yaml"))["columns"])


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=CANONICAL_COLUMNS)


def align_columns(df: pd.DataFrame) -> pd.DataFrame:
    aligned = df.copy()
    for column in CANONICAL_COLUMNS:
        if column not in aligned.columns:
            aligned[column] = pd.NA
    return aligned[CANONICAL_COLUMNS]


def _missing_required(df: pd.DataFrame, config: SchemaConfig) -> list[str]:
    missing: list[str] = []
    for column, spec in config.columns.items():
        if spec.get("nullable", True):
            continue
        if column not in df.columns or df[column].isna().any():
            missing.append(column)
    return missing


def validate_schema(df: pd.DataFrame, *, require_safe: bool = True) -> None:
    config = SchemaConfig.load()
    missing_columns = [column for column in CANONICAL_COLUMNS if column not in df.columns]
    if missing_columns:
        raise DatasetValidationError(f"Missing columns: {missing_columns}")

    required = _missing_required(df, config)
    if required:
        raise DatasetValidationError(f"Required columns contain nulls or are absent: {required}")

    if df["record_id"].duplicated().any():
        dupes = df.loc[df["record_id"].duplicated(), "record_id"].head(5).tolist()
        raise DatasetValidationError(f"Duplicate record_id values: {dupes}")

    for column, spec in config.columns.items():
        allowed = spec.get("values")
        if not allowed:
            continue
        values = set(df[column].dropna().tolist())
        invalid = sorted(values - set(allowed))
        if invalid:
            raise DatasetValidationError(f"Invalid values for {column}: {invalid[:10]}")

    binary_values = set(df["binary_label"].dropna().astype(int).tolist())
    if not binary_values.issubset({0, 1}):
        raise DatasetValidationError(f"Invalid binary_label values: {sorted(binary_values)}")

    if require_safe and not df["is_safe_representation"].astype(bool).all():
        raise DatasetValidationError("All gold/silver rows must be marked as safe representations.")


def normalize_types(df: pd.DataFrame) -> pd.DataFrame:
    out = align_columns(df)
    out["is_synthetic"] = out["is_synthetic"].astype(bool)
    out["is_safe_representation"] = out["is_safe_representation"].astype(bool)
    out["binary_label"] = out["binary_label"].astype("Int64")
    return out

