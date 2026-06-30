from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from cyberdataset.normalize import finalize_records
from cyberdataset.schema import validate_schema
from cyberdataset.utils import read_table, write_table


DATA_EXTENSIONS = {".csv", ".json", ".jsonl", ".parquet"}


def canonical_name(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def data_files(input_path: str | Path) -> list[Path]:
    path = Path(input_path)
    if path.is_file():
        return [path]
    return sorted(file for file in path.rglob("*") if file.is_file() and file.suffix.lower() in DATA_EXTENSIONS)


def load_tables(input_path: str | Path, limit: int | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    remaining = limit
    for path in data_files(input_path):
        df = read_table(path)
        if remaining is not None:
            df = df.head(remaining)
            remaining -= len(df)
        df["__source_file"] = path.name
        frames.append(df)
        if remaining is not None and remaining <= 0:
            break
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def first_present(row: pd.Series, names: Iterable[str], default: Any = None) -> Any:
    canonical_to_original = {canonical_name(column): column for column in row.index}
    for name in names:
        original = canonical_to_original.get(canonical_name(name))
        if original is not None and pd.notna(row[original]):
            return row[original]
    return default


def compact_features(row: pd.Series, *, exclude: set[str], max_fields: int = 80) -> dict[str, Any]:
    features: dict[str, Any] = {}
    excluded = {canonical_name(column) for column in exclude}
    for column, value in row.items():
        if canonical_name(column) in excluded or column.startswith("__"):
            continue
        if pd.isna(value):
            continue
        if len(features) >= max_fields:
            break
        if hasattr(value, "item"):
            value = value.item()
        features[str(column).strip()] = value
    return features


def normalize_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    df = finalize_records(records)
    validate_schema(df)
    return df


def write_silver_frame(df: pd.DataFrame, output_path: str | Path) -> None:
    validate_schema(df)
    write_table(df, output_path)


def safe_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)

