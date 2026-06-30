from __future__ import annotations

import hashlib

import pandas as pd


VALID_SPLITS = {"train", "validation", "test"}


def assign_split(record_id: str, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> str:
    train_ratio, validation_ratio, _ = ratios
    digest = hashlib.sha256(record_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + validation_ratio:
        return "validation"
    return "test"


def add_splits(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["split"] = out["record_id"].map(assign_split)
    return out

