from __future__ import annotations

from typing import Any

import pandas as pd


def coverage_report(df: pd.DataFrame) -> dict[str, Any]:
    missing = {column: int(df[column].isna().sum()) for column in df.columns}
    return {
        "rows": int(len(df)),
        "source_dataset_counts": df["source_dataset"].value_counts(dropna=False).to_dict(),
        "source_type_counts": df["source_type"].value_counts(dropna=False).to_dict(),
        "main_category_counts": df["main_category"].value_counts(dropna=False).to_dict(),
        "label_counts": df["label"].value_counts(dropna=False).to_dict(),
        "split_counts": df["split"].value_counts(dropna=False).to_dict(),
        "missing_values": missing,
    }

