from __future__ import annotations

import pandas as pd

from cyberdataset.schema import normalize_types, validate_schema


def clean_normalized(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_types(df)
    validate_schema(out)
    return out

