from __future__ import annotations

from pathlib import Path

import pandas as pd

from cyberdataset.schema import validate_schema
from cyberdataset.utils import write_table


MANUAL_IMPORT_MESSAGE = (
    "This v1 scaffold does not download or parse this source automatically. "
    "Place licensed raw exports under data/bronze_raw/ and implement source-specific parsing here."
)


def load_raw(input_path: str | Path, limit: int | None = None) -> pd.DataFrame:
    raise NotImplementedError(MANUAL_IMPORT_MESSAGE)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError(MANUAL_IMPORT_MESSAGE)


def write_silver(df: pd.DataFrame, output_path: str | Path) -> None:
    validate_schema(df)
    write_table(df, output_path)

