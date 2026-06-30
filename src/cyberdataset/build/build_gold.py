from __future__ import annotations

from pathlib import Path

import pandas as pd

from cyberdataset.reporting import coverage_report
from cyberdataset.schema import empty_frame, normalize_types, validate_schema
from cyberdataset.sampling import quota_sample
from cyberdataset.utils import DATA_DIR, dump_json, read_table, write_table


def silver_files() -> list[Path]:
    silver_dir = DATA_DIR / "silver_normalized"
    return sorted(
        [
            *silver_dir.glob("*.csv"),
            *silver_dir.glob("*.jsonl"),
            *silver_dir.glob("*.parquet"),
            *silver_dir.glob("*/*.csv"),
            *silver_dir.glob("*/*.csv.gz"),
            *silver_dir.glob("*/*.jsonl"),
            *silver_dir.glob("*/*.parquet"),
        ]
    )


def build_gold(max_rows: int = 100000) -> pd.DataFrame:
    files = silver_files()
    if not files:
        df = empty_frame()
    else:
        df = pd.concat([read_table(path) for path in files], ignore_index=True)
        df = normalize_types(df)
        df = quota_sample(df, max_rows=max_rows)
    validate_schema(df)
    gold_csv = DATA_DIR / "gold_unified" / "ultimate_cybersecurity_dataset.csv"
    gold_parquet = DATA_DIR / "gold_unified" / "ultimate_cybersecurity_dataset.parquet"
    write_table(df, gold_csv)
    write_table(df, gold_parquet)
    dump_json(coverage_report(df), DATA_DIR / "reports" / "gold_coverage_report.json")
    print(f"Wrote {len(df)} rows to {gold_csv} and {gold_parquet}")
    return df


def main() -> None:
    build_gold()


if __name__ == "__main__":
    main()
