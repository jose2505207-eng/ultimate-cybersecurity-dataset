from __future__ import annotations

import argparse
from pathlib import Path

from cyberdataset.schema import validate_schema
from cyberdataset.utils import read_table


def validate_file(path: str | Path) -> None:
    df = read_table(path)
    validate_schema(df)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a normalized cybersecurity dataset file.")
    parser.add_argument("--input", required=True, help="CSV, JSONL, JSON, or parquet file to validate.")
    args = parser.parse_args()
    validate_file(args.input)
    print(f"Validated {args.input}")


if __name__ == "__main__":
    main()

