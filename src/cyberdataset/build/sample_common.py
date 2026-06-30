from __future__ import annotations

from cyberdataset.sampling import quota_sample
from cyberdataset.schema import validate_schema
from cyberdataset.utils import DATA_DIR, read_table, write_table


def build_sample(size: int) -> None:
    source = DATA_DIR / "gold_unified" / "ultimate_cybersecurity_dataset.csv"
    df = read_table(source)
    sample = quota_sample(df, max_rows=size)
    validate_schema(sample)
    output = DATA_DIR / "gold_unified" / f"ultimate_cybersecurity_dataset_{size}.csv"
    write_table(sample, output)
    print(f"Wrote {len(sample)} rows to {output}")

