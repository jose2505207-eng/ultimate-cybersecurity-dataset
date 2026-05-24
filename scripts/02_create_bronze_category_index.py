"""Create a compact category index from the bronze inventory."""

from __future__ import annotations

import argparse

import pandas as pd

from scripts.normalizers.common import PROJECT_ROOT


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    catalog = PROJECT_ROOT / "data" / "bronze_catalog"
    inv = pd.read_csv(catalog / "bronze_inventory.csv")
    idx = inv[["dataset_folder", "source_type", "main_category", "license_string", "license_compatibility"]].copy()
    idx.to_csv(catalog / "bronze_category_index.csv", index=False)


if __name__ == "__main__":
    main()
