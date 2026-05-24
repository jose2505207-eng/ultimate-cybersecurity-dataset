"""Report cross-source CVE overlap without deleting duplicates."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.normalizers.common import PROJECT_ROOT


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    silver = PROJECT_ROOT / "data" / "silver_normalized"
    frames = []
    for module in ("advisory_nvd_cve", "supply_chain_osv", "supply_chain_github_advisory"):
        path = silver / module / f"{module}.parquet"
        if path.exists():
            df = pd.read_parquet(path, columns=["record_id", "cve_id"])
            df["source"] = module
            frames.append(df.dropna(subset=["cve_id"]))
    out_dir = silver / "_dedup"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not frames:
        pd.DataFrame(columns=["cve_id", "sources", "record_ids"]).to_csv(out_dir / "cross_source_cve_overlap.csv", index=False)
        return
    all_df = pd.concat(frames, ignore_index=True)
    grouped = all_df.groupby("cve_id").agg(sources=("source", lambda s: "|".join(sorted(set(s)))), record_ids=("record_id", lambda s: "|".join(sorted(set(s))))).reset_index()
    grouped[grouped["sources"].str.contains(r"\|", regex=True)].to_csv(out_dir / "cross_source_cve_overlap.csv", index=False)


if __name__ == "__main__":
    main()
