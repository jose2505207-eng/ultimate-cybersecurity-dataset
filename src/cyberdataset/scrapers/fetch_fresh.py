"""CLI for the fresh-data scraper.

Example::

    python -m cyberdataset.scrapers.fetch_fresh \\
        --sources cisa_kev,osv,nvd \\
        --out-dir data/bronze_raw/fresh \\
        --cache-dir .cache/fresh_scraper \\
        --limit 1000

Writes raw records to ``<out-dir>/<source_id>/<date>/raw.jsonl`` plus a per-run
``metadata.json``, and a combined ``<out-dir>/manifest.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cyberdataset.scrapers.adapters import available_sources, build_adapters
from cyberdataset.scrapers.base import FreshDataScraper

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "bronze_raw" / "fresh"
DEFAULT_CACHE_DIR = PROJECT_ROOT / ".cache" / "fresh_scraper"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch fresh public cybersecurity data into the bronze layer.",
    )
    parser.add_argument(
        "--sources",
        default="cisa_kev,osv,nvd",
        help=f"Comma-separated source names. Available: {','.join(available_sources())}",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                        help="Bronze output directory for fresh records.")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR),
                        help="On-disk HTTP cache directory.")
    parser.add_argument("--limit", type=int, default=1000,
                        help="Max records to keep per source (None-like 0 means no cap).")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable HTTP response caching for this run.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    source_names = [name for name in args.sources.split(",") if name.strip()]
    try:
        adapters = build_adapters(source_names)
    except KeyError as exc:
        print(f"error: {exc}")
        return 2

    cache_dir = None if args.no_cache else args.cache_dir
    limit = args.limit if args.limit and args.limit > 0 else None

    scraper = FreshDataScraper(adapters, cache_dir=cache_dir)
    print(f"Fetching fresh data from: {', '.join(a.source_id for a in adapters)}")
    manifest = scraper.run(args.out_dir, limit=limit)

    print(json.dumps(
        {
            "total_records": manifest["total_records"],
            "sources": [
                {
                    "source_id": entry["source_id"],
                    "record_count": entry["record_count"],
                    "errors": entry["errors"],
                    "warnings": entry["warnings"],
                }
                for entry in manifest["sources"]
            ],
        },
        indent=2,
    ))
    print(f"Manifest: {Path(args.out_dir) / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
