"""Silver -> Gold unified builder and CLI.

Scans the per-source silver datasets, normalizes every row into the gold
unified canonical schema, deduplicates by stable content hash, scores quality,
assigns deterministic seeded splits, and writes JSONL (always) plus Parquet
(when ``pyarrow`` is available), a manifest, and a dataset card.

CLI::

    python -m cyberdataset.gold.build_gold \\
        --silver-dir data/silver_normalized \\
        --out-dir data/gold \\
        --min-quality 0.50 --seed 42
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cyberdataset.gold.dataset_card import render_dataset_card
from cyberdataset.gold.schema import (
    GOLD_SCHEMA_VERSION,
    GOLD_UNIFIED_COLUMNS,
    UnifiedGoldRecord,
)
from cyberdataset.gold.transform import silver_row_to_record, slugify_source
from cyberdataset.gold.validate import (
    assert_valid_gold_records,
    validate_manifest_consistency,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SILVER_DIR = PROJECT_ROOT / "data" / "silver_normalized"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "gold"

# Files that are not data-bearing silver rows.
_SKIP_NAME_FRAGMENTS = ("manifest", "metadata", "_dedup", "report", "summary")
# Preference order: read the richest single representation per source directory.
_READ_EXTENSIONS = (".parquet", ".csv.gz", ".jsonl", ".csv")


def _pyarrow_available() -> bool:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass
class SilverFile:
    """A single discovered, data-bearing silver file."""

    path: Path
    source_id: str
    source_name: str
    source_license: str


def _matches_extension(path: Path) -> str | None:
    name = path.name.lower()
    for ext in _READ_EXTENSIONS:
        if name.endswith(ext):
            return ext
    return None


def _is_skippable(path: Path) -> bool:
    name = path.name.lower()
    return any(fragment in name for fragment in _SKIP_NAME_FRAGMENTS)


def _read_metadata(source_dir: Path) -> dict[str, Any]:
    for meta in sorted(source_dir.glob("*metadata.json")):
        try:
            return json.loads(meta.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
    return {}


def discover_silver_files(silver_dir: Path) -> list[SilverFile]:
    """Discover one data-bearing file per silver source directory (and top-level files).

    Within a source directory the richest single representation is chosen
    (parquet > csv.gz > jsonl > csv) so rows are never counted twice.
    """
    silver_dir = Path(silver_dir)
    discovered: list[SilverFile] = []
    have_pyarrow = _pyarrow_available()

    def pick_best(files: list[Path]) -> Path | None:
        for ext in _READ_EXTENSIONS:
            if ext == ".parquet" and not have_pyarrow:
                continue
            for f in files:
                if f.name.lower().endswith(ext):
                    return f
        # Last resort: a parquet file even if pyarrow may be missing.
        return files[0] if files else None

    if not silver_dir.exists():
        return discovered

    # Per-source subdirectories.
    for source_dir in sorted(p for p in silver_dir.iterdir() if p.is_dir()):
        if source_dir.name.startswith("_"):
            continue
        candidates = [
            f
            for f in sorted(source_dir.iterdir())
            if f.is_file() and _matches_extension(f) and not _is_skippable(f)
        ]
        chosen = pick_best(candidates)
        if chosen is None:
            continue
        metadata = _read_metadata(source_dir)
        discovered.append(
            SilverFile(
                path=chosen,
                source_id=slugify_source(source_dir.name),
                source_name=metadata.get("source_dataset") or source_dir.name,
                source_license=metadata.get("license") or "unknown",
            )
        )

    # Top-level files placed directly under the silver dir.
    for f in sorted(silver_dir.glob("*")):
        if f.is_file() and _matches_extension(f) and not _is_skippable(f):
            discovered.append(
                SilverFile(
                    path=f,
                    source_id=slugify_source(f.stem.split(".")[0]),
                    source_name=f.stem.split(".")[0],
                    source_license="unknown",
                )
            )

    return discovered


def _read_rows(path: Path, limit_per_source: int | None) -> list[dict[str, Any]]:
    """Read a silver file into a list of row dicts, tolerating missing engines."""
    import pandas as pd

    name = path.name.lower()
    try:
        if name.endswith(".parquet"):
            frame = pd.read_parquet(path)
        elif name.endswith(".jsonl"):
            frame = pd.read_json(path, lines=True)
        elif name.endswith(".csv.gz"):
            frame = pd.read_csv(path, compression="gzip", low_memory=False)
        else:
            frame = pd.read_csv(path, low_memory=False)
    except ImportError as exc:  # e.g. parquet engine missing
        print(f"  ! skipping {path.name}: {exc}")
        return []
    except (ValueError, OSError) as exc:
        print(f"  ! failed to read {path.name}: {exc}")
        return []

    if limit_per_source is not None:
        frame = frame.head(limit_per_source)
    frame = frame.where(frame.notna(), None)
    return frame.to_dict(orient="records")


def build_records(
    silver_files: list[SilverFile],
    *,
    seed: int,
    min_quality: float,
    limit_per_source: int | None,
    processed_at: str,
) -> tuple[list[UnifiedGoldRecord], dict[str, int]]:
    """Transform silver files into deduplicated, quality-filtered gold records."""
    records: list[UnifiedGoldRecord] = []
    seen_hashes: set[str] = set()
    stats = Counter()

    for silver in silver_files:
        rows = _read_rows(silver.path, limit_per_source)
        stats["rows_scanned"] += len(rows)
        for index, row in enumerate(rows):
            record = silver_row_to_record(
                row,
                source_id=silver.source_id,
                source_name=silver.source_name,
                source_license=silver.source_license,
                seed=seed,
                row_index=index,
                processed_at=processed_at,
            )
            if record is None:
                stats["dropped_empty"] += 1
                continue
            if record.quality_score < min_quality:
                stats["dropped_low_quality"] += 1
                continue
            if record.dedup_hash in seen_hashes:
                stats["duplicates_removed"] += 1
                continue
            seen_hashes.add(record.dedup_hash)
            records.append(record)

    return records, dict(stats)


def _counts(records: list[UnifiedGoldRecord], key: str) -> dict[str, int]:
    counter: Counter = Counter(getattr(r, key) for r in records)
    return dict(sorted(counter.items()))


def build_manifest(
    records: list[UnifiedGoldRecord],
    *,
    stats: dict[str, int],
    seed: int,
    min_quality: float,
    silver_dir: Path,
    silver_files: list[SilverFile],
    processed_at: str,
    parquet_written: bool,
) -> dict[str, Any]:
    """Assemble the gold manifest with counts by source/domain/category/label/split."""
    quality_values = [r.quality_score for r in records]
    return {
        "schema_version": GOLD_SCHEMA_VERSION,
        "generated_at": processed_at,
        "seed": seed,
        "min_quality": min_quality,
        "silver_dir": str(silver_dir),
        "sources_scanned": [s.source_id for s in silver_files],
        "rows_scanned": stats.get("rows_scanned", 0),
        "dropped_empty": stats.get("dropped_empty", 0),
        "dropped_low_quality": stats.get("dropped_low_quality", 0),
        "duplicates_removed": stats.get("duplicates_removed", 0),
        "total_records": len(records),
        "parquet_written": parquet_written,
        "mean_quality_score": round(sum(quality_values) / len(quality_values), 4)
        if quality_values
        else 0.0,
        "counts_by_source": _counts(records, "source_id"),
        "counts_by_domain": _counts(records, "domain"),
        "counts_by_category": _counts(records, "category"),
        "counts_by_label": _counts(records, "label"),
        "counts_by_split": _counts(records, "split"),
    }


def _write_jsonl(records: list[UnifiedGoldRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_jsonl_dict(), ensure_ascii=False))
            handle.write("\n")


def _write_parquet(records: list[UnifiedGoldRecord], path: Path) -> bool:
    if not _pyarrow_available():
        return False
    import pandas as pd

    frame = pd.DataFrame([r.to_row_dict() for r in records], columns=GOLD_UNIFIED_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return True


def build_gold(
    *,
    silver_dir: Path | str = DEFAULT_SILVER_DIR,
    out_dir: Path | str = DEFAULT_OUT_DIR,
    min_quality: float = 0.50,
    seed: int = 42,
    limit_per_source: int | None = None,
    write_parquet: bool = True,
    write_card: bool = True,
) -> dict[str, Any]:
    """Run the full Silver -> Gold pipeline and return the manifest dict."""
    silver_dir = Path(silver_dir)
    out_dir = Path(out_dir)
    processed_at = datetime.now(UTC).isoformat()

    silver_files = discover_silver_files(silver_dir)
    print(f"Discovered {len(silver_files)} silver source file(s) under {silver_dir}")

    records, stats = build_records(
        silver_files,
        seed=seed,
        min_quality=min_quality,
        limit_per_source=limit_per_source,
        processed_at=processed_at,
    )
    print(
        f"Built {len(records)} gold records "
        f"({stats.get('duplicates_removed', 0)} duplicates removed, "
        f"{stats.get('dropped_low_quality', 0)} below quality {min_quality})"
    )

    if records:
        assert_valid_gold_records(records)
    else:
        print("  ! no records met the quality threshold; writing an empty gold dataset")

    jsonl_path = out_dir / "gold_unified.jsonl"
    parquet_path = out_dir / "gold_unified.parquet"
    manifest_path = out_dir / "manifest.json"
    card_path = out_dir / "dataset_card.md"

    _write_jsonl(records, jsonl_path)
    parquet_written = _write_parquet(records, parquet_path) if write_parquet else False
    if write_parquet and not parquet_written:
        print("  ! pyarrow not available: wrote JSONL only (skipped Parquet)")

    manifest = build_manifest(
        records,
        stats=stats,
        seed=seed,
        min_quality=min_quality,
        silver_dir=silver_dir,
        silver_files=silver_files,
        processed_at=processed_at,
        parquet_written=parquet_written,
    )

    consistency_issues = validate_manifest_consistency(manifest, output_row_count=len(records))
    if consistency_issues:
        raise ValueError(f"Manifest inconsistency: {consistency_issues}")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    if write_card:
        card_path.write_text(render_dataset_card(manifest), encoding="utf-8")

    print(f"Wrote: {jsonl_path}")
    if parquet_written:
        print(f"Wrote: {parquet_path}")
    print(f"Wrote: {manifest_path}")
    if write_card:
        print(f"Wrote: {card_path}")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the gold unified cybersecurity dataset from silver outputs.",
    )
    parser.add_argument("--silver-dir", default=str(DEFAULT_SILVER_DIR),
                        help="Directory containing per-source silver datasets.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                        help="Directory to write gold outputs into.")
    parser.add_argument("--min-quality", type=float, default=0.50,
                        help="Drop records below this quality score (0..1).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for deterministic train/val/test splits.")
    parser.add_argument("--limit-per-source", type=int, default=None,
                        help="Optional cap on rows read per silver file (bounds memory).")
    parser.add_argument("--no-parquet", action="store_true",
                        help="Skip Parquet output even if pyarrow is available.")
    parser.add_argument("--no-card", action="store_true",
                        help="Skip writing dataset_card.md.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    build_gold(
        silver_dir=args.silver_dir,
        out_dir=args.out_dir,
        min_quality=args.min_quality,
        seed=args.seed,
        limit_per_source=args.limit_per_source,
        write_parquet=not args.no_parquet,
        write_card=not args.no_card,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
