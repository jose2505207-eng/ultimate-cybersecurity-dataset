"""Quality checks and validation for the gold unified layer.

These checks are the contract the builder must satisfy. They run automatically
at the end of :func:`cyberdataset.gold.build_gold.build_gold` and are also used
by the test-suite against tiny fixtures.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from cyberdataset.gold.schema import (
    DOMAINS,
    GOLD_UNIFIED_COLUMNS,
    VALID_SPLITS,
    GoldValidationError,
    UnifiedGoldRecord,
)

#: Fields that must always be present and non-empty on every gold record.
REQUIRED_FIELDS: tuple[str, ...] = (
    "record_id",
    "source_id",
    "domain",
    "category",
    "task_type",
    "split",
    "dedup_hash",
)


def _as_dict(record: UnifiedGoldRecord | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(record, UnifiedGoldRecord):
        return record.to_jsonl_dict()
    return dict(record)


def validate_gold_records(
    records: Iterable[UnifiedGoldRecord | Mapping[str, Any]],
) -> list[str]:
    """Validate a collection of gold records and return a list of issues.

    The list is empty when the dataset is valid. The function never raises for
    data problems (use :func:`assert_valid_gold_records` for that); it raises
    only for a structurally impossible input.
    """
    issues: list[str] = []
    seen_record_ids: set[str] = set()
    seen_dedup_hashes: set[str] = set()
    row_count = 0

    for index, raw in enumerate(records):
        row = _as_dict(raw)
        row_count += 1

        missing_columns = [c for c in GOLD_UNIFIED_COLUMNS if c not in row]
        if missing_columns:
            issues.append(f"row {index}: missing columns {missing_columns}")
            continue

        for field_name in REQUIRED_FIELDS:
            value = row.get(field_name)
            if value is None or (isinstance(value, str) and not value.strip()):
                issues.append(f"row {index}: required field '{field_name}' is empty")

        raw_text = (row.get("raw_text") or "").strip() if isinstance(row.get("raw_text"), str) else row.get("raw_text")
        norm_text = (row.get("normalized_text") or "").strip() if isinstance(row.get("normalized_text"), str) else row.get("normalized_text")
        if not raw_text and not norm_text:
            issues.append(f"row {index}: both raw_text and normalized_text are empty")

        domain = row.get("domain")
        if domain not in DOMAINS:
            issues.append(f"row {index}: invalid domain {domain!r}")

        if not (row.get("category") or "").strip():
            issues.append(f"row {index}: empty category")

        split = row.get("split")
        if split not in VALID_SPLITS:
            issues.append(f"row {index}: invalid split {split!r}")

        quality = row.get("quality_score")
        if not isinstance(quality, (int, float)) or not (0.0 <= float(quality) <= 1.0):
            issues.append(f"row {index}: quality_score out of range: {quality!r}")

        record_id = row.get("record_id")
        if record_id in seen_record_ids:
            issues.append(f"row {index}: duplicate record_id {record_id!r}")
        else:
            seen_record_ids.add(record_id)

        dedup_hash = row.get("dedup_hash")
        if dedup_hash in seen_dedup_hashes:
            issues.append(f"row {index}: duplicate dedup_hash {dedup_hash!r}")
        elif dedup_hash:
            seen_dedup_hashes.add(dedup_hash)

    if row_count == 0:
        issues.append("dataset is empty: no gold records produced")

    return issues


def assert_valid_gold_records(
    records: Iterable[UnifiedGoldRecord | Mapping[str, Any]],
    *,
    max_reported: int = 20,
) -> None:
    """Raise :class:`GoldValidationError` if any validation issue is found."""
    issues = validate_gold_records(records)
    if issues:
        shown = "; ".join(issues[:max_reported])
        more = "" if len(issues) <= max_reported else f" (+{len(issues) - max_reported} more)"
        raise GoldValidationError(f"Gold validation failed: {shown}{more}")


def validate_manifest_consistency(
    manifest: Mapping[str, Any],
    *,
    output_row_count: int,
) -> list[str]:
    """Confirm manifest totals agree with the number of rows written."""
    issues: list[str] = []
    total = manifest.get("total_records")
    if total != output_row_count:
        issues.append(
            f"manifest total_records ({total}) != output row count ({output_row_count})"
        )
    for group in ("counts_by_domain", "counts_by_split", "counts_by_label", "counts_by_source"):
        counts = manifest.get(group) or {}
        summed = sum(counts.values())
        if summed != output_row_count:
            issues.append(
                f"manifest {group} sums to {summed}, expected {output_row_count}"
            )
    return issues
