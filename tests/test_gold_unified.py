"""Tests for the gold unified layer: schema, determinism, dedup, splits, manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cyberdataset.gold.build_gold import build_gold, discover_silver_files
from cyberdataset.gold.schema import GOLD_UNIFIED_COLUMNS, VALID_SPLITS
from cyberdataset.gold.transform import (
    assign_split,
    compute_dedup_hash,
    infer_domain,
    silver_row_to_record,
)
from cyberdataset.gold.validate import (
    assert_valid_gold_records,
    validate_gold_records,
    validate_manifest_consistency,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_SILVER_DIR = PROJECT_ROOT / "examples" / "silver_sample"


def _sample_row(**overrides):
    row = {
        "record_id": "nvd_cve::0001",
        "source_dataset": "nvd_cve",
        "source_type": "vulnerability_advisory",
        "main_category": "Threat Intelligence, CVE, Advisory & Taxonomy",
        "attack_name": "Known Exploited Vulnerability",
        "label": "advisory",
        "cwe_id": "CWE-79",
        "cve_id": "CVE-2024-0001",
        "severity": "high",
        "raw_text": "Cross-site scripting in the sample console allows script injection.",
        "license": "NVD-Public-Domain",
    }
    row.update(overrides)
    return row


# --------------------------------------------------------------------------- #
# Schema / record construction                                                #
# --------------------------------------------------------------------------- #


def test_record_has_exact_canonical_schema():
    record = silver_row_to_record(
        _sample_row(), source_id="nvd_cve", source_name="NVD CVE"
    )
    assert record is not None
    assert list(record.to_jsonl_dict().keys()) == GOLD_UNIFIED_COLUMNS


def test_empty_text_row_is_dropped():
    record = silver_row_to_record(
        _sample_row(raw_text="", features_json="{}"),
        source_id="nvd_cve",
        source_name="NVD CVE",
    )
    assert record is None


def test_domain_inference():
    assert infer_domain("nvd_cve", "vulnerability_advisory") == "vulnerabilities_exposures"
    assert infer_domain("phishing_email_curated", "email") == "phishing_social"
    assert infer_domain("giskard_prompt_injections", "prompt") == "prompt_injection_jailbreak"
    assert infer_domain("unsw_nb15", "network_flow") == "network_intrusion"
    assert infer_domain("totally_unknown_thing", "mystery") == "miscellaneous"


def test_entities_and_identifiers_extracted():
    record = silver_row_to_record(
        _sample_row(), source_id="nvd_cve", source_name="NVD CVE"
    )
    assert record.cve == "CVE-2024-0001"
    assert "CVE-2024-0001" in record.entities["cve_ids"]
    assert "CWE-79" in record.entities["cwe_ids"]


# --------------------------------------------------------------------------- #
# Determinism                                                                 #
# --------------------------------------------------------------------------- #


def test_record_id_is_deterministic():
    a = silver_row_to_record(_sample_row(), source_id="nvd_cve", source_name="x")
    b = silver_row_to_record(_sample_row(), source_id="nvd_cve", source_name="x")
    assert a.record_id == b.record_id


def test_split_assignment_is_deterministic_and_valid():
    assert assign_split("nvd_cve::abc") == assign_split("nvd_cve::abc")
    for rid in (f"s::{i}" for i in range(50)):
        assert assign_split(rid) in VALID_SPLITS


def test_split_changes_with_seed():
    rids = [f"s::{i}" for i in range(200)]
    seed1 = {rid: assign_split(rid, seed=1) for rid in rids}
    seed2 = {rid: assign_split(rid, seed=2) for rid in rids}
    assert seed1 != seed2  # seeds must influence the partition


# --------------------------------------------------------------------------- #
# Deduplication                                                               #
# --------------------------------------------------------------------------- #


def test_identical_text_yields_identical_dedup_hash():
    text = "Cross-site scripting in the sample console."
    assert compute_dedup_hash(text) == compute_dedup_hash("  cross-site   SCRIPTING in the sample console.  ")


# --------------------------------------------------------------------------- #
# End-to-end build against the example fixtures                               #
# --------------------------------------------------------------------------- #


@pytest.fixture
def built(tmp_path):
    manifest = build_gold(
        silver_dir=SAMPLE_SILVER_DIR,
        out_dir=tmp_path,
        min_quality=0.40,
        seed=42,
        write_parquet=False,
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "gold_unified.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    return manifest, rows, tmp_path


def test_discover_finds_all_sample_sources():
    discovered = discover_silver_files(SAMPLE_SILVER_DIR)
    ids = {s.source_id for s in discovered}
    assert {"advisory_nvd_cve", "phishing_social", "ai_security_prompt_injection",
            "network_unsw_nb15", "supply_chain_osv"}.issubset(ids)


def test_build_produces_valid_dataset(built):
    _manifest, rows, _ = built
    assert rows
    assert validate_gold_records(rows) == []
    assert_valid_gold_records(rows)


def test_build_deduplicates_cross_source(built):
    manifest, rows, _ = built
    # The XSS advisory appears in both nvd and osv fixtures -> exactly one survives.
    hashes = [r["dedup_hash"] for r in rows]
    assert len(hashes) == len(set(hashes))
    assert manifest["duplicates_removed"] >= 1


def test_manifest_counts_are_consistent(built):
    manifest, rows, _ = built
    assert manifest["total_records"] == len(rows)
    assert validate_manifest_consistency(manifest, output_row_count=len(rows)) == []


def test_manifest_and_card_written(built):
    _manifest, _rows, out_dir = built
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "dataset_card.md").exists()
    card = (out_dir / "dataset_card.md").read_text(encoding="utf-8")
    assert "Gold Unified Cybersecurity Dataset" in card


def test_min_quality_filter_reduces_rows(tmp_path):
    low = build_gold(silver_dir=SAMPLE_SILVER_DIR, out_dir=tmp_path / "low",
                     min_quality=0.0, seed=42, write_parquet=False)
    high = build_gold(silver_dir=SAMPLE_SILVER_DIR, out_dir=tmp_path / "high",
                      min_quality=0.95, seed=42, write_parquet=False)
    assert high["total_records"] <= low["total_records"]


def test_invalid_record_is_reported():
    record = silver_row_to_record(_sample_row(), source_id="nvd_cve", source_name="x")
    bad = record.to_jsonl_dict()
    bad["domain"] = "not_a_domain"
    issues = validate_gold_records([bad])
    assert any("invalid domain" in issue for issue in issues)
