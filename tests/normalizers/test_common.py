from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts.normalizers.common import (
    DuplicateRecordIdError,
    ensure_unified_schema,
    extract_domain_from_url,
    make_record_id,
    normalize_binary_label,
    normalize_severity,
    normalize_timestamp,
    safe_json_dumps,
    validate_against_schema,
    write_silver,
)
from scripts.normalizers.schema import COLUMN_ORDER, SCHEMA_VERSION, SilverRecord, assert_schema_sync


def test_schema_model_and_column_order_stay_in_sync():
    assert_schema_sync()
    assert list(SilverRecord.model_fields.keys()) == COLUMN_ORDER


def test_make_record_id_is_deterministic():
    assert make_record_id("x", "abc") == make_record_id("x", "abc")
    assert make_record_id("x", "abc").startswith("x::")


def test_safe_json_dumps_validates_cap():
    assert json.loads(safe_json_dumps({"b": 1, "a": 2})) == {"a": 2, "b": 1}
    assert safe_json_dumps({"x": "y" * 100}, max_bytes=10) is None


def test_binary_label_normalization_rejects_ambiguous():
    assert normalize_binary_label("phishing") == 1
    assert normalize_binary_label("benign") == 0
    with pytest.raises(ValueError):
        normalize_binary_label("maybe")


def test_normalize_severity_score_precedence():
    assert normalize_severity(cvss_v2=9.0, cvss_v3=5.0, cvss_v4=1.0) == ("low", 1.0)
    assert normalize_severity(vendor_severity="moderate") == ("medium", None)


def test_extract_domain_from_url_local_only():
    assert extract_domain_from_url("https://Example.com/a") == "example.com"
    assert extract_domain_from_url("example.org/path") == "example.org"
    assert extract_domain_from_url("http://[broken") is None


def test_ensure_and_validate_unified_schema():
    df = pd.DataFrame(
        [
            {
                "record_id": make_record_id("unit", "1"),
                "label": "benign",
                "binary_label": 0,
                "source_file": "fixture.csv",
            }
        ]
    )
    out = ensure_unified_schema(df, "unit", "other", "Miscellaneous / Needs Review", "UNKNOWN")
    assert list(out.columns) == COLUMN_ORDER
    assert out.loc[0, "schema_version"] == SCHEMA_VERSION
    validate_against_schema(out)


def test_duplicate_record_ids_are_rejected():
    rid = make_record_id("unit", "1")
    df = pd.DataFrame(
        [
            {"record_id": rid, "label": "benign", "binary_label": 0, "source_file": "a.csv"},
            {"record_id": rid, "label": "benign", "binary_label": 0, "source_file": "b.csv"},
        ]
    )
    out = ensure_unified_schema(df, "unit", "other", "Miscellaneous / Needs Review", "UNKNOWN")
    with pytest.raises(DuplicateRecordIdError):
        validate_against_schema(out)


def test_timestamp_normalization_is_utc():
    ts = normalize_timestamp("2026-05-24T01:02:03")
    assert ts is not None
    assert str(ts.tz) == "UTC"


def test_write_silver_is_atomic_and_reports_bytes(tmp_path):
    df = pd.DataFrame(
        [
            {
                "record_id": make_record_id("unit", "1"),
                "label": "benign",
                "binary_label": 0,
                "source_file": "fixture.csv",
            }
        ]
    )
    out = ensure_unified_schema(df, "unit", "other", "Miscellaneous / Needs Review", "UNKNOWN")
    meta = write_silver(out, tmp_path / "unit")
    assert (tmp_path / "unit.parquet").exists()
    assert (tmp_path / "unit.csv.gz").exists()
    assert meta["parquet_bytes"] > 0
    assert meta["csv_gz_bytes"] > 0


def test_runner_dry_run_does_not_write_metadata(tmp_path):
    output = tmp_path / "silver"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.normalizers.ai_security_hackaprompt",
            "--input",
            str(Path("data/bronze_raw/ai_security_prompt_injection").resolve()),
            "--output",
            str(output),
            "--dry-run",
            "--max-rows",
            "10",
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=True,
    )
    assert '"would_status": "blocked"' in proc.stdout
    assert not list(output.rglob("*_metadata.json"))
