from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_manifest_has_required_status_and_license_columns():
    manifest = Path("data/silver_normalized/silver_manifest.csv")
    assert manifest.exists()
    df = pd.read_csv(manifest)
    for column in ("silver_module", "status", "row_count", "license", "license_compatibility", "notes"):
        assert column in df.columns
    assert not ((df["row_count"] == 0) & (df["status"] == "ok")).any()
    for module in df.loc[df["status"] != "ok", "silver_module"]:
        assert not (Path("data/silver_normalized") / module / f"{module}.parquet").exists()
        assert not (Path("data/silver_normalized") / module / f"{module}.csv.gz").exists()


def test_report_contains_required_hardening_sections():
    report = Path("docs/silver_layer_report.md")
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    for heading in (
        "## Succeeded Modules",
        "## Skipped or Blocked Modules",
        "## Rows by Module",
        "## License Summary",
        "## Cross-Source CVE Overlap",
        "## Failures, Skips, and Blockers",
    ):
        assert heading in text
    assert "EXCLUDED FROM PUBLIC RELEASE" in text
