"""Build silver manifest and markdown report."""

from __future__ import annotations

import argparse
import json
from datetime import UTC
from pathlib import Path

import pandas as pd

from scripts.normalizers.common import PROJECT_ROOT, license_compatibility


def markdown_table(df: pd.DataFrame) -> str:
    """Render a small dataframe as a Markdown table without optional deps."""

    if df.empty:
        return "None"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for rec in df.astype(str).to_dict("records"):
        lines.append("| " + " | ".join(rec[c].replace("|", "\\|") for c in cols) + " |")
    return "\n".join(lines)


def build_manifest() -> pd.DataFrame:
    """Collect module metadata JSON files into the manifest."""

    silver = PROJECT_ROOT / "data" / "silver_normalized"
    rows = []
    for meta_path in sorted(silver.glob("*/*_metadata.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        row_count = int(meta.get("row_count", 0) or 0)
        status = str(meta.get("status") or ("ok" if row_count > 0 else "skipped"))
        compat = license_compatibility(str(meta.get("license") or "UNKNOWN"))
        rows.append(
            {
                "silver_module": meta_path.parent.name,
                "category": meta.get("main_category"),
                "source_dataset": meta.get("source_dataset"),
                "source_type": meta.get("source_type"),
                "main_category": meta.get("main_category"),
                "output_path_parquet": meta.get("output_paths", {}).get("parquet", ""),
                "output_path_csv_gz": meta.get("output_paths", {}).get("csv_gz", ""),
                "output_path_metadata": str(meta_path.relative_to(PROJECT_ROOT)),
                "row_count": row_count,
                "binary_label_distribution": json.dumps(meta.get("binary_label_distribution", {}), sort_keys=True),
                "label_distribution": json.dumps(meta.get("label_distribution", {}), sort_keys=True),
                "schema_version": meta.get("schema_version"),
                "license": meta.get("license"),
                "license_compatibility": compat,
                "input_hash": meta.get("input_hash"),
                "status": status,
                "duration_seconds": meta.get("duration_seconds", 0),
                "created_at_utc": meta.get("created_at_utc"),
                "notes": meta.get("notes", ""),
            }
        )
    df = pd.DataFrame(rows)
    out = silver / "silver_manifest.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def build_report(manifest: pd.DataFrame) -> None:
    """Write docs/silver_layer_report.md."""

    docs = PROJECT_ROOT / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    silver = PROJECT_ROOT / "data" / "silver_normalized"
    overlap = silver / "_dedup" / "cross_source_cve_overlap.csv"
    overlap_count = len(pd.read_csv(overlap)) if overlap.exists() else 0
    parquet_bytes = sum(p.stat().st_size for p in silver.glob("*/*.parquet"))
    csv_bytes = sum(p.stat().st_size for p in silver.glob("*/*.csv.gz"))
    lines = ["# Silver Layer Report", "", f"Generated: {pd.Timestamp.now(tz=UTC).isoformat()}", ""]
    lines.append("## Module Status")
    if manifest.empty:
        lines.append("No silver modules have completed yet.")
    else:
        lines.append(markdown_table(manifest[["silver_module", "status", "row_count", "license", "license_compatibility", "notes"]]))
    lines.extend(["", "## Succeeded Modules"])
    ok = manifest[manifest["status"] == "ok"] if not manifest.empty else pd.DataFrame()
    lines.append(markdown_table(ok[["silver_module", "row_count", "license"]]) if not ok.empty else "None")
    lines.extend(["", "## Skipped or Blocked Modules"])
    blocked = manifest[manifest["status"].isin(["skipped", "blocked"])] if not manifest.empty else pd.DataFrame()
    lines.append(markdown_table(blocked[["silver_module", "status", "notes"]]) if not blocked.empty else "None")
    lines.extend(["", "## Rows by Category"])
    lines.append(markdown_table(manifest.groupby("main_category")["row_count"].sum().reset_index()) if not manifest.empty else "None")
    lines.extend(["", "## Rows by Module"])
    lines.append(markdown_table(manifest[["silver_module", "row_count"]].sort_values("row_count", ascending=False)) if not manifest.empty else "None")
    lines.extend(["", "## Label Distribution"])
    for _, row in manifest.iterrows():
        lines.append(f"- {row['silver_module']}: {row['label_distribution']}")
    lines.extend(["", "## Silver Size on Disk", f"- Parquet bytes: {parquet_bytes}", f"- CSV.GZ bytes: {csv_bytes}", ""])
    lines.append("## License Summary")
    for _, row in manifest.iterrows():
        flag = " EXCLUDED FROM PUBLIC RELEASE" if str(row["license_compatibility"]).startswith("restricted") else ""
        lines.append(f"- {row['source_dataset']}: {row['license']} ({row['license_compatibility']}){flag}")
    lines.extend(["", "## Cross-Source CVE Overlap", f"- Overlapping CVE rows: {overlap_count}", ""])
    lines.append("## Failures, Skips, and Blockers")
    err = silver / "normalization_errors.jsonl"
    lines.append("### Failed")
    failed = manifest[manifest["status"] == "failed"] if not manifest.empty else pd.DataFrame()
    lines.append(markdown_table(failed[["silver_module", "notes"]]) if not failed.empty else "No failed modules in manifest.")
    lines.append("")
    lines.append("### Skipped or Blocked")
    lines.append(markdown_table(blocked[["silver_module", "status", "notes"]]) if not blocked.empty else "No skipped or blocked modules in manifest.")
    lines.append("")
    lines.append("### Error Log Tail")
    lines.append(err.read_text(encoding="utf-8")[-4000:] if err.exists() else "No errors logged.")
    lines.extend(
        [
            "",
            "## Next Recommended Normalizers",
            "- Remaining Priority 2: malware_bodmas, network_cic_ids_2017, supply_chain_datadog_malicious_packages.",
            "- Remaining Priority 3: auth_lanl_authentication, iot_iot23, ics_swat_wadi_epic_batadal, insider_cert_insider_threat.",
            "",
            "## Open Data Quality Issues",
            "- CAPEC source is missing locally; module is blocked until a CSV/XML source is added.",
            "- HackAPrompt source has README/metadata only; module is blocked until local prompt records are added.",
            "- PhishTank local file is a rate-limit response, not a CSV export; module is blocked until a valid local export is added.",
            "- Review preflight loose files, suspicious binaries/scripts, and incomplete download records before benchmark work.",
        ]
    )
    (docs / "silver_layer_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    build_report(build_manifest())


if __name__ == "__main__":
    main()
