from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from cyberdataset.inventory import write_inventory_report
from cyberdataset.normalize import finalize_records, make_record
from cyberdataset.schema import validate_schema
from cyberdataset.utils import DATA_DIR, config_path, load_yaml, write_table


SUPPORTED_INGEST_MODULES = {
    "ingest_cicids2017",
    "ingest_unsw_nb15",
    "ingest_phishtank",
    "ingest_urlhaus",
    "ingest_nvd",
    "ingest_cisa_kev",
}

SILVER_ROOT = DATA_DIR / "silver_normalized"


def smoke_records() -> list[dict]:
    return [
        make_record(
            source_dataset="SyntheticCodeSmoke",
            source_type="code",
            main_category="vulnerable_code",
            attack_name="CWE Metadata Example",
            source_label="vulnerable",
            raw_text_or_features={"summary": "redacted vulnerable code metadata", "features": {"cwe": "CWE-79"}},
            source_key="code-1",
            cwe_id="CWE-79",
            severity="high",
            license_note="Synthetic fixture for tests only.",
            is_synthetic=True,
        ),
        make_record(
            source_dataset="SyntheticFlowSmoke",
            source_type="network_flow",
            main_category="network_intrusion",
            attack_name="Benign Flow",
            source_label="benign",
            raw_text_or_features={"duration": 1.2, "bytes_in": 128, "bytes_out": 256},
            source_key="flow-1",
            severity="unknown",
            license_note="Synthetic fixture for tests only.",
            is_synthetic=True,
        ),
        make_record(
            source_dataset="SyntheticPromptSmoke",
            source_type="prompt",
            main_category="prompt_injection_ai_security",
            attack_name="Prompt Injection",
            source_label="prompt_attack",
            raw_text_or_features={"summary": "redacted prompt attack pattern", "risk": "instruction override attempt"},
            source_key="prompt-1",
            mitre_tactic="Initial Access",
            severity="medium",
            license_note="Synthetic fixture for tests only.",
            is_synthetic=True,
        ),
    ]


def build_smoke_silver() -> None:
    df = finalize_records(smoke_records())
    validate_schema(df)
    output = SILVER_ROOT / "_legacy_smoke" / "synthetic_smoke.csv"
    write_table(df, output)
    print(f"Wrote {len(df)} rows to {output}")


def _dataset_configs() -> list[dict]:
    return load_yaml(config_path("datasets.yaml"))["datasets"]


def _source_output_name(source_name: str) -> str:
    return source_name.lower().replace(" ", "_").replace("/", "_")


def build_available_silver(*, limit_per_source: int | None = None) -> list[Path]:
    report_path = write_inventory_report()
    print(f"Wrote bronze inventory report to {report_path}")

    bronze_root = DATA_DIR / "bronze_raw"
    silver_root = SILVER_ROOT / "_legacy_ingest"
    silver_root.mkdir(parents=True, exist_ok=True)
    smoke_output = silver_root / "synthetic_smoke.csv"
    if smoke_output.exists():
        smoke_output.unlink()

    written: list[Path] = []
    for config in _dataset_configs():
        source_dir = bronze_root / config["name"]
        if not source_dir.exists():
            continue

        module_name = config["ingest_module"]
        if module_name not in SUPPORTED_INGEST_MODULES:
            print(f"Skipping {config['name']}: parser not implemented yet.")
            continue

        module = importlib.import_module(f"cyberdataset.ingest.{module_name}")
        raw = module.load_raw(source_dir, limit=limit_per_source)
        if raw.empty:
            print(f"Skipping {config['name']}: no supported raw files found in {source_dir}.")
            continue

        normalized = module.normalize(raw)
        validate_schema(normalized)
        output = silver_root / f"{_source_output_name(config['name'])}.csv"
        module.write_silver(normalized, output)
        written.append(output)
        print(f"Wrote {len(normalized)} rows to {output}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Build source-separated silver datasets.")
    parser.add_argument("--smoke", action="store_true", help="Write tiny safe synthetic silver fixtures.")
    parser.add_argument("--limit-per-source", type=int, default=None, help="Optional row cap per source for dry runs.")
    args = parser.parse_args()
    if args.smoke:
        build_smoke_silver()
        return
    written = build_available_silver(limit_per_source=args.limit_per_source)
    if not written:
        print("No silver files were written. Place raw files under data/bronze_raw/<DatasetName>/ for supported parsers.")


if __name__ == "__main__":
    main()
