"""Inventory bronze dataset folders for silver planning."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.normalizers.common import PROJECT_ROOT

BRONZE = PROJECT_ROOT / "data" / "bronze_raw"
CATALOG = PROJECT_ROOT / "data" / "bronze_catalog"
IGNORE = {"venv", ".venv", "__pycache__", "node_modules", ".cache", ".ipynb_checkpoints", ".git"}


def ignored(path: Path) -> bool:
    """Return true for globally ignored dataset folders."""

    return path.name in IGNORE or "broken" in path.name.lower()


def guess_category(name: str) -> tuple[str, str]:
    """Guess source type and category from folder name."""

    low = name.lower()
    if "mitre" in low or "nvd" in low or "capec" in low:
        return "cti_taxonomy" if "nvd" not in low else "vulnerability_advisory", "Threat Intelligence, CVE, Advisory & Taxonomy"
    if "github_advisory" in low or "osv" in low:
        return "package_metadata", "Supply Chain & Open Source Package Security"
    if "phish" in low:
        return "phishing_url", "Phishing, Social Engineering & Fraud"
    if "huggingface_ai_security" in low or low == "huggingface":
        return "prompt_text", "AI, LLM & ML Security"
    if "prompt" in low or "gandalf" in low or "giskard" in low or "genai" in low:
        return "prompt_text", "AI, LLM & ML Security"
    if "smartbugs" in low or "defi" in low:
        return "smart_contract_code", "Cryptocurrency & Blockchain Attacks"
    return "other", "Miscellaneous / Needs Review"


def build_inventory() -> pd.DataFrame:
    """Build the canonical bronze inventory table."""

    rows = []
    for folder in sorted(p for p in BRONZE.iterdir() if p.is_dir() and not ignored(p)):
        files = [p for p in folder.rglob("*") if p.is_file() and not any(part in IGNORE for part in p.parts)]
        source_type, category = guess_category(folder.name)
        suffixes = sorted({p.suffix.lower() or "<none>" for p in files})
        size = sum(p.stat().st_size for p in files if not p.is_symlink())
        rows.append(
            {
                "dataset_folder": folder.name,
                "detected_dataset_name": folder.name.replace("_", " "),
                "source_type": source_type,
                "main_category": category,
                "file_count": len(files),
                "total_size_bytes": size,
                "primary_file_types": "|".join(suffixes[:12]),
                "sample_file_paths": "|".join(str(p.relative_to(BRONZE)) for p in files[:5]),
                "license_known": any("license" in p.name.lower() for p in files),
                "license_string": "UNKNOWN",
                "license_compatibility": "unknown",
                "has_partial_downloads": any(p.name.endswith(".crdownload") for p in files),
                "has_suspicious_binaries": any(p.suffix.lower() in {".exe", ".dll", ".so", ".elf", ".apk", ".bin", ".sh", ".js"} for p in files),
                "notes": "",
            }
        )
    df = pd.DataFrame(rows)
    CATALOG.mkdir(parents=True, exist_ok=True)
    df.to_csv(CATALOG / "bronze_inventory.csv", index=False)
    return df


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    build_inventory()


if __name__ == "__main__":
    main()
