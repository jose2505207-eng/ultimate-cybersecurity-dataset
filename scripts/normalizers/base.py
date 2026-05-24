"""Reusable CLI and parsers for Priority-1 normalizers."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import zipfile
from collections.abc import Callable
from datetime import UTC
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import pandas as pd

from scripts.normalizers.common import (
    BRONZE_ROOT,
    DEFAULT_ROW_CAP,
    PROJECT_ROOT,
    clean_text,
    compute_input_hash,
    ensure_unified_schema,
    extract_domain_from_url,
    license_compatibility,
    make_record_id,
    normalize_severity,
    safe_json_dumps,
    safe_read_csv_chunks,
    safe_read_json_stream,
    safe_read_yaml_dir,
    sample_balanced,
    write_metadata_json,
    write_silver,
)
from scripts.normalizers.schema import SCHEMA_VERSION

THREAT_CAT = "Threat Intelligence, CVE, Advisory & Taxonomy"
SUPPLY_CAT = "Supply Chain & Open Source Package Security"
PHISH_CAT = "Phishing, Social Engineering & Fraud"
AI_CAT = "AI, LLM & ML Security"
CRYPTO_CAT = "Cryptocurrency & Blockchain Attacks"
MALWARE_CAT = "Malware & PE/Memory Features"
NETWORK_CAT = "Network Intrusion & Traffic Attacks"
VULN_CAT = "Vulnerable Code & Software Weaknesses"
WEB_CAT = "Web Application Security"
API_CAT = "API Security"
HOST_CAT = "Endpoint, Host & Windows/Sysmon Telemetry"

ZERO_ROW_BLOCKERS = {
    "cti_capec_attack_patterns": "No CAPEC CSV/XML source is present under data/bronze_raw/capec.",
    "ai_security_hackaprompt": "HackAPrompt bronze input contains README/metadata only; no local CSV/JSON/Parquet prompt records were found.",
    "phishing_phishtank": "Local PhishTank file is not a valid CSV export; it contains a rate-limit response instead of URL records.",
}


def rel(path: Path) -> str:
    """Return bronze-relative path."""

    try:
        return str(path.relative_to(BRONZE_ROOT))
    except ValueError:
        return str(path)


def latest_files(root: Path, patterns: list[str]) -> list[Path]:
    """Find files matching patterns under a root."""

    out: list[Path] = []
    for pattern in patterns:
        out.extend(root.rglob(pattern))
    return sorted({p for p in out if p.is_file() and ".git" not in p.parts})


def _deduplicate_for_write(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop repeated record IDs after recording duplicate counts."""

    if "record_id" not in df.columns or df.empty:
        return df, {"duplicate_record_ids": 0, "duplicate_rows_removed": 0, "sample_duplicate_record_ids": []}
    duplicated = df["record_id"].duplicated(keep="first")
    duplicate_ids = sorted(set(df.loc[duplicated, "record_id"].astype(str).head(20).tolist()))
    deduped = df.loc[~duplicated].copy()
    return deduped, {
        "duplicate_record_ids": int(df.loc[duplicated, "record_id"].nunique()),
        "duplicate_rows_removed": int(duplicated.sum()),
        "sample_duplicate_record_ids": duplicate_ids,
    }


def run_module(
    *,
    module: str,
    source_dataset: str,
    source_type: str,
    main_category: str,
    license_name: str,
    parser: Callable[[Path, int | None, int], pd.DataFrame],
) -> None:
    """Run a normalizer with the canonical CLI."""

    argp = argparse.ArgumentParser(description=f"Normalize {module}")
    argp.add_argument("--input", type=Path, default=BRONZE_ROOT)
    argp.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "silver_normalized")
    argp.add_argument("--max-rows", type=int, default=None)
    argp.add_argument("--sample", type=int, default=None)
    argp.add_argument("--random-state", type=int, default=42)
    argp.add_argument("--force", action="store_true")
    argp.add_argument("--dry-run", action="store_true")
    args = argp.parse_args()
    started = time.time()
    input_path = args.input
    if input_path == BRONZE_ROOT:
        input_path = BRONZE_ROOT / source_dataset
    output_dir = args.output / module
    output_stem = output_dir / module
    metadata_path = output_stem.with_name(f"{module}_metadata.json")
    input_hash = compute_input_hash([input_path])
    if not args.force and metadata_path.exists():
        existing_meta = json.loads(metadata_path.read_text())
        has_outputs = output_stem.with_suffix(".parquet").exists() and output_stem.with_suffix(".csv.gz").exists()
        existing_status = existing_meta.get("status")
        if existing_meta.get("input_hash") == input_hash and existing_status == "ok" and has_outputs:
            return
        if existing_meta.get("input_hash") == input_hash and existing_status in {"blocked", "skipped"} and not has_outputs:
            return
    parse_limit = args.max_rows or args.sample or DEFAULT_ROW_CAP
    df = parser(input_path, parse_limit, args.random_state)
    parsed_rows = len(df)
    df, duplicate_report = _deduplicate_for_write(df)
    sampling = {"applied": False, "method": None, "original_rows": parsed_rows, "kept_rows": len(df), "seed": args.random_state}
    if args.sample and len(df) > args.sample:
        df = sample_balanced(df, "binary_label", args.sample, args.random_state)
        sampling.update({"applied": True, "method": "balanced", "kept_rows": len(df)})
    if len(df) > DEFAULT_ROW_CAP and args.max_rows is None:
        df = sample_balanced(df, "binary_label", DEFAULT_ROW_CAP, args.random_state)
        sampling.update({"applied": True, "method": "balanced", "kept_rows": len(df)})
    if df.empty:
        status = "blocked" if module in ZERO_ROW_BLOCKERS else "skipped"
        notes = ZERO_ROW_BLOCKERS.get(module, "No usable records were produced from local input.")
        df = ensure_unified_schema(df, source_dataset, source_type, main_category, license_name)
    else:
        status = "ok"
        notes = ""
        df = ensure_unified_schema(df, source_dataset, source_type, main_category, license_name)
    output_bytes = {"parquet": 0, "csv_gz": 0}
    meta = {
        "schema_version": SCHEMA_VERSION,
        "source_dataset": source_dataset,
        "source_type": source_type,
        "main_category": main_category,
        "input_paths": [rel(input_path)],
        "input_hash": input_hash,
        "output_paths": {},
        "row_count": len(df),
        "label_distribution": {str(k): int(v) for k, v in df["label"].value_counts(dropna=False).to_dict().items()},
        "binary_label_distribution": {str(k): int(v) for k, v in df["binary_label"].value_counts(dropna=False).to_dict().items()},
        "sampling": sampling,
        "deduplication": duplicate_report,
        "created_at_utc": pd.Timestamp.now(tz=UTC).isoformat(),
        "duration_seconds": round(time.time() - started, 3),
        "normalizer_script": f"scripts/normalizers/{module}.py",
        "normalizer_version": "0.1.0",
        "python_version": sys.version.split()[0],
        "pandas_version": pd.__version__,
        "pyarrow_version": __import__("pyarrow").__version__,
        "license": license_name,
        "license_compatibility": license_compatibility(license_name),
        "known_limitations": [notes] if notes else [],
        "notes": notes,
        "status": status,
        "output_bytes": output_bytes,
    }
    if args.dry_run:
        print(json.dumps({"module": module, "would_write_rows": len(df), "would_status": status, "output": str(output_stem)}, indent=2))
        return
    if status == "ok":
        outputs = write_silver(df, output_stem, max_rows=args.max_rows or DEFAULT_ROW_CAP)
        meta["output_paths"] = {"parquet": outputs["parquet"], "csv_gz": outputs["csv_gz"]}
        meta["output_bytes"] = {"parquet": outputs["parquet_bytes"], "csv_gz": outputs["csv_gz_bytes"]}
    else:
        for stale in (output_stem.with_suffix(".parquet"), output_stem.with_suffix(".csv.gz")):
            if stale.exists():
                stale.unlink()
    write_metadata_json(meta, metadata_path)


def parse_mitre(input_path: Path, limit: int | None, _: int) -> pd.DataFrame:
    """Parse current ATT&CK STIX bundles."""

    files = [p for p in latest_files(input_path, ["enterprise-attack.json", "ics-attack.json", "mobile-attack.json"]) if "broken" not in p.parts]
    rows = []
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        for obj in data.get("objects", []):
            if obj.get("type") != "attack-pattern" or obj.get("revoked") or obj.get("x_mitre_deprecated"):
                continue
            ext_id = next((r.get("external_id") for r in obj.get("external_references", []) if r.get("source_name") == "mitre-attack"), None)
            tactics = "|".join(k.get("phase_name", "") for k in obj.get("kill_chain_phases", []) if k.get("phase_name")) or None
            raw = "\n\n".join(x for x in [obj.get("description"), f"Detection: {obj.get('x_mitre_detection')}" if obj.get("x_mitre_detection") else None] if x)
            rows.append(
                {
                    "record_id": make_record_id("cti_mitre_attack_stix", ext_id or obj.get("id", "")),
                    "attack_name": obj.get("name"),
                    "attack_family": ext_id.split(".")[0] if ext_id and "." in ext_id else None,
                    "label": "attack_technique",
                    "binary_label": 1,
                    "mitre_tactic": tactics,
                    "mitre_technique_id": ext_id,
                    "platform": "|".join(obj.get("x_mitre_platforms", [])) or None,
                    "raw_text": raw,
                    "source_file": rel(path),
                }
            )
            if limit and len(rows) >= limit:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def parse_nvd(input_path: Path, limit: int | None, _: int) -> pd.DataFrame:
    """Parse NVD 2.0 JSON/JSONL feeds."""

    rows = []
    for path in latest_files(input_path, ["*.jsonl", "*.json"]):
        for item in safe_read_json_stream(path):
            cve = item.get("cve", item)
            cve_id = cve.get("id") or cve.get("CVE_data_meta", {}).get("ID")
            descs = cve.get("descriptions") or cve.get("description", {}).get("description_data", [])
            desc = next((d.get("value") for d in descs if d.get("lang") == "en"), None)
            metrics = cve.get("metrics", {})
            cvss4 = _metric_score(metrics, "cvssMetricV40")
            cvss3 = _metric_score(metrics, "cvssMetricV31") or _metric_score(metrics, "cvssMetricV30")
            cvss2 = _metric_score(metrics, "cvssMetricV2")
            sev, score = normalize_severity(cvss_v2=cvss2, cvss_v3=cvss3, cvss_v4=cvss4)
            cwes = [d.get("value") for w in cve.get("weaknesses", []) for d in w.get("description", []) if d.get("value", "").startswith("CWE-")]
            ref_obj = cve.get("references", {})
            refs = ref_obj if isinstance(ref_obj, list) else ref_obj.get("referenceData") or ref_obj.get("references") or []
            rows.append(
                {
                    "record_id": make_record_id("advisory_nvd_cve", cve_id or json.dumps(cve)[:200]),
                    "label": "vulnerability_advisory",
                    "binary_label": 1,
                    "cve_id": cve_id,
                    "cwe_id": cwes[0] if cwes else None,
                    "severity": sev,
                    "severity_score": score,
                    "timestamp": cve.get("published") or cve.get("publishedDate"),
                    "raw_text": desc,
                    "url": refs[0].get("url") if refs else None,
                    "features_json": safe_json_dumps({"cwe_ids": cwes}),
                    "source_file": rel(path),
                }
            )
            if limit and len(rows) >= limit:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def _xml_text(element: ET.Element | None) -> str | None:
    """Return normalized text from an XML element."""

    if element is None:
        return None
    return clean_text(" ".join(part.strip() for part in element.itertext() if part and part.strip()))


def _find_child(element: ET.Element, local_name: str) -> ET.Element | None:
    """Find the first direct child by namespace-insensitive local name."""

    suffix = f"}}{local_name}"
    for child in list(element):
        if child.tag == local_name or child.tag.endswith(suffix):
            return child
    return None


def _find_descendants(element: ET.Element, local_name: str) -> list[ET.Element]:
    """Find descendants by namespace-insensitive local name."""

    suffix = f"}}{local_name}"
    return [child for child in element.iter() if child.tag == local_name or child.tag.endswith(suffix)]


def parse_capec(input_path: Path, limit: int | None, _: int) -> pd.DataFrame:
    """Parse local CAPEC XML/CSV attack pattern exports."""

    rows = []
    files = latest_files(input_path, ["*.xml", "*.csv"]) if input_path.exists() else []
    for path in files:
        if path.suffix.lower() == ".xml":
            rows.extend(_parse_capec_xml(path, limit, len(rows)))
        elif path.suffix.lower() == ".csv":
            rows.extend(_parse_capec_csv(path, limit, len(rows)))
        if limit and len(rows) >= limit:
            return pd.DataFrame(rows[:limit])
    return pd.DataFrame(rows)


def _parse_capec_xml(path: Path, limit: int | None, existing: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = ET.parse(path).getroot()
    for pattern in _find_descendants(root, "Attack_Pattern"):
        status = str(pattern.attrib.get("Status", "")).lower()
        if status in {"deprecated", "obsolete"}:
            continue
        capec_id = pattern.attrib.get("ID") or pattern.attrib.get("Name")
        cwes = [f"CWE-{w.attrib.get('CWE_ID')}" for w in _find_descendants(pattern, "Related_Weakness") if w.attrib.get("CWE_ID")]
        mitigations = [_xml_text(m) for m in _find_descendants(pattern, "Mitigation")]
        descriptions = [
            _xml_text(_find_child(pattern, "Description")),
            _xml_text(_find_child(pattern, "Extended_Description")),
            "Mitigations: " + " ".join(m for m in mitigations if m) if mitigations else None,
        ]
        rows.append(
            {
                "record_id": make_record_id("cti_capec_attack_patterns", str(capec_id)),
                "attack_name": pattern.attrib.get("Name"),
                "attack_family": pattern.attrib.get("Abstraction"),
                "label": "attack_pattern",
                "binary_label": 1,
                "cwe_id": cwes[0] if cwes else None,
                "severity": normalize_severity(vendor_severity=pattern.attrib.get("Typical_Severity"))[0] if pattern.attrib.get("Typical_Severity") else None,
                "raw_text": "\n\n".join(part for part in descriptions if part),
                "features_json": safe_json_dumps({"capec_id": capec_id, "cwe_ids": cwes, "status": pattern.attrib.get("Status")}),
                "source_file": rel(path),
            }
        )
        if limit and existing + len(rows) >= limit:
            return rows
    return rows


def _parse_capec_csv(path: Path, limit: int | None, existing: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for chunk in safe_read_csv_chunks(path):
        for rec in chunk.to_dict("records"):
            lower = {str(k).lower().strip(): v for k, v in rec.items()}
            capec_id = lower.get("id") or lower.get("capec id") or lower.get("capec_id") or lower.get("capec-id")
            name = lower.get("name") or lower.get("attack pattern name")
            cwe_raw = str(lower.get("related weaknesses") or lower.get("related_weaknesses") or lower.get("cwe_id") or "")
            cwes = [f"CWE-{m}" for m in re.findall(r"(?:CWE-)?(\\d+)", cwe_raw)]
            severity = lower.get("typical severity") or lower.get("severity")
            sev, _score = normalize_severity(vendor_severity=severity) if severity else (None, None)
            rows.append(
                {
                    "record_id": make_record_id("cti_capec_attack_patterns", str(capec_id or name)),
                    "attack_name": name,
                    "attack_family": lower.get("abstraction") or lower.get("mechanisms of attack"),
                    "label": "attack_pattern",
                    "binary_label": 1,
                    "cwe_id": cwes[0] if cwes else None,
                    "severity": sev,
                    "raw_text": "\n\n".join(
                        str(part)
                        for part in [
                            lower.get("description"),
                            lower.get("extended description") or lower.get("extended_description"),
                            lower.get("mitigations"),
                        ]
                        if pd.notna(part) and str(part).strip()
                    ),
                    "features_json": safe_json_dumps({"capec_id": capec_id, "cwe_ids": cwes}),
                    "source_file": rel(path),
                }
            )
            if limit and existing + len(rows) >= limit:
                return rows
    return rows


def _metric_score(metrics: dict[str, Any], key: str) -> float | None:
    vals = metrics.get(key) or []
    if not vals:
        return None
    return vals[0].get("cvssData", {}).get("baseScore")


def parse_osv(input_path: Path, limit: int | None, _: int) -> pd.DataFrame:
    """Parse OSV zip JSON entries without writing extracted files to bronze."""

    rows = []
    for zp in latest_files(input_path, ["*.zip"]):
        with zipfile.ZipFile(zp) as zf:
            for name in sorted(n for n in zf.namelist() if n.endswith(".json")):
                data = json.loads(zf.read(name).decode("utf-8", errors="ignore"))
                affected = data.get("affected") or [{}]
                pkg = (affected[0].get("package") or {}) if affected else {}
                aliases = data.get("aliases") or []
                cve = next((a for a in aliases if str(a).startswith("CVE-")), None)
                sev, score = normalize_severity(vendor_severity=(data.get("database_specific") or {}).get("severity"))
                rows.append(
                    {
                        "record_id": make_record_id("supply_chain_osv", data.get("id", name)),
                        "attack_name": data.get("id"),
                        "label": "vulnerable_dependency",
                        "binary_label": 1,
                        "cve_id": cve,
                        "severity": sev,
                        "severity_score": score,
                        "timestamp": data.get("modified"),
                        "ecosystem": pkg.get("ecosystem"),
                        "package_name": pkg.get("name"),
                        "raw_text": "\n\n".join(x for x in [data.get("summary"), data.get("details")] if x),
                        "features_json": safe_json_dumps({"affected_versions": affected[:3], "aliases": aliases}),
                        "source_file": f"{rel(zp)}:{name}",
                    }
                )
                if limit and len(rows) >= limit:
                    return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def parse_ghsa(input_path: Path, limit: int | None, _: int) -> pd.DataFrame:
    """Parse GitHub Advisory YAML files."""

    rows = []
    json_paths = (input_path / "advisories").rglob("*.json") if (input_path / "advisories").exists() else iter(())
    for path in json_paths:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        _append_ghsa_row(rows, path, data)
        if limit and len(rows) >= limit:
            return pd.DataFrame(rows)
    for path, data in safe_read_yaml_dir(input_path / "advisories" if (input_path / "advisories").exists() else input_path):
        _append_ghsa_row(rows, path, data)
        if limit and len(rows) >= limit:
            return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def _append_ghsa_row(rows: list[dict[str, Any]], path: Path, data: dict[str, Any]) -> None:
    """Append one GHSA row from JSON/YAML data."""

    adv_id = data.get("id") or data.get("ghsaId") or data.get("ghsa_id") or path.stem
    aliases = data.get("aliases") or data.get("identifiers") or []
    if aliases and isinstance(aliases[0], dict):
        aliases = [a.get("value") for a in aliases]
    cve = next((a for a in aliases if str(a).startswith("CVE-")), None)
    affected = data.get("affected") or [{}]
    pkg = affected[0].get("package", {}) if isinstance(affected, list) and affected else {}
    db = data.get("database_specific") or {}
    sev, score = normalize_severity(vendor_severity=db.get("severity") or (data.get("severity") if isinstance(data.get("severity"), str) else None))
    rows.append(
        {
            "record_id": make_record_id("supply_chain_github_advisory", adv_id),
            "attack_name": adv_id,
            "label": "vulnerable_dependency",
            "binary_label": 1,
            "cve_id": cve,
            "severity": sev,
            "severity_score": score,
            "ecosystem": pkg.get("ecosystem"),
            "package_name": pkg.get("name"),
            "raw_text": "\n\n".join(x for x in [data.get("summary"), data.get("details")] if x),
            "features_json": safe_json_dumps({"aliases": aliases}),
            "source_file": rel(path),
        }
    )


def parse_phishtank(input_path: Path, limit: int | None, _: int) -> pd.DataFrame:
    """Parse PhishTank CSV.GZ without requesting URLs."""

    rows = []
    for path in latest_files(input_path, ["*.csv.gz", "*.csv"]):
        for chunk in safe_read_csv_chunks(path):
            for rec in chunk.to_dict("records"):
                url = rec.get("url") or rec.get("phish_detail_url")
                if not url:
                    continue
                rows.append(
                    {
                        "record_id": make_record_id("phishing_phishtank", str(rec.get("phish_id") or url)),
                        "label": "phishing",
                        "binary_label": 1,
                        "url": url,
                        "domain": extract_domain_from_url(url),
                        "timestamp": rec.get("submission_time"),
                        "source_file": rel(path),
                    }
                )
                if limit and len(rows) >= limit:
                    return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def parse_balanced_urls(input_path: Path, limit: int | None, _: int) -> pd.DataFrame:
    """Parse local phishing URL JSON datasets."""

    rows = []
    for path in latest_files(input_path, ["*.json"]):
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        if not isinstance(data, list):
            continue
        for rec in data:
            url = rec.get("text") or rec.get("url")
            binary = int(rec.get("label", 0))
            rows.append(
                {
                    "record_id": make_record_id("phishing_balanced_urls", f"{url}:{binary}"),
                    "label": "phishing" if binary else "benign_url",
                    "binary_label": binary,
                    "url": url,
                    "domain": extract_domain_from_url(url),
                    "source_file": rel(path),
                }
            )
            if limit and len(rows) >= limit:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def parse_owasp_genai(input_path: Path, limit: int | None, _: int) -> pd.DataFrame:
    """Parse OWASP GenAI Top 10 local HTML/Markdown text."""

    rows = []
    pattern = __import__("re").compile(r"(LLM\d{2})[:\s-]+([^<\n]+)", __import__("re").I)
    for path in latest_files(input_path, ["*.md", "*.html", "*.txt"]):
        text = path.read_text(encoding="utf-8", errors="ignore")
        seen: set[str] = set()
        for match in pattern.finditer(text):
            code = match.group(1).upper()
            if code in seen:
                continue
            seen.add(code)
            start = max(match.start() - 200, 0)
            end = min(match.end() + 1800, len(text))
            rows.append(
                {
                    "record_id": make_record_id("ai_security_owasp_genai_top10", code),
                    "attack_name": f"{code}: {clean_text(match.group(2))}",
                    "attack_family": "genai_application_risk",
                    "label": "ai_security_risk",
                    "binary_label": 1,
                    "raw_text": clean_text(text[start:end]),
                    "source_file": rel(path),
                }
            )
            if limit and len(rows) >= limit:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def parse_giskard(input_path: Path, limit: int | None, _: int) -> pd.DataFrame:
    """Parse Giskard prompt injection CSV."""

    rows = []
    for path in latest_files(input_path, ["*.csv"]):
        if "prompt" not in path.name:
            continue
        for chunk in safe_read_csv_chunks(path):
            for rec in chunk.to_dict("records"):
                prompt = rec.get("prompt")
                rows.append(
                    {
                        "record_id": make_record_id("ai_security_giskard_prompt_injections", str(rec.get("index") or prompt)),
                        "attack_name": "jailbreak" if "jailbreak" in str(rec.get("group", "")).lower() else "prompt_injection",
                        "attack_family": rec.get("group"),
                        "label": "jailbreak_prompt" if "jailbreak" in str(rec.get("group", "")).lower() else "malicious_prompt",
                        "binary_label": 1,
                        "language": rec.get("language"),
                        "raw_text": prompt,
                        "source_file": rel(path),
                    }
                )
                if limit and len(rows) >= limit:
                    return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def parse_lakera(input_path: Path, limit: int | None, _: int) -> pd.DataFrame:
    """Parse Lakera Gandalf parquet files."""

    rows = []
    for path in latest_files(input_path, ["*.parquet"]):
        if "gandalf" not in str(path).lower() and "ignore_instructions" not in str(path).lower():
            continue
        import pyarrow.parquet as pq

        df = pq.read_table(path, columns=["text", "similarity"]).slice(0, limit or DEFAULT_ROW_CAP).to_pandas()
        for i, rec in enumerate(df.to_dict("records")):
            text = rec.get("text") or rec.get("prompt")
            rows.append(
                {
                    "record_id": make_record_id("ai_security_lakera_gandalf", f"{path.name}:{i}:{text}"),
                    "attack_name": "instruction_override",
                    "label": "malicious_prompt",
                    "binary_label": 1,
                    "raw_text": text,
                    "features_json": safe_json_dumps({"similarity": rec.get("similarity")}),
                    "source_file": rel(path),
                }
            )
            if limit and len(rows) >= limit:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def parse_hf_prompt(input_path: Path, limit: int | None, _: int) -> pd.DataFrame:
    """Parse local Hugging Face prompt injection datasets."""

    rows = []
    for path in latest_files(input_path, ["*.csv"]):
        if "prompt" not in str(path).lower() and "antijection" not in str(path).lower():
            continue
        for chunk in safe_read_csv_chunks(path):
            for rec in chunk.to_dict("records"):
                prompt = rec.get("prompt") or rec.get("text")
                if not prompt:
                    continue
                label = str(rec.get("label", "malicious")).lower()
                binary = 0 if "benign" in label else 1
                rows.append(
                    {
                        "record_id": make_record_id("ai_security_huggingface_prompt_injection", f"{path}:{prompt}"),
                        "attack_name": rec.get("attack_category") or "prompt_injection",
                        "attack_family": rec.get("context"),
                        "label": "benign_prompt" if binary == 0 else "malicious_prompt",
                        "binary_label": binary,
                        "raw_text": prompt,
                        "source_file": rel(path),
                    }
                )
                if limit and len(rows) >= limit:
                    return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def parse_hackaprompt(input_path: Path, limit: int | None, _: int) -> pd.DataFrame:
    """Parse HackAPrompt-like local files if present."""

    rows = []
    for path in latest_files(input_path, ["*.csv", "*.json", "*.jsonl", "*.parquet"]):
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix == ".csv":
            df = pd.concat(list(safe_read_csv_chunks(path)), ignore_index=True)
        elif path.suffix == ".jsonl":
            df = pd.DataFrame(list(safe_read_json_stream(path)))
        else:
            data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            df = pd.DataFrame(data if isinstance(data, list) else data.get("data", []))
        for i, rec in enumerate(df.to_dict("records")):
            prompt = rec.get("prompt") or rec.get("text") or rec.get("user_input")
            if not prompt:
                continue
            rows.append(
                {
                    "record_id": make_record_id("ai_security_hackaprompt", f"{path}:{i}:{prompt}"),
                    "attack_name": "prompt_injection",
                    "label": "malicious_prompt",
                    "binary_label": 1,
                    "raw_text": prompt,
                    "source_file": rel(path),
                }
            )
            if limit and len(rows) >= limit:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def parse_smartbugs(input_path: Path, limit: int | None, _: int) -> pd.DataFrame:
    """Parse SmartBugs Curated Solidity sources without compiling them."""

    rows = []
    meta_path = input_path / "vulnerabilities.json"
    records = json.loads(meta_path.read_text(encoding="utf-8", errors="ignore")) if meta_path.exists() else []
    for rec in records:
        rel_path = rec.get("path")
        if not rel_path:
            continue
        source_path = input_path / rel_path
        if not source_path.exists() or source_path.suffix.lower() != ".sol":
            continue
        vulns = rec.get("vulnerabilities") or []
        categories = sorted({v.get("category") for v in vulns if v.get("category")})
        lines = [line for v in vulns for line in v.get("lines", [])]
        source = source_path.read_text(encoding="utf-8", errors="ignore")
        rows.append(
            {
                "record_id": make_record_id("blockchain_smartbugs_curated", rel_path),
                "attack_name": source_path.stem,
                "attack_family": "|".join(categories) or source_path.parent.name,
                "label": "vulnerable_code",
                "binary_label": 1,
                "language": "Solidity",
                "platform": "Ethereum",
                "raw_text": source,
                "features_json": safe_json_dumps({"pragma": rec.get("pragma"), "source": rec.get("source"), "vulnerability_lines": lines, "categories": categories}),
                "source_file": rel(source_path),
                "notes": "Solidity source only; never compiled or deployed.",
            }
        )
        if limit and len(rows) >= limit:
            return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def parse_defihacklabs(input_path: Path, limit: int | None, _: int) -> pd.DataFrame:
    """Parse DeFiHackLabs incident and RCA metadata without fetching links."""

    incidents_path = input_path / "incidents.json"
    rootcause_path = input_path / "rootcause_data.json"
    incidents = json.loads(incidents_path.read_text(encoding="utf-8", errors="ignore")) if incidents_path.exists() else []
    rootcauses = json.loads(rootcause_path.read_text(encoding="utf-8", errors="ignore")) if rootcause_path.exists() else {}
    rows = []
    for rec in incidents:
        name = rec.get("name") or "unknown"
        rca = rootcauses.get(name, {}) if isinstance(rootcauses, dict) else {}
        incident_type = rec.get("type") or rca.get("type")
        raw_parts = [f"Incident: {name}", f"Type: {incident_type}" if incident_type else None, rca.get("rootCause")]
        rows.append(
            {
                "record_id": make_record_id("blockchain_defihacklabs_incidents", f"{rec.get('date')}::{name}::{rec.get('chain')}"),
                "attack_name": name,
                "attack_family": incident_type,
                "label": "exploit_incident",
                "binary_label": 1,
                "timestamp": rec.get("date") or rca.get("date"),
                "platform": rec.get("chain"),
                "raw_text": "\n\n".join(str(x) for x in raw_parts if x),
                "features_json": safe_json_dumps({"lost": rec.get("Lost") or rca.get("Lost"), "loss_type": rec.get("lossType"), "contract": rec.get("Contract") or rca.get("Contract")}),
                "source_file": rel(incidents_path),
                "notes": "Metadata and local RCA text only; no on-chain or external fetch performed.",
            }
        )
        if limit and len(rows) >= limit:
            return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def _family_from_malmem_filename(filename: str) -> tuple[str | None, str]:
    parts = str(filename).split("-", 2)
    if len(parts) >= 2:
        return parts[0], parts[1]
    return None, str(filename)


def parse_cic_malmem(input_path: Path, limit: int | None, _: int) -> pd.DataFrame:
    """Parse available CICMalMem2022 memory-feature CSVs without malware execution."""

    rows = []
    if input_path.is_file():
        files = [input_path]
    else:
        files = [input_path / "output2.csv"] if (input_path / "output2.csv").exists() else latest_files(input_path, ["*output*.csv", "*Output*.csv"])
    for path in files:
        for chunk in safe_read_csv_chunks(path):
            feature_cols = [c for c in chunk.columns if c != "Filename"]
            for rec in chunk.to_dict("records"):
                family, name = _family_from_malmem_filename(rec.get("Filename", ""))
                binary = 0 if str(family).lower() == "benign" else 1
                rows.append(
                    {
                        "record_id": make_record_id("malware_cic_malmem_2022", str(rec.get("Filename"))),
                        "attack_name": name,
                        "attack_family": family,
                        "label": "benign" if binary == 0 else "malicious",
                        "binary_label": binary,
                        "platform": "Windows memory",
                        "raw_text": str(rec.get("Filename")),
                        "features_json": safe_json_dumps({col: rec.get(col) for col in feature_cols}),
                        "source_file": rel(path),
                        "notes": "Memory feature CSV only; companion files may be missing per preflight.",
                    }
                )
                if limit and len(rows) >= limit:
                    return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def parse_unsw_nb15(input_path: Path, limit: int | None, _: int) -> pd.DataFrame:
    """Parse UNSW-NB15 CSVs from a verified CSV-only zip without extracting files."""

    rows = []
    zip_candidates = [input_path] if input_path.is_file() else latest_files(input_path, ["OneDrive_1_5-23-2026.zip", "*UNSW*.zip"])
    for archive in zip_candidates:
        if archive.suffix.lower() != ".zip":
            continue
        with zipfile.ZipFile(archive) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv") and "UNSW_NB15" in n]
            if not csv_names:
                continue
            for name in sorted(csv_names):
                with zf.open(name) as fh:
                    for chunk in pd.read_csv(fh, chunksize=50_000):
                        feature_cols = [c for c in chunk.columns if c not in {"id", "attack_cat", "label"}]
                        for rec in chunk.to_dict("records"):
                            binary = int(rec.get("label", 0))
                            attack = None if str(rec.get("attack_cat", "")).lower() == "normal" else rec.get("attack_cat")
                            rows.append(
                                {
                                    "record_id": make_record_id("network_unsw_nb15", f"{name}:{rec.get('id')}"),
                                    "attack_name": attack,
                                    "attack_family": attack,
                                    "label": "benign" if binary == 0 else "intrusion",
                                    "binary_label": binary,
                                    "protocol": rec.get("proto"),
                                    "raw_text": f"UNSW-NB15 flow {name}:{rec.get('id')}",
                                    "features_json": safe_json_dumps({col: rec.get(col) for col in feature_cols}),
                                    "source_file": f"{rel(archive)}:{name}",
                                }
                            )
                            if limit and len(rows) >= limit:
                                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def parse_sard_juliet(input_path: Path, limit: int | None, _: int) -> pd.DataFrame:
    """Parse Juliet C/C++ source files from ZIP without extracting or compiling."""

    rows = []
    archives = [input_path] if input_path.is_file() else latest_files(input_path, ["*juliet*.zip", "*Juliet*.zip"])
    for archive in archives:
        if archive.suffix.lower() != ".zip":
            continue
        with zipfile.ZipFile(archive) as zf:
            names = sorted(
                n
                for n in zf.namelist()
                if n.lower().endswith((".c", ".cpp", ".h", ".hpp"))
                and "/testcases/" in n.lower()
                and "__macosx" not in n.lower()
            )
            for name in names:
                base = Path(name).name
                cwe_match = re.search(r"CWE(\d+)", name)
                is_good = bool(re.search(r"(^|[_-])good", base, re.IGNORECASE))
                label = "non_vulnerable_code" if is_good else "vulnerable_code"
                try:
                    source = zf.read(name).decode("utf-8", errors="ignore")
                except (KeyError, UnicodeDecodeError):
                    continue
                rows.append(
                    {
                        "record_id": make_record_id("vulnerable_code_sard_juliet", name),
                        "attack_name": Path(base).stem,
                        "attack_family": f"CWE-{cwe_match.group(1)}" if cwe_match else None,
                        "label": label,
                        "binary_label": 0 if is_good else 1,
                        "cwe_id": f"CWE-{cwe_match.group(1)}" if cwe_match else None,
                        "language": "C++" if name.lower().endswith((".cpp", ".hpp")) else "C",
                        "raw_text": source,
                        "features_json": safe_json_dumps({"zip_member": name, "juliet_variant": "good" if is_good else "bad"}),
                        "source_file": f"{rel(archive)}:{name}",
                        "notes": "Source code only; never compiled or executed.",
                    }
                )
                if limit and len(rows) >= limit:
                    return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def parse_owasp_benchmark(input_path: Path, limit: int | None, _: int) -> pd.DataFrame:
    """Parse OWASP Benchmark expected results and local Java sources."""

    rows = []
    expected = input_path / "BenchmarkJava" / "expectedresults-1.2.csv"
    source_root = input_path / "BenchmarkJava" / "src" / "main" / "java" / "org" / "owasp" / "benchmark" / "testcode"
    if not expected.exists():
        return pd.DataFrame(rows)
    cols = ["test_name", "category", "real_vulnerability", "cwe"]
    df = pd.read_csv(expected, comment="#", header=None, names=cols)
    for rec in df.to_dict("records"):
        test_name = str(rec.get("test_name"))
        source_path = source_root / f"{test_name}.java"
        source = source_path.read_text(encoding="utf-8", errors="ignore") if source_path.exists() else None
        vulnerable = str(rec.get("real_vulnerability")).lower() == "true"
        rows.append(
            {
                "record_id": make_record_id("web_owasp_benchmark", test_name),
                "attack_name": test_name,
                "attack_family": rec.get("category"),
                "label": "vulnerable_code" if vulnerable else "non_vulnerable_code",
                "binary_label": 1 if vulnerable else 0,
                "cwe_id": f"CWE-{int(rec.get('cwe'))}" if pd.notna(rec.get("cwe")) else None,
                "language": "Java",
                "raw_text": source,
                "features_json": safe_json_dumps({"benchmark_category": rec.get("category"), "expected_vulnerable": vulnerable}),
                "source_file": rel(source_path if source_path.exists() else expected),
                "notes": "Java source and expected-result metadata only; never built or executed.",
            }
        )
        if limit and len(rows) >= limit:
            return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def _markdown_sections(text: str, heading_pattern: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(heading_pattern, text, flags=re.MULTILINE))
    sections: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        sections.append((clean_text(match.group(1)) or "Untitled", clean_text(text[match.end() : end]) or ""))
    return sections


def parse_owasp_crapi(input_path: Path, limit: int | None, _: int) -> pd.DataFrame:
    """Parse local OWASP crAPI challenge and OpenAPI documentation."""

    rows = []
    challenges = input_path / "crAPI" / "docs" / "challenges.md"
    openapi = input_path / "crAPI" / "openapi-spec" / "crapi-openapi-spec.json"
    endpoint_count = None
    if openapi.exists():
        spec = json.loads(openapi.read_text(encoding="utf-8", errors="ignore"))
        endpoint_count = sum(len(methods) for methods in spec.get("paths", {}).values() if isinstance(methods, dict))
    if challenges.exists():
        text = challenges.read_text(encoding="utf-8", errors="ignore")
        for title, body in _markdown_sections(text, r"^###\s+(Challenge\s+\d+\s+-\s+.+)$"):
            rows.append(
                {
                    "record_id": make_record_id("api_owasp_crapi", title),
                    "attack_name": title,
                    "attack_family": "OWASP API Security",
                    "label": "attack_pattern",
                    "binary_label": 1,
                    "raw_text": body,
                    "features_json": safe_json_dumps({"openapi_endpoint_count": endpoint_count}),
                    "source_file": rel(challenges),
                    "notes": "Local challenge/OpenAPI documentation only; crAPI services were not started.",
                }
            )
            if limit and len(rows) >= limit:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def parse_otrf_security_datasets(input_path: Path, limit: int | None, _: int) -> pd.DataFrame:
    """Parse OTRF Security Datasets YAML metadata without downloading linked telemetry."""

    rows = []
    metadata_roots = [input_path / "Security-Datasets" / "datasets", input_path]
    seen: set[Path] = set()
    for root in metadata_roots:
        if not root.exists():
            continue
        for path, data in safe_read_yaml_dir(root):
            if path in seen or path.parent.name != "_metadata":
                continue
            seen.add(path)
            mappings = data.get("attack_mappings") or []
            techniques: list[str] = []
            tactics: list[str] = []
            for mapping in mappings:
                tech = str(mapping.get("technique") or "").strip()
                sub = str(mapping.get("sub-technique") or "").strip()
                if tech:
                    techniques.append(f"{tech}.{sub}" if sub and sub.lower() != "none" else tech)
                tactics.extend(str(t) for t in (mapping.get("tactics") or []) if t)
            platforms = data.get("platform") or []
            file_types = sorted({f.get("type") for f in data.get("files", []) if isinstance(f, dict) and f.get("type")})
            source_type = "sysmon_event" if any(str(t).lower() == "host" for t in file_types) else "host_telemetry"
            rows.append(
                {
                    "record_id": make_record_id("host_otrf_security_datasets", data.get("id") or str(path)),
                    "attack_name": data.get("title"),
                    "attack_family": data.get("type"),
                    "label": "attack_technique",
                    "binary_label": 1,
                    "mitre_tactic": "|".join(sorted(set(tactics))) or None,
                    "mitre_technique_id": "|".join(sorted(set(techniques))) or None,
                    "platform": "|".join(str(p) for p in platforms) or None,
                    "timestamp": data.get("creation_date"),
                    "raw_text": data.get("description"),
                    "features_json": safe_json_dumps({"dataset_id": data.get("id"), "file_types": file_types, "tags": data.get("tags")}),
                    "source_file": rel(path),
                    "notes": f"Metadata only; linked telemetry archives were not downloaded. source_type_hint={source_type}",
                }
            )
            if limit and len(rows) >= limit:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)
