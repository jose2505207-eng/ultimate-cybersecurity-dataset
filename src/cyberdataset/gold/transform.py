"""Silver-row -> gold-unified-record transformation helpers.

These functions are deliberately pure and side-effect free so they are easy to
unit test and reason about. The builder in :mod:`cyberdataset.gold.build_gold`
orchestrates IO around them.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from cyberdataset.gold.schema import DOMAINS, VALID_SPLITS, UnifiedGoldRecord

# --------------------------------------------------------------------------- #
# Source attribution registry (public, stable source URLs).                   #
# --------------------------------------------------------------------------- #

#: Best-effort attribution URLs for known silver sources. Keyed by a substring
#: that appears in the silver ``source_dataset``/source-id. Unknown sources get
#: an empty URL (preserved metadata still records the source name).
SOURCE_URL_REGISTRY: dict[str, str] = {
    "nvd": "https://nvd.nist.gov/",
    "cisa_kev": "https://www.cisa.gov/known-exploited-vulnerabilities-catalog",
    "kev": "https://www.cisa.gov/known-exploited-vulnerabilities-catalog",
    "osv": "https://osv.dev/",
    "github_advisor": "https://github.com/advisories",
    "ghsa": "https://github.com/advisories",
    "mitre_attack": "https://attack.mitre.org/",
    "capec": "https://capec.mitre.org/",
    "cwe": "https://cwe.mitre.org/",
    "phishtank": "https://phishtank.org/",
    "unsw_nb15": "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
    "sard": "https://samate.nist.gov/SARD/",
    "juliet": "https://samate.nist.gov/SARD/",
    "owasp": "https://owasp.org/",
    "gandalf": "https://www.lakera.ai/",
    "hackaprompt": "https://www.hackaprompt.com/",
    "giskard": "https://www.giskard.ai/",
    "smartbugs": "https://github.com/smartbugs/smartbugs",
    "defihacklabs": "https://github.com/SunWeb3Sec/DeFiHackLabs",
    "otrf": "https://github.com/OTRF/Security-Datasets",
}

# --------------------------------------------------------------------------- #
# Domain inference. Order matters: the first matching rule wins, so the more   #
# specific signals are listed before broad ones.                              #
# --------------------------------------------------------------------------- #

_DOMAIN_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("prompt_injection_jailbreak", ("prompt", "jailbreak", "gandalf", "hackaprompt",
                                    "genai", "llm", "giskard")),
    ("phishing_social", ("phish", "spam", "social_engineering", "nazario", "nigerian")),
    ("blockchain_web3", ("blockchain", "web3", "smart_contract", "ethereum", "defi",
                         "smartbugs", "solidity", "crypto")),
    ("malware_code", ("malware", "malmem", "bodmas", "ember", "ransomware",
                      "vulnerable_code", "sard", "juliet", "diversevul", "cvefixes",
                      "code_weakness")),
    ("network_intrusion", ("network", "nb15", "cicids", "cic_ids", "ids", "iot", "ics",
                           "telemetry", "swat", "wadi", "ton_iot", "awid", "flow",
                           "intrusion")),
    ("cloud_infrastructure", ("cloud", "mordor", "otrf", "host_", "k8s", "kubernetes",
                              "aws", "azure", "gcp", "infrastructure")),
    ("threat_intelligence", ("mitre", "att&ck", "attack_stix", "capec", "cti",
                             "threat_intel", "adversary_tactic")),
    ("vulnerabilities_exposures", ("nvd", "cve", "advisory", "kev", "osv", "ghsa",
                                   "supply_chain", "vulnerability", "exposure", "cwe")),
    ("web_app_security", ("web", "owasp", "crapi", "api", "xss", "sqli", "benchmark",
                          "appsec")),
)

#: Map silver ``source_type`` values to a coarse gold ``task_type``.
_TASK_TYPE_BY_SOURCE_TYPE: dict[str, str] = {
    "code": "vulnerability_classification",
    "vulnerable_code": "vulnerability_classification",
    "network_flow": "intrusion_classification",
    "iot_flow": "intrusion_classification",
    "ics_telemetry": "intrusion_classification",
    "malware_features": "malware_classification",
    "email": "phishing_classification",
    "url": "phishing_classification",
    "prompt": "prompt_attack_detection",
    "prompt_text": "prompt_attack_detection",
    "cloud_log": "anomaly_detection",
    "mobile_features": "malware_classification",
    "cti": "knowledge_reference",
    "advisory": "knowledge_reference",
    "vulnerability_advisory": "knowledge_reference",
    "package_metadata": "knowledge_reference",
}

_TASK_TYPE_BY_DOMAIN: dict[str, str] = {
    "web_app_security": "vulnerability_classification",
    "network_intrusion": "intrusion_classification",
    "malware_code": "malware_classification",
    "phishing_social": "phishing_classification",
    "prompt_injection_jailbreak": "prompt_attack_detection",
    "vulnerabilities_exposures": "knowledge_reference",
    "cloud_infrastructure": "anomaly_detection",
    "blockchain_web3": "vulnerability_classification",
    "threat_intelligence": "knowledge_reference",
    "miscellaneous": "classification",
}

_WHITESPACE_RE = re.compile(r"\s+")
_CVE_RE = re.compile(r"CVE-\d{4}-\d{3,7}", re.IGNORECASE)
_CWE_RE = re.compile(r"CWE-\d{1,5}", re.IGNORECASE)
_MITRE_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")
_URL_RE = re.compile(r"https?://[^\s\"'<>]+|hxxps?://[^\s\"'<>]+", re.IGNORECASE)

# Columns that are bookkeeping rather than content; surfaced via metadata.
_METADATA_CANDIDATES: tuple[str, ...] = (
    "source_dataset",
    "source_type",
    "main_category",
    "attack_name",
    "attack_family",
    "binary_label",
    "severity_score",
    "platform",
    "language",
    "protocol",
    "ecosystem",
    "package_name",
    "source_file",
    "schema_version",
    "ingested_at",
)


def _clean_str(value: Any) -> str:
    """Coerce a value to a stripped string, treating NA/None/'nan' as empty."""
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "<na>", "null"}:
        return ""
    return text


def clean_text(value: Any) -> str:
    """Collapse whitespace and strip control noise from free text."""
    text = _clean_str(value)
    if not text:
        return ""
    text = text.replace("\x00", " ")
    return _WHITESPACE_RE.sub(" ", text).strip()


def slugify_source(value: str) -> str:
    """Turn a source name into a stable lowercase slug used as ``source_id``."""
    slug = re.sub(r"[^a-z0-9]+", "_", _clean_str(value).lower()).strip("_")
    return slug or "unknown_source"


def lookup_source_url(source_id: str, source_name: str = "") -> str:
    """Resolve a public attribution URL for a known source, else ``''``."""
    haystack = f"{source_id} {source_name}".lower()
    for key, url in SOURCE_URL_REGISTRY.items():
        if key in haystack:
            return url
    return ""


def infer_domain(*signals: Any) -> str:
    """Infer a canonical security domain from free-text signals.

    Signals are typically the source id, ``source_dataset``, ``main_category``,
    and ``source_type``. Returns ``"miscellaneous"`` when nothing matches.
    """
    haystack = " ".join(_clean_str(signal).lower() for signal in signals)
    for domain, keywords in _DOMAIN_RULES:
        if any(keyword in haystack for keyword in keywords):
            return domain
    return "miscellaneous"


def infer_task_type(source_type: str, domain: str) -> str:
    """Infer a coarse ML task type from the source type and inferred domain."""
    key = _clean_str(source_type).lower()
    if key in _TASK_TYPE_BY_SOURCE_TYPE:
        return _TASK_TYPE_BY_SOURCE_TYPE[key]
    return _TASK_TYPE_BY_DOMAIN.get(domain, "classification")


def extract_entities(text: str, row: Mapping[str, Any]) -> dict[str, Any]:
    """Pull structured identifiers from text and explicit silver columns."""
    entities: dict[str, Any] = {}

    cves = sorted({m.upper() for m in _CVE_RE.findall(text)})
    explicit_cve = _clean_str(row.get("cve_id"))
    if explicit_cve:
        cves = sorted(set(cves) | {explicit_cve.upper()})
    if cves:
        entities["cve_ids"] = cves

    cwes = sorted({m.upper() for m in _CWE_RE.findall(text)})
    explicit_cwe = _clean_str(row.get("cwe_id"))
    if explicit_cwe:
        cwes = sorted(set(cwes) | {explicit_cwe.upper()})
    if cwes:
        entities["cwe_ids"] = cwes

    urls = sorted(set(_URL_RE.findall(text)))[:25]
    explicit_url = _clean_str(row.get("url"))
    if explicit_url:
        urls = sorted(set(urls) | {explicit_url})[:25]
    if urls:
        entities["urls"] = urls

    for key in ("domain", "ip", "package_name", "ecosystem"):
        value = _clean_str(row.get(key))
        if value:
            entities[key] = value

    return entities


def extract_mitre_ids(text: str, row: Mapping[str, Any]) -> list[str]:
    """Collect MITRE ATT&CK technique ids from the explicit column and text."""
    ids = {m.upper() for m in _MITRE_RE.findall(text)}
    explicit = _clean_str(row.get("mitre_technique_id"))
    if explicit:
        ids.add(explicit.upper())
    return sorted(ids)


def compute_dedup_hash(normalized_text: str) -> str:
    """Stable content hash used for cross-source exact deduplication."""
    key = _WHITESPACE_RE.sub(" ", normalized_text.lower()).strip()
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def compute_quality_score(
    *,
    normalized_text: str,
    label: str,
    has_identifier: bool,
    severity: str,
) -> float:
    """Heuristic 0..1 quality score favoring informative, labeled content."""
    if not normalized_text:
        return 0.0
    score = 0.5
    length = len(normalized_text)
    if length >= 40:
        score += 0.15
    if length >= 200:
        score += 0.10
    if length < 10:
        score -= 0.30
    if label and label.lower() not in {"unknown", ""}:
        score += 0.10
    if has_identifier:
        score += 0.10
    if severity and severity.lower() not in {"unknown", ""}:
        score += 0.05
    return round(max(0.0, min(1.0, score)), 4)


def assign_split(
    record_id: str,
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> str:
    """Deterministic, seeded train/val/test assignment.

    Uses a stable SHA-256 hash of ``seed:record_id`` so the same record always
    lands in the same split for a given seed, independent of input ordering.
    """
    train_ratio, val_ratio, _ = ratios
    digest = hashlib.sha256(f"{seed}:{record_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + val_ratio:
        return "val"
    return "test"


def _build_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in _METADATA_CANDIDATES:
        value = _clean_str(row.get(key))
        if value:
            metadata[key] = value
    # Preserve any pre-existing structured features payload if present.
    features = row.get("features_json")
    if isinstance(features, str) and features.strip() and features.strip() not in {"{}", "[]"}:
        try:
            metadata["features"] = json.loads(features)
        except (ValueError, TypeError):
            metadata["features"] = features
    elif isinstance(features, (dict, list)) and features:
        metadata["features"] = features
    return metadata


def silver_row_to_record(
    row: Mapping[str, Any],
    *,
    source_id: str,
    source_name: str,
    source_url: str = "",
    source_license: str = "",
    seed: int = 42,
    row_index: int = 0,
    processed_at: str | None = None,
) -> UnifiedGoldRecord | None:
    """Normalize a single silver row into a :class:`UnifiedGoldRecord`.

    Returns ``None`` when the row carries no usable text content (both
    ``raw_text`` and any reconstructable text are empty), so the builder can
    drop empties before quality filtering.
    """
    processed_at = processed_at or datetime.now(UTC).isoformat()

    raw_text = _clean_str(row.get("raw_text"))
    if not raw_text:
        # Fall back to a compact rendering of structured features so feature-only
        # silver rows still carry inspectable content.
        features = row.get("features_json")
        if isinstance(features, str):
            raw_text = _clean_str(features)
        elif isinstance(features, (dict, list)):
            raw_text = json.dumps(features, ensure_ascii=False, sort_keys=True)
    raw_text = raw_text if raw_text and raw_text not in {"{}", "[]"} else ""

    normalized_text = clean_text(raw_text)
    if not normalized_text:
        return None

    domain = infer_domain(
        source_id,
        row.get("source_dataset"),
        row.get("main_category"),
        row.get("attack_name"),
        row.get("source_type"),
    )
    category = _clean_str(row.get("main_category")) or domain
    subcategory = (
        _clean_str(row.get("attack_family"))
        or _clean_str(row.get("attack_name"))
        or category
    )
    source_type = _clean_str(row.get("source_type"))
    task_type = infer_task_type(source_type, domain)

    label = _clean_str(row.get("label")) or "unknown"
    severity = _clean_str(row.get("severity")) or "unknown"
    cwe = _clean_str(row.get("cwe_id")) or None
    cve = _clean_str(row.get("cve_id")) or None

    mitre_ids = extract_mitre_ids(normalized_text, row)
    entities = extract_entities(normalized_text, row)
    metadata = _build_metadata(row)

    silver_record_id = _clean_str(row.get("record_id")) or f"row{row_index}"
    fingerprint = hashlib.sha1(
        f"{source_id}|{silver_record_id}|{normalized_text}".encode("utf-8")
    ).hexdigest()[:16]
    record_id = f"{source_id}::{fingerprint}"

    has_identifier = bool(cwe or cve or mitre_ids or entities.get("cve_ids"))
    quality_score = compute_quality_score(
        normalized_text=normalized_text,
        label=label,
        has_identifier=has_identifier,
        severity=severity,
    )
    dedup_hash = compute_dedup_hash(normalized_text)

    return UnifiedGoldRecord(
        record_id=record_id,
        source_id=source_id,
        source_name=source_name or source_id,
        source_url=source_url or lookup_source_url(source_id, source_name),
        source_license=source_license or _clean_str(row.get("license")) or "unknown",
        collected_at=_clean_str(row.get("ingested_at")) or _clean_str(row.get("timestamp")) or None,
        processed_at=processed_at,
        domain=domain,
        category=category,
        subcategory=subcategory,
        task_type=task_type,
        raw_text=raw_text,
        normalized_text=normalized_text,
        label=label,
        severity=severity,
        cwe=cwe,
        cve=cve,
        mitre_attack_ids=mitre_ids,
        language="en",
        entities=entities,
        metadata=metadata,
        quality_score=quality_score,
        dedup_hash=dedup_hash,
        split=assign_split(record_id, seed=seed),
    )


def is_valid_domain(domain: str) -> bool:
    """Return whether ``domain`` is one of the canonical gold domains."""
    return domain in DOMAINS


def is_valid_split(split: str) -> bool:
    """Return whether ``split`` is one of ``train``/``val``/``test``."""
    return split in VALID_SPLITS
