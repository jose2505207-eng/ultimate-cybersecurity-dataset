"""Canonical schema for the gold unified cybersecurity layer.

The gold unified schema is intentionally flat and self-describing so it can be
consumed directly as JSONL for model training/benchmarking or loaded into a
dataframe for analytics. Nested fields (``mitre_attack_ids``, ``entities``,
``metadata``) are kept as native Python objects in memory and JSON-serialized
when written to columnar formats such as Parquet/CSV.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

GOLD_SCHEMA_VERSION = "gold-unified-v1"

#: Column order for the gold unified dataset. Every output row carries exactly
#: these keys, in this order.
GOLD_UNIFIED_COLUMNS: list[str] = [
    "record_id",
    "source_id",
    "source_name",
    "source_url",
    "source_license",
    "collected_at",
    "processed_at",
    "domain",
    "category",
    "subcategory",
    "task_type",
    "raw_text",
    "normalized_text",
    "label",
    "severity",
    "cwe",
    "cve",
    "mitre_attack_ids",
    "language",
    "entities",
    "metadata",
    "quality_score",
    "dedup_hash",
    "split",
]

#: Object-valued columns. They stay native in JSONL and are JSON-encoded for
#: columnar formats (Parquet/CSV) to keep a stable, engine-agnostic schema.
OBJECT_COLUMNS: tuple[str, ...] = ("mitre_attack_ids", "entities", "metadata")

#: Canonical security domains covered by the unified layer. Domain inference
#: maps every silver source into exactly one of these buckets.
DOMAINS: tuple[str, ...] = (
    "web_app_security",
    "network_intrusion",
    "malware_code",
    "phishing_social",
    "prompt_injection_jailbreak",
    "vulnerabilities_exposures",
    "cloud_infrastructure",
    "blockchain_web3",
    "threat_intelligence",
    "miscellaneous",
)

#: Human-readable labels for each domain, used in the dataset card.
DOMAIN_LABELS: dict[str, str] = {
    "web_app_security": "Web & application security",
    "network_intrusion": "Network intrusion & IDS/IPS",
    "malware_code": "Malware, vulnerable code & software weaknesses",
    "phishing_social": "Phishing & social engineering",
    "prompt_injection_jailbreak": "Prompt injection & LLM jailbreaks",
    "vulnerabilities_exposures": "Vulnerabilities & exposures (CVE/advisories)",
    "cloud_infrastructure": "Cloud & infrastructure security",
    "blockchain_web3": "Blockchain & web3 security",
    "threat_intelligence": "Threat intelligence (ATT&CK/CAPEC/CTI)",
    "miscellaneous": "Miscellaneous / uncategorized",
}

#: Valid split labels for the gold unified layer.
VALID_SPLITS: tuple[str, ...] = ("train", "val", "test")


class GoldValidationError(ValueError):
    """Raised when a gold unified record or dataset violates the contract."""


@dataclass
class UnifiedGoldRecord:
    """One row of the gold unified dataset.

    Attributes mirror :data:`GOLD_UNIFIED_COLUMNS`. ``mitre_attack_ids`` is a
    list, ``entities`` and ``metadata`` are dicts; everything else is a scalar.
    """

    record_id: str
    source_id: str
    source_name: str
    source_url: str
    source_license: str
    collected_at: str | None
    processed_at: str
    domain: str
    category: str
    subcategory: str
    task_type: str
    raw_text: str
    normalized_text: str
    label: str
    severity: str
    cwe: str | None
    cve: str | None
    mitre_attack_ids: list[str] = field(default_factory=list)
    language: str = "en"
    entities: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    dedup_hash: str = ""
    split: str = "train"

    def to_jsonl_dict(self) -> dict[str, Any]:
        """Return a dict with native nested types, ordered for JSONL output."""
        data = asdict(self)
        return {column: data[column] for column in GOLD_UNIFIED_COLUMNS}

    def to_row_dict(self) -> dict[str, Any]:
        """Return a dict where object columns are JSON-encoded strings.

        This keeps a single, engine-agnostic schema for Parquet/CSV writers,
        matching the repository's existing ``features_json`` convention.
        """
        row = self.to_jsonl_dict()
        for column in OBJECT_COLUMNS:
            row[column] = json.dumps(row[column], ensure_ascii=False, sort_keys=True)
        return row
