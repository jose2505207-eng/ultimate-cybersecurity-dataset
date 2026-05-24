"""Unified silver schema and controlled vocabularies."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


SCHEMA_VERSION = "1.0.0"

SOURCE_TYPES = {
    "network_flow",
    "malware_features",
    "malware_binary_metadata",
    "vulnerable_code",
    "smart_contract_code",
    "smart_contract_bytecode_metadata",
    "web_app_request",
    "api_request",
    "auth_event",
    "host_telemetry",
    "sysmon_event",
    "dns_event",
    "cti_taxonomy",
    "vulnerability_advisory",
    "package_metadata",
    "phishing_url",
    "phishing_email",
    "prompt_text",
    "llm_io_pair",
    "defi_incident",
    "ics_telemetry",
    "iot_telemetry",
    "mobile_app_metadata",
    "insider_event",
    "other",
}

LABELS = {
    "benign",
    "malicious",
    "phishing",
    "benign_url",
    "vulnerable_dependency",
    "vulnerable_code",
    "non_vulnerable_code",
    "malicious_package",
    "attack_technique",
    "attack_pattern",
    "vulnerability_advisory",
    "ai_security_risk",
    "malicious_prompt",
    "benign_prompt",
    "jailbreak_prompt",
    "exploit_incident",
    "intrusion",
    "anomaly",
    "unknown",
}

CATEGORIES = [
    "Vulnerable Code & Software Weaknesses",
    "Web Application Security",
    "API Security",
    "Network Intrusion & Traffic Attacks",
    "Identity & Credential Attacks",
    "Malware & PE/Memory Features",
    "Phishing, Social Engineering & Fraud",
    "Threat Intelligence, CVE, Advisory & Taxonomy",
    "Supply Chain & Open Source Package Security",
    "IoT, Embedded & Hardware Attacks",
    "ICS, OT & Critical Infrastructure Attacks",
    "Cloud, SaaS & Identity Abuse",
    "Endpoint, Host & Windows/Sysmon Telemetry",
    "Insider Threat & User Behavior Analytics",
    "Cryptocurrency & Blockchain Attacks",
    "AI, LLM & ML Security",
    "DNS, Exfiltration & C2 Abuse",
    "Mobile Security",
    "Dark Web, Abuse & Underground Activity",
    "Miscellaneous / Needs Review",
]

SEVERITIES = {"critical", "high", "medium", "low", "info", "unknown"}

COLUMN_ORDER = [
    "record_id",
    "source_dataset",
    "source_type",
    "main_category",
    "attack_name",
    "attack_family",
    "label",
    "binary_label",
    "mitre_tactic",
    "mitre_technique_id",
    "cwe_id",
    "cve_id",
    "severity",
    "severity_score",
    "platform",
    "language",
    "protocol",
    "ecosystem",
    "package_name",
    "url",
    "domain",
    "ip",
    "timestamp",
    "raw_text",
    "features_json",
    "source_file",
    "license",
    "notes",
    "schema_version",
    "ingested_at",
]

NULLABLE_COLUMNS = {
    "attack_name",
    "attack_family",
    "mitre_tactic",
    "mitre_technique_id",
    "cwe_id",
    "cve_id",
    "severity",
    "severity_score",
    "platform",
    "language",
    "protocol",
    "ecosystem",
    "package_name",
    "url",
    "domain",
    "ip",
    "timestamp",
    "raw_text",
    "features_json",
    "notes",
}


class SilverRecord(BaseModel):
    """Pydantic model for one unified silver row."""

    model_config = ConfigDict(extra="forbid")

    record_id: str
    source_dataset: str
    source_type: str
    main_category: str
    attack_name: str | None = None
    attack_family: str | None = None
    label: str
    binary_label: int = Field(ge=0, le=1)
    mitre_tactic: str | None = None
    mitre_technique_id: str | None = None
    cwe_id: str | None = None
    cve_id: str | None = None
    severity: str | None = None
    severity_score: float | None = None
    platform: str | None = None
    language: str | None = None
    protocol: str | None = None
    ecosystem: str | None = None
    package_name: str | None = None
    url: str | None = None
    domain: str | None = None
    ip: str | None = None
    timestamp: datetime | None = None
    raw_text: str | None = None
    features_json: str | None = None
    source_file: str
    license: str
    notes: str | None = None
    schema_version: str
    ingested_at: datetime

    @field_validator("source_type")
    @classmethod
    def _valid_source_type(cls, value: str) -> str:
        if value not in SOURCE_TYPES:
            raise ValueError(f"invalid source_type: {value}")
        return value

    @field_validator("main_category")
    @classmethod
    def _valid_category(cls, value: str) -> str:
        if value not in CATEGORIES:
            raise ValueError(f"invalid main_category: {value}")
        return value

    @field_validator("label")
    @classmethod
    def _valid_label(cls, value: str) -> str:
        if value not in LABELS:
            raise ValueError(f"invalid label: {value}")
        return value

    @field_validator("schema_version")
    @classmethod
    def _valid_schema_version(cls, value: str) -> str:
        if value != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        return value

    @field_validator("severity")
    @classmethod
    def _valid_severity(cls, value: str | None) -> str | None:
        if value is not None and value not in SEVERITIES:
            raise ValueError(f"invalid severity: {value}")
        return value


def assert_schema_sync() -> None:
    """Raise if model fields and ordered columns diverge."""

    fields = list(SilverRecord.model_fields.keys())
    if fields != COLUMN_ORDER:
        raise AssertionError(f"schema mismatch: model={fields} columns={COLUMN_ORDER}")
