"""OSV (Open Source Vulnerabilities) adapter.

Uses the public OSV query API (https://api.osv.dev/v1/query), which requires no
API key. By default it queries vulnerabilities for a well-known package so the
adapter returns useful data out of the box; the package/ecosystem can be
overridden via ``SourceConfig.extra``.
"""

from __future__ import annotations

from typing import Any

from cyberdataset.scrapers.base import BaseSourceAdapter, SourceConfig
from cyberdataset.scrapers.http import HttpClient

OSV_QUERY_URL = "https://api.osv.dev/v1/query"


class OsvAdapter(BaseSourceAdapter):
    """Fetch and parse OSV vulnerabilities via the public query API."""

    def __init__(self, config: SourceConfig | None = None) -> None:
        super().__init__(
            config
            or SourceConfig(
                source_id="osv",
                source_name="OSV (Open Source Vulnerabilities)",
                source_url="https://osv.dev/",
                license_note="OSV records are CC-BY-4.0; upstream advisory terms may vary.",
                rate_limit_seconds=1.0,
                extra={"ecosystem": "PyPI", "package": "django"},
            )
        )

    def fetch(self, client: HttpClient, *, limit: int | None = None) -> dict[str, Any]:
        ecosystem = self.config.extra.get("ecosystem", "PyPI")
        package = self.config.extra.get("package", "django")
        body = {"package": {"ecosystem": ecosystem, "name": package}}
        return client.post_json(OSV_QUERY_URL, body)

    def parse(self, raw: dict[str, Any]) -> list[dict[str, Any]]:
        vulns = raw.get("vulns", []) if isinstance(raw, dict) else []
        records: list[dict[str, Any]] = []
        for vuln in vulns:
            aliases = vuln.get("aliases", []) or []
            cve_id = next((a for a in aliases if str(a).upper().startswith("CVE-")), None)
            records.append(
                {
                    "osv_id": vuln.get("id"),
                    "cve_id": cve_id,
                    "aliases": aliases,
                    "summary": vuln.get("summary"),
                    "details": vuln.get("details"),
                    "published": vuln.get("published"),
                    "modified": vuln.get("modified"),
                    "affected": vuln.get("affected", []),
                    "references": vuln.get("references", []),
                }
            )
        return records
