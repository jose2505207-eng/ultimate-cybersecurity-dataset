"""CISA Known Exploited Vulnerabilities (KEV) catalog adapter.

Public JSON feed, no API key required. The KEV catalog lists CVEs that CISA has
observed being actively exploited in the wild.
"""

from __future__ import annotations

from typing import Any

from cyberdataset.scrapers.base import BaseSourceAdapter, SourceConfig
from cyberdataset.scrapers.http import HttpClient

KEV_FEED_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"


class CisaKevAdapter(BaseSourceAdapter):
    """Fetch and parse the CISA KEV catalog JSON feed."""

    def __init__(self, config: SourceConfig | None = None) -> None:
        super().__init__(
            config
            or SourceConfig(
                source_id="cisa_kev",
                source_name="CISA Known Exploited Vulnerabilities",
                source_url=KEV_FEED_URL,
                license_note="CISA KEV catalog is U.S. Government public-domain content.",
                rate_limit_seconds=1.0,
            )
        )

    def fetch(self, client: HttpClient, *, limit: int | None = None) -> dict[str, Any]:
        return client.fetch_json(self.source_url)

    def parse(self, raw: dict[str, Any]) -> list[dict[str, Any]]:
        vulnerabilities = raw.get("vulnerabilities", []) if isinstance(raw, dict) else []
        records: list[dict[str, Any]] = []
        for item in vulnerabilities:
            records.append(
                {
                    "cve_id": item.get("cveID"),
                    "vendor_project": item.get("vendorProject"),
                    "product": item.get("product"),
                    "vulnerability_name": item.get("vulnerabilityName"),
                    "date_added": item.get("dateAdded"),
                    "short_description": item.get("shortDescription"),
                    "required_action": item.get("requiredAction"),
                    "due_date": item.get("dueDate"),
                    "known_ransomware_use": item.get("knownRansomwareCampaignUse"),
                }
            )
        return records
