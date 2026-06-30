"""NVD CVE feed adapter (NVD REST API 2.0).

The NVD API works without an API key; supplying ``NVD_API_KEY`` (optional) only
raises the rate limit. The adapter requests a single bounded page so it stays
respectful by default.
"""

from __future__ import annotations

import os
import urllib.parse
from typing import Any

from cyberdataset.scrapers.base import BaseSourceAdapter, SourceConfig
from cyberdataset.scrapers.http import HttpClient

NVD_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
MAX_RESULTS_PER_PAGE = 2000


class NvdCveAdapter(BaseSourceAdapter):
    """Fetch and parse recent CVEs from the NVD REST API."""

    def __init__(self, config: SourceConfig | None = None) -> None:
        super().__init__(
            config
            or SourceConfig(
                source_id="nvd_cve",
                source_name="NVD CVE",
                source_url=NVD_API_URL,
                license_note="NVD data is U.S. Government public-domain; see NVD terms of use.",
                rate_limit_seconds=6.0,  # NVD recommends >=6s between keyless requests
                max_retries=5,
                requires_api_key=False,
                api_key_env="NVD_API_KEY",
            )
        )

    def fetch(self, client: HttpClient, *, limit: int | None = None) -> dict[str, Any]:
        results_per_page = min(limit or MAX_RESULTS_PER_PAGE, MAX_RESULTS_PER_PAGE)
        params = {"resultsPerPage": results_per_page, "startIndex": 0}
        url = f"{NVD_API_URL}?{urllib.parse.urlencode(params)}"
        headers: dict[str, str] = {}
        api_key = os.getenv(self.config.api_key_env or "NVD_API_KEY")
        if api_key:
            headers["apiKey"] = api_key
        return client.fetch_json(url, headers=headers)

    def parse(self, raw: dict[str, Any]) -> list[dict[str, Any]]:
        vulnerabilities = raw.get("vulnerabilities", []) if isinstance(raw, dict) else []
        records: list[dict[str, Any]] = []
        for entry in vulnerabilities:
            cve = entry.get("cve", {}) if isinstance(entry, dict) else {}
            descriptions = cve.get("descriptions", []) or []
            english = next(
                (d.get("value") for d in descriptions if d.get("lang") == "en"),
                descriptions[0].get("value") if descriptions else None,
            )
            metrics = cve.get("metrics", {}) or {}
            severity = _first_severity(metrics)
            records.append(
                {
                    "cve_id": cve.get("id"),
                    "published": cve.get("published"),
                    "last_modified": cve.get("lastModified"),
                    "vuln_status": cve.get("vulnStatus"),
                    "description": english,
                    "severity": severity,
                    "weaknesses": _weaknesses(cve.get("weaknesses", [])),
                }
            )
        return records


def _first_severity(metrics: dict[str, Any]) -> str | None:
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        items = metrics.get(key) or []
        if items:
            data = items[0].get("cvssData", {})
            return data.get("baseSeverity") or items[0].get("baseSeverity")
    return None


def _weaknesses(weaknesses: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for weakness in weaknesses or []:
        for desc in weakness.get("description", []) or []:
            value = desc.get("value")
            if value and value not in out:
                out.append(value)
    return out
