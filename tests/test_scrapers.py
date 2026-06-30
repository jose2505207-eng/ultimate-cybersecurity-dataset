"""Tests for the fresh-data scraper: mocked HTTP, parsing, cache, rate limiting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cyberdataset.scrapers.adapters import available_sources, build_adapters
from cyberdataset.scrapers.adapters.cisa_kev import CisaKevAdapter
from cyberdataset.scrapers.base import FreshDataScraper
from cyberdataset.scrapers.http import HttpClient, HttpError

KEV_PAYLOAD = {
    "title": "CISA Catalog of Known Exploited Vulnerabilities",
    "vulnerabilities": [
        {
            "cveID": "CVE-2024-1234",
            "vendorProject": "ExampleVendor",
            "product": "ExampleProduct",
            "vulnerabilityName": "Example RCE",
            "dateAdded": "2024-01-15",
            "shortDescription": "Remote code execution in ExampleProduct.",
            "requiredAction": "Apply updates.",
            "dueDate": "2024-02-05",
            "knownRansomwareCampaignUse": "Known",
        },
        {
            "cveID": "CVE-2024-5678",
            "vendorProject": "OtherVendor",
            "product": "OtherProduct",
            "vulnerabilityName": "Example SQLi",
            "dateAdded": "2024-01-20",
            "shortDescription": "SQL injection in OtherProduct.",
            "requiredAction": "Apply updates.",
            "dueDate": "2024-02-10",
            "knownRansomwareCampaignUse": "Unknown",
        },
    ],
}


def _transport_from(payload: dict, counter: list[int]):
    def transport(url, data, headers, timeout):
        counter.append(url)
        return json.dumps(payload).encode("utf-8")

    return transport


# --------------------------------------------------------------------------- #
# Adapter registry                                                            #
# --------------------------------------------------------------------------- #


def test_registry_exposes_known_sources():
    assert {"cisa_kev", "osv", "nvd"}.issubset(set(available_sources()))


def test_unknown_source_raises():
    with pytest.raises(KeyError):
        build_adapters(["not_a_real_source"])


# --------------------------------------------------------------------------- #
# Adapter fetch/parse with a mocked transport                                 #
# --------------------------------------------------------------------------- #


def test_cisa_kev_fetch_and_parse():
    calls: list[str] = []
    client = HttpClient(transport=_transport_from(KEV_PAYLOAD, calls), rate_limit_seconds=0)
    adapter = CisaKevAdapter()

    raw = adapter.fetch(client)
    records = adapter.parse(raw)

    assert len(calls) == 1
    assert len(records) == 2
    assert records[0]["cve_id"] == "CVE-2024-1234"
    assert records[0]["known_ransomware_use"] == "Known"


def test_normalize_to_bronze_attaches_provenance():
    adapter = CisaKevAdapter()
    native = adapter.parse(KEV_PAYLOAD)
    bronze = adapter.normalize_to_bronze(native)
    assert all(item["source_id"] == "cisa_kev" for item in bronze)
    assert all("fetched_at" in item and "raw" in item for item in bronze)


# --------------------------------------------------------------------------- #
# Orchestrator end-to-end (mocked client) writes bronze + manifest            #
# --------------------------------------------------------------------------- #


def test_scraper_run_writes_outputs(tmp_path):
    calls: list[str] = []
    client = HttpClient(transport=_transport_from(KEV_PAYLOAD, calls), rate_limit_seconds=0)
    scraper = FreshDataScraper([CisaKevAdapter()], client=client)

    manifest = scraper.run(tmp_path, limit=1)

    assert manifest["total_records"] == 1  # limit applied
    entry = manifest["sources"][0]
    assert entry["errors"] == []
    raw_path = Path(entry["output_path"])
    assert raw_path.exists() and raw_path.name == "raw.jsonl"
    assert (raw_path.parent / "metadata.json").exists()
    assert (tmp_path / "manifest.json").exists()


def test_scraper_records_errors_without_aborting(tmp_path):
    def failing_transport(url, data, headers, timeout):
        raise OSError("network down")

    client = HttpClient(transport=failing_transport, rate_limit_seconds=0,
                        max_retries=1, backoff_factor=0, sleep=lambda _s: None)
    scraper = FreshDataScraper([CisaKevAdapter()], client=client)

    manifest = scraper.run(tmp_path)
    entry = manifest["sources"][0]
    assert entry["errors"]  # the failure is captured, not raised
    assert manifest["total_records"] == 0


# --------------------------------------------------------------------------- #
# HttpClient cache + rate limiting                                            #
# --------------------------------------------------------------------------- #


def test_cache_avoids_second_network_call(tmp_path):
    calls: list[str] = []
    client = HttpClient(
        transport=_transport_from(KEV_PAYLOAD, calls),
        rate_limit_seconds=0,
        cache_dir=tmp_path / "cache",
    )
    first = client.fetch_json("https://example.test/kev.json")
    second = client.fetch_json("https://example.test/kev.json")

    assert first == second
    assert len(calls) == 1  # second call served from cache
    assert client.cache_hits == 1


def test_rate_limit_sleeps_between_requests():
    slept: list[float] = []
    ticks = iter([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    client = HttpClient(
        transport=lambda *a: b"{}",
        rate_limit_seconds=2.0,
        sleep=slept.append,
        _clock=lambda: next(ticks),
    )
    client.fetch("https://example.test/a")  # first request: no wait
    client.fetch("https://example.test/b")  # second: must wait ~2s
    assert slept and slept[0] > 0


def test_http_error_after_retries():
    def failing_transport(url, data, headers, timeout):
        raise OSError("boom")

    client = HttpClient(transport=failing_transport, rate_limit_seconds=0,
                        max_retries=2, backoff_factor=0, sleep=lambda _s: None)
    with pytest.raises(HttpError):
        client.fetch("https://example.test/x")
