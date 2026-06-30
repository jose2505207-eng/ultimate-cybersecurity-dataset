"""Fresh-data web scraper for respectful public cybersecurity source ingestion.

This package fetches *fresh* data from public, key-optional cybersecurity feeds
and APIs into the bronze layer. It is intentionally dependency-light (standard
library ``urllib`` only) and does **not** depend on any paid or proxy service.

Design goals:

* Respectful by default -- rate limiting, retry/backoff, timeouts, a descriptive
  user agent, robots.txt awareness for page scraping, and local caching.
* Public data only -- never authenticate, bypass anti-bot, or harvest PII.
* Pluggable -- new sources are added by subclassing
  :class:`cyberdataset.scrapers.base.BaseSourceAdapter`.

Entry points:

* :class:`cyberdataset.scrapers.base.FreshDataScraper` -- the orchestrator.
* ``python -m cyberdataset.scrapers.fetch_fresh`` -- the CLI.
"""

from __future__ import annotations

from cyberdataset.scrapers.base import (
    BaseSourceAdapter,
    FreshDataScraper,
    SourceConfig,
)
from cyberdataset.scrapers.http import HttpClient, HttpError

__all__ = [
    "BaseSourceAdapter",
    "FreshDataScraper",
    "SourceConfig",
    "HttpClient",
    "HttpError",
]
