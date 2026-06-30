"""Base interfaces for the fresh-data scraper: config, adapter, orchestrator."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cyberdataset.scrapers.http import DEFAULT_USER_AGENT, HttpClient, HttpError
from cyberdataset.scrapers.robots import is_allowed


@dataclass
class SourceConfig:
    """Configuration for a single public source adapter.

    Attributes mirror respectful-scraping knobs. ``extra`` carries adapter
    specific options (e.g. an OSV query package) without changing this contract.
    """

    source_id: str
    source_name: str
    source_url: str
    license_note: str = "Verify upstream terms before redistribution."
    request_timeout: float = 30.0
    max_retries: int = 4
    backoff_factor: float = 1.5
    rate_limit_seconds: float = 1.0
    user_agent: str = DEFAULT_USER_AGENT
    respect_robots: bool = False  # APIs/feeds default off; HTML adapters set True
    requires_api_key: bool = False
    api_key_env: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class BaseSourceAdapter(ABC):
    """Abstract base for a public-source fresh-data adapter.

    Subclasses implement :meth:`fetch` and :meth:`parse`. ``normalize_to_bronze``
    and ``save`` have sensible defaults but may be overridden. Each adapter
    exposes ``source_id``, ``source_name``, and ``source_url`` via its config.
    """

    #: Subclasses set this. The orchestrator instantiates adapters with no args.
    config: SourceConfig

    def __init__(self, config: SourceConfig | None = None) -> None:
        if config is not None:
            self.config = config
        if not getattr(self, "config", None):
            raise ValueError(f"{type(self).__name__} requires a SourceConfig")

    @property
    def source_id(self) -> str:
        return self.config.source_id

    @property
    def source_name(self) -> str:
        return self.config.source_name

    @property
    def source_url(self) -> str:
        return self.config.source_url

    # --- adapter contract --------------------------------------------------- #

    @abstractmethod
    def fetch(self, client: HttpClient, *, limit: int | None = None) -> Any:
        """Retrieve raw payload(s) from the public source."""

    @abstractmethod
    def parse(self, raw: Any) -> list[dict[str, Any]]:
        """Parse the raw payload into a list of source-native record dicts."""

    def normalize_to_bronze(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Wrap native records with consistent bronze provenance fields.

        The default keeps the native record intact under ``raw`` and attaches
        source attribution so downstream silver ingestion can trace every row.
        Adapters may override to flatten or enrich.
        """
        fetched_at = datetime.now(UTC).isoformat()
        bronze: list[dict[str, Any]] = []
        for record in records:
            bronze.append(
                {
                    "source_id": self.source_id,
                    "source_name": self.source_name,
                    "source_url": self.source_url,
                    "license_note": self.config.license_note,
                    "fetched_at": fetched_at,
                    "raw": record,
                }
            )
        return bronze

    def save(
        self,
        records: list[dict[str, Any]],
        out_dir: Path | str,
        *,
        fetched_at: datetime | None = None,
    ) -> dict[str, Any]:
        """Write bronze records to ``out_dir/<source_id>/<date>/`` with metadata.

        Returns a manifest fragment describing what was written.
        """
        fetched_at = fetched_at or datetime.now(UTC)
        date_str = fetched_at.strftime("%Y-%m-%d")
        target_dir = Path(out_dir) / self.source_id / date_str
        target_dir.mkdir(parents=True, exist_ok=True)

        raw_path = target_dir / "raw.jsonl"
        with raw_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")

        metadata = {
            "source_id": self.source_id,
            "source_name": self.source_name,
            "source_url": self.source_url,
            "license_note": self.config.license_note,
            "fetched_at": fetched_at.isoformat(),
            "record_count": len(records),
            "output_path": str(raw_path),
            "requires_api_key": self.config.requires_api_key,
            "api_key_env": self.config.api_key_env,
        }
        (target_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        return metadata

    # --- convenience -------------------------------------------------------- #

    def check_robots(self, url: str) -> bool:
        """Return True if scraping ``url`` is permitted (or checks are disabled)."""
        if not self.config.respect_robots:
            return True
        return is_allowed(url, self.config.user_agent)


class FreshDataScraper:
    """Orchestrates a set of adapters into a single fresh-data run.

    Each adapter is run independently: failures are captured per source and do
    not abort the run, so a flaky feed never blocks the others.
    """

    def __init__(
        self,
        adapters: list[BaseSourceAdapter],
        *,
        cache_dir: Path | str | None = None,
        client: HttpClient | None = None,
    ) -> None:
        self.adapters = adapters
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._client = client

    def _client_for(self, adapter: BaseSourceAdapter) -> HttpClient:
        if self._client is not None:
            return self._client
        return HttpClient(
            timeout=adapter.config.request_timeout,
            max_retries=adapter.config.max_retries,
            backoff_factor=adapter.config.backoff_factor,
            rate_limit_seconds=adapter.config.rate_limit_seconds,
            user_agent=adapter.config.user_agent,
            cache_dir=self.cache_dir,
        )

    def run(
        self,
        out_dir: Path | str,
        *,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Run all adapters, writing bronze output and a combined manifest."""
        out_dir = Path(out_dir)
        fetched_at = datetime.now(UTC)
        entries: list[dict[str, Any]] = []

        for adapter in self.adapters:
            entry: dict[str, Any] = {
                "source_id": adapter.source_id,
                "source_name": adapter.source_name,
                "source_url": adapter.source_url,
                "fetched_at": fetched_at.isoformat(),
                "record_count": 0,
                "output_path": None,
                "license_note": adapter.config.license_note,
                "errors": [],
                "warnings": [],
            }
            try:
                client = self._client_for(adapter)
                if not adapter.check_robots(adapter.source_url):
                    entry["warnings"].append("robots.txt disallows scraping; source skipped")
                    entries.append(entry)
                    continue
                raw = adapter.fetch(client, limit=limit)
                native = adapter.parse(raw)
                if limit is not None and len(native) > limit:
                    native = native[:limit]
                bronze = adapter.normalize_to_bronze(native)
                metadata = adapter.save(bronze, out_dir, fetched_at=fetched_at)
                entry.update(
                    record_count=metadata["record_count"],
                    output_path=metadata["output_path"],
                )
                if metadata["record_count"] == 0:
                    entry["warnings"].append("source returned zero records")
            except HttpError as exc:
                entry["errors"].append(f"http error: {exc}")
            except (ValueError, KeyError, OSError) as exc:
                entry["errors"].append(f"{type(exc).__name__}: {exc}")
            entries.append(entry)

        manifest = {
            "generated_at": fetched_at.isoformat(),
            "out_dir": str(out_dir),
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
            "sources": entries,
            "total_records": sum(e["record_count"] for e in entries),
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        return manifest
