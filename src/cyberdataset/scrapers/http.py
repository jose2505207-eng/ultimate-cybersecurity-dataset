"""A small, respectful HTTP client built on the standard library.

Features: configurable timeout, retry with exponential backoff, a descriptive
user agent, per-client rate limiting, and on-disk response caching. The actual
network call is isolated behind a ``transport`` callable so tests can inject a
fake transport without monkeypatching ``urllib``.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_USER_AGENT = (
    "ultimate-cybersecurity-dataset-research/0.1 "
    "(+defensive security dataset; contact: repo maintainer)"
)

#: A transport takes (url, data, headers, timeout) and returns response bytes.
Transport = Callable[[str, bytes | None, dict[str, str], float], bytes]


class HttpError(RuntimeError):
    """Raised when an HTTP request ultimately fails after all retries."""


def _urllib_transport(url: str, data: bytes | None, headers: dict[str, str], timeout: float) -> bytes:
    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (public URLs only)
        return response.read()


@dataclass
class HttpClient:
    """Respectful HTTP client with retries, rate limiting, and caching.

    Parameters
    ----------
    timeout:
        Per-request timeout in seconds.
    max_retries:
        Number of additional attempts after the first failure.
    backoff_factor:
        Base seconds for exponential backoff between retries.
    rate_limit_seconds:
        Minimum spacing enforced between successive requests.
    user_agent:
        Value sent in the ``User-Agent`` header.
    cache_dir:
        When set, GET responses are cached on disk keyed by URL.
    cache_ttl_seconds:
        Maximum age of a cached response before it is refetched.
    transport:
        Network call implementation; defaults to a ``urllib`` transport.
    sleep:
        Sleep function (injectable for tests).
    """

    timeout: float = 30.0
    max_retries: int = 4
    backoff_factor: float = 1.5
    rate_limit_seconds: float = 1.0
    user_agent: str = DEFAULT_USER_AGENT
    cache_dir: Path | None = None
    cache_ttl_seconds: float = 86_400.0
    transport: Transport = _urllib_transport
    sleep: Callable[[float], None] = time.sleep
    _clock: Callable[[], float] = field(default=time.monotonic, repr=False)
    _last_request_at: float | None = field(default=None, init=False, repr=False)
    request_count: int = field(default=0, init=False)
    cache_hits: int = field(default=0, init=False)

    def _cache_path(self, url: str, data: bytes | None) -> Path | None:
        if self.cache_dir is None:
            return None
        key = hashlib.sha256((url + "|" + (data.decode("utf-8", "replace") if data else "")).encode("utf-8")).hexdigest()
        return Path(self.cache_dir) / f"{key}.cache"

    def _read_cache(self, path: Path | None) -> bytes | None:
        if path is None or not path.exists():
            return None
        age = time.time() - path.stat().st_mtime
        if age > self.cache_ttl_seconds:
            return None
        self.cache_hits += 1
        return path.read_bytes()

    def _write_cache(self, path: Path | None, payload: bytes) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    def _respect_rate_limit(self) -> None:
        if self.rate_limit_seconds <= 0:
            return
        now = self._clock()
        if self._last_request_at is not None:
            wait = self.rate_limit_seconds - (now - self._last_request_at)
            if wait > 0:
                self.sleep(wait)
                now = self._clock()
        self._last_request_at = now

    def fetch(
        self,
        url: str,
        *,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
        accept: str = "application/json,*/*",
        use_cache: bool = True,
    ) -> bytes:
        """Fetch ``url`` and return the raw response bytes.

        Honors the cache (for cacheable requests), rate limiting, and retries
        with exponential backoff. Raises :class:`HttpError` on final failure.
        """
        cacheable = use_cache and self.cache_dir is not None
        cache_path = self._cache_path(url, data) if cacheable else None
        cached = self._read_cache(cache_path) if cacheable else None
        if cached is not None:
            return cached

        request_headers = {"User-Agent": self.user_agent, "Accept": accept}
        if headers:
            request_headers.update(headers)
        if data is not None:
            request_headers.setdefault("Content-Type", "application/json")

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._respect_rate_limit()
            try:
                payload = self.transport(url, data, request_headers, self.timeout)
                self.request_count += 1
                if cacheable:
                    self._write_cache(cache_path, payload)
                return payload
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    self.sleep(self.backoff_factor * (2**attempt))
        raise HttpError(f"GET {url} failed after {self.max_retries + 1} attempts: {last_error}")

    def fetch_json(self, url: str, **kwargs: Any) -> Any:
        """Fetch ``url`` and parse the response body as JSON."""
        return json.loads(self.fetch(url, **kwargs).decode("utf-8"))

    def post_json(self, url: str, body: dict[str, Any], **kwargs: Any) -> Any:
        """POST a JSON body to ``url`` and parse the JSON response."""
        data = json.dumps(body).encode("utf-8")
        return json.loads(self.fetch(url, data=data, **kwargs).decode("utf-8"))
