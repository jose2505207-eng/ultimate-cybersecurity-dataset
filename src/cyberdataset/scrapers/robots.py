"""robots.txt awareness for respectful page scraping.

Used only for HTML page scraping; structured public APIs/feeds (CISA KEV, OSV,
NVD) are explicit machine-readable distributions and are fetched directly. The
checker fails *open* on network errors but logs that it could not verify, so a
missing robots.txt never silently blocks legitimate public data.
"""

from __future__ import annotations

import urllib.error
import urllib.robotparser
from urllib.parse import urljoin, urlparse


def robots_url_for(url: str) -> str:
    """Return the robots.txt URL for the host serving ``url``."""
    parsed = urlparse(url)
    return urljoin(f"{parsed.scheme}://{parsed.netloc}", "/robots.txt")


def is_allowed(url: str, user_agent: str, *, default: bool = True) -> bool:
    """Return whether ``user_agent`` may fetch ``url`` per the host robots.txt.

    Returns ``default`` (allow) when robots.txt cannot be retrieved, so transient
    network issues do not block access to clearly public resources.
    """
    parser = urllib.robotparser.RobotFileParser()
    parser.set_url(robots_url_for(url))
    try:
        parser.read()
    except (urllib.error.URLError, OSError):
        return default
    # When robots.txt is empty/unparsed, can_fetch returns True by default.
    return parser.can_fetch(user_agent, url)
