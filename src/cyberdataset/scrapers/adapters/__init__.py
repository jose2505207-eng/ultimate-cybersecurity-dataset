"""Built-in public-source adapters and a name-based registry.

All adapters here use public, key-optional sources. Where a source supports an
API key (e.g. NVD), the key only raises rate limits -- it is never required.
"""

from __future__ import annotations

from collections.abc import Callable

from cyberdataset.scrapers.adapters.cisa_kev import CisaKevAdapter
from cyberdataset.scrapers.adapters.nvd import NvdCveAdapter
from cyberdataset.scrapers.adapters.osv import OsvAdapter
from cyberdataset.scrapers.base import BaseSourceAdapter

#: Maps a stable CLI source name to an adapter factory.
ADAPTER_REGISTRY: dict[str, Callable[[], BaseSourceAdapter]] = {
    "cisa_kev": CisaKevAdapter,
    "osv": OsvAdapter,
    "nvd": NvdCveAdapter,
}

__all__ = [
    "ADAPTER_REGISTRY",
    "CisaKevAdapter",
    "NvdCveAdapter",
    "OsvAdapter",
    "build_adapters",
    "available_sources",
]


def available_sources() -> list[str]:
    """Return the sorted list of known source names."""
    return sorted(ADAPTER_REGISTRY)


def build_adapters(names: list[str]) -> list[BaseSourceAdapter]:
    """Instantiate adapters for the given source names.

    Raises ``KeyError`` (with the list of valid names) for an unknown source.
    """
    adapters: list[BaseSourceAdapter] = []
    for name in names:
        key = name.strip().lower()
        if key not in ADAPTER_REGISTRY:
            raise KeyError(f"unknown source {name!r}; valid: {available_sources()}")
        adapters.append(ADAPTER_REGISTRY[key]())
    return adapters
