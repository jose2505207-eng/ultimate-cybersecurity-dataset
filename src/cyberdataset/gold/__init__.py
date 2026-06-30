"""Gold unified layer.

This subpackage builds a single, AI-training-ready "gold unified" dataset from
the per-source silver datasets under ``data/silver_normalized/``.

Unlike :mod:`cyberdataset.build.build_gold` (which produces the legacy
``gold_unified`` benchmark) and ``scripts/build_gold_benchmark.py`` (which
produces the multi-head evaluation benchmark), this layer normalizes every
silver row into one flat canonical schema that spans all cybersecurity domains,
deduplicates it with a stable hash, scores quality, and assigns deterministic
train/val/test splits.

Primary entry points:

* :func:`cyberdataset.gold.build_gold.build_gold` -- the Silver -> Gold builder.
* ``python -m cyberdataset.gold.build_gold`` -- the CLI.
* :func:`cyberdataset.gold.validate.validate_gold_records` -- quality checks.
"""

from __future__ import annotations

from cyberdataset.gold.schema import (
    DOMAINS,
    GOLD_SCHEMA_VERSION,
    GOLD_UNIFIED_COLUMNS,
    VALID_SPLITS,
    UnifiedGoldRecord,
)

__all__ = [
    "DOMAINS",
    "GOLD_SCHEMA_VERSION",
    "GOLD_UNIFIED_COLUMNS",
    "VALID_SPLITS",
    "UnifiedGoldRecord",
]
