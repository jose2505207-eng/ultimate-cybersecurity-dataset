"""Thin CLI wrapper around :mod:`cyberdataset.gold.build_gold`.

Provided for parity with the repository's ``scripts/``-style entry points
(e.g. ``scripts/build_gold_benchmark.py``). It simply delegates to the package
CLI so both invocation styles work:

    python -m cyberdataset.gold.build_gold --silver-dir data/silver_normalized --out-dir data/gold
    python -m scripts.build_gold_unified  --silver-dir data/silver_normalized --out-dir data/gold
"""

from __future__ import annotations

from cyberdataset.gold.build_gold import main

if __name__ == "__main__":
    raise SystemExit(main())
