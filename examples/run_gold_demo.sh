#!/usr/bin/env bash
#
# Tiny end-to-end demo of the gold unified layer. Builds a small gold dataset
# from the bundled example silver fixtures (examples/silver_sample/) -- no need
# for the full ~200GB bronze/silver data.
#
# Usage:
#   bash examples/run_gold_demo.sh
#
set -euo pipefail

# Resolve repo root from this script's location so it runs from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="src:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

SILVER_DIR="examples/silver_sample"
OUT_DIR="examples/gold_demo"

echo "==> Building gold unified demo from ${SILVER_DIR}"
"${PYTHON_BIN}" -m cyberdataset.gold.build_gold \
  --silver-dir "${SILVER_DIR}" \
  --out-dir "${OUT_DIR}" \
  --min-quality 0.40 \
  --seed 42

echo
echo "==> Manifest summary (${OUT_DIR}/manifest.json):"
"${PYTHON_BIN}" - "${OUT_DIR}/manifest.json" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
for key in ("total_records", "duplicates_removed", "mean_quality_score",
            "counts_by_domain", "counts_by_split", "counts_by_label"):
    print(f"  {key}: {manifest[key]}")
PY

echo
echo "==> Sample gold record (first line of ${OUT_DIR}/gold_unified.jsonl):"
head -n 1 "${OUT_DIR}/gold_unified.jsonl"
echo
echo "Demo complete. See ${OUT_DIR}/dataset_card.md for the generated dataset card."
