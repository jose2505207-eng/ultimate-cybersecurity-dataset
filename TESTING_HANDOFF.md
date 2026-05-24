# Ultimate Cybersecurity Dataset — Silver Layer Testing Handoff

## Project Status

This repository contains the silver-layer version of the Ultimate Cybersecurity Dataset project.

This is not the final gold benchmark yet.

The goal of this phase is to validate that the normalized silver modules are clean, usable, documented, and ready to become the final gold benchmark.

## Canonical Dataset Path

Use this folder as the source of truth:

data/silver_normalized/

Do not use silver_clean. The active silver layer is data/silver_normalized/.

## Key Files

data/silver_normalized/silver_manifest.csv
docs/silver_layer_report.md
docs/schema_v1.md
docs/label_taxonomy.md
requirements.txt
tests/
scripts/

## Setup

Run:

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

## Run Tests

Run:

python -m compileall -q scripts tests
python -m pytest tests/ -q

## Testing Goals

Please validate:

1. Every manifest row points to a real output file.
2. Every normalized file follows the unified schema.
3. Required columns are present across all modules.
4. Labels and binary_label values are consistent.
5. Row counts by module, source type, category, and label are reasonable.
6. Duplicates and near-duplicates are identified.
7. Any possible train/test leakage risk is documented.
8. Broken, weak, or suspicious modules are flagged before gold-layer creation.

## Expected Output From Testing

Please return a short testing report with:

- Tests passed/failed
- Total usable rows
- Rows per module
- Rows per category
- Rows per label
- Schema issues
- Label issues
- Missing file issues
- Duplicate/leakage concerns
- Recommended fixes before building the gold layer

## Gold Layer Comes Later

The gold layer should only be built after the silver layer passes review.

Gold will eventually include:

data/gold/final_benchmark.parquet
data/gold/final_benchmark.csv.gz
data/gold/train.parquet
data/gold/validation.parquet
data/gold/test.parquet
data/gold/gold_manifest.csv
docs/gold_benchmark_report.md
