# Unified Silver Schema v1

The authoritative schema lives in `scripts/normalizers/schema.py`.

Schema version: `1.0.0`

Every silver output uses the ordered columns defined by `COLUMN_ORDER`, validates controlled vocabularies for `source_type`, `main_category`, and `label`, and writes UTC timestamps.
