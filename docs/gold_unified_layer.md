# Gold Unified Layer

The gold unified layer merges every per-source silver dataset into a single,
flat, AI-training-ready dataset with one canonical schema across all domains.
It complements (does not replace) the multi-head evaluation benchmark in
`scripts/build_gold_benchmark.py`.

## Modules

| Module | Responsibility |
| --- | --- |
| `cyberdataset.gold.schema` | Canonical schema, domains, `UnifiedGoldRecord`. |
| `cyberdataset.gold.transform` | Pure silver-row → gold-record mapping: domain/task inference, text cleaning, entity extraction, quality scoring, dedup hashing, seeded splits. |
| `cyberdataset.gold.build_gold` | Builder + CLI: discover silver, normalize, dedup, filter, write JSONL/Parquet, manifest, card. |
| `cyberdataset.gold.validate` | Quality checks for records and manifest consistency. |
| `cyberdataset.gold.dataset_card` | Renders `dataset_card.md` from the manifest. |

## Canonical schema

```
record_id, source_id, source_name, source_url, source_license, collected_at,
processed_at, domain, category, subcategory, task_type, raw_text,
normalized_text, label, severity, cwe, cve, mitre_attack_ids, language,
entities, metadata, quality_score, dedup_hash, split
```

- `mitre_attack_ids` is a list; `entities` and `metadata` are JSON objects.
  In JSONL they are native; in Parquet/CSV they are JSON-encoded strings (same
  convention as silver's `features_json`).
- `domain` is one of the canonical security domains (see
  `cyberdataset.gold.schema.DOMAINS`).
- `split` is one of `train` / `val` / `test`, assigned deterministically from a
  SHA-256 hash of `seed:record_id`.

## Domain inference

`transform.infer_domain` maps free-text signals (source id, `source_dataset`,
`main_category`, `attack_name`, `source_type`) to exactly one domain using an
ordered keyword ruleset. The first matching rule wins; unmatched rows become
`miscellaneous`. Order is deliberate: specific signals (prompt/phishing/
blockchain/malware/network) precede the broad CVE/advisory and threat-intel
buckets so, e.g., MITRE/CAPEC sources route to `threat_intelligence` while
NVD/OSV/KEV route to `vulnerabilities_exposures`.

## Quality score

A transparent 0..1 heuristic (`transform.compute_quality_score`): base 0.5,
with bonuses for text length, a non-`unknown` label, presence of identifiers
(CVE/CWE/ATT&CK), and a known severity; penalized for very short text. Empty
text scores 0. Tune the build cutoff with `--min-quality`.

## Deduplication

`transform.compute_dedup_hash` hashes the lowercased, whitespace-collapsed
`normalized_text`. Identical content across different sources collapses to one
row (the first seen is kept); the count is reported as `duplicates_removed`.

## Validation contract

`validate.validate_gold_records` enforces: all canonical columns present;
required fields non-empty; `raw_text` or `normalized_text` non-empty; valid
`domain`; non-empty `category`; `split` in {train,val,test}; `quality_score` in
[0,1]; unique `record_id`; no duplicate `dedup_hash`. The builder also runs
`validate.validate_manifest_consistency` so manifest totals always equal the
written row count.

## CLI

```bash
python -m cyberdataset.gold.build_gold \
  --silver-dir data/silver_normalized \
  --out-dir data/gold \
  --min-quality 0.50 --seed 42 \
  [--limit-per-source N] [--no-parquet] [--no-card]
```

`--limit-per-source` caps rows read per silver file to bound memory when running
against the full ~200GB+ corpus.

## Outputs

- `data/gold/gold_unified.jsonl` — always written.
- `data/gold/gold_unified.parquet` — written when `pyarrow` is available.
- `data/gold/manifest.json` — counts by source/domain/category/label/split,
  duplicates removed, mean quality, seed, and threshold.
- `data/gold/dataset_card.md` — generated dataset card.
