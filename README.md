# Ultimate Cybersecurity Research Dataset

This repository is the data and benchmark layer for a hosted cybersecurity assessment model or agent. The datasets here are not the end product by themselves. They support model training, evaluation, benchmarking, and regression testing across major cybersecurity categories such as phishing, intrusion detection, threat intelligence, vulnerable code, prompt injection, and ICS or IoT attack data.

The practical goal is to make it possible for a teammate to:

- place raw cybersecurity datasets into the repo without mutating them
- normalize each source into a shared silver schema
- build benchmark-ready gold datasets
- run predictions against those benchmarks
- evaluate model or agent behavior consistently over time

## Project Purpose

Use this repository when you need the data pipeline behind a cybersecurity pentesting or security-assessment model. The core workflow is:

1. ingest raw source datasets
2. normalize them into schema-aligned silver rows
3. build benchmark-ready gold slices
4. run model predictions
5. score and compare results

This repo is intentionally defensive. It is for data engineering, benchmark generation, and model assessment. It is not for shipping offensive exploit instructions or live operational payloads.

## Dataset Architecture

The repository follows a layered architecture:

- `data/bronze_raw/`
  Untouched raw datasets. These are the original files placed into the repo by a human or copied from an approved local source. Do not clean, rewrite, or relabel bronze data in place.
- `data/silver_normalized/`
  Cleaned, normalized, schema-aligned outputs. Each silver module is source-specific but conforms to the unified project schema so downstream scripts can consume it consistently.
- `data/gold/`
  Benchmark-ready evaluation datasets. Gold files are built from silver rows and are intended for model benchmarking and regression tracking.
- `reports/`
  Integration reports, manifests, summaries, and diagnostics describing what was ingested, what was skipped, and what was written.

Related paths you will see in the repo:

- `data/gold_unified/`
  Legacy merged benchmark outputs.
- `docs/`
  Additional usage and benchmark notes.
- `tests/`
  Validation coverage for normalization, benchmark construction, and evaluation.

## Pipeline Architecture

The project is a layered, reproducible data pipeline. Each layer has one job and
a stable contract with the next, so sources can be added or refreshed without
rewriting downstream code.

```
  public sources                     data/bronze_raw/            data/silver_normalized/        data/gold/
  (feeds, APIs, dumps)   ───────►    raw, immutable      ──────► cleaned, per-category    ──────► unified, curated
                          ingest      ~200GB+             normalize  schema-aligned         build   AI/benchmark-ready
        ▲                                                                                    │
        │  cyberdataset.scrapers.fetch_fresh (public, key-optional, no proxy service)        │
        └────────────────── fresh refresh into data/bronze_raw/fresh/ ────────────────────────┘
```

- **Bronze (`data/bronze_raw/`)** — untouched raw cybersecurity datasets,
  collectively **~200GB+**. Immutable input; never cleaned or relabeled in place.
- **Silver (`data/silver_normalized/`)** — each source cleaned and normalized
  into a shared, source-specific schema with safe representations.
- **Gold unified (`data/gold/gold_unified.*`)** — every silver row mapped into
  one flat canonical schema spanning all domains, deduplicated by stable content
  hash, quality-scored, and split deterministically into train/val/test. Built
  by `cyberdataset.gold.build_gold`.
- **Gold benchmark (`data/gold/benchmark_gold.*`)** — the existing multi-head
  evaluation benchmark (see "Rebuild Full Gold Benchmark" below). Complementary
  to the unified layer.
- **Fresh-data scraper (`cyberdataset.scrapers`)** — a respectful, public-source
  refresh layer (CISA KEV, OSV, NVD, …) that pulls new records into
  `data/bronze_raw/fresh/`. It uses only the standard library and **requires no
  paid or proxy service**; source API keys (e.g. NVD) are always optional.

### Build the Gold Unified Layer

```bash
python -m cyberdataset.gold.build_gold \
  --silver-dir data/silver_normalized \
  --out-dir data/gold \
  --min-quality 0.50 \
  --seed 42
# equivalent: python -m scripts.build_gold_unified ...
```

This is the single script that unifies the Silver layer into the Gold layer.
It scans every silver source, maps each row into the canonical schema,
deduplicates, quality-scores, and assigns deterministic splits.

Outputs:

- `data/gold/gold_unified.jsonl` (always)
- `data/gold/gold_unified.parquet` (when `pyarrow` is installed)
- `data/gold/manifest.json` — counts by source/domain/category/label/split and duplicates removed
- `data/gold/dataset_card.md` — generated dataset card

Try it instantly on the bundled fixtures (no 200GB needed) — same script,
pointed at the tracked silver sample:

```bash
python -m cyberdataset.gold.build_gold \
  --silver-dir examples/silver_sample \
  --out-dir examples/gold_demo \
  --min-quality 0.40 --seed 42
# or simply: bash examples/run_gold_demo.sh
```

Only the silver fixtures (`examples/silver_sample/`) are tracked in git — they
show the per-source modules. The gold output is regenerable, so it is written to
the gitignored `examples/gold_demo/` on demand rather than committed.

Canonical schema (one flat row per example):

```
record_id, source_id, source_name, source_url, source_license, collected_at,
processed_at, domain, category, subcategory, task_type, raw_text,
normalized_text, label, severity, cwe, cve, mitre_attack_ids, language,
entities, metadata, quality_score, dedup_hash, split
```

### Add a New Module to the Silver Layer

A "module" is one source directory under `data/silver_normalized/`. The gold
builder auto-discovers every subdirectory, so adding a source needs **no code
changes** — just drop a normalized file into a new directory:

```
data/silver_normalized/
  <source_id>/
    <source_id>.parquet          # or .csv.gz / .jsonl / .csv
    <source_id>_metadata.json    # optional: source_dataset, license, source_url, ...
```

Rules the builder follows:

- The **directory name becomes `source_id`** (slugified). Use one directory per
  source, e.g. `cti_capec_attack_patterns/`.
- One **richest representation per directory** is read (`parquet` > `csv.gz` >
  `jsonl` > `csv`), so duplicate formats of the same data are never double-counted.
- The optional `*_metadata.json` supplies `source_name` (`source_dataset` key)
  and `source_license` (`license` key); missing values fall back to the directory
  name and `"unknown"`.
- Directories whose name starts with `_` are skipped (reserved for shared assets
  like manifests).

You can populate a module two ways:

1. **From the scraper:** fetch into Bronze, then normalize with the matching
   ingest adapter (see below) — `fresh bronze → silver`.
2. **Directly:** write an already-normalized file into the new directory.

To preview that a new module is picked up before a full build:

```bash
python -c "from cyberdataset.gold.build_gold import discover_silver_files; \
print([s.source_id for s in discover_silver_files('data/silver_normalized')])"
```

### Merge All Modules into Gold

Merging is a single re-run of the builder over the whole silver directory — it
scans **every** module, maps each row into the canonical schema, deduplicates
**across sources** by stable content hash, quality-scores, splits, and writes the
combined output:

```bash
python -m cyberdataset.gold.build_gold \
  --silver-dir data/silver_normalized \
  --out-dir data/gold \
  --min-quality 0.50 --seed 42
```

After it finishes, `data/gold/manifest.json` lists `sources_scanned` and the
per-source / per-domain / per-split counts, so you can confirm the new module was
merged. Because dedup and splits are deterministic (seeded content hashing),
re-running after adding a module yields a stable, reproducible gold set — only the
new, non-duplicate records are added.

### Fetch Fresh Public Data

```bash
python -m cyberdataset.scrapers.fetch_fresh \
  --sources cisa_kev,osv,nvd \
  --out-dir data/bronze_raw/fresh \
  --cache-dir .cache/fresh_scraper \
  --limit 1000
```

This writes raw records to `data/bronze_raw/fresh/<source_id>/<date>/raw.jsonl`
plus a per-run `metadata.json`, and a combined
`data/bronze_raw/fresh/manifest.json` (with per-source counts, attribution, and
any errors/warnings). The scraper respects rate limits, retries with backoff,
applies timeouts, caches responses, sends a descriptive user agent, and honors
robots.txt for HTML page scraping. It never authenticates, bypasses anti-bot, or
collects PII.

**Fresh bronze → silver → gold:** fresh raw under `data/bronze_raw/fresh/` is
normalized by the matching `cyberdataset.ingest.*` adapter (e.g.
`ingest_cisa_kev`, `ingest_nvd`, `ingest_osv`) into a silver dataset, then merged
into the unified layer by re-running `cyberdataset.gold.build_gold`.

## What This Demonstrates

- **Large-scale data engineering** — a Bronze→Silver→Gold pipeline over ~200GB+
  of heterogeneous public security data.
- **Schema design & normalization** — many disparate sources mapped into one
  flat, self-describing canonical schema.
- **Dedup & quality scoring** — stable content hashing for cross-source
  deduplication and a transparent quality score per record.
- **Reproducible pipelines** — deterministic, seeded train/val/test splits and
  manifests that make every build auditable.
- **AI / benchmark readiness** — JSONL + Parquet outputs ready for model
  training, benchmarking, and analytics.
- **Fresh public-data ingestion** — a dependency-light, respectful scraper for
  refreshing the dataset from public sources, with no paid or proxy dependency.

## Quickstart

```bash
source venv/bin/activate
```

If `venv/` does not exist yet:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Build Flow

The common end-to-end workflow is:

1. place raw files into `data/bronze_raw/...`
2. run the relevant normalizer to produce silver
3. build the full gold benchmark
4. optionally build a focused benchmark slice such as phishing
5. run predictions
6. evaluate predictions

## Phishing/Social Integration

The curated phishing email integration uses this raw folder:

```text
data/bronze_raw/phishing_email_11_curated/
```

Expected raw files:

- `Assassin.csv`
- `Assassin_vectorized_data.csv`
- `CEAS-08.csv`
- `Enron.csv`
- `Ling.csv`
- `Nazario_5.csv`
- `Nigerian_5.csv`
- `TREC-05.csv`
- `TREC-06.csv`
- `TREC-07.csv`

Silver output:

```text
data/silver_normalized/phishing_social/phishing_email_11_curated_silver.csv
data/silver_normalized/phishing_social/phishing_email_11_curated_silver.parquet
```

Current phishing silver stats:

- final row count: `109,994`
- label distribution:
  - `benign`: `102,758`
  - `phishing`: `7,236`

Source mapping policy:

- `Nazario_5.csv` and `Nigerian_5.csv` are treated as phishing or fraud positives.
- `Enron.csv` is treated as a benign business hard negative source.
- `Assassin.csv`, `CEAS-08.csv`, `Ling.csv`, and `TREC-*` are inspected and mapped conservatively.
- spam is not blindly treated as phishing
- vectorized-only inputs such as `Assassin_vectorized_data.csv` are excluded from LLM text training if they do not contain reconstructable human-readable email text

## How To Normalize Phishing Data

```bash
source venv/bin/activate

python -m scripts.normalize_phishing_email_11 \
  --raw-dir data/bronze_raw/phishing_email_11_curated \
  --out data/silver_normalized/phishing_social/phishing_email_11_curated_silver.csv \
  --report-dir reports \
  --seed 42 \
  --max-rows-per-source 20000
```

This produces:

- `data/silver_normalized/phishing_social/phishing_email_11_curated_silver.csv`
- `data/silver_normalized/phishing_social/phishing_email_11_curated_silver.parquet`
- `reports/phishing_email_11_curated_integration_report.md`
- `reports/phishing_email_11_curated_integration_summary.json`

## How To Rebuild Full Gold Benchmark

Use the full gold benchmark when you want cross-domain evaluation across the repo’s broader cybersecurity coverage.

```bash
python -m scripts.build_gold_benchmark \
  --silver-dir data/silver_normalized \
  --out-dir data/gold \
  --max-rows 100000 \
  --seed 42 \
  --format both
```

Outputs:

- `data/gold/benchmark_gold.csv`
- `data/gold/benchmark_gold.parquet`
- `data/gold/benchmark_manifest.json`

## How To Build Phishing-Focused Benchmark

Use the phishing-focused benchmark when you want a targeted phishing or social-engineering evaluation instead of the full multi-domain benchmark.

```bash
python -m scripts.build_phishing_social_gold \
  --silver-file data/silver_normalized/phishing_social/phishing_email_11_curated_silver.csv \
  --gold-file data/gold/benchmark_gold.csv \
  --out data/gold/benchmark_phishing_social_gold.csv \
  --max-rows 2000 \
  --seed 42
```

Outputs:

- `data/gold/benchmark_phishing_social_gold.csv`
- `data/gold/benchmark_phishing_social_gold.parquet`
- `data/gold/benchmark_phishing_social_manifest.json`

Current phishing-focused gold stats:

- `2,000` rows
- `1,000` phishing positives
- `500` benign business emails
- `500` hard negatives
- `0` duplicate `raw_text`
- deterministic `train`, `validation`, and `test` splits

## How To Run Predictions

Local stub dry run:

```bash
python -m scripts.run_model_predictions \
  --gold-file data/gold/benchmark_phishing_social_gold.csv \
  --out-dir data/gold \
  --provider local_stub \
  --model-name local_stub \
  --limit 100 \
  --dry-run
```

`local_stub` is only a pipeline sanity check. It verifies that benchmark loading, prompt construction, prediction writing, and evaluation wiring all work. It is not a real model evaluation.

## How To Evaluate Predictions

```bash
python -m scripts.evaluate_benchmark \
  --gold-file data/gold/benchmark_phishing_social_gold.csv \
  --predictions-file data/gold/predictions_local_stub.csv \
  --out-dir data/gold
```

## Real Model Benchmark Guidance

After the stub flow passes, run the same benchmark against a real provider supported by `scripts/run_model_predictions.py`, such as:

- OpenAI
- OpenRouter
- Ollama
- vLLM-compatible local OpenAI-style endpoints

Before choosing provider arguments, inspect the current CLI:

```bash
python -m scripts.run_model_predictions --help
```

Use the phishing-focused gold file when you want phishing-specific results:

```text
data/gold/benchmark_phishing_social_gold.csv
```

Use the full gold file when you want broader multi-domain results:

```text
data/gold/benchmark_gold.csv
```

## Reports And Manifests

Important report and manifest files:

- `reports/phishing_email_11_curated_integration_report.md`
  Human-readable integration report. Use this to inspect files discovered, files skipped, mapping decisions, included rows, excluded rows, and known limitations.
- `reports/phishing_email_11_curated_integration_summary.json`
  Machine-readable summary of the phishing integration. Use this for automated checks, dashboards, or quick validation of row counts and label distributions.
- `data/gold/benchmark_manifest.json`
  Manifest for the full benchmark. Use this to confirm row counts, source coverage, and benchmark composition after a full gold rebuild.
- `data/gold/benchmark_phishing_social_manifest.json`
  Manifest for the phishing-focused benchmark. Use this to confirm the 50/25/25-style sampling target, split distribution, and source-file coverage.

## Testing

Run the project validation commands after docs or pipeline changes:

```bash
python -m compileall -q scripts tests
python -m pytest tests/ -q
```

Current status after the gold unified layer and fresh-data scraper work:

- tests pass: `95 passed`
- new coverage: `tests/test_gold_unified.py` (schema, deterministic IDs, dedup,
  splits, manifest, quality filter) and `tests/test_scrapers.py` (adapter
  fetch/parse with mocked HTTP, orchestrator output, cache, rate limiting)

## Troubleshooting

`python: command not found`

```bash
source venv/bin/activate
python3 -m compileall -q scripts tests
```

Use `python3` directly if `python` is not available on `PATH`.

Missing YAML dependency:

```bash
pip install pyyaml
```

Missing Pydantic dependency:

```bash
pip install pydantic
```

Missing parquet engine:

```bash
pip install pyarrow
```

Silver output has `0` rows:

- confirm the raw CSVs exist under `data/bronze_raw/phishing_email_11_curated/`
- rerun the normalizer with the `--report-dir reports` flag
- inspect `reports/phishing_email_11_curated_integration_report.md`

`local_stub` is not a real benchmark:

- it is only a dry-run or pipeline sanity check
- use a real provider after the stub flow is working

Wrong benchmark file:

- use `data/gold/benchmark_phishing_social_gold.csv` for phishing-focused evaluation
- do not rely only on `data/gold/benchmark_gold.csv` when you want phishing-specific performance numbers

## Safety Position

This repo exists for defensive cybersecurity research, model assessment, and benchmark construction. Do not use it to add live exploit steps, credentials, or malware execution workflows to benchmark outputs or documentation.
