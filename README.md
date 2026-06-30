# Ultimate Cybersecurity Research Dataset

Framework-first repository for building a normalized mixed cybersecurity research benchmark capped at 100,000 rows.

This project separates raw source acquisition, source-specific cleaning, and final gold dataset assembly:

- `data/bronze_raw/`: manually placed original datasets or source exports.
- `data/silver_normalized/`: canonical silver-layer outputs. This is the only supported silver output path.
- `data/gold/`: multi-head evaluation benchmark files.
- `data/gold_unified/`: legacy merged benchmark files.
- `data/reports/`: validation and coverage reports.

The v1 scaffold does not download large, gated, credentialed, or license-sensitive datasets. Dataset access requirements are tracked in `config/datasets.yaml` and `LICENSE_NOTES.md`.
<img width="1672" height="941" alt="dataset" src="https://github.com/user-attachments/assets/d91384ab-6635-4921-9c67-ce0f73eea703" />

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
make test
make smoke
```

## Build Flow

```bash
make inventory
make build-silver
make build-gold
make sample-10k
make validate
```

`make smoke` creates a tiny safe synthetic silver fixture under `data/silver_normalized/` and a gold dataset for contract verification.

## Multi-Head Gold Benchmark

The current gold benchmark layer maps canonical silver rows into evaluation heads:

- `malware_code`
- `cti`
- `prompt_injection_jailbreaks`

It is extensible for `network_intrusion`, `phishing_social`, `cloud_saas_abuse`, `iot_ics`, and `supply_chain`.

Build the multi-head benchmark:

```bash
python -m scripts.build_gold_benchmark \
  --silver-dir data/silver_normalized \
  --out-dir data/gold \
  --max-rows 100000 \
  --seed 42 \
  --format both
```

Run local stub predictions:

```bash
python -m scripts.run_model_predictions \
  --gold-file data/gold/benchmark_gold.csv \
  --out-dir data/gold \
  --provider local_stub \
  --model-name local_stub \
  --limit 100 \
  --dry-run
```

Evaluate predictions:

```bash
python -m scripts.evaluate_benchmark \
  --gold-file data/gold/benchmark_gold.csv \
  --predictions-file path/to/predictions.csv \
  --out-dir data/gold
```

Prediction files must include `record_id` and `prediction`. Optional columns are `model_name`, `score`, `probability`, `confidence`, and `explanation`.

Run OpenAI predictions:

```bash
export OPENAI_API_KEY="your-openai-api-key"

python -m scripts.run_model_predictions \
  --gold-file data/gold/benchmark_gold.csv \
  --out-dir data/gold \
  --provider openai \
  --model-name gpt-4o-mini \
  --limit 100 \
  --resume
```

Run OpenRouter predictions:

```bash
export OPENROUTER_API_KEY="your-openrouter-api-key"

python -m scripts.run_model_predictions \
  --gold-file data/gold/benchmark_gold.csv \
  --out-dir data/gold \
  --provider openrouter \
  --model-name "qwen/qwen-2.5-14b-instruct" \
  --limit 100 \
  --resume
```

See `docs/gold_benchmark.md` for schema, metrics, task types, and Qwen2.5-14B adapter evaluation notes.

## Bronze Raw Layout

Place licensed raw datasets under one folder per source, using names from `config/datasets.yaml`:

```text
data/bronze_raw/
  CICIDS2017/
    README.txt
    LICENSE.txt
    Monday-WorkingHours.pcap_ISCX.csv
  UNSW_NB15/
    README.txt
    UNSW_NB15_training-set.csv
  PhishTank/
    README.txt
    phishtank.csv
  URLhaus/
    README.txt
    urlhaus.csv
  NVD/
    README.txt
    nvdcve-2.0-2024.json
  CISA_KEV/
    README.txt
    known_exploited_vulnerabilities.json
```

Supported first-wave parsers: `CICIDS2017`, `UNSW_NB15`, `PhishTank`, `URLhaus`, `NVD`, and `CISA_KEV`. Other source folders are inventoried and skipped until their parser is implemented.

## Canonical Schema

The normalized schema lives in `config/schema.yaml` and is mirrored by `cyberdataset.schema`. Every row includes source metadata, normalized attack taxonomy, labels, MITRE/CAPEC/CWE/CVE references where available, safe text/features, safety flags, license notes, and a deterministic split.

## Safety Position

This repo is designed for defensive research dataset engineering. It must not store live malware binaries, credentials, exploit-ready operational payloads, or unsafe prompt injection instructions in gold outputs. Use safe summaries, redacted text, feature vectors, hashes, or pointers to source rows.
