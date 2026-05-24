# Ultimate Cybersecurity Research Dataset

Framework-first repository for building a normalized mixed cybersecurity research benchmark capped at 100,000 rows.

This project separates raw source acquisition, source-specific cleaning, and final gold dataset assembly:

- `data/bronze_raw/`: manually placed original datasets or source exports.
- `data/silver_normalized/`: canonical silver-layer outputs. This is the only supported silver output path.
- `data/gold_unified/`: merged benchmark files.
- `data/reports/`: validation and coverage reports.

The v1 scaffold does not download large, gated, credentialed, or license-sensitive datasets. Dataset access requirements are tracked in `config/datasets.yaml` and `LICENSE_NOTES.md`.

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
