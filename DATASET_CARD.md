# Dataset Card

## Intended Use

This benchmark is intended for cybersecurity research across heterogeneous defensive data types: vulnerable code metadata, network flows, malware feature vectors, phishing/social engineering indicators, cloud and SaaS abuse logs, supply chain advisories, prompt injection and AI security examples, CTI references, IoT/ICS telemetry, mobile features, and smart contract weaknesses.

## Out of Scope

- Operational exploit development.
- Malware distribution or execution.
- Credential collection or abuse.
- Re-identification of users, organizations, or systems.
- Production security decisions without source-specific validation.

## Data Sources

Sources are registered in `config/datasets.yaml`. Raw data is not bundled by default because many upstream datasets are large, gated, credentialed, or license-restricted.

## Labels

Labels are normalized into `label` and, where meaningful, `binary_label`.

- `0`: benign or non-vulnerable.
- `1`: malicious, vulnerable, phishing, malware, abuse, or attack-like.
- `null`: reference-only records where a binary interpretation is not meaningful.

## Safety

Gold rows must set `is_safe_representation=true`. Unsafe payloads should be represented with redacted text, feature JSON, hashes, aggregate telemetry, or source-row pointers.

## Known Limitations

This v1 scaffold provides the build system and contracts. Dataset-specific high-fidelity parsers should be implemented incrementally after raw data access and license review.

Current implemented parsers cover structured first-wave sources: CICIDS2017, UNSW-NB15, PhishTank, URLhaus, NVD, and CISA KEV. Remaining registered sources require additional parser work after raw files and license notes are available.
