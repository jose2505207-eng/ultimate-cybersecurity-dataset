# License Notes

This repository does not redistribute upstream datasets.

Each source in `config/datasets.yaml` includes a `license` or `usage_note` field. Before using a source, verify the current upstream license, citation requirements, redistribution limits, and access terms.

Default policy:

- Store raw files locally under `data/bronze_raw/`.
- Do not commit raw datasets.
- Do not commit malware binaries, credentials, exploit-ready payloads, or private logs.
- Prefer derived safe features, hashes, redacted text, and source-row pointers.
- Keep source-specific license notes in silver and gold metadata.

