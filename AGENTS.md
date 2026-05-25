# Agent Instructions

This repository uses a bronze -> silver -> gold architecture.

- Preserve `data/bronze_raw/` as immutable input. Do not modify, delete, move, or execute bronze dataset content.
- Treat silver as normalized, source-specific data under `data/silver_normalized/`.
- Treat gold as model-evaluation benchmark data under `data/gold/`.
- Do not train models from this repo task flow unless explicitly requested.
- Do not generate harmful cybersecurity instructions, live exploit steps, credentials, or malware execution workflows.
- Keep benchmark scripts deterministic, testable, and CLI-friendly.
- Prefer small clear functions over hidden global behavior.
- Preserve existing schemas; add transformation layers instead of breaking silver compatibility.
- Run validation after changes:

```bash
python -m compileall -q scripts tests
python -m pytest tests/ -q
```
