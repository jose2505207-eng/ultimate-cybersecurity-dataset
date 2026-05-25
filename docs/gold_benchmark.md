# Gold Multi-Head Benchmark

The project now has three layers:

- `data/bronze_raw/`: immutable source data as downloaded or manually placed.
- `data/silver_normalized/`: canonical normalized per-source modules using the unified silver schema.
- `data/gold/`: evaluation-ready benchmark rows and evaluation outputs.

The gold layer does not train models. It transforms silver rows into benchmark tasks so later Qwen2.5-14B adapters, LoRA runs, or other models can be evaluated by cybersecurity domain.

## Evaluation Heads

Current heads are configured in `config/benchmark_heads.yml`:

- `malware_code`: vulnerable code, malware features, static/dynamic malware labels, code-security classification, and CWE/CVE mapping when available.
- `cti`: threat intelligence, MITRE ATT&CK, CVE/advisory/NVD/OSV records, tactic/technique classification, and threat knowledge tasks.
- `prompt_injection_jailbreaks`: prompt injection, jailbreaks, AI security risk, prompt attack classification, and future safe refusal/compliance metadata.

Future heads are represented in config so they can be promoted without redesign:

- `network_intrusion`
- `phishing_social`
- `cloud_saas_abuse`
- `iot_ics`
- `supply_chain`

## Task Types

Every gold row has one task type:

- `classification`
- `generation`
- `knowledge`
- `reasoning`

The builder assigns task type deterministically from `config/benchmark_heads.yml`. Missing fields are filled with safe defaults such as `unknown` or null.

## Metrics

Classification metrics:

- precision
- recall
- f1_macro
- f1_weighted
- roc_auc when probability scores are available
- confusion matrix

Generation metrics:

- BLEU when optional dependencies are installed
- ROUGE-L when optional dependencies are installed
- BERTScore when optional dependencies are installed
- sentence-transformer semantic similarity when installed
- fallback token overlap / normalized similarity

Knowledge and reasoning metrics:

- exact_match
- normalized_match
- semantic_similarity fallback
- explanation_quality placeholder for future human grading

## Build The Benchmark

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
- `data/gold/benchmark_gold.parquet` when `pyarrow` is available and `--format parquet|both`
- `data/gold/benchmark_manifest.json`

Dry run:

```bash
python -m scripts.build_gold_benchmark --dry-run
```

## Prediction Format

Predictions must contain:

```text
record_id,prediction
```

Optional columns:

```text
model_name,score,probability,confidence,explanation
```

CSV and JSONL are supported.

## Evaluate Predictions

```bash
python -m scripts.evaluate_benchmark \
  --gold-file data/gold/benchmark_gold.csv \
  --predictions-file path/to/predictions.csv \
  --out-dir data/gold
```

Outputs:

- `data/gold/evaluation_results.json`
- `data/gold/evaluation_results.csv`

The evaluator prints a leaderboard-style summary and computes a weighted overall score using `config/benchmark_heads.yml`.

## Qwen2.5-14B Adapter Support

The gold layer is model-agnostic. Future Qwen2.5-14B adapters should emit prediction files with `record_id` and `prediction`; optional `probability`, `confidence`, and `explanation` fields improve scoring and analysis. Domain-specific adapters can be compared by filtering `evaluation_head`.
