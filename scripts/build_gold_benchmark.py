"""Build the multi-head gold cybersecurity benchmark from silver outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "benchmark_heads.yml"
DEFAULT_SILVER_DIR = PROJECT_ROOT / "data" / "silver_normalized"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "gold"

GOLD_COLUMNS = [
    "record_id",
    "source_dataset",
    "source_type",
    "main_category",
    "attack_name",
    "attack_family",
    "label",
    "binary_label",
    "mitre_tactic",
    "mitre_technique_id",
    "benchmark_domain",
    "task_type",
    "evaluation_head",
    "metric_group",
    "input_text",
    "expected_output",
    "gold_label",
    "label_set",
    "difficulty",
    "split",
    "requires_probability",
    "scoring_notes",
    "safety_notes",
    "created_at",
]

TASK_TYPES = {"classification", "generation", "knowledge", "reasoning"}
METRIC_GROUPS = {"classification", "generation", "knowledge", "reasoning"}


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load benchmark-head configuration."""

    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def read_table(path: Path) -> pd.DataFrame:
    """Read CSV, JSONL, or parquet files."""

    suffixes = "".join(path.suffixes[-2:])
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if suffixes == ".csv.gz" or path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(f"unsupported input file: {path}")


def discover_silver_files(silver_dir: Path) -> list[Path]:
    """Discover data-bearing silver files without reading metadata/report files."""

    manifest = silver_dir / "silver_manifest.csv"
    if manifest.exists():
        df = pd.read_csv(manifest)
        files = []
        for value in df.get("output_path_parquet", pd.Series(dtype=str)).dropna():
            if value:
                path = PROJECT_ROOT / str(value)
                if path.exists():
                    files.append(path)
        if files:
            return sorted(files)
    candidates: list[Path] = []
    for pattern in ("*.parquet", "*.csv", "*.csv.gz", "*.jsonl", "*/*.parquet", "*/*.csv", "*/*.csv.gz", "*/*.jsonl"):
        candidates.extend(silver_dir.glob(pattern))
    return sorted(
        path
        for path in set(candidates)
        if "metadata" not in path.name
        and path.name != "silver_manifest.csv"
        and "_dedup" not in path.parts
        and path.is_file()
    )


def load_silver_rows(silver_dir: Path) -> pd.DataFrame:
    """Load all available silver rows."""

    frames = []
    for path in discover_silver_files(silver_dir):
        df = read_table(path)
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _safe_str(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _matches_head(row: pd.Series, head: dict[str, Any]) -> bool:
    category = _safe_str(row.get("main_category"))
    source_type = _safe_str(row.get("source_type"))
    label = _safe_str(row.get("label"))
    return (
        category in set(head.get("mapped_main_categories", []))
        or source_type in set(head.get("mapped_source_types", []))
        or label in set(head.get("mapped_labels", []))
    )


def assign_head(row: pd.Series, config: dict[str, Any]) -> tuple[str, str, str]:
    """Assign evaluation head, task type, and metric group deterministically."""

    for name, head in (config.get("evaluation_heads") or {}).items():
        if _matches_head(row, head):
            task_type = str(head.get("default_task_type", "classification"))
            metric_group = str(head.get("metric_group", task_type))
            return name, task_type, metric_group
    for name, head in (config.get("future_heads") or {}).items():
        if _matches_head(row, head):
            task_type = str(head.get("default_task_type", "classification"))
            metric_group = str(head.get("metric_group", task_type))
            return name, task_type, metric_group
    return "miscellaneous", "classification", "classification"


def make_input_text(row: pd.Series) -> str:
    """Create safe benchmark input text from silver fields."""

    raw = _safe_str(row.get("raw_text"))
    if raw:
        return raw[:4000]
    parts = [
        f"Dataset: {_safe_str(row.get('source_dataset')) or 'unknown'}",
        f"Category: {_safe_str(row.get('main_category')) or 'unknown'}",
        f"Attack: {_safe_str(row.get('attack_name')) or 'unknown'}",
        f"Family: {_safe_str(row.get('attack_family')) or 'unknown'}",
        f"Features: {_safe_str(row.get('features_json')) or 'unavailable'}",
    ]
    return "\n".join(parts)


def make_expected_output(row: pd.Series, task_type: str) -> str | None:
    """Create expected output for non-training benchmark tasks."""

    if task_type == "classification":
        return _safe_str(row.get("label")) or _safe_str(row.get("binary_label")) or "unknown"
    if task_type == "knowledge":
        for col in ("mitre_technique_id", "cve_id", "cwe_id", "attack_name", "label"):
            value = _safe_str(row.get(col))
            if value:
                return value
        return "unknown"
    if task_type == "reasoning":
        return _safe_str(row.get("notes")) or _safe_str(row.get("label")) or "unknown"
    return _safe_str(row.get("expected_output"))


def label_set_for(group: pd.DataFrame, row: pd.Series) -> str:
    """Return deterministic JSON label set for a benchmark head/task group."""

    labels = sorted(str(x) for x in group["gold_label"].dropna().unique())
    if not labels:
        labels = sorted(str(x) for x in row.index if x)
    return json.dumps(labels, sort_keys=True)


def difficulty(row: pd.Series) -> str:
    """Assign a simple deterministic difficulty bucket."""

    if _safe_str(row.get("mitre_technique_id")) or _safe_str(row.get("cve_id")):
        return "hard"
    if _safe_str(row.get("cwe_id")) or _safe_str(row.get("features_json")):
        return "medium"
    return "easy"


def split_for_record(record_id: str, seed: int) -> str:
    """Assign stable train/validation/test split."""

    import hashlib

    digest = hashlib.sha1(f"{seed}:{record_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < 0.70:
        return "train"
    if bucket < 0.85:
        return "validation"
    return "test"


def transform_silver_to_gold(silver: pd.DataFrame, config: dict[str, Any], seed: int) -> pd.DataFrame:
    """Map silver rows into the gold benchmark schema."""

    rows: list[dict[str, Any]] = []
    created_at = pd.Timestamp.now(tz=UTC).isoformat()
    for _, row in silver.iterrows():
        record_id = _safe_str(row.get("record_id"))
        if not record_id:
            continue
        evaluation_head, task_type, metric_group = assign_head(row, config)
        if task_type not in TASK_TYPES:
            task_type = "classification"
        if metric_group not in METRIC_GROUPS:
            metric_group = task_type
        gold_label = _safe_str(row.get("label")) or "unknown"
        expected_output = make_expected_output(row, task_type)
        rows.append(
            {
                "record_id": record_id,
                "source_dataset": _safe_str(row.get("source_dataset")) or "unknown",
                "source_type": _safe_str(row.get("source_type")) or "other",
                "main_category": _safe_str(row.get("main_category")) or "Miscellaneous / Needs Review",
                "attack_name": _safe_str(row.get("attack_name")),
                "attack_family": _safe_str(row.get("attack_family")),
                "label": gold_label,
                "binary_label": row.get("binary_label") if pd.notna(row.get("binary_label")) else None,
                "mitre_tactic": _safe_str(row.get("mitre_tactic")),
                "mitre_technique_id": _safe_str(row.get("mitre_technique_id")),
                "benchmark_domain": evaluation_head,
                "task_type": task_type,
                "evaluation_head": evaluation_head,
                "metric_group": metric_group,
                "input_text": make_input_text(row),
                "expected_output": expected_output,
                "gold_label": gold_label,
                "label_set": None,
                "difficulty": difficulty(row),
                "split": split_for_record(record_id, seed),
                "requires_probability": bool(task_type == "classification"),
                "scoring_notes": f"Use {metric_group} metrics for {task_type}.",
                "safety_notes": "Benchmark input is derived from local normalized silver rows; do not execute code, visit URLs, or run samples.",
                "created_at": created_at,
            }
        )
    gold = pd.DataFrame(rows, columns=GOLD_COLUMNS)
    if gold.empty:
        return gold
    grouped = gold.groupby(["evaluation_head", "task_type"], dropna=False)
    label_sets = {
        key: json.dumps(sorted(str(x) for x in group["gold_label"].dropna().unique()), sort_keys=True)
        for key, group in grouped
    }
    gold["label_set"] = [label_sets[(row.evaluation_head, row.task_type)] for row in gold.itertuples()]
    return gold


def stratified_cap(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    """Cap rows with deterministic stratified sampling across benchmark dimensions."""

    if max_rows <= 0 or len(df) <= max_rows:
        return df.sort_values("record_id").reset_index(drop=True)
    strata = ["evaluation_head", "task_type", "main_category", "gold_label"]
    grouped = list(df.groupby(strata, dropna=False))
    base = max(1, max_rows // max(1, len(grouped)))
    samples = []
    remaining = max_rows
    for _key, group in grouped:
        take = min(len(group), base, remaining)
        if take > 0:
            samples.append(group.sample(n=take, random_state=seed))
            remaining -= take
        if remaining <= 0:
            break
    if remaining > 0:
        used = pd.concat(samples, ignore_index=False).index if samples else []
        rest = df.drop(index=used).sample(n=min(remaining, len(df) - len(used)), random_state=seed)
        samples.append(rest)
    return pd.concat(samples, ignore_index=True).sort_values("record_id").reset_index(drop=True)


def write_outputs(gold: pd.DataFrame, out_dir: Path, output_format: str, manifest: dict[str, Any]) -> None:
    """Write benchmark artifacts."""

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "benchmark_gold.csv"
    gold.to_csv(csv_path, index=False)
    manifest["outputs"]["csv"] = _display_path(csv_path)
    if output_format in {"parquet", "both"}:
        try:
            parquet_path = out_dir / "benchmark_gold.parquet"
            gold.to_parquet(parquet_path, index=False)
            manifest["outputs"]["parquet"] = _display_path(parquet_path)
        except Exception as exc:
            manifest.setdefault("warnings", []).append(f"parquet output skipped: {exc}")
    (out_dir / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _display_path(path: Path) -> str:
    """Return repo-relative path when possible."""

    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def build_benchmark(silver_dir: Path, out_dir: Path, max_rows: int, seed: int, output_format: str, dry_run: bool, config_path: Path = DEFAULT_CONFIG) -> pd.DataFrame:
    """Build and optionally write the gold benchmark."""

    config = load_config(config_path)
    silver = load_silver_rows(silver_dir)
    gold = transform_silver_to_gold(silver, config, seed)
    original_rows = len(gold)
    gold = stratified_cap(gold, max_rows=max_rows, seed=seed)
    manifest = {
        "created_at": pd.Timestamp.now(tz=UTC).isoformat(),
        "config_path": str(config_path.relative_to(PROJECT_ROOT)),
        "silver_dir": str(silver_dir),
        "row_count": int(len(gold)),
        "original_row_count": int(original_rows),
        "max_rows": int(max_rows),
        "seed": int(seed),
        "outputs": {},
        "counts": {
            "evaluation_head": gold["evaluation_head"].value_counts().to_dict() if not gold.empty else {},
            "task_type": gold["task_type"].value_counts().to_dict() if not gold.empty else {},
            "label": gold["gold_label"].value_counts().to_dict() if not gold.empty else {},
            "source_dataset": gold["source_dataset"].value_counts().to_dict() if not gold.empty else {},
        },
    }
    print_summary(gold)
    if not dry_run:
        write_outputs(gold, out_dir, output_format, manifest)
    return gold


def print_summary(gold: pd.DataFrame) -> None:
    """Print clean build summary."""

    print(f"Gold benchmark rows: {len(gold)}")
    for column in ("evaluation_head", "task_type", "gold_label", "source_dataset"):
        print(f"\nCounts by {column}:")
        counts = Counter(gold[column].fillna("unknown")) if not gold.empty else Counter()
        for key, count in counts.most_common(20):
            print(f"  {key}: {count}")


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--silver-dir", type=Path, default=DEFAULT_SILVER_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-rows", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--format", choices=["csv", "parquet", "both"], default="both")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    build_benchmark(args.silver_dir, args.out_dir, args.max_rows, args.seed, args.format, args.dry_run, args.config)


if __name__ == "__main__":
    main()
