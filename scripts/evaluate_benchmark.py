"""Evaluate model predictions against the multi-head gold benchmark."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from scripts.build_gold_benchmark import DEFAULT_CONFIG, DEFAULT_OUT_DIR, PROJECT_ROOT, read_table


GROUP_COLUMNS = ["evaluation_head", "task_type", "main_category", "source_dataset"]


def normalize_text(value: Any) -> str:
    """Normalize text for matching."""

    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def token_set(value: Any) -> set[str]:
    """Tokenize safely for fallback text overlap."""

    return set(re.findall(r"[a-z0-9_.:-]+", normalize_text(value)))


def read_predictions(path: Path) -> pd.DataFrame:
    """Read CSV or JSONL predictions."""

    df = read_table(path)
    required = {"record_id", "prediction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"prediction file missing columns: {sorted(missing)}")
    return df


def classification_metrics(y_true: list[str], y_pred: list[str], probabilities: list[float | None] | None = None) -> dict[str, Any]:
    """Compute multiclass precision/recall/F1 without sklearn."""

    labels = sorted(set(y_true) | set(y_pred))
    per_label: dict[str, dict[str, float]] = {}
    supports = {}
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred, strict=False))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred, strict=False))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred, strict=False))
        support = sum(t == label for t in y_true)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        supports[label] = support
    total = max(1, len(y_true))
    macro_precision = sum(v["precision"] for v in per_label.values()) / max(1, len(labels))
    macro_recall = sum(v["recall"] for v in per_label.values()) / max(1, len(labels))
    f1_macro = sum(v["f1"] for v in per_label.values()) / max(1, len(labels))
    f1_weighted = sum(per_label[label]["f1"] * supports[label] for label in labels) / total
    confusion = {label: {other: 0 for other in labels} for label in labels}
    for true, pred in zip(y_true, y_pred, strict=False):
        confusion[true][pred] += 1
    out: dict[str, Any] = {
        "precision": macro_precision,
        "recall": macro_recall,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": confusion,
    }
    if probabilities and len(set(y_true)) == 2 and any(p is not None for p in probabilities):
        out["roc_auc"] = binary_auc(y_true, probabilities)
    return out


def binary_auc(y_true: list[str], probabilities: list[float | None]) -> float | None:
    """Compute binary ROC AUC by rank when scores are available."""

    labels = sorted(set(y_true))
    if len(labels) != 2:
        return None
    positive = labels[-1]
    pairs = [(float(score), true == positive) for true, score in zip(y_true, probabilities, strict=False) if score is not None and not math.isnan(float(score))]
    positives = sum(is_pos for _score, is_pos in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return None
    pairs.sort(key=lambda item: item[0])
    rank_sum = sum(rank for rank, (_score, is_pos) in enumerate(pairs, start=1) if is_pos)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def normalized_similarity(expected: Any, prediction: Any) -> float:
    """Fallback normalized token similarity."""

    a = token_set(expected)
    b = token_set(prediction)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def exact_match(expected: Any, prediction: Any) -> float:
    """Exact normalized match score."""

    return float(normalize_text(expected) == normalize_text(prediction))


def generation_metrics(expected: list[Any], predictions: list[Any]) -> dict[str, Any]:
    """Compute generation metrics with optional dependency fallbacks."""

    warnings: list[str] = []
    overlap_scores = [normalized_similarity(e, p) for e, p in zip(expected, predictions, strict=False)]
    out: dict[str, Any] = {
        "token_overlap": sum(overlap_scores) / max(1, len(overlap_scores)),
        "normalized_similarity": sum(overlap_scores) / max(1, len(overlap_scores)),
    }
    try:
        from rouge_score import rouge_scorer  # type: ignore

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [scorer.score(str(e), str(p))["rougeL"].fmeasure for e, p in zip(expected, predictions, strict=False)]
        out["rouge_l"] = sum(scores) / max(1, len(scores))
    except Exception:
        warnings.append("ROUGE-L dependency not installed; used fallback similarity.")
    try:
        import nltk  # type: ignore

        scores = [nltk.translate.bleu_score.sentence_bleu([str(e).split()], str(p).split()) for e, p in zip(expected, predictions, strict=False)]
        out["bleu"] = sum(scores) / max(1, len(scores))
    except Exception:
        warnings.append("BLEU dependency not installed; used fallback similarity.")
    try:
        from bert_score import score as bert_score  # type: ignore

        _p, _r, f1 = bert_score([str(p) for p in predictions], [str(e) for e in expected], lang="en", verbose=False)
        out["bertscore_f1"] = float(f1.mean())
    except Exception:
        warnings.append("BERTScore dependency not installed; used fallback similarity.")
    try:
        from sentence_transformers import SentenceTransformer, util  # type: ignore

        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb_e = model.encode([str(e) for e in expected], convert_to_tensor=True)
        emb_p = model.encode([str(p) for p in predictions], convert_to_tensor=True)
        sims = util.cos_sim(emb_e, emb_p).diagonal()
        out["semantic_similarity"] = float(sims.mean())
    except Exception:
        warnings.append("sentence-transformers dependency not installed; used fallback similarity.")
    if warnings:
        out["warnings"] = sorted(set(warnings))
    return out


def knowledge_metrics(expected: list[Any], predictions: list[Any]) -> dict[str, Any]:
    """Compute exact/normalized matching for knowledge and reasoning tasks."""

    exact = [exact_match(e, p) for e, p in zip(expected, predictions, strict=False)]
    similarity = [normalized_similarity(e, p) for e, p in zip(expected, predictions, strict=False)]
    return {
        "exact_match": sum(exact) / max(1, len(exact)),
        "normalized_match": sum(similarity) / max(1, len(similarity)),
        "semantic_similarity": sum(similarity) / max(1, len(similarity)),
        "explanation_quality": None,
    }


def score_group(df: pd.DataFrame) -> dict[str, Any]:
    """Score one grouped slice."""

    if df["metric_group"].nunique(dropna=False) > 1:
        parts = []
        for metric_group, group in df.groupby("metric_group", dropna=False):
            scored = score_group(group)
            parts.append((str(metric_group), len(group), float(scored["primary_score"])))
        total = max(1, sum(size for _metric_group, size, _score in parts))
        primary = sum(size * score for _metric_group, size, score in parts) / total
        return {"metric_group": "mixed", "primary_score": primary, "component_scores": parts}
    metric_group = str(df["metric_group"].iloc[0])
    if metric_group == "classification":
        probs = None
        for col in ("probability", "score", "confidence"):
            if col in df.columns:
                probs = pd.to_numeric(df[col], errors="coerce").where(pd.notna(df[col]), None).tolist()
                break
        metrics = classification_metrics(df["gold_label"].astype(str).tolist(), df["prediction"].astype(str).tolist(), probs)
        primary = metrics["f1_macro"]
    elif metric_group == "generation":
        metrics = generation_metrics(df["expected_output"].tolist(), df["prediction"].tolist())
        primary = float(metrics.get("semantic_similarity") or metrics.get("normalized_similarity") or 0.0)
    else:
        metrics = knowledge_metrics(df["expected_output"].tolist(), df["prediction"].tolist())
        primary = float(metrics.get("normalized_match", 0.0))
    return {"metric_group": metric_group, "primary_score": primary, **metrics}


def load_weights(config_path: Path) -> dict[str, float]:
    """Load metric group weights from config."""

    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    return {str(k): float(v) for k, v in (config.get("metric_weights") or {}).items()}


def evaluate(gold_path: Path, predictions_path: Path, out_dir: Path, config_path: Path = DEFAULT_CONFIG) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate predictions and write result artifacts."""

    gold = read_table(gold_path)
    predictions = read_predictions(predictions_path)
    joined = gold.merge(predictions, on="record_id", how="inner", suffixes=("", "_pred"))
    if joined.empty:
        raise ValueError("no predictions matched gold record_id values")
    result_rows: list[dict[str, Any]] = []
    slices: list[tuple[str, str, pd.DataFrame]] = [("overall", "overall", joined)]
    for col in GROUP_COLUMNS:
        for key, group in joined.groupby(col, dropna=False):
            slices.append((col, str(key), group))
    for group_by, group_value, group in slices:
        scored = score_group(group)
        result_rows.append(
            {
                "group_by": group_by,
                "group_value": group_value,
                "n": int(len(group)),
                **{k: v for k, v in scored.items() if not isinstance(v, dict)},
                "metrics_json": json.dumps(scored, sort_keys=True, default=str),
            }
        )
    results = pd.DataFrame(result_rows)
    weights = load_weights(config_path)
    by_group = joined.groupby("metric_group", dropna=False)
    weighted_parts = []
    for metric_group, group in by_group:
        primary = score_group(group)["primary_score"]
        weighted_parts.append((primary, weights.get(str(metric_group), 1.0)))
    overall = sum(score * weight for score, weight in weighted_parts) / max(1e-12, sum(weight for _score, weight in weighted_parts))
    payload = {
        "gold_path": str(gold_path),
        "predictions_path": str(predictions_path),
        "matched_rows": int(len(joined)),
        "coverage": int(len(joined)) / max(1, len(gold)),
        "weighted_overall_score": overall,
        "results": result_rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "evaluation_results.csv", index=False)
    (out_dir / "evaluation_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print_leaderboard(results, overall)
    return results, payload


def print_leaderboard(results: pd.DataFrame, weighted_overall_score: float) -> None:
    """Print leaderboard-style summary."""

    print(f"Weighted overall score: {weighted_overall_score:.4f}")
    keep = results[results["group_by"].isin(["overall", "evaluation_head", "task_type"])]
    cols = ["group_by", "group_value", "n", "metric_group", "primary_score"]
    print(keep[cols].sort_values(["group_by", "group_value"]).to_string(index=False))


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-file", type=Path, default=DEFAULT_OUT_DIR / "benchmark_gold.csv")
    parser.add_argument("--predictions-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    evaluate(args.gold_file, args.predictions_file, args.out_dir, args.config)


if __name__ == "__main__":
    main()
