"""Evaluate local Qwen QLoRA adapters against gold benchmark rows."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.build_gold_benchmark import PROJECT_ROOT
from scripts.evaluate_benchmark import classification_metrics, normalize_text, read_table
from scripts.run_model_predictions import build_messages, parse_model_response, safe_model_name
from scripts.train_qlora import DEFAULT_CONFIG, detect_and_load_dataset, load_config


DEFAULT_ADAPTER = PROJECT_ROOT / "outputs" / "qlora" / "qwen25_coder_7b_local_final_512_step100" / "final_adapter"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "qlora_eval" / "qwen25_coder_7b_final_adapter_eval"


def now_utc() -> str:
    return datetime.now(tz=UTC).isoformat()


def select_gold_subset(gold: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    """Select a deterministic mixed benchmark subset."""

    gold = gold.dropna(subset=["record_id", "input_text"]).copy()
    if "expected_output" not in gold.columns:
        gold["expected_output"] = gold.get("gold_label", "unknown")
    gold["expected_output"] = gold["expected_output"].fillna(gold.get("gold_label", "unknown")).fillna("unknown")
    strata = [col for col in ("evaluation_head", "task_type", "main_category", "gold_label") if col in gold.columns]
    if not strata or len(gold) <= limit:
        return gold.sort_values("record_id").head(limit).reset_index(drop=True)
    groups = list(gold.groupby(strata, dropna=False))
    per = max(1, limit // max(1, len(groups)))
    samples = []
    remaining = limit
    for _key, group in groups:
        take = min(len(group), per, remaining)
        if take > 0:
            samples.append(group.sample(n=take, random_state=seed))
            remaining -= take
        if remaining <= 0:
            break
    if remaining > 0:
        used = pd.concat(samples).index if samples else []
        rest = gold.drop(index=used)
        if not rest.empty:
            samples.append(rest.sample(n=min(remaining, len(rest)), random_state=seed))
    return pd.concat(samples, ignore_index=True).sort_values("record_id").reset_index(drop=True)


def setup_tokenizer(model_name: str, local_files_only: bool):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def torch_dtype():
    import torch

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_model(model_name: str, local_files_only: bool, adapter_path: Path | None = None):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype(),
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=local_files_only,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = True
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path), local_files_only=local_files_only)
    model.eval()
    return model


def release_model(model: Any) -> None:
    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def target_for_row(row: pd.Series) -> str:
    if str(row.get("task_type") or "") == "classification":
        return str(row.get("gold_label") or row.get("expected_output") or "").strip()
    return str(row.get("expected_output") or row.get("gold_label") or "").strip()


def generate_variant_predictions(
    *,
    gold: pd.DataFrame,
    variant_name: str,
    model_name: str,
    adapter_path: Path | None,
    out_dir: Path,
    max_input_chars: int,
    max_new_tokens: int,
    max_seq_length: int,
    local_files_only: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate predictions for one model variant."""

    import torch

    tokenizer = setup_tokenizer(model_name, local_files_only=local_files_only)
    model = load_model(model_name, local_files_only=local_files_only, adapter_path=adapter_path)
    safe_name = safe_model_name(variant_name)
    rows = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for index, rec in enumerate(gold.to_dict("records"), start=1):
        row = pd.Series(rec)
        messages = build_messages(row, max_input_chars=max_input_chars)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokenized = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_length,
        )
        device = next(model.parameters()).device
        tokenized = {key: value.to(device) for key, value in tokenized.items()}
        item_started = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(
                **tokenized,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        runtime = max(time.perf_counter() - item_started, 1e-9)
        generated_ids = output_ids[0][tokenized["input_ids"].shape[1] :]
        raw_response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        parsed = parse_model_response(raw_response, row)
        target = target_for_row(row)
        rows.append(
            {
                "record_id": row.get("record_id"),
                "prompt": prompt,
                "prediction": parsed.prediction,
                "label": target,
                "gold_label": row.get("gold_label"),
                "expected_output": row.get("expected_output"),
                "raw_response": raw_response,
                "model_name": variant_name,
                "provider": "local_qlora",
                "task_type": row.get("task_type"),
                "evaluation_head": row.get("evaluation_head"),
                "main_category": row.get("main_category"),
                "source_dataset": row.get("source_dataset"),
                "correct": normalize_text(parsed.prediction) == normalize_text(target),
                "confidence": parsed.confidence,
                "score": parsed.score,
                "probability": parsed.probability,
                "generation_runtime_s": runtime,
                "prompt_tokens": int(tokenized["attention_mask"].sum().item()),
                "completion_tokens": int(len(generated_ids)),
                "created_at": now_utc(),
            }
        )
        print(
            f"[eval] {variant_name} row={index}/{len(gold)} "
            f"pred={parsed.prediction!r} label={target!r} correct={rows[-1]['correct']} "
            f"tok_s={(rows[-1]['completion_tokens'] / runtime):.2f}",
            flush=True,
        )
    elapsed = max(time.perf_counter() - started, 1e-9)
    df = pd.DataFrame(rows)
    detailed_path = out_dir / f"predictions_detailed_{safe_name}.csv"
    compat_path = out_dir / f"predictions_{safe_name}.csv"
    df.to_csv(detailed_path, index=False)
    df[["record_id", "prediction", "model_name", "score", "probability", "confidence", "provider", "task_type", "evaluation_head", "created_at"]].to_csv(
        compat_path, index=False
    )
    summary = {
        "variant": variant_name,
        "adapter_path": str(adapter_path) if adapter_path else None,
        "rows": int(len(df)),
        "elapsed_seconds": elapsed,
        "rows_per_second": float(len(df) / elapsed),
        "vram_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        "vram_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024**2, 2),
        "detailed_predictions": str(detailed_path),
        "compatible_predictions": str(compat_path),
    }
    release_model(model)
    return df, summary


def majority_baseline(gold: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    target = gold.apply(target_for_row, axis=1)
    majority = Counter(target).most_common(1)[0][0]
    df = gold.copy()
    df["prompt"] = ""
    df["prediction"] = majority
    df["label"] = target
    df["model_name"] = "majority_label_baseline"
    df["provider"] = "baseline"
    df["raw_response"] = majority
    df["correct"] = df["prediction"].map(normalize_text) == df["label"].map(normalize_text)
    df["created_at"] = now_utc()
    df.to_csv(out_dir / "predictions_detailed_majority_label_baseline.csv", index=False)
    df[["record_id", "prediction", "model_name", "provider", "task_type", "evaluation_head", "created_at"]].to_csv(
        out_dir / "predictions_majority_label_baseline.csv", index=False
    )
    return df


def metrics_for_predictions(df: pd.DataFrame) -> dict[str, Any]:
    y_true = df["label"].astype(str).tolist()
    y_pred = df["prediction"].astype(str).tolist()
    class_metrics = classification_metrics(y_true, y_pred)
    accuracy = sum(normalize_text(t) == normalize_text(p) for t, p in zip(y_true, y_pred, strict=False)) / max(1, len(y_true))
    rows = []
    for group_by in ("evaluation_head", "main_category", "source_dataset", "task_type"):
        if group_by not in df.columns:
            continue
        for key, group in df.groupby(group_by, dropna=False):
            correct = sum(group["correct"].astype(bool))
            rows.append(
                {
                    "group_by": group_by,
                    "group_value": str(key),
                    "n": int(len(group)),
                    "accuracy": float(correct / max(1, len(group))),
                }
            )
    return {
        "n": int(len(df)),
        "accuracy": float(accuracy),
        "precision_macro": float(class_metrics["precision"]),
        "recall_macro": float(class_metrics["recall"]),
        "f1_macro": float(class_metrics["f1_macro"]),
        "f1_weighted": float(class_metrics["f1_weighted"]),
        "category_metrics": rows,
        "confusion_matrix": class_metrics["confusion_matrix"],
    }


def compare_variants(variant_frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    metrics = {name: metrics_for_predictions(df) for name, df in variant_frames.items()}
    names = list(metrics)
    comparisons = []
    if len(names) >= 2:
        base_name = names[0]
        base = metrics[base_name]
        for name in names[1:]:
            item = metrics[name]
            comparisons.append(
                {
                    "baseline": base_name,
                    "candidate": name,
                    "accuracy_delta": item["accuracy"] - base["accuracy"],
                    "f1_macro_delta": item["f1_macro"] - base["f1_macro"],
                }
            )
    return {"metrics": metrics, "comparisons": comparisons}


def load_prior_baselines(paths: list[Path], gold: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    target = gold[["record_id"]].copy()
    target["label"] = gold.apply(target_for_row, axis=1)
    for path in paths:
        df = read_table(path)
        if not {"record_id", "prediction"} <= set(df.columns):
            continue
        joined = target.merge(df, on="record_id", how="inner")
        if joined.empty:
            continue
        joined["prompt"] = ""
        joined["model_name"] = df.get("model_name", pd.Series([path.stem])).iloc[0] if "model_name" in df.columns else path.stem
        joined["provider"] = "prior_baseline"
        joined["correct"] = joined["prediction"].map(normalize_text) == joined["label"].map(normalize_text)
        for col in ("task_type", "evaluation_head", "main_category", "source_dataset"):
            if col in gold.columns:
                joined = joined.merge(gold[["record_id", col]], on="record_id", how="left")
        out[path.stem] = joined
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", default="auto")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-input-chars", type=int, default=3500)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--skip-adapter", action="store_true")
    parser.add_argument("--baseline-predictions", type=Path, nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.local_files_only:
        cfg["model"]["local_files_only"] = True
    model_name = str(cfg["model"].get("name") or "Qwen/Qwen2.5-Coder-7B-Instruct")
    gold, detected, source_path = detect_and_load_dataset(args.dataset, cfg)
    gold = select_gold_subset(gold, args.limit, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gold_path = args.out_dir / "evaluation_gold_subset.csv"
    gold.to_csv(gold_path, index=False)

    variant_frames: dict[str, pd.DataFrame] = {}
    run_summaries = {
        "created_at": now_utc(),
        "model_name": model_name,
        "detected_format": detected,
        "source_path": source_path,
        "gold_subset": str(gold_path),
        "limit": int(args.limit),
        "variants": [],
    }
    variant_frames["majority_label_baseline"] = majority_baseline(gold, args.out_dir)
    variant_frames.update(load_prior_baselines(args.baseline_predictions, gold))

    if not args.skip_base:
        df, summary = generate_variant_predictions(
            gold=gold,
            variant_name="base_qwen25_coder_7b_instruct",
            model_name=model_name,
            adapter_path=None,
            out_dir=args.out_dir,
            max_input_chars=args.max_input_chars,
            max_new_tokens=args.max_new_tokens,
            max_seq_length=args.max_seq_length,
            local_files_only=args.local_files_only,
        )
        variant_frames["base_qwen25_coder_7b_instruct"] = df
        run_summaries["variants"].append(summary)

    if not args.skip_adapter:
        if not args.adapter_path.exists():
            raise SystemExit(f"adapter path does not exist: {args.adapter_path}")
        df, summary = generate_variant_predictions(
            gold=gold,
            variant_name="qlora_adapter_final_local",
            model_name=model_name,
            adapter_path=args.adapter_path,
            out_dir=args.out_dir,
            max_input_chars=args.max_input_chars,
            max_new_tokens=args.max_new_tokens,
            max_seq_length=args.max_seq_length,
            local_files_only=args.local_files_only,
        )
        variant_frames["qlora_adapter_final_local"] = df
        run_summaries["variants"].append(summary)

    comparison = compare_variants(variant_frames)
    (args.out_dir / "evaluation_summary.json").write_text(json.dumps({**run_summaries, **comparison}, indent=2, sort_keys=True), encoding="utf-8")
    metric_rows = []
    for name, values in comparison["metrics"].items():
        metric_rows.append(
            {
                "variant": name,
                "n": values["n"],
                "accuracy": values["accuracy"],
                "precision_macro": values["precision_macro"],
                "recall_macro": values["recall_macro"],
                "f1_macro": values["f1_macro"],
                "f1_weighted": values["f1_weighted"],
            }
        )
    pd.DataFrame(metric_rows).to_csv(args.out_dir / "model_comparison_metrics.csv", index=False)
    print(json.dumps({**run_summaries, **comparison}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
