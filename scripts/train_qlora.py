"""Local QLoRA validation training for defensive cybersecurity benchmark rows."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from scripts.build_gold_benchmark import (
    DEFAULT_CONFIG as DEFAULT_BENCHMARK_CONFIG,
    PROJECT_ROOT,
    build_benchmark,
    read_table,
    transform_silver_to_gold,
)
from scripts.run_model_predictions import build_messages


DEFAULT_CONFIG = PROJECT_ROOT / "config" / "qlora_local_qwen25_coder_7b.yml"
DEFAULT_GOLD = PROJECT_ROOT / "data" / "gold" / "benchmark_gold.csv"
DEFAULT_SILVER = PROJECT_ROOT / "data" / "silver_normalized"

GOLD_REQUIRED_COLUMNS = {"record_id", "input_text", "expected_output", "task_type", "evaluation_head", "gold_label"}
SILVER_HINT_COLUMNS = {"record_id", "source_dataset", "source_type", "main_category", "label", "binary_label"}
SFT_CHAT_COLUMNS = {"messages"}


class QLoRASetupError(RuntimeError):
    """Raised for actionable local setup failures."""


@dataclass(frozen=True)
class DatasetBundle:
    """Prepared train/eval rows and source metadata."""

    train: pd.DataFrame
    eval: pd.DataFrame
    detected_format: str
    source_path: str


class ChatSFTDataset:
    """Small tokenized causal-LM dataset with prompt tokens masked from loss."""

    def __init__(self, examples: list[dict[str, list[int]]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


class DataCollatorForCausalChat:
    """Pad Qwen chat examples and keep ignored labels as -100."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_len = max(len(item["input_ids"]) for item in features)
        input_ids, attention_mask, labels = [], [], []
        for item in features:
            pad = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * pad)
            attention_mask.append(item["attention_mask"] + [0] * pad)
            labels.append(item["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def parameter_counts(model: Any) -> dict[str, int]:
    """Return total and trainable parameter counts."""

    trainable = 0
    total = 0
    for param in model.parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
    return {"trainable_parameters": int(trainable), "total_parameters": int(total)}


def validate_adapter_artifacts(path: Path) -> dict[str, Any]:
    """Validate that a PEFT adapter directory has the expected non-empty files."""

    expected = ["adapter_config.json", "adapter_model.safetensors"]
    files = {}
    missing = []
    for name in expected:
        item = path / name
        exists = item.exists()
        size = item.stat().st_size if exists else 0
        files[name] = {"exists": exists, "bytes": int(size)}
        if not exists or size <= 0:
            missing.append(name)
    return {
        "path": str(path),
        "ok": not missing,
        "missing_or_empty": missing,
        "files": files,
    }


def summarize_stability(step_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize loss, throughput, and memory stability for local runs."""

    if not step_metrics:
        return {}
    losses = [float(item["loss"]) for item in step_metrics]
    tps = [float(item["tokens_per_second"]) for item in step_metrics]
    allocated = [float(item["vram_allocated_mb"]) for item in step_metrics]
    reserved = [float(item["vram_reserved_mb"]) for item in step_metrics]
    median_loss = statistics.median(losses)
    spike_threshold = max(median_loss * 4.0, median_loss + 5.0, 1.0)
    spikes = [
        {"step": item["step"], "loss": item["loss"]}
        for item in step_metrics
        if float(item["loss"]) > spike_threshold
    ]
    return {
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_min": min(losses),
        "loss_max": max(losses),
        "loss_mean": statistics.mean(losses),
        "loss_median": median_loss,
        "loss_spike_threshold": spike_threshold,
        "loss_spikes": spikes,
        "throughput_mean": statistics.mean(tps),
        "throughput_min": min(tps),
        "throughput_max": max(tps),
        "throughput_stdev": statistics.pstdev(tps) if len(tps) > 1 else 0.0,
        "vram_allocated_min_mb": min(allocated),
        "vram_allocated_max_mb": max(allocated),
        "vram_allocated_delta_mb": max(allocated) - min(allocated),
        "vram_reserved_min_mb": min(reserved),
        "vram_reserved_max_mb": max(reserved),
        "vram_reserved_delta_mb": max(reserved) - min(reserved),
    }


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML config with simple defaults."""

    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    cfg.setdefault("model", {})
    cfg.setdefault("dataset", {})
    cfg.setdefault("qlora", {})
    cfg.setdefault("training", {})
    return cfg


def _read_manifest(manifest_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    frames = []
    for value in manifest.get("output_path_parquet", pd.Series(dtype=str)).dropna():
        path = PROJECT_ROOT / str(value)
        if path.exists():
            frames.append(read_table(path))
    if not frames:
        for value in manifest.get("output_path_csv_gz", pd.Series(dtype=str)).dropna():
            path = PROJECT_ROOT / str(value)
            if path.exists():
                frames.append(read_table(path))
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _discover_silver_files(path: Path) -> list[Path]:
    patterns = ("*/*.parquet", "*/*.csv.gz", "*.parquet", "*.csv.gz", "*.csv", "*.jsonl")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(path.glob(pattern))
    return sorted(
        item
        for item in set(files)
        if item.is_file()
        and "metadata" not in item.name
        and item.name != "silver_manifest.csv"
        and "_dedup" not in item.parts
    )


def _load_sft_chat_dir(path: Path) -> pd.DataFrame | None:
    frames = []
    for split_name in ("train", "eval", "validation", "test"):
        split_path = path / f"{split_name}.jsonl"
        if split_path.exists():
            df = read_table(split_path)
            if not df.empty and SFT_CHAT_COLUMNS <= set(df.columns):
                df = df.copy()
                if "split" not in df.columns:
                    df["split"] = "eval" if split_name == "validation" else split_name
                frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True, sort=False)


def detect_and_load_dataset(source: str | Path, cfg: dict[str, Any]) -> tuple[pd.DataFrame, str, str]:
    """Detect gold, silver, silver manifest, SFT chat JSONL, or auto source."""

    source_text = str(source)
    if source_text == "auto":
        if DEFAULT_GOLD.exists():
            source = DEFAULT_GOLD
        elif (DEFAULT_SILVER / "silver_manifest.csv").exists():
            source = DEFAULT_SILVER / "silver_manifest.csv"
        else:
            source = DEFAULT_SILVER
    source_path = Path(source)

    if source_path.is_file():
        if source_path.name == "silver_manifest.csv":
            silver = _read_manifest(source_path)
            return transform_silver_to_gold(silver, _benchmark_config(), int(cfg["dataset"].get("seed", 42))), "silver_manifest", str(source_path)
        table = read_table(source_path)
        columns = set(table.columns)
        if SFT_CHAT_COLUMNS <= columns:
            return table, "sft_chat", str(source_path)
        if GOLD_REQUIRED_COLUMNS <= columns:
            return table, "gold", str(source_path)
        if SILVER_HINT_COLUMNS <= columns:
            return transform_silver_to_gold(table, _benchmark_config(), int(cfg["dataset"].get("seed", 42))), "silver_table", str(source_path)
        raise QLoRASetupError(f"Could not detect dataset format for {source_path}; columns={sorted(columns)}")

    if source_path.is_dir():
        sft = _load_sft_chat_dir(source_path)
        if sft is not None:
            return sft, "sft_chat_dir", str(source_path)
        gold = source_path / "benchmark_gold.csv"
        if gold.exists():
            return read_table(gold), "gold_dir", str(gold)
        manifest = source_path / "silver_manifest.csv"
        if manifest.exists():
            silver = _read_manifest(manifest)
            return transform_silver_to_gold(silver, _benchmark_config(), int(cfg["dataset"].get("seed", 42))), "silver_manifest_dir", str(manifest)
        frames = [read_table(path) for path in _discover_silver_files(source_path)]
        if frames:
            silver = pd.concat(frames, ignore_index=True, sort=False)
            return transform_silver_to_gold(silver, _benchmark_config(), int(cfg["dataset"].get("seed", 42))), "silver_dir", str(source_path)

    if source_text == "auto" and DEFAULT_SILVER.exists():
        output_dir = Path(cfg["training"].get("output_dir", "outputs/qlora")) / "prepared_gold"
        gold = build_benchmark(
            DEFAULT_SILVER,
            output_dir,
            max_rows=int(cfg["dataset"].get("gold_build_max_rows", 256)),
            seed=int(cfg["dataset"].get("seed", 42)),
            output_format="csv",
            dry_run=False,
        )
        return gold, "built_gold_from_silver", str(output_dir / "benchmark_gold.csv")

    raise QLoRASetupError(f"No usable dataset found at {source_path}")


def _benchmark_config() -> dict[str, Any]:
    with DEFAULT_BENCHMARK_CONFIG.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def split_train_eval(gold: pd.DataFrame, cfg: dict[str, Any]) -> DatasetBundle:
    """Select deterministic train/eval splits from gold-shaped or SFT chat rows."""

    seed = int(cfg["dataset"].get("seed", 42))
    train_rows = int(cfg["dataset"].get("train_rows", 8))
    eval_rows = int(cfg["dataset"].get("eval_rows", 4))
    if SFT_CHAT_COLUMNS <= set(gold.columns):
        sft = gold.copy()
        if "record_id" not in sft.columns:
            sft["record_id"] = sft.get("example_id", sft.get("source_record_id", pd.Series([f"sft_{i}" for i in range(len(sft))])))
        sft = sft.dropna(subset=["record_id", "messages"]).sort_values("record_id").sample(frac=1.0, random_state=seed).reset_index(drop=True)
        if "split" in sft.columns:
            split_values = sft["split"].astype(str)
            train_pool = sft[split_values == "train"]
            eval_pool = sft[split_values.isin({"eval", "validation", "test"})]
        else:
            train_pool = sft
            eval_pool = sft.iloc[0:0]
        if train_pool.empty:
            train_pool = sft
        train = train_pool.head(train_rows).copy()
        eval_pool = eval_pool[~eval_pool["record_id"].astype(str).isin(set(train["record_id"].astype(str)))]
        if eval_pool.empty:
            eval_pool = sft[~sft["record_id"].astype(str).isin(set(train["record_id"].astype(str)))]
        eval_df = eval_pool.head(eval_rows).copy()
        if train.empty:
            raise QLoRASetupError("No SFT training rows were available after dataset detection.")
        return DatasetBundle(train=train, eval=eval_df, detected_format="", source_path="")

    gold = gold.dropna(subset=["record_id", "input_text"]).copy()
    gold["expected_output"] = gold["expected_output"].fillna(gold.get("gold_label", "unknown")).fillna("unknown")
    gold = gold.sort_values("record_id").sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if "split" in gold.columns:
        train_pool = gold[~gold["split"].astype(str).isin({"test", "validation"})]
        eval_pool = gold[gold["split"].astype(str).isin({"validation", "test"})]
    else:
        train_pool = gold
        eval_pool = gold.iloc[0:0]
    if train_pool.empty:
        train_pool = gold
    train = train_pool.head(train_rows).copy()
    eval_pool = eval_pool[~eval_pool["record_id"].astype(str).isin(set(train["record_id"].astype(str)))]
    if eval_pool.empty:
        eval_pool = gold[~gold["record_id"].astype(str).isin(set(train["record_id"].astype(str)))]
    eval_df = eval_pool.head(eval_rows).copy()
    if train.empty:
        raise QLoRASetupError("No training rows were available after dataset detection.")
    return DatasetBundle(train=train, eval=eval_df, detected_format="", source_path="")


def assistant_payload(row: pd.Series) -> str:
    """Build the supervised assistant target in benchmark-compatible JSON."""

    task_type = str(row.get("task_type") or "classification")
    prediction = str(row.get("expected_output") or row.get("gold_label") or row.get("label") or "unknown").strip()
    if task_type == "classification":
        prediction = str(row.get("gold_label") or prediction)
    payload = {
        "prediction": prediction,
        "confidence": 1.0,
        "explanation": "Gold supervised target for local pipeline validation.",
    }
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def make_chat_messages(row: pd.Series, max_input_chars: int) -> list[dict[str, str]]:
    """Reuse benchmark-safe prompt construction and append the supervised answer."""

    messages = build_messages(row, max_input_chars=max_input_chars)
    messages.append({"role": "assistant", "content": assistant_payload(row)})
    return messages


def coerce_messages(value: Any) -> list[dict[str, str]]:
    """Return a normalized list of chat messages from JSON or Python objects."""

    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise QLoRASetupError("SFT chat rows must contain a messages list.")
    messages = []
    for item in value:
        if not isinstance(item, dict):
            raise QLoRASetupError("Each SFT chat message must be an object with role/content.")
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "")
        if not role or not content:
            raise QLoRASetupError("Each SFT chat message must include non-empty role and content.")
        messages.append({"role": role, "content": content})
    return messages


def supervised_prompt_and_answer(record: dict[str, Any], max_input_chars: int) -> tuple[list[dict[str, str]], str]:
    """Return prompt messages and supervised assistant answer for either schema."""

    if "messages" in record and record.get("messages") is not None:
        messages = coerce_messages(record["messages"])
        if messages and messages[-1]["role"] == "assistant":
            return messages[:-1], messages[-1]["content"]
        return messages, ""
    row = pd.Series(record)
    return build_messages(row, max_input_chars=max_input_chars), assistant_payload(row)


def write_prepared_jsonl(rows: pd.DataFrame, path: Path, max_input_chars: int) -> None:
    """Persist prepared chat examples for inspection/reproducibility."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows.to_dict("records"):
            prompt_messages, answer = supervised_prompt_and_answer(row, max_input_chars=max_input_chars)
            messages = [*prompt_messages, {"role": "assistant", "content": answer}]
            fh.write(
                json.dumps(
                    {
                        "record_id": row.get("record_id") or row.get("example_id") or row.get("source_record_id"),
                        "messages": messages,
                    },
                    sort_keys=True,
                )
            )
            fh.write("\n")


def setup_tokenizer(model_name: str, cfg: dict[str, Any]):
    """Load Qwen tokenizer and ensure a pad token exists."""

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise QLoRASetupError("transformers is required for QLoRA training. Install transformers and retry.") from exc
    print(f"[qlora] loading tokenizer: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=bool(cfg["model"].get("trust_remote_code", True)),
        local_files_only=bool(cfg["model"].get("local_files_only", False)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def tokenize_rows(rows: pd.DataFrame, tokenizer: Any, cfg: dict[str, Any]) -> ChatSFTDataset:
    """Tokenize chat rows and mask prompt tokens from supervised loss."""

    max_len = int(cfg["model"].get("max_seq_length", 512))
    max_input_chars = int(cfg["dataset"].get("max_input_chars", 3500))
    examples: list[dict[str, list[int]]] = []
    for record in rows.to_dict("records"):
        prompt_messages, answer = supervised_prompt_and_answer(record, max_input_chars=max_input_chars)
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        answer_text = answer + (tokenizer.eos_token or "")
        answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
        if len(answer_ids) >= max_len:
            input_ids = answer_ids[:max_len]
            labels = input_ids.copy()
        else:
            prompt_budget = max_len - len(answer_ids)
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False, truncation=True, max_length=prompt_budget)["input_ids"]
            input_ids = prompt_ids + answer_ids
            labels = [-100] * len(prompt_ids) + answer_ids.copy()
        if all(label == -100 for label in labels) and labels:
            labels[-1] = input_ids[-1]
        examples.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels})
    return ChatSFTDataset(examples)


def _torch_dtype() -> Any:
    import torch

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def setup_model(cfg: dict[str, Any]):
    """Load Qwen in 4-bit and attach LoRA adapters."""

    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError as exc:
        raise QLoRASetupError("Missing QLoRA dependency. Install torch, transformers, peft, accelerate, and bitsandbytes.") from exc
    if not torch.cuda.is_available():
        raise QLoRASetupError("CUDA is not available; this local QLoRA path requires a CUDA GPU.")

    dtype = _torch_dtype()
    quant = BitsAndBytesConfig(
        load_in_4bit=bool(cfg["qlora"].get("load_in_4bit", True)),
        bnb_4bit_quant_type=str(cfg["qlora"].get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=bool(cfg["qlora"].get("bnb_4bit_use_double_quant", True)),
    )
    model_kwargs = {
        "quantization_config": quant,
        "device_map": "auto",
        "trust_remote_code": bool(cfg["model"].get("trust_remote_code", True)),
        "local_files_only": bool(cfg["model"].get("local_files_only", False)),
        "low_cpu_mem_usage": bool(cfg["training"].get("low_cpu_mem_usage", True)),
    }
    attn = cfg["model"].get("attn_implementation")
    if attn:
        model_kwargs["attn_implementation"] = attn

    print("[qlora] loading base model with 4-bit quantization", flush=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(str(cfg["model"]["name"]), **model_kwargs)
    except Exception as exc:
        raise QLoRASetupError(
            "Failed to load the 4-bit model. Check network/cache access for "
            f"{cfg['model']['name']} and verify bitsandbytes CUDA support. Original error: {exc}"
        ) from exc

    model.config.use_cache = False
    if bool(cfg["training"].get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
    print("[qlora] preparing model for k-bit training", flush=True)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=bool(cfg["training"].get("gradient_checkpointing", True)))
    lora_config = LoraConfig(
        r=int(cfg["qlora"].get("lora_r", 16)),
        lora_alpha=int(cfg["qlora"].get("lora_alpha", 32)),
        lora_dropout=float(cfg["qlora"].get("lora_dropout", 0.05)),
        target_modules=list(cfg["qlora"].get("target_modules", [])),
        bias="none",
        task_type="CAUSAL_LM",
    )
    print("[qlora] injecting LoRA adapters", flush=True)
    return get_peft_model(model, lora_config)


def build_training_args(cfg: dict[str, Any]):
    """Build TrainingArguments across recent Transformers versions."""

    from transformers import TrainingArguments

    training = cfg["training"]
    kwargs = {
        "output_dir": str(PROJECT_ROOT / str(training.get("output_dir", "outputs/qlora/local"))),
        "num_train_epochs": float(training.get("num_train_epochs", 1)),
        "per_device_train_batch_size": int(training.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(training.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(training.get("gradient_accumulation_steps", 1)),
        "learning_rate": float(training.get("learning_rate", 2e-4)),
        "weight_decay": float(training.get("weight_decay", 0.0)),
        "warmup_ratio": float(training.get("warmup_ratio", 0.0)),
        "max_grad_norm": float(training.get("max_grad_norm", 0.3)),
        "logging_steps": int(training.get("logging_steps", 1)),
        "save_strategy": str(training.get("save_strategy", "epoch")),
        "save_total_limit": int(training.get("save_total_limit", 2)),
        "optim": str(training.get("optim", "paged_adamw_8bit")),
        "report_to": list(training.get("report_to", [])),
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
        "fp16": False,
        "bf16": _torch_dtype().__str__().endswith("bfloat16"),
    }
    if training.get("max_steps") is not None:
        kwargs["max_steps"] = int(training["max_steps"])
    import inspect

    params = inspect.signature(TrainingArguments.__init__).parameters
    strategy_key = "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"
    kwargs[strategy_key] = str(training.get("eval_strategy", "epoch"))
    return TrainingArguments(**kwargs)


def _prepare_datasets(cfg: dict[str, Any], bundle: DatasetBundle) -> tuple[Any, ChatSFTDataset, ChatSFTDataset | None, dict[str, Any], Path]:
    """Prepare tokenizer, tokenized datasets, and persisted JSONL examples."""

    output_dir = PROJECT_ROOT / str(cfg["training"].get("output_dir", "outputs/qlora/local"))
    prepared_dir = output_dir / "prepared"
    max_input_chars = int(cfg["dataset"].get("max_input_chars", 3500))
    write_prepared_jsonl(bundle.train, prepared_dir / "train_chat.jsonl", max_input_chars)
    write_prepared_jsonl(bundle.eval, prepared_dir / "eval_chat.jsonl", max_input_chars)
    print(
        f"[qlora] dataset ready: train_rows={len(bundle.train)} eval_rows={len(bundle.eval)} "
        f"format={bundle.detected_format}",
        flush=True,
    )

    model_name = str(cfg["model"].get("name") or "Qwen/Qwen2.5-Coder-7B-Instruct")
    tokenizer = setup_tokenizer(model_name, cfg)
    train_dataset = tokenize_rows(bundle.train, tokenizer, cfg)
    eval_dataset = tokenize_rows(bundle.eval, tokenizer, cfg) if not bundle.eval.empty else None
    prep_summary = {
        "detected_format": bundle.detected_format,
        "source_path": bundle.source_path,
        "train_rows": len(bundle.train),
        "eval_rows": len(bundle.eval),
        "train_tokens": int(sum(sum(x["attention_mask"]) for x in train_dataset.examples)),
        "eval_tokens": int(sum(sum(x["attention_mask"]) for x in eval_dataset.examples)) if eval_dataset else 0,
        "prepared_train": str(prepared_dir / "train_chat.jsonl"),
        "prepared_eval": str(prepared_dir / "eval_chat.jsonl"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prepare_summary.json").write_text(json.dumps(prep_summary, indent=2, sort_keys=True), encoding="utf-8")
    return tokenizer, train_dataset, eval_dataset, prep_summary, output_dir


def evaluate_dataset_loss(model: Any, collator: DataCollatorForCausalChat, dataset: ChatSFTDataset | None, device: Any, limit: int) -> dict[str, float]:
    """Evaluate a bounded number of batches for cheap local tracking."""

    if dataset is None or len(dataset) == 0 or limit <= 0:
        return {}
    import torch

    was_training = model.training
    model.eval()
    losses = []
    tokens = 0
    with torch.no_grad():
        for index in range(min(limit, len(dataset))):
            batch = collator([dataset[index]])
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            losses.append(float(outputs.loss.detach().float().cpu().item()))
            tokens += int(batch["attention_mask"].sum().item())
    if was_training:
        model.train()
    return {
        "eval_loss": float(statistics.mean(losses)) if losses else math.nan,
        "eval_batches": float(len(losses)),
        "eval_tokens": float(tokens),
    }


def learning_rate_for_step(base_lr: float, step: int, max_steps: int, warmup_ratio: float, scheduler_type: str) -> float:
    """Small deterministic LR schedule for the manual local loop."""

    warmup_steps = int(max_steps * warmup_ratio)
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * (step / warmup_steps)
    if scheduler_type == "cosine" and max_steps > warmup_steps:
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
    if scheduler_type == "linear" and max_steps > warmup_steps:
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return base_lr * max(0.0, 1.0 - progress)
    return base_lr


def _generate_validation_sample(model: Any, tokenizer: Any, row: pd.Series, cfg: dict[str, Any]) -> dict[str, Any]:
    """Generate one short adapter-backed response for qualitative validation."""

    import torch

    max_input_chars = int(cfg["dataset"].get("max_input_chars", 3500))
    record = row.to_dict()
    messages, expected_output = supervised_prompt_and_answer(record, max_input_chars=max_input_chars)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokenized = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=True,
        max_length=int(cfg["model"].get("max_seq_length", 512)),
    )
    device = next(model.parameters()).device
    tokenized = {key: value.to(device) for key, value in tokenized.items()}
    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            **tokenized,
            max_new_tokens=int(cfg["training"].get("validation_max_new_tokens", 96)),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output_ids[0][tokenized["input_ids"].shape[1] :]
    return {
        "record_id": str(row.get("record_id") or row.get("example_id") or row.get("source_record_id") or ""),
        "expected_output": expected_output or str(row.get("expected_output") or row.get("gold_label") or ""),
        "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=True).strip(),
    }


def manual_smoke_train(cfg: dict[str, Any], bundle: DatasetBundle, prepare_only: bool = False) -> dict[str, Any]:
    """Run an explicit tiny QLoRA loop and save adapter artifacts."""

    import torch

    tokenizer, train_dataset, eval_dataset, prep_summary, output_dir = _prepare_datasets(cfg, bundle)
    if prepare_only:
        return prep_summary

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = setup_model(cfg)
    counts = parameter_counts(model)
    print(
        f"[qlora] trainable parameters: {counts['trainable_parameters']:,} / {counts['total_parameters']:,}",
        flush=True,
    )

    collator = DataCollatorForCausalChat(pad_token_id=int(tokenizer.pad_token_id))
    device = next(model.parameters()).device
    params = [param for param in model.parameters() if param.requires_grad]
    base_learning_rate = float(cfg["training"].get("learning_rate", 2e-4))
    optimizer = torch.optim.AdamW(
        params,
        lr=base_learning_rate,
        weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
    )
    checkpoint_every = int(cfg["training"].get("checkpoint_every") or 0)
    eval_every = int(cfg["training"].get("eval_every") or 0)
    eval_batches = int(cfg["training"].get("eval_batches") or 4)
    grad_accum = max(1, int(cfg["training"].get("gradient_accumulation_steps", 1)))
    warmup_ratio = float(cfg["training"].get("warmup_ratio", 0.0))
    scheduler_type = str(cfg["training"].get("lr_scheduler_type", "constant"))

    model.train()
    started = time.perf_counter()
    max_steps = int(cfg["training"].get("max_steps") or 1)
    step_metrics = []
    checkpoint_validations = []
    total_train_tokens = 0
    last_loss = None
    for step in range(max_steps):
        batch_started = time.perf_counter()
        learning_rate = learning_rate_for_step(base_learning_rate, step + 1, max_steps, warmup_ratio, scheduler_type)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        micro_losses = []
        step_tokens = 0
        for micro_step in range(grad_accum):
            row_index = ((step * grad_accum) + micro_step) % len(train_dataset)
            batch = collator([train_dataset[row_index]])
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            raw_loss = outputs.loss
            (raw_loss / grad_accum).backward()
            micro_losses.append(float(raw_loss.detach().float().cpu().item()))
            step_tokens += int(batch["attention_mask"].sum().item())
        optimizer.step()
        step_runtime = max(time.perf_counter() - batch_started, 1e-9)
        total_train_tokens += step_tokens
        last_loss = statistics.mean(micro_losses)
        current = {
            "step": step + 1,
            "loss": float(last_loss),
            "runtime_seconds": float(step_runtime),
            "tokens": step_tokens,
            "tokens_per_second": float(step_tokens / step_runtime),
            "learning_rate": learning_rate,
            "gradient_accumulation_steps": grad_accum,
            "vram_allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 2),
            "vram_reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 2),
        }
        if eval_every > 0 and eval_dataset is not None and (step + 1) % eval_every == 0:
            current.update(evaluate_dataset_loss(model, collator, eval_dataset, device, eval_batches))
        step_metrics.append(current)
        current = step_metrics[-1]
        eval_text = f" eval_loss={current['eval_loss']:.6f}" if "eval_loss" in current else ""
        print(
            "[qlora] "
            f"step={current['step']}/{max_steps} "
            f"loss={current['loss']:.6f} "
            f"lr={current['learning_rate']:.8f} "
            f"tok_s={current['tokens_per_second']:.2f} "
            f"vram_alloc_mb={current['vram_allocated_mb']:.2f} "
            f"vram_reserved_mb={current['vram_reserved_mb']:.2f}"
            f"{eval_text}",
            flush=True,
        )
        if checkpoint_every > 0 and current["step"] % checkpoint_every == 0:
            step_checkpoint = output_dir / f"checkpoint-{current['step']}"
            model.save_pretrained(str(step_checkpoint))
            validation = validate_adapter_artifacts(step_checkpoint)
            checkpoint_validations.append(validation)
            print(
                f"[qlora] cadence checkpoint step={current['step']} ok={validation['ok']} path={step_checkpoint}",
                flush=True,
            )
    runtime = max(time.perf_counter() - started, 1e-9)
    checkpoint_dir = output_dir / f"checkpoint-{max_steps}"
    final_dir = output_dir / "final_adapter"
    model.save_pretrained(str(checkpoint_dir))
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    checkpoint_validation = validate_adapter_artifacts(checkpoint_dir)
    final_adapter_validation = validate_adapter_artifacts(final_dir)
    checkpoint_validations.append(checkpoint_validation)
    print(
        f"[qlora] checkpoint validation: checkpoint_ok={checkpoint_validation['ok']} "
        f"final_adapter_ok={final_adapter_validation['ok']}",
        flush=True,
    )

    eval_metrics: dict[str, float] = {}
    if eval_dataset is not None and len(eval_dataset) > 0:
        eval_metrics.update(evaluate_dataset_loss(model, collator, eval_dataset, device, eval_batches))

    generation_sample = _generate_validation_sample(model, tokenizer, bundle.eval.iloc[0] if not bundle.eval.empty else bundle.train.iloc[0], cfg)

    metrics = {
        **prep_summary,
        **counts,
        "train_loss": float(last_loss if last_loss is not None else 0.0),
        "loss_history": [item["loss"] for item in step_metrics],
        "step_metrics": step_metrics,
        "train_runtime": float(runtime),
        "gradient_accumulation_steps": grad_accum,
        "tokens_per_second": float(total_train_tokens / runtime),
        "vram_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        "vram_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024**2, 2),
        "eval_metrics": eval_metrics,
        "generation_sample": generation_sample,
        "output_dir": str(output_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "final_adapter": str(final_dir),
        "checkpoint_validation": checkpoint_validation,
        "checkpoint_validations": checkpoint_validations,
        "final_adapter_validation": final_adapter_validation,
        "stability_summary": summarize_stability(step_metrics),
        "smoke_mode": "manual_tiny_loop",
    }
    (output_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    return metrics


def train(cfg: dict[str, Any], bundle: DatasetBundle, prepare_only: bool = False) -> dict[str, Any]:
    """Prepare data, optionally run tiny local QLoRA training, and return metrics."""

    import torch
    from transformers import Trainer

    tokenizer, train_dataset, eval_dataset, prep_summary, output_dir = _prepare_datasets(cfg, bundle)
    if prepare_only:
        return prep_summary

    torch.cuda.reset_peak_memory_stats()
    model = setup_model(cfg)
    counts = parameter_counts(model)
    model.print_trainable_parameters()
    args = build_training_args(cfg)
    collator = DataCollatorForCausalChat(pad_token_id=int(tokenizer.pad_token_id))
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    print("[qlora] starting Trainer.train()", flush=True)
    started = time.perf_counter()
    try:
        train_result = trainer.train()
    except Exception as exc:
        optim = str(cfg["training"].get("optim", ""))
        if optim != "adamw_torch" and any(text in str(exc).lower() for text in ("bitsandbytes", "8-bit", "paged_adamw")):
            print(f"[qlora] optimizer failed with {optim}; retrying once with adamw_torch. root_cause={exc}", flush=True)
            cfg["training"]["optim"] = "adamw_torch"
            args = build_training_args(cfg)
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=collator,
            )
            train_result = trainer.train()
        else:
            raise
    runtime = max(time.perf_counter() - started, 1e-9)
    trainer.save_model(str(output_dir / "final_adapter"))
    tokenizer.save_pretrained(str(output_dir / "final_adapter"))
    eval_metrics = trainer.evaluate() if eval_dataset else {}
    train_metrics = dict(train_result.metrics)
    train_tokens = prep_summary["train_tokens"] * float(cfg["training"].get("num_train_epochs", 1))
    metrics = {
        **prep_summary,
        **counts,
        "train_loss": float(train_metrics.get("train_loss", math.nan)),
        "train_runtime": float(train_metrics.get("train_runtime", runtime)),
        "tokens_per_second": float(train_tokens / max(float(train_metrics.get("train_runtime", runtime)), 1e-9)),
        "vram_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        "vram_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024**2, 2),
        "eval_metrics": {k: float(v) if isinstance(v, (int, float)) else v for k, v in eval_metrics.items()},
        "output_dir": str(output_dir),
        "final_adapter": str(output_dir / "final_adapter"),
    }
    (output_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=str, default=None, help="auto, a gold/silver file, a silver manifest, or a directory.")
    parser.add_argument("--prepare-only", action="store_true", help="Only detect/format/tokenize data; do not load or train the model.")
    parser.add_argument("--train-rows", type=int, default=None)
    parser.add_argument("--eval-rows", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--local-files-only", action="store_true", help="Use only the local Hugging Face cache.")
    parser.add_argument("--max-steps", type=int, default=None, help="Cap training steps for smoke tests.")
    parser.add_argument("--max-seq-length", type=int, default=None, help="Override tokenized sequence length.")
    parser.add_argument("--checkpoint-every", type=int, default=None, help="Save and validate LoRA adapter checkpoints every N steps.")
    parser.add_argument("--eval-every", type=int, default=None, help="Evaluate a small bounded set every N optimizer steps in manual loop.")
    parser.add_argument("--eval-batches", type=int, default=None, help="Number of eval examples to score at each manual eval point.")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=None)
    parser.add_argument("--manual-smoke", action="store_true", help="Run a direct one-step smoke loop and save adapter metrics.")
    parser.add_argument("--manual-loop", action="store_true", help="Run the direct instrumented local training loop.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.train_rows is not None:
        cfg["dataset"]["train_rows"] = args.train_rows
    if args.eval_rows is not None:
        cfg["dataset"]["eval_rows"] = args.eval_rows
    if args.output_dir:
        cfg["training"]["output_dir"] = args.output_dir
    if args.local_files_only:
        cfg["model"]["local_files_only"] = True
    if args.max_steps is not None:
        cfg["training"]["max_steps"] = args.max_steps
    if args.max_seq_length is not None:
        cfg["model"]["max_seq_length"] = args.max_seq_length
    if args.checkpoint_every is not None:
        cfg["training"]["checkpoint_every"] = args.checkpoint_every
    if args.eval_every is not None:
        cfg["training"]["eval_every"] = args.eval_every
    if args.eval_batches is not None:
        cfg["training"]["eval_batches"] = args.eval_batches
    if args.learning_rate is not None:
        cfg["training"]["learning_rate"] = args.learning_rate
    if args.gradient_accumulation_steps is not None:
        cfg["training"]["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.lora_r is not None:
        cfg["qlora"]["lora_r"] = args.lora_r
    if args.lora_alpha is not None:
        cfg["qlora"]["lora_alpha"] = args.lora_alpha
    if args.lora_dropout is not None:
        cfg["qlora"]["lora_dropout"] = args.lora_dropout
    source = args.dataset or cfg["dataset"].get("source", "auto")

    try:
        gold, detected, source_path = detect_and_load_dataset(source, cfg)
        bundle = split_train_eval(gold, cfg)
        bundle = DatasetBundle(bundle.train, bundle.eval, detected, source_path)
        runner = manual_smoke_train if (args.manual_smoke or args.manual_loop) else train
        metrics = runner(cfg, bundle, prepare_only=args.prepare_only)
        if args.prepare_only:
            print(json.dumps(metrics, indent=2, sort_keys=True))
    except QLoRASetupError as exc:
        raise SystemExit(f"QLoRA setup failed: {exc}") from exc


if __name__ == "__main__":
    main()
