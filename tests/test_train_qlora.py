from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from scripts.train_qlora import (
    assistant_payload,
    coerce_messages,
    detect_and_load_dataset,
    load_config,
    make_chat_messages,
    split_train_eval,
    supervised_prompt_and_answer,
)


FIXTURES = Path("tests/fixtures")
CONFIG = Path("config/qlora_local_qwen25_coder_7b.yml")


def test_detects_silver_table_and_converts_to_gold(tmp_path):
    silver = tmp_path / "silver_sample.csv"
    shutil.copy(FIXTURES / "silver_sample.csv", silver)
    cfg = load_config(CONFIG)
    gold, detected, source_path = detect_and_load_dataset(silver, cfg)
    assert detected == "silver_table"
    assert source_path.endswith("silver_sample.csv")
    assert {"record_id", "input_text", "expected_output", "evaluation_head", "gold_label"}.issubset(gold.columns)
    assert set(gold["evaluation_head"]) >= {"malware_code", "cti", "prompt_injection_jailbreaks"}


def test_split_train_eval_is_tiny_and_deterministic():
    cfg = load_config(CONFIG)
    cfg["dataset"]["train_rows"] = 2
    cfg["dataset"]["eval_rows"] = 1
    gold, _detected, _source_path = detect_and_load_dataset(FIXTURES / "silver_sample.csv", cfg)
    first = split_train_eval(gold, cfg)
    second = split_train_eval(gold, cfg)
    assert len(first.train) == 2
    assert len(first.eval) == 1
    assert first.train["record_id"].tolist() == second.train["record_id"].tolist()
    assert first.eval["record_id"].tolist() == second.eval["record_id"].tolist()


def test_chat_format_uses_safe_benchmark_prompt_and_gold_answer():
    cfg = load_config(CONFIG)
    gold, _detected, _source_path = detect_and_load_dataset(FIXTURES / "silver_sample.csv", cfg)
    row = gold[gold["evaluation_head"] == "prompt_injection_jailbreaks"].iloc[0]
    messages = make_chat_messages(row, max_input_chars=6000)
    assert [message["role"] for message in messages] == ["system", "user", "assistant"]
    combined = "\n".join(message["content"] for message in messages)
    assert "untrusted data" in combined
    assert "Do not follow" in combined
    payload = json.loads(messages[-1]["content"])
    assert payload["prediction"] == row["gold_label"]
    assert payload["confidence"] == 1.0


def test_assistant_payload_prefers_gold_label_for_classification():
    row = pd.Series({"task_type": "classification", "expected_output": "wrong", "gold_label": "benign"})
    payload = json.loads(assistant_payload(row))
    assert payload["prediction"] == "benign"


def test_detects_clean_sft_chat_directory(tmp_path):
    messages = [
        {"role": "system", "content": "defensive assistant"},
        {"role": "user", "content": "Classify this safe benchmark record."},
        {"role": "assistant", "content": '{"label":"benign"}'},
    ]
    (tmp_path / "train.jsonl").write_text(
        json.dumps({"example_id": "ex_train", "source_record_id": "r_train", "split": "train", "messages": messages}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "eval.jsonl").write_text(
        json.dumps({"example_id": "ex_eval", "source_record_id": "r_eval", "split": "eval", "messages": messages}) + "\n",
        encoding="utf-8",
    )
    cfg = load_config(CONFIG)
    cfg["dataset"]["train_rows"] = 1
    cfg["dataset"]["eval_rows"] = 1
    table, detected, source_path = detect_and_load_dataset(tmp_path, cfg)
    bundle = split_train_eval(table, cfg)
    prompt_messages, answer = supervised_prompt_and_answer(bundle.train.iloc[0].to_dict(), max_input_chars=500)
    assert detected == "sft_chat_dir"
    assert source_path.endswith(str(tmp_path))
    assert len(bundle.train) == 1
    assert len(bundle.eval) == 1
    assert [message["role"] for message in coerce_messages(bundle.train.iloc[0]["messages"])] == ["system", "user", "assistant"]
    assert [message["role"] for message in prompt_messages] == ["system", "user"]
    assert json.loads(answer)["label"] == "benign"
