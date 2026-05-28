from __future__ import annotations

import json

import pandas as pd

from scripts.prepare_sft_dataset import prepare_examples


def _cfg():
    return {
        "seed": 42,
        "split_ratios": {"train": 0.8, "eval": 0.1, "test": 0.1},
        "quality": {"min_prompt_chars": 20, "min_target_chars": 2, "max_input_chars": 500, "label_leakage_action": "exclude_all"},
    }


def test_prepare_examples_removes_duplicate_prompt_target_pairs():
    gold = pd.DataFrame(
        [
            {
                "record_id": "r1",
                "source_dataset": "unit",
                "source_type": "other",
                "main_category": "Miscellaneous / Needs Review",
                "evaluation_head": "misc",
                "task_type": "classification",
                "input_text": "A defensive benchmark record with enough context.",
                "expected_output": "benign",
                "gold_label": "benign",
                "label_set": '["benign"]',
                "split": "train",
            },
            {
                "record_id": "r2",
                "source_dataset": "unit",
                "source_type": "other",
                "main_category": "Miscellaneous / Needs Review",
                "evaluation_head": "misc",
                "task_type": "classification",
                "input_text": "A defensive benchmark record with enough context.",
                "expected_output": "benign",
                "gold_label": "benign",
                "label_set": '["benign"]',
                "split": "train",
            },
        ]
    )
    prepared, removed, summary = prepare_examples(gold, _cfg())
    assert len(prepared) == 1
    assert len(removed) == 1
    assert summary["removed_rows"] == 1


def test_prepared_example_has_traceable_chat_schema():
    gold = pd.DataFrame(
        [
            {
                "record_id": "r1",
                "source_dataset": "unit",
                "source_type": "other",
                "main_category": "Miscellaneous / Needs Review",
                "evaluation_head": "malware_code",
                "task_type": "classification",
                "input_text": "A safe summarized code-security benchmark record.",
                "expected_output": "benign",
                "gold_label": "benign",
                "label_set": '["benign","malicious"]',
                "split": "train",
            }
        ]
    )
    prepared, removed, _summary = prepare_examples(gold, _cfg())
    assert removed.empty
    rec = prepared.iloc[0]
    messages = json.loads(rec["messages_json"])
    target = json.loads(rec["target_json"])
    assert [m["role"] for m in messages] == ["system", "user", "assistant"]
    assert target["label"] == "benign"
    assert rec["source_record_id"] == "r1"
    assert "source_type" not in messages[1]["content"]


def test_label_leakage_is_removed_from_all_splits():
    gold = pd.DataFrame(
        [
            {
                "record_id": "r_leak",
                "source_dataset": "unit",
                "source_type": "other",
                "main_category": "Miscellaneous / Needs Review",
                "evaluation_head": "misc",
                "task_type": "classification",
                "input_text": "This benchmark input explicitly says the label is benign.",
                "expected_output": "benign",
                "gold_label": "benign",
                "label_set": '["benign","malicious"]',
                "split": "test",
            }
        ]
    )
    prepared, removed, summary = prepare_examples(gold, _cfg())
    assert prepared.empty
    assert removed.iloc[0]["reason"] == "label_leakage"
    assert summary["flag_counts"]["label_leakage"] == 1
