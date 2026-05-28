from __future__ import annotations

import pandas as pd

from scripts.evaluate_qlora_adapters import metrics_for_predictions, select_gold_subset


def test_select_gold_subset_is_deterministic_and_bounded():
    gold = pd.DataFrame(
        [
            {
                "record_id": f"r{i}",
                "input_text": "x",
                "expected_output": "a" if i % 2 else "b",
                "gold_label": "a" if i % 2 else "b",
                "evaluation_head": "h1" if i < 5 else "h2",
                "task_type": "classification",
                "main_category": "cat",
            }
            for i in range(10)
        ]
    )
    first = select_gold_subset(gold, limit=4, seed=7)
    second = select_gold_subset(gold, limit=4, seed=7)
    assert len(first) == 4
    assert first["record_id"].tolist() == second["record_id"].tolist()


def test_metrics_for_predictions_reports_accuracy_and_f1():
    df = pd.DataFrame(
        [
            {"prediction": "a", "label": "a", "correct": True, "evaluation_head": "h1", "main_category": "cat", "source_dataset": "s", "task_type": "classification"},
            {"prediction": "b", "label": "a", "correct": False, "evaluation_head": "h1", "main_category": "cat", "source_dataset": "s", "task_type": "classification"},
            {"prediction": "b", "label": "b", "correct": True, "evaluation_head": "h2", "main_category": "cat", "source_dataset": "s", "task_type": "classification"},
        ]
    )
    metrics = metrics_for_predictions(df)
    assert metrics["accuracy"] == 2 / 3
    assert metrics["f1_macro"] > 0
    assert metrics["category_metrics"]

