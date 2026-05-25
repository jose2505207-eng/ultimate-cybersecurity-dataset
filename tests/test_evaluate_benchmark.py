from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from scripts.build_gold_benchmark import build_benchmark
from scripts.evaluate_benchmark import evaluate


FIXTURES = Path("tests/fixtures")


def silver_fixture_dir(tmp_path: Path) -> Path:
    silver_dir = tmp_path / "silver"
    silver_dir.mkdir()
    shutil.copy(FIXTURES / "silver_sample.csv", silver_dir / "silver_sample.csv")
    return silver_dir


def test_evaluator_computes_classification_metrics_and_writes_outputs(tmp_path):
    gold_dir = tmp_path / "gold"
    build_benchmark(silver_fixture_dir(tmp_path), gold_dir, max_rows=10, seed=42, output_format="csv", dry_run=False)
    results, payload = evaluate(gold_dir / "benchmark_gold.csv", FIXTURES / "predictions_sample.csv", gold_dir)
    assert (gold_dir / "evaluation_results.csv").exists()
    assert (gold_dir / "evaluation_results.json").exists()
    overall = results[(results["group_by"] == "overall") & (results["group_value"] == "overall")].iloc[0]
    assert overall["primary_score"] >= 0.9
    assert payload["matched_rows"] == 5
    saved = json.loads((gold_dir / "evaluation_results.json").read_text(encoding="utf-8"))
    assert saved["weighted_overall_score"] >= 0.9


def test_generation_fallback_metrics_do_not_require_optional_dependencies(tmp_path):
    gold = pd.DataFrame(
        [
            {
                "record_id": "gen::1",
                "source_dataset": "safe_generation_fixture",
                "source_type": "llm_io_pair",
                "main_category": "AI, LLM & ML Security",
                "attack_name": "Toy safe response",
                "attack_family": "prompt safety",
                "label": "benign_prompt",
                "binary_label": 0,
                "mitre_tactic": None,
                "mitre_technique_id": None,
                "benchmark_domain": "prompt_injection_jailbreaks",
                "task_type": "generation",
                "evaluation_head": "prompt_injection_jailbreaks",
                "metric_group": "generation",
                "input_text": "Classify this safe toy prompt.",
                "expected_output": "This is a safe benign prompt.",
                "gold_label": "benign_prompt",
                "label_set": '["benign_prompt"]',
                "difficulty": "easy",
                "split": "test",
                "requires_probability": False,
                "scoring_notes": "Use generation metrics.",
                "safety_notes": "Safe toy fixture.",
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        ]
    )
    preds = pd.DataFrame([{"record_id": "gen::1", "prediction": "safe benign prompt"}])
    gold_path = tmp_path / "gold.csv"
    preds_path = tmp_path / "preds.csv"
    gold.to_csv(gold_path, index=False)
    preds.to_csv(preds_path, index=False)
    results, payload = evaluate(gold_path, preds_path, tmp_path)
    assert payload["matched_rows"] == 1
    assert (tmp_path / "evaluation_results.json").exists()
    overall = results[(results["group_by"] == "overall") & (results["group_value"] == "overall")].iloc[0]
    assert overall["primary_score"] >= 0.0
