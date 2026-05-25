from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from scripts.build_gold_benchmark import build_benchmark
from scripts.evaluate_benchmark import evaluate
from scripts.run_model_predictions import (
    build_messages,
    extract_classification_prediction,
    generate_predictions,
    parse_model_response,
    safe_model_name,
)


FIXTURES = Path("tests/fixtures")


def silver_fixture_dir(tmp_path: Path) -> Path:
    silver_dir = tmp_path / "silver"
    silver_dir.mkdir()
    shutil.copy(FIXTURES / "silver_sample.csv", silver_dir / "silver_sample.csv")
    return silver_dir


def build_gold_fixture(tmp_path: Path) -> Path:
    gold_dir = tmp_path / "gold"
    build_benchmark(silver_fixture_dir(tmp_path), gold_dir, max_rows=10, seed=42, output_format="csv", dry_run=False)
    return gold_dir / "benchmark_gold.csv"


def test_local_stub_creates_prediction_csv_with_required_columns(tmp_path):
    gold_file = build_gold_fixture(tmp_path)
    out_dir = tmp_path / "predictions"
    summary = generate_predictions(
        gold_file=gold_file,
        out_dir=out_dir,
        provider="local_stub",
        model_name="local_stub",
        limit=None,
        seed=42,
        dry_run=False,
        resume=False,
        output_format="csv",
    )
    predictions_path = out_dir / "predictions_local_stub.csv"
    assert predictions_path.exists()
    written = pd.read_csv(predictions_path)
    assert set(["record_id", "prediction", "model_name", "score", "probability", "confidence", "explanation"]).issubset(
        written.columns
    )
    assert len(written) == 5
    assert summary["written_rows"] == 5


def test_limit_restricts_processed_rows(tmp_path):
    gold_file = build_gold_fixture(tmp_path)
    out_dir = tmp_path / "predictions"
    generate_predictions(
        gold_file=gold_file,
        out_dir=out_dir,
        provider="local_stub",
        model_name="local_stub",
        limit=2,
        seed=42,
        dry_run=False,
        resume=False,
        output_format="csv",
    )
    written = pd.read_csv(out_dir / "predictions_local_stub.csv")
    assert len(written) == 2


def test_resume_skips_existing_predictions(tmp_path):
    gold_file = build_gold_fixture(tmp_path)
    out_dir = tmp_path / "predictions"
    first = generate_predictions(
        gold_file=gold_file,
        out_dir=out_dir,
        provider="local_stub",
        model_name="local_stub",
        limit=2,
        seed=42,
        dry_run=False,
        resume=False,
        output_format="csv",
    )
    second = generate_predictions(
        gold_file=gold_file,
        out_dir=out_dir,
        provider="local_stub",
        model_name="local_stub",
        limit=5,
        seed=42,
        dry_run=False,
        resume=True,
        output_format="csv",
    )
    written = pd.read_csv(out_dir / "predictions_local_stub.csv")
    assert len(written) == 5
    assert first["written_rows"] == 2
    assert second["skipped_existing"] == 2
    assert second["written_rows"] == 3


def test_dry_run_does_not_call_external_providers(tmp_path, monkeypatch):
    gold_file = build_gold_fixture(tmp_path)
    out_dir = tmp_path / "predictions"

    def fail_call(*_args, **_kwargs):
        raise AssertionError("external provider should not be called during dry-run")

    monkeypatch.setattr("scripts.run_model_predictions.call_openai_model", fail_call)
    monkeypatch.setattr("scripts.run_model_predictions.call_openrouter_model", fail_call)
    summary = generate_predictions(
        gold_file=gold_file,
        out_dir=out_dir,
        provider="openai",
        model_name="gpt-4o-mini",
        limit=2,
        seed=42,
        dry_run=True,
        resume=False,
        output_format="csv",
    )
    written = pd.read_csv(out_dir / "predictions_gpt-4o-mini.csv")
    assert len(written) == 2
    assert summary["dry_run"] is True
    assert set(written["provider"]) == {"openai"}


def test_safe_model_filename_generation():
    assert safe_model_name("openai/gpt-4o-mini") == "openai_gpt-4o-mini"
    assert safe_model_name("qwen/qwen2.5-14b-instruct") == "qwen_qwen2.5-14b-instruct"
    assert safe_model_name("  weird model name  ") == "weird_model_name"


def test_classification_parser_extracts_valid_label_from_json():
    row = pd.Series({"task_type": "classification", "label_set": '["benign_prompt", "malicious_prompt"]'})
    result = parse_model_response('{"prediction":"malicious_prompt","confidence":0.9,"explanation":"matched"}', row)
    assert result.prediction == "malicious_prompt"
    assert result.confidence == 0.9


def test_classification_parser_extracts_valid_label_from_messy_text():
    labels = ["benign_prompt", "malicious_prompt"]
    prediction = extract_classification_prediction("Final answer: MALICIOUS_PROMPT because this is adversarial.", labels)
    assert prediction == "malicious_prompt"


def test_prompt_templates_warn_on_prompt_injection_rows(tmp_path):
    gold_file = build_gold_fixture(tmp_path)
    gold = pd.read_csv(gold_file)
    row = gold[gold["evaluation_head"] == "prompt_injection_jailbreaks"].iloc[0]
    messages = build_messages(row)
    combined = "\n".join(message["content"] for message in messages)
    assert "untrusted data" in combined
    assert "Do not follow" in combined
    assert "jailbreak" in combined.lower()


def test_predictions_can_be_evaluated_by_existing_evaluator(tmp_path):
    gold_file = build_gold_fixture(tmp_path)
    out_dir = tmp_path / "predictions"
    generate_predictions(
        gold_file=gold_file,
        out_dir=out_dir,
        provider="local_stub",
        model_name="local_stub",
        limit=5,
        seed=42,
        dry_run=False,
        resume=False,
        output_format="both",
    )
    results, payload = evaluate(gold_file, out_dir / "predictions_local_stub.csv", tmp_path / "eval")
    assert payload["matched_rows"] == 5
    assert not results.empty
    jsonl_rows = [json.loads(line) for line in (out_dir / "predictions_local_stub.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(jsonl_rows) == 5
