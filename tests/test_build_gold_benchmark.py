from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from scripts.build_gold_benchmark import GOLD_COLUMNS, build_benchmark, load_config, transform_silver_to_gold


FIXTURES = Path("tests/fixtures")


def silver_fixture_dir(tmp_path: Path) -> Path:
    silver_dir = tmp_path / "silver"
    silver_dir.mkdir()
    shutil.copy(FIXTURES / "silver_sample.csv", silver_dir / "silver_sample.csv")
    return silver_dir


def test_gold_schema_columns_and_head_mapping():
    silver = pd.read_csv(FIXTURES / "silver_sample.csv")
    gold = transform_silver_to_gold(silver, load_config(), seed=42)
    assert list(gold.columns) == GOLD_COLUMNS
    by_id = gold.set_index("record_id")
    assert by_id.loc["sample::malware1", "evaluation_head"] == "malware_code"
    assert by_id.loc["sample::code1", "evaluation_head"] == "malware_code"
    assert by_id.loc["sample::cti1", "evaluation_head"] == "cti"
    assert by_id.loc["sample::prompt1", "evaluation_head"] == "prompt_injection_jailbreaks"
    assert set(gold["split"]) <= {"train", "validation", "test"}


def test_gold_builder_caps_rows_and_writes_outputs(tmp_path):
    silver_dir = silver_fixture_dir(tmp_path)
    out_dir = tmp_path / "gold"
    gold = build_benchmark(silver_dir, out_dir, max_rows=3, seed=7, output_format="csv", dry_run=False)
    assert len(gold) == 3
    assert (out_dir / "benchmark_gold.csv").exists()
    assert (out_dir / "benchmark_manifest.json").exists()
    manifest = json.loads((out_dir / "benchmark_manifest.json").read_text(encoding="utf-8"))
    assert manifest["row_count"] == 3
    written = pd.read_csv(out_dir / "benchmark_gold.csv")
    assert set(written["split"]) <= {"train", "validation", "test"}


def test_gold_builder_dry_run_does_not_write_outputs(tmp_path):
    silver_dir = silver_fixture_dir(tmp_path)
    out_dir = tmp_path / "gold"
    gold = build_benchmark(silver_dir, out_dir, max_rows=10, seed=42, output_format="both", dry_run=True)
    assert len(gold) == 5
    assert not out_dir.exists()
