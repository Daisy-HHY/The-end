from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from four_flower.scripts.aggregate_results import aggregate_backbone_comparison


def test_aggregate_writes_csv_with_expected_columns(tmp_path: Path):
    runs_dir = Path("D:/Users/hhyan/Desktop/universityWork/2025-2026第二学期/人工智能创新/The-end/four_flower/runs")
    if not (runs_dir / "eff_b0_full").exists():
        pytest.skip("Backbone experiments not yet run")

    csv_path = tmp_path / "backbone_comparison.csv"
    png_path = tmp_path / "backbone_acc_f1.png"

    rows = aggregate_backbone_comparison(runs_dir, csv_path, png_path)

    assert rows
    assert csv_path.exists()
    assert png_path.exists()

    with csv_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        loaded_rows = list(reader)

    assert "backbone" in loaded_rows[0]
    assert "preset" in loaded_rows[0]
    assert "test_acc" in loaded_rows[0]
    assert "test_macro_f1" in loaded_rows[0]
