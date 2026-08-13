from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

import pytest

from publication_pipeline_draft.walk_forward_windows import (
    WindowProtocolError, build_windows, write_release
)
from publication_pipeline_draft.export_window_periods import export


ROOT = Path(__file__).resolve().parents[2]
PROGRAM = ROOT / "publication_pipeline_draft/config/publication_research_program_v2.json"


def development_panel(tmp_path: Path, months: int = 204) -> tuple[Path, Path]:
    monthly = tmp_path / "monthly.csv"
    rows = []
    year, month = 2009, 1
    for index in range(months):
        next_month = month + 1
        next_year = year
        if next_month == 13:
            next_month, next_year = 1, year + 1
        rows.append({"decision_date": date(year, month, 1).isoformat(),
                     "holding_end_date": date(next_year, next_month, 1).isoformat(),
                     "g_SPY": 1.0})
        year, month = next_year, next_month
    with monthly.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "release_status": "frozen_development_panel_no_test_access",
        "panel_id": "global_liquid_etf_18", "test_data_accessed": False,
    }), encoding="utf-8")
    return monthly, manifest


def test_nonoverlapping_retrospective_windows_are_materialized(tmp_path: Path) -> None:
    monthly, manifest = development_panel(tmp_path)
    windows, summary = build_windows(
        PROGRAM, "retrospective_expanding_24m_v1", monthly, manifest)
    assert len(windows) >= 4
    assert not summary["confirmatory_claim_permitted"]
    for previous, current in zip(windows, windows[1:]):
        assert previous["test_end"] <= current["test_start"]
    output = tmp_path / "release"
    write_release(output, windows, summary)
    assert (output / "window_schedule.csv").is_file()
    period_output = tmp_path / "periods"
    export(output / "window_schedule.csv", monthly, period_output)
    assert len(list(period_output.glob("evaluation_periods_*.csv"))) == len(windows)


def test_future_window_requires_separate_access_ledger(tmp_path: Path) -> None:
    monthly, manifest = development_panel(tmp_path)
    with pytest.raises(WindowProtocolError, match="custodian"):
        build_windows(PROGRAM, "future_nonoverlapping_24m_v1", monthly, manifest)
