from __future__ import annotations

import csv
import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from publication_pipeline_draft.asset_panel_protocol import AssetPanelError, materialize


ASSETS = [f"ETF{i:02d}" for i in range(18)]


def make_panel(tmp_path: Path, *, contains_test: bool = False) -> tuple[Path, Path]:
    levels = tmp_path / "levels.csv"
    start = date(2015, 1, 2)
    dates = []
    current = start
    while len(dates) < 252 * 6:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    with levels.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["date", *ASSETS])
        for index, day in enumerate(dates):
            writer.writerow([day.isoformat(), *[100 + index * 0.01 + j for j in range(18)]])
    metadata = {
        "schema_version": 1,
        "panel_id": "test_panel",
        "data_role": "development_train_validation",
        "source_uri": "doi:test",
        "license": "test",
        "retrieved_utc": "2026-01-01T00:00:00Z",
        "base_currency": "USD",
        "price_type": "adjusted_total_return_index_or_adjusted_close",
        "point_in_time_universe_evidence": "hash:test",
        "date_start": dates[0].isoformat(),
        "date_end": dates[-1].isoformat(),
        "asset_order": ASSETS,
        "selection_locked_before_return_access": True,
        "contains_test_returns": contains_test,
        "missing_data_rule": "no_return_interpolation",
        "duplicate_date_rule": "forbidden",
    }
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return levels, metadata_path


def test_materializes_development_only_panel(tmp_path: Path) -> None:
    levels, metadata = make_panel(tmp_path)
    output = tmp_path / "release"
    result = materialize(
        levels,
        metadata,
        output,
        validation_end=date(2025, 12, 31),
        earliest_future_test_start=date(2026, 7, 7),
    )
    assert result["release_status"] == "frozen_development_panel_no_test_access"
    assert not result["test_data_accessed"]
    assert result["monthly_periods"] >= 60
    assert (output / "development_monthly_asset_gross.csv").is_file()
    assert (output / "development_daily_log_returns.csv").is_file()


def test_rejects_test_role_data(tmp_path: Path) -> None:
    levels, metadata = make_panel(tmp_path, contains_test=True)
    with pytest.raises(AssetPanelError, match="declares test returns"):
        materialize(
            levels,
            metadata,
            tmp_path / "release",
            validation_end=date(2025, 12, 31),
            earliest_future_test_start=date(2026, 7, 7),
        )


def test_rejects_future_dates_before_reading_as_development(tmp_path: Path) -> None:
    levels, metadata = make_panel(tmp_path)
    with pytest.raises(AssetPanelError, match="validation end"):
        materialize(
            levels,
            metadata,
            tmp_path / "release",
            validation_end=date(2019, 1, 1),
            earliest_future_test_start=date(2026, 7, 7),
        )
