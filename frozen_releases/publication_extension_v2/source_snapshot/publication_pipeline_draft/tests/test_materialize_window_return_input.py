from __future__ import annotations

import csv
import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from publication_pipeline_draft.materialize_window_return_input import (
    WindowInputError,
    materialize,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    assets = [f"A{i:02d}" for i in range(17)] + ["BIL"]
    daily = tmp_path / "daily.csv"
    rows = []
    # Month-end-only observations are sufficient to prove calendar selection.
    for year in range(2010, 2021):
        for month in range(1, 13):
            current = date(year, month, 28)
            rows.append({"date": current.isoformat(), **{asset: "0.001" for asset in assets}})
    with daily.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["date", *assets])
        writer.writeheader(); writer.writerows(rows)
    panel_manifest = tmp_path / "panel.json"
    panel_manifest.write_text(json.dumps({
        "release_status": "frozen_development_panel_no_test_access",
        "test_data_accessed": False,
        "panel_id": "panel18",
        "asset_order": assets,
        "daily_log_returns_sha256": digest(daily),
    }), encoding="utf-8")
    schedule = tmp_path / "schedule.csv"
    with schedule.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "window_id", "evidence_class", "claim_limit", "test_data_role",
            "test_start", "test_end", "test_months",
        ])
        writer.writeheader(); writer.writerow({
            "window_id": "w1", "evidence_class": "retrospective_walk_forward",
            "claim_limit": "development_and_robustness_only",
            "test_data_role": "retrospective_development_only",
            "test_start": "2018-12-28", "test_end": "2020-12-28",
            "test_months": 24,
        })
    return daily, panel_manifest, schedule


def test_materializes_exact_dimension_and_final_24_periods(tmp_path: Path) -> None:
    daily, panel, schedule = build_inputs(tmp_path)
    output = tmp_path / "release"
    result = materialize(
        daily_returns=daily, panel_manifest_path=panel, schedule_path=schedule,
        window_id="w1", output=output, reference_asset="BIL",
    )
    assert result["asset_count"] == 18
    assert result["reference_asset_index_1based"] == 18
    assert result["vine_truncation_level"] == 17
    assert result["expected_evaluation_periods"] == 24
    assert result["date_end"] == "2020-12-28"
    assert result["post_window_rows_included"] is False
    assert digest(output / "window_daily_log_returns.csv") == result["return_file_sha256"]


def test_rejects_parent_hash_mismatch(tmp_path: Path) -> None:
    daily, panel, schedule = build_inputs(tmp_path)
    metadata = json.loads(panel.read_text(encoding="utf-8"))
    metadata["daily_log_returns_sha256"] = "0" * 64
    panel.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(WindowInputError, match="do not match"):
        materialize(
            daily_returns=daily, panel_manifest_path=panel, schedule_path=schedule,
            window_id="w1", output=tmp_path / "bad", reference_asset="BIL",
        )
