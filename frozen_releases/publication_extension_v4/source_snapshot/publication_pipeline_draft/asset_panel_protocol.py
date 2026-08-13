#!/usr/bin/env python3
"""Validate and freeze a development-only external asset panel.

Input is a wide CSV of positive adjusted total-return levels with a leading
``date`` column.  The command refuses test-role data, future-confirmatory dates,
interpolation, duplicate dates, implicit asset reordering, and incomplete
cross-sections.  It produces a development release only; a separate data
custodian must commit future test bytes under ``future_confirmatory_protocol``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Any


class AssetPanelError(RuntimeError):
    """Raised when an external panel violates the causal data contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_date(value: str, field: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as error:
        raise AssetPanelError(f"{field} must be YYYY-MM-DD: {value!r}") from error


def month_key(value: date) -> tuple[int, int]:
    return value.year, value.month


def read_metadata(path: Path) -> dict[str, Any]:
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AssetPanelError(f"Could not read panel metadata: {error}") from error
    required = {
        "schema_version", "panel_id", "data_role", "source_uri", "license",
        "base_currency", "price_type", "point_in_time_universe_evidence",
        "date_start", "date_end", "asset_order",
        "selection_locked_before_return_access", "contains_test_returns",
        "missing_data_rule", "duplicate_date_rule",
    }
    missing = sorted(required - metadata.keys())
    if missing:
        raise AssetPanelError(f"Metadata is missing fields: {', '.join(missing)}")
    if metadata["schema_version"] != 1:
        raise AssetPanelError("External panel metadata schema must be 1.")
    if metadata["data_role"] != "development_train_validation":
        raise AssetPanelError("Only development train/validation data may be materialized here.")
    if metadata["contains_test_returns"] is not False:
        raise AssetPanelError("Development panel declares test returns.")
    if metadata["selection_locked_before_return_access"] is not True:
        raise AssetPanelError("Asset selection was not locked before return access.")
    if metadata["missing_data_rule"] != "no_return_interpolation":
        raise AssetPanelError("Return interpolation is forbidden.")
    if metadata["duplicate_date_rule"] != "forbidden":
        raise AssetPanelError("Duplicate dates must be forbidden.")
    assets = [str(value) for value in metadata["asset_order"]]
    if len(assets) < 15 or len(assets) > 50 or len(assets) != len(set(assets)):
        raise AssetPanelError("Asset order must contain 15-50 unique symbols.")
    return metadata


def read_levels(path: Path, metadata: dict[str, Any]) -> tuple[list[date], list[list[float]]]:
    if not path.is_file():
        raise AssetPanelError(f"Panel CSV not found: {path}")
    assets = metadata["asset_order"]
    expected_header = ["date", *assets]
    dates: list[date] = []
    values: list[list[float]] = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as error:
            raise AssetPanelError("Panel CSV is empty.") from error
        if header != expected_header:
            raise AssetPanelError("Panel columns or asset order do not match metadata.")
        for row_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise AssetPanelError(f"CSV row {row_number} has the wrong width.")
            current = parse_date(row[0], f"row {row_number} date")
            try:
                numeric = [float(value) for value in row[1:]]
            except ValueError as error:
                raise AssetPanelError(f"CSV row {row_number} contains non-numeric data.") from error
            if not all(math.isfinite(value) and value > 0 for value in numeric):
                raise AssetPanelError(f"CSV row {row_number} contains missing/non-positive levels.")
            dates.append(current)
            values.append(numeric)
    if len(dates) < 252 * 5:
        raise AssetPanelError("Panel has fewer than five years of daily observations.")
    if dates != sorted(dates) or len(dates) != len(set(dates)):
        raise AssetPanelError("Panel dates are not strictly increasing and unique.")
    start = parse_date(metadata["date_start"], "metadata date_start")
    end = parse_date(metadata["date_end"], "metadata date_end")
    if dates[0] != start or dates[-1] != end:
        raise AssetPanelError("Metadata date bounds do not match the CSV.")
    return dates, values


def monthly_gross(
    dates: list[date], values: list[list[float]], assets: list[str]
) -> list[dict[str, Any]]:
    month_end_indices: list[int] = []
    for index, current in enumerate(dates):
        if index == len(dates) - 1 or month_key(dates[index + 1]) != month_key(current):
            month_end_indices.append(index)
    if len(month_end_indices) < 61:
        raise AssetPanelError("Panel has fewer than 61 month-end observations.")
    rows: list[dict[str, Any]] = []
    for previous, current in zip(month_end_indices[:-1], month_end_indices[1:]):
        row: dict[str, Any] = {
            "decision_date": dates[previous].isoformat(),
            "holding_end_date": dates[current].isoformat(),
            "trading_days": current - previous,
        }
        for asset_index, asset in enumerate(assets):
            row[f"g_{asset}"] = values[current][asset_index] / values[previous][asset_index]
        rows.append(row)
    return rows


def daily_log_returns(
    dates: list[date], values: list[list[float]], assets: list[str]
) -> list[dict[str, Any]]:
    """Create the canonical daily log-return panel without imputing returns."""
    rows: list[dict[str, Any]] = []
    for previous in range(len(dates) - 1):
        current = previous + 1
        row: dict[str, Any] = {"date": dates[current].isoformat()}
        for asset_index, asset in enumerate(assets):
            row[asset] = math.log(
                values[current][asset_index] / values[previous][asset_index]
            )
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def materialize(
    levels: Path,
    metadata_path: Path,
    output: Path,
    *,
    validation_end: date,
    earliest_future_test_start: date,
) -> dict[str, Any]:
    if output.exists():
        raise AssetPanelError(f"Output already exists: {output}")
    metadata = read_metadata(metadata_path)
    dates, values = read_levels(levels, metadata)
    if dates[-1] > validation_end:
        raise AssetPanelError("Development panel extends beyond the declared validation end.")
    if dates[-1] >= earliest_future_test_start:
        raise AssetPanelError("Development materialization would access future test dates.")
    rows = monthly_gross(dates, values, metadata["asset_order"])
    daily_rows = daily_log_returns(dates, values, metadata["asset_order"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        copied_levels = temporary / "development_adjusted_levels.csv"
        copied_metadata = temporary / "panel_metadata.json"
        shutil.copyfile(levels, copied_levels)
        shutil.copyfile(metadata_path, copied_metadata)
        write_csv(temporary / "development_monthly_asset_gross.csv", rows)
        write_csv(temporary / "development_daily_log_returns.csv", daily_rows)
        manifest = {
            "schema_version": 1,
            "release_status": "frozen_development_panel_no_test_access",
            "panel_id": metadata["panel_id"],
            "data_role": metadata["data_role"],
            "asset_order": metadata["asset_order"],
            "daily_rows": len(dates),
            "monthly_periods": len(rows),
            "date_start": dates[0].isoformat(),
            "date_end": dates[-1].isoformat(),
            "validation_end": validation_end.isoformat(),
            "earliest_future_test_start": earliest_future_test_start.isoformat(),
            "test_data_accessed": False,
            "levels_sha256": sha256_file(copied_levels),
            "metadata_sha256": sha256_file(copied_metadata),
            "monthly_gross_sha256": sha256_file(
                temporary / "development_monthly_asset_gross.csv"
            ),
            "daily_log_returns_sha256": sha256_file(
                temporary / "development_daily_log_returns.csv"
            ),
        }
        (temporary / "development_panel_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temporary / "READ_ONLY_DEVELOPMENT_RELEASE.txt").write_text(
            "Contains train/validation data only. It is not a confirmatory test release.\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--validation-end", required=True)
    parser.add_argument("--earliest-future-test-start", default="2026-07-07")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = materialize(
            args.levels,
            args.metadata,
            args.output,
            validation_end=parse_date(args.validation_end, "validation end"),
            earliest_future_test_start=parse_date(
                args.earliest_future_test_start, "earliest future test start"
            ),
        )
    except AssetPanelError as error:
        print(f"ASSET PANEL FAILURE: {error}")
        return 1
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
