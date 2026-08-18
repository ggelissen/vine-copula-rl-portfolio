#!/usr/bin/env python3
"""Freeze the existing seven-asset adjusted-level panel for retrospective use."""

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


class FocusedPanelError(RuntimeError):
    pass


ASSETS = ["SP500", "NASDAQ", "DOW", "SSE50", "DIVIDEND", "CHINEXT", "GOLD"]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FocusedPanelError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as error:
        raise FocusedPanelError(f"Invalid YYYY-MM-DD date: {value}") from error


def read_levels(path: Path) -> tuple[list[date], list[list[float]]]:
    require(path.is_file(), f"Adjusted-level file not found: {path}")
    dates: list[date] = []
    values: list[list[float]] = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = next(reader, [])
        require(header == ["date", *ASSETS],
                "Seven-asset columns/order differ from the frozen protocol.")
        for number, row in enumerate(reader, start=2):
            require(len(row) == len(header), f"Row {number} has the wrong width.")
            current = parse_date(row[0])
            try:
                numeric = [float(value) for value in row[1:]]
            except ValueError as error:
                raise FocusedPanelError(
                    f"Row {number} contains non-numeric levels.") from error
            require(all(math.isfinite(value) and value > 0 for value in numeric),
                    f"Row {number} contains missing/non-positive levels.")
            dates.append(current)
            values.append(numeric)
    require(len(dates) >= 252 * 10 and dates == sorted(dates) and
            len(dates) == len(set(dates)),
            "Seven-asset panel needs ten years of unique sorted daily observations.")
    require(dates[-1] <= date(2026, 7, 6),
            "Seven-asset retrospective panel extends past the consumed sample.")
    return dates, values


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def materialize(levels: Path, output: Path) -> dict[str, Any]:
    require(not output.exists(), f"Output already exists: {output}")
    dates, values = read_levels(levels)
    month_ends = [index for index, current in enumerate(dates)
                  if index == len(dates) - 1 or
                  (dates[index + 1].year, dates[index + 1].month) !=
                  (current.year, current.month)]
    require(len(month_ends) >= 133,
            "Seven-asset panel cannot supply two 24-month focused windows.")
    monthly: list[dict[str, Any]] = []
    for previous, current in zip(month_ends[:-1], month_ends[1:]):
        row: dict[str, Any] = {
            "decision_date": dates[previous].isoformat(),
            "holding_end_date": dates[current].isoformat(),
            "calendar_days": (dates[current] - dates[previous]).days,
        }
        for index, asset in enumerate(ASSETS):
            row[f"g_{asset}"] = values[current][index] / values[previous][index]
        monthly.append(row)
    daily: list[dict[str, Any]] = []
    for previous in range(len(dates) - 1):
        current = previous + 1
        row = {"date": dates[current].isoformat()}
        for index, asset in enumerate(ASSETS):
            row[asset] = math.log(values[current][index] / values[previous][index])
        daily.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        copied = temporary / "development_adjusted_levels.csv"
        shutil.copy2(levels, copied)
        monthly_path = temporary / "development_monthly_asset_gross.csv"
        daily_path = temporary / "development_daily_log_returns.csv"
        write_csv(monthly_path, monthly)
        write_csv(daily_path, daily)
        manifest = {
            "schema_version": 1,
            "release_status": "frozen_development_panel_no_test_access",
            "panel_id": "original_seven_asset_panel",
            "evidence_class": "retrospective_walk_forward",
            "data_role": "retrospective_development_only",
            "asset_order": ASSETS,
            "asset_count": 7,
            "source_level_rows": len(dates),
            "daily_return_rows": len(daily),
            "monthly_periods": len(monthly),
            "date_start": dates[0].isoformat(), "date_end": dates[-1].isoformat(),
            "test_data_accessed": False,
            "contains_previously_consumed_holdout": True,
            "fresh_confirmatory_data_accessed": False,
            "claim_limit": "same_market_retrospective_robustness_only",
            "levels_sha256": sha256(copied),
            "monthly_gross_sha256": sha256(monthly_path),
            "daily_log_returns_sha256": sha256(daily_path),
        }
        (temporary / "development_panel_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        (temporary / "READ_ONLY_DEVELOPMENT_RELEASE.txt").write_text(
            "Same-market retrospective development evidence; not confirmation.\n",
            encoding="utf-8")
        checksum = [f"{sha256(path)}  {path.name}" for path in
                    sorted(temporary.iterdir()) if path.is_file() and
                    path.name != "CONTENTS.sha256"]
        (temporary / "CONTENTS.sha256").write_text(
            "\n".join(checksum) + "\n", encoding="ascii")
        os.replace(temporary, output)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=Path,
                        default=Path("data/portfolio_B_7assets_2013.csv"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = materialize(args.levels.resolve(), args.output)
    except (OSError, ValueError, FocusedPanelError) as error:
        print(f"FOCUSED SEVEN-ASSET PANEL FAILURE: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
