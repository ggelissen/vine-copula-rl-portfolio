#!/usr/bin/env python3
"""Freeze one dimension-matched, causally truncated RL return input.

The source development panel can contain several retrospective windows.  This
command exposes only observations through one window's test end and writes the
exact final 24 periods expected to remain untouched by synthetic generation and
training.  It never labels retrospective evidence as confirmatory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from datetime import date
from pathlib import Path
from typing import Any


class WindowInputError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise WindowInputError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    require(bool(rows), f"CSV is empty: {path}")
    return rows


def month_ends(dates: list[date]) -> list[date]:
    return [
        current
        for index, current in enumerate(dates)
        if index == len(dates) - 1
        or (dates[index + 1].year, dates[index + 1].month)
        != (current.year, current.month)
    ]


def materialize(
    *,
    daily_returns: Path,
    panel_manifest_path: Path,
    schedule_path: Path,
    window_id: str,
    output: Path,
    reference_asset: str,
    vine_truncation_level: int = 0,
) -> dict[str, Any]:
    require(not output.exists(), f"Output already exists: {output}")
    panel = json.loads(panel_manifest_path.read_text(encoding="utf-8"))
    require(
        panel.get("release_status") == "frozen_development_panel_no_test_access",
        "Parent panel is not a frozen development release.",
    )
    require(panel.get("test_data_accessed") is False,
            "Parent development panel declares test access.")
    require(
        sha256(daily_returns) == panel.get("daily_log_returns_sha256"),
        "Daily return bytes do not match the parent panel manifest.",
    )
    schedule = read_csv(schedule_path)
    matches = [row for row in schedule if row.get("window_id") == window_id]
    require(len(matches) == 1, "window_id is absent or duplicated in the schedule.")
    window = matches[0]
    require(window.get("test_data_role") == "retrospective_development_only",
            "Only retrospective development windows may use this materializer.")
    require(str(window.get("confirmatory_claim_permitted", "")).lower() ==
            "false",
            "Retrospective window does not explicitly prohibit confirmation.")
    expected_periods = int(window["test_months"])
    require(expected_periods == 24,
            "The current RL protocol requires exactly 24 reserved periods.")

    rows = read_csv(daily_returns)
    header = list(rows[0])
    asset_order = [str(value) for value in panel["asset_order"]]
    require(header == ["date", *asset_order],
            "Daily return columns/order do not match the panel manifest.")
    require(reference_asset in asset_order,
            "Reference asset is absent from the frozen asset order.")
    require(vine_truncation_level == 0 or
            1 <= vine_truncation_level < len(asset_order),
            "Vine truncation must be 0 (all trees) or between 1 and d-1.")
    all_dates = [date.fromisoformat(row["date"]) for row in rows]
    require(all_dates == sorted(all_dates) and len(all_dates) == len(set(all_dates)),
            "Daily return dates are not strictly increasing and unique.")
    test_end = date.fromisoformat(window["test_end"])
    selected = [row for row, current in zip(rows, all_dates) if current <= test_end]
    require(bool(selected) and date.fromisoformat(selected[-1]["date"]) == test_end,
            "Daily panel has no observation at the scheduled test end.")
    selected_dates = [date.fromisoformat(row["date"]) for row in selected]
    endpoints = month_ends(selected_dates)
    require(len(endpoints) >= expected_periods + 1,
            "Window input has insufficient monthly history.")
    evaluation_pairs = list(zip(endpoints[-(expected_periods + 1):-1],
                                endpoints[-expected_periods:]))
    expected_start = date.fromisoformat(window["test_start"])
    # The return loader's decision date is the prior observed month-end.
    require(evaluation_pairs[0][0] == expected_start,
            "Computed first evaluation decision does not match the schedule.")
    require(evaluation_pairs[-1][1] == test_end,
            "Computed final evaluation holding end does not match the schedule.")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        return_file = temporary / "window_daily_log_returns.csv"
        with return_file.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=header)
            writer.writeheader()
            writer.writerows(selected)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "release_status": "frozen_window_return_input_no_confirmation",
            "panel_id": panel["panel_id"],
            "window_id": window_id,
            "evidence_class": window["evidence_class"],
            "claim_limit": window["claim_limit"],
            "confirmatory_claim_permitted": False,
            "asset_order": asset_order,
            "asset_count": len(asset_order),
            "reference_asset": reference_asset,
            "reference_asset_index_1based": asset_order.index(reference_asset) + 1,
            "vine_truncation_level": (
                len(asset_order) - 1 if vine_truncation_level == 0
                else vine_truncation_level
            ),
            "return_kind": "daily_log_returns",
            "return_rows": len(selected),
            "date_start": selected[0]["date"],
            "date_end": selected[-1]["date"],
            "expected_evaluation_periods": expected_periods,
            "expected_evaluation_start": window["test_start"],
            "expected_evaluation_end": window["test_end"],
            "return_file_sha256": sha256(return_file),
            "parent_panel_manifest_sha256": sha256(panel_manifest_path),
            "parent_daily_returns_sha256": sha256(daily_returns),
            "window_schedule_sha256": sha256(schedule_path),
            "post_window_rows_included": False,
        }
        (temporary / "return_input_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        shutil.copy2(schedule_path, temporary / "source_window_schedule.csv")
        (temporary / "READ_ONLY_WINDOW_INPUT.txt").write_text(
            "Development/robustness input only. Final 24 rows are reserved by the RL split.\n",
            encoding="utf-8",
        )
        checksum_lines = [
            f"{sha256(path)}  {path.name}"
            for path in sorted(temporary.iterdir())
            if path.is_file() and path.name != "CONTENTS.sha256"
        ]
        (temporary / "CONTENTS.sha256").write_text(
            "\n".join(checksum_lines) + "\n", encoding="ascii"
        )
        os.replace(temporary, output)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-returns", required=True, type=Path)
    parser.add_argument("--panel-manifest", required=True, type=Path)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--window-id", required=True)
    parser.add_argument("--reference-asset", default="BIL")
    parser.add_argument("--vine-truncation-level", default=0, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = materialize(
            daily_returns=args.daily_returns,
            panel_manifest_path=args.panel_manifest,
            schedule_path=args.schedule,
            window_id=args.window_id,
            output=args.output,
            reference_asset=args.reference_asset,
            vine_truncation_level=args.vine_truncation_level,
        )
    except (OSError, ValueError, WindowInputError, json.JSONDecodeError) as error:
        print(f"WINDOW INPUT FAILURE: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
