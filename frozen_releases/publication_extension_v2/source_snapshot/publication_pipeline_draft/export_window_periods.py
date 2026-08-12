#!/usr/bin/env python3
"""Export canonical per-window benchmark calendars from a frozen schedule."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path


class ExportError(RuntimeError):
    pass


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def export(schedule_path: Path, monthly_path: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise ExportError(f"Output already exists: {output}")
    schedule, monthly = read(schedule_path), read(monthly_path)
    if not schedule or not monthly:
        raise ExportError("Schedule/monthly panel is empty.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        inventory = []
        for window in schedule:
            start, end = window["test_start"], window["test_end"]
            selected = [row for row in monthly
                        if row["decision_date"] >= start and
                        row["holding_end_date"] <= end]
            expected = int(window["test_months"])
            if len(selected) != expected:
                raise ExportError(
                    f"{window['window_id']} has {len(selected)} periods; expected {expected}.")
            path = temporary / f"evaluation_periods_{window['window_id']}.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=["window_id", "decision_date", "holding_end_date"])
                writer.writeheader()
                writer.writerows({"window_id": window["window_id"],
                                  "decision_date": row["decision_date"],
                                  "holding_end_date": row["holding_end_date"]}
                                 for row in selected)
            inventory.append({"window_id": window["window_id"],
                              "periods": len(selected), "file": path.name})
        with (temporary / "period_inventory.csv").open(
                "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(inventory[0]))
            writer.writeheader(); writer.writerows(inventory)
        manifest = {"schema_version": 1, "status": "canonical_periods_exported",
                    "window_count": len(inventory), "calendar_reestimated": False}
        (temporary / "period_export_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, output)
        return manifest
    except Exception:
        import shutil
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--monthly-panel", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = export(args.schedule, args.monthly_panel, args.output)
    except (OSError, ExportError) as error:
        print(f"PERIOD EXPORT FAILURE: {error}"); return 1
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
