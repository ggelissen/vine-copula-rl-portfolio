#!/usr/bin/env python3
"""Freeze two non-overlapping windows for the focused seven-asset protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from publication_pipeline_draft.focused_window_training_protocol import (
    validate_protocol,
)


class FocusedWindowScheduleError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FocusedWindowScheduleError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    require(rows, f"CSV is empty: {path}")
    return rows


def materialize(protocol_path: Path, monthly_path: Path,
                panel_manifest_path: Path, output: Path) -> dict[str, Any]:
    require(not output.exists(), f"Output already exists: {output}")
    protocol, protocol_sha256 = validate_protocol(protocol_path)
    panel = json.loads(panel_manifest_path.read_text(encoding="utf-8"))
    require(panel.get("release_status") ==
            "frozen_development_panel_no_test_access" and
            panel.get("panel_id") == protocol["panel_id"] and
            panel.get("evidence_class") == protocol["evidence_class"] and
            panel.get("fresh_confirmatory_data_accessed") is False,
            "Focused panel manifest differs from the protocol.")
    require(sha256(monthly_path) == panel.get("monthly_gross_sha256"),
            "Monthly panel bytes differ from the frozen panel manifest.")
    rows = read_csv(monthly_path)
    design = protocol["window_design"]
    train = int(design["minimum_train_months"])
    validation = int(design["validation_months"])
    test = int(design["test_months"])
    step = int(design["step_months"])
    windows: list[dict[str, Any]] = []
    test_start = train + validation
    number = 1
    while test_start + test <= len(rows):
        validation_start = test_start - validation
        stop = test_start + test - 1
        windows.append({
            "window_id": f"{protocol['window_design_id']}_w{number:02d}",
            "design_id": protocol["window_design_id"],
            "evidence_class": protocol["evidence_class"],
            "claim_limit": protocol["claim_limit"],
            "confirmatory_claim_permitted": False,
            "train_start": rows[0]["decision_date"],
            "train_end": rows[validation_start - 1]["holding_end_date"],
            "validation_start": rows[validation_start]["decision_date"],
            "validation_end": rows[test_start - 1]["holding_end_date"],
            "test_start": rows[test_start]["decision_date"],
            "test_end": rows[stop]["holding_end_date"],
            "train_months": validation_start,
            "validation_months": validation,
            "test_months": test,
            "test_start_index": test_start + 1,
            "test_end_index": stop + 1,
            "test_data_role": "retrospective_development_only",
        })
        number += 1
        test_start += step
    require(len(windows) >= int(design["minimum_windows"]),
            f"Only {len(windows)} focused windows are available.")
    # Freeze exactly the earliest two windows. This selection is deterministic
    # and leaves any later consumed period unused rather than cherry-picking it.
    windows = windows[:2]
    require(windows[0]["test_end"] <= windows[1]["test_start"],
            "Focused test windows overlap.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        schedule = temporary / "window_schedule.csv"
        with schedule.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(windows[0]))
            writer.writeheader()
            writer.writerows(windows)
        manifest = {
            "schema_version": 1,
            "status": "focused_retrospective_windows_frozen",
            "design_id": protocol["window_design_id"],
            "panel_id": protocol["panel_id"],
            "window_count": 2, "test_months_per_window": 24,
            "windows_nonoverlapping": True,
            "protocol_sha256": protocol_sha256,
            "panel_manifest_sha256": sha256(panel_manifest_path),
            "monthly_panel_sha256": sha256(monthly_path),
            "same_market_retrospective": True,
            "confirmatory_claim_permitted": False,
        }
        (temporary / "window_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        (temporary / "READ_ONLY_DEVELOPMENT_WINDOWS.txt").write_text(
            "Two deterministic non-overlapping retrospective windows.\n",
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
    parser.add_argument("--protocol", type=Path, default=Path(
        "publication_pipeline_draft/config/focused_walk_forward_mechanisms_v1.json"))
    parser.add_argument("--monthly-panel", required=True, type=Path)
    parser.add_argument("--panel-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = materialize(args.protocol.resolve(), args.monthly_panel.resolve(),
                             args.panel_manifest.resolve(), args.output)
    except (OSError, ValueError, KeyError, FocusedWindowScheduleError) as error:
        print(f"FOCUSED WINDOW SCHEDULE FAILURE: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
