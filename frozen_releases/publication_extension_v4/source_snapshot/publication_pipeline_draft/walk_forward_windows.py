#!/usr/bin/env python3
"""Materialize non-overlapping development/external walk-forward windows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

from publication_pipeline_draft.publication_research_program import validate_program


class WindowProtocolError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise WindowProtocolError(message)


def load_monthly(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    require(bool(rows), "Monthly panel is empty.")
    required = {"decision_date", "holding_end_date"}
    require(required <= set(rows[0]), "Monthly panel lacks canonical dates.")
    for row in rows:
        row["decision_date"] = date.fromisoformat(row["decision_date"]).isoformat()
        row["holding_end_date"] = date.fromisoformat(row["holding_end_date"]).isoformat()
    require([row["decision_date"] for row in rows] == sorted(
        row["decision_date"] for row in rows), "Monthly periods are not sorted.")
    require(len({row["decision_date"] for row in rows}) == len(rows),
            "Monthly decision dates are duplicated.")
    return rows


def build_windows(program_path: Path, design_id: str, monthly_path: Path,
                  panel_manifest_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validated = validate_program(program_path)
    designs = {row["design_id"]: row for row in validated.raw["window_designs"]}
    require(design_id in designs, f"Unknown design: {design_id}")
    design = designs[design_id]
    require(design["evidence_class"] != "future_temporal",
            "Future temporal windows require the separate custodian/access-ledger protocol.")
    manifest = json.loads(panel_manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("release_status") == "frozen_development_panel_no_test_access",
            "Panel is not a frozen development release.")
    require(manifest.get("test_data_accessed") is False,
            "Development schedule cannot contain confirmatory test data.")
    require(manifest.get("panel_id") == design["panel_id"],
            "Window design and panel ID disagree.")
    rows = load_monthly(monthly_path)
    train = int(design["minimum_train_months"])
    validation = int(design["validation_months"])
    test = int(design["test_months"])
    step = int(design["step_months"])
    windows: list[dict[str, Any]] = []
    test_start = train + validation
    number = 1
    while test_start + test <= len(rows):
        validation_start = test_start - validation
        test_stop = test_start + test - 1
        windows.append({
            "window_id": f"{design_id}_w{number:02d}",
            "design_id": design_id,
            "evidence_class": design["evidence_class"],
            "claim_limit": design["claim_limit"],
            "train_start": rows[0]["decision_date"],
            "train_end": rows[validation_start - 1]["holding_end_date"],
            "validation_start": rows[validation_start]["decision_date"],
            "validation_end": rows[test_start - 1]["holding_end_date"],
            "test_start": rows[test_start]["decision_date"],
            "test_end": rows[test_stop]["holding_end_date"],
            "train_months": validation_start,
            "validation_months": validation,
            "test_months": test,
            "test_start_index": test_start + 1,
            "test_end_index": test_stop + 1,
            "test_data_role": "retrospective_development_only",
        })
        number += 1
        test_start += step
    require(len(windows) >= int(design["minimum_windows"]),
            f"Only {len(windows)} windows are available; design requires "
            f"{design['minimum_windows']}.")
    for previous, current in zip(windows, windows[1:]):
        require(previous["test_end"] <= current["test_start"],
                "Test windows overlap; this would understate uncertainty.")
    summary = {
        "schema_version": 1,
        "status": "materialized_development_walk_forward_windows",
        "design_id": design_id,
        "evidence_class": design["evidence_class"],
        "claim_limit": design["claim_limit"],
        "panel_id": manifest["panel_id"],
        "window_count": len(windows),
        "program_sha256": validated.sha256,
        "panel_manifest_sha256": hashlib.sha256(
            panel_manifest_path.read_bytes()).hexdigest(),
        "monthly_panel_sha256": hashlib.sha256(monthly_path.read_bytes()).hexdigest(),
        "confirmatory_claim_permitted": False,
        "test_data_accessed": False,
    }
    return windows, summary


def write_release(output: Path, windows: list[dict[str, Any]],
                  summary: dict[str, Any]) -> None:
    require(not output.exists(), f"Window release already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        with (temporary / "window_schedule.csv").open(
                "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(windows[0]))
            writer.writeheader(); writer.writerows(windows)
        (temporary / "window_manifest.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (temporary / "READ_ONLY_DEVELOPMENT_WINDOWS.txt").write_text(
            "Retrospective/external development evidence only; not fresh confirmation.\n",
            encoding="utf-8")
        os.replace(temporary, output)
    except Exception:
        import shutil
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, default=Path(
        "publication_pipeline_draft/config/publication_research_program_v2.json"))
    parser.add_argument("--design-id", required=True)
    parser.add_argument("--monthly-panel", required=True, type=Path)
    parser.add_argument("--panel-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        windows, summary = build_windows(
            args.program, args.design_id, args.monthly_panel, args.panel_manifest)
        write_release(args.output, windows, summary)
    except (OSError, ValueError, WindowProtocolError) as error:
        print(f"WINDOW PROTOCOL FAILURE: {error}")
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
