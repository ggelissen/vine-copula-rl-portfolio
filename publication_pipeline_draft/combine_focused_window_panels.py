#!/usr/bin/env python3
"""Combine immutable non-overlapping focused window period panels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


class CombineError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def combine(inputs: list[Path], output: Path) -> dict[str, object]:
    if output.exists() or output.with_suffix(output.suffix + ".manifest.json").exists():
        raise CombineError(f"Output already exists: {output}")
    if len(inputs) < 2:
        raise CombineError("At least two focused windows are required.")
    frames = []
    inventory = []
    for root in inputs:
        manifest_path = root / "focused_score_manifest.json"
        panel_path = root / "focused_scored_period_panel.csv"
        contents = root / "CONTENTS.sha256"
        if not all(path.is_file() for path in (manifest_path, panel_path, contents)):
            raise CombineError(f"Focused score release is incomplete: {root}")
        for line in contents.read_text(encoding="ascii").splitlines():
            if line.strip():
                expected, relative = line.split("  ", 1)
                target = root / relative
                if not target.is_file() or sha256(target) != expected:
                    raise CombineError(f"Focused score checksum mismatch: {target}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "focused_window_common_accounting_complete":
            raise CombineError(f"Focused score status is invalid: {root}")
        frame = pd.read_csv(panel_path)
        if len(frame) != 24 * 24 or frame["strategy_id"].nunique() != 24:
            raise CombineError(f"Focused score cardinality is invalid: {root}")
        frames.append(frame)
        inventory.append({"window_id": manifest["window_id"],
                          "score_root": str(root.resolve()),
                          "panel_sha256": sha256(panel_path),
                          "manifest_sha256": sha256(manifest_path)})
    combined = pd.concat(frames, ignore_index=True)
    intervals = combined.groupby("window_id").agg(
        start=("decision_date", "min"), end=("holding_end_date", "max")
    ).sort_values("start")
    previous = None
    for row in intervals.itertuples():
        if previous is not None and row.start < previous:
            raise CombineError("Focused window periods overlap.")
        previous = row.end
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    combined.to_csv(temporary, index=False)
    os.replace(temporary, output)
    result = {"schema_version": 1,
              "status": "focused_window_period_panels_combined",
              "window_count": len(intervals), "strategy_count_per_window": 24,
              "rows": len(combined), "windows_nonoverlapping": True,
              "inputs": inventory,
              "combined_panel_sha256": sha256(output),
              "confirmatory_claim_permitted": False}
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = combine([path.resolve() for path in args.inputs], args.output)
    except (OSError, ValueError, KeyError, CombineError) as error:
        print(f"FOCUSED PANEL COMBINE FAILURE: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
