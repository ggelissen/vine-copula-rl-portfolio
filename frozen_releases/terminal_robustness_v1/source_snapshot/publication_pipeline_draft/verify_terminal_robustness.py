#!/usr/bin/env python3
"""Fail-closed semantic and clean-room verification of terminal robustness results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class TerminalVerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TerminalVerificationError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_contents(root: Path) -> None:
    contents = root / "CONTENTS.sha256"
    require(contents.is_file(), f"CONTENTS.sha256 is missing: {root}")
    for line in contents.read_text(encoding="ascii").splitlines():
        if line.strip():
            expected, relative = line.split("  ", 1)
            target = root / relative.removeprefix("./")
            require(target.is_file() and sha256(target) == expected,
                    f"Result hash mismatch: {relative}")


def semantic_verify(root: Path) -> dict[str, Any]:
    verify_contents(root)
    manifest = json.loads((root / "terminal_robustness_manifest.json").read_text(
        encoding="utf-8"))
    require(manifest.get("status") == "terminal_robustness_campaign_complete",
            "Terminal campaign is not complete.")
    require(manifest.get("policy_retraining_performed") is False and
            manifest.get("model_selection_performed") is False and
            manifest.get("confirmatory_claim_created") is False,
            "Terminal campaign exceeded its frozen claim authority.")
    require(manifest.get("evidence_classes_kept_separate") is True and
            manifest.get("all_daily_paths_reconciled") is True,
            "Evidence separation or daily reconciliation failed.")
    tables = root / "tables"
    reconciliation = pd.read_csv(tables / "daily_monthly_reconciliation.csv")
    require(bool(len(reconciliation)) and
            reconciliation["reconciliation_pass"].astype(bool).all(),
            "A daily path failed monthly reconciliation.")
    require(reconciliation[["gross_error", "net_error"]].abs().to_numpy().max() <=
            1e-8, "Daily reconciliation exceeds the frozen tolerance.")
    risk = pd.read_csv(tables / "daily_tail_risk_metrics.csv")
    require(bool(len(risk)) and np.isfinite(risk.select_dtypes(
        include=[np.number]).to_numpy()).all(), "Daily risk metrics are incomplete.")
    primary_tail = risk[(risk["scope"] == "complete_periods") &
                        (risk["daily_observations"] >= 400)]
    require(bool(len(primary_tail)) and
            (primary_tail["daily_tail_95_event_count"] >= 20).all(),
            "Five-percent daily tail metrics lack twenty events.")
    friction = pd.read_csv(tables / "friction_surface.csv")
    require({0, 10, 25, 50} <= set(
        friction["transaction_cost_bps_one_way"].astype(int)),
        "Transaction-cost sensitivity grid is incomplete.")
    require({0, 3, 6, 10} <= set(
        friction["annual_short_borrow_percent"].astype(int)),
        "Short-borrow sensitivity grid is incomplete.")
    resampling = pd.read_csv(tables / "resampling_robustness.csv")
    require({"moving_block", "stationary"} == set(resampling["method"]),
            "Both registered resampling methods are required.")
    require({1, 2, 3, 4, 6} <= set(resampling.loc[
        resampling["method"] == "moving_block", "block_length"].astype(int)),
        "Moving-block sensitivity grid is incomplete.")
    require({2, 3, 6, 12} <= set(resampling.loc[
        resampling["method"] == "stationary", "block_length"].astype(int)),
        "Stationary-bootstrap sensitivity grid is incomplete.")
    require((resampling["bootstrap_replications"] >= 50000).all(),
            "Resampling replications fall below the frozen contract.")
    summary = pd.read_csv(tables / "registered_contrast_robustness_summary.csv")
    require(bool(len(summary)) and (summary["resampling_specifications"] == 9).all(),
            "Registered contrast summary is incomplete.")
    primary = pd.read_csv(tables / "primary_economic_metrics.csv")
    require(bool(len(primary)) and
            (primary["transaction_cost_bps_one_way"] == 10).all() and
            (primary["annual_short_borrow_percent"] == 3).all() and
            (primary["annual_cash_borrow_percent"] == 2).all(),
            "Primary economic metric extraction differs from the contract.")
    ledger = pd.read_csv(tables / "evidence_ledger.csv")
    require({"frozen_primary_evaluation", "post_holdout_explanatory",
             "retrospective_walk_forward"} <= set(ledger["evidence_class"]),
            "Evidence ledger omits a registered evidence class.")
    return {
        "status": "terminal_robustness_semantic_verification_passed",
        "source_count": int(len(ledger)),
        "daily_reconciled_period_paths": int(len(reconciliation)),
        "daily_risk_rows": int(len(risk)),
        "friction_surface_rows": int(len(friction)),
        "resampling_rows": int(len(resampling)),
        "registered_contrasts": int(len(summary)),
        "minimum_five_percent_tail_events": int(
            primary_tail["daily_tail_95_event_count"].min()),
        "contents_sha256": sha256(root / "CONTENTS.sha256"),
    }


def cleanroom_compare(reference: Path, candidate: Path) -> dict[str, Any]:
    left = semantic_verify(reference)
    right = semantic_verify(candidate)
    left_tables = reference / "tables"
    right_tables = candidate / "tables"
    names = sorted(path.name for path in left_tables.iterdir() if path.is_file())
    require(names == sorted(path.name for path in right_tables.iterdir()
                            if path.is_file()),
            "Clean-room table inventory differs from the reference.")
    mismatches = [name for name in names if sha256(left_tables / name) !=
                  sha256(right_tables / name)]
    require(not mismatches,
            "Clean-room table bytes differ: " + ", ".join(mismatches))
    return {
        "status": "terminal_robustness_cleanroom_reproduction_passed",
        "table_count": len(names), "byte_identical_tables": True,
        "reference_contents_sha256": left["contents_sha256"],
        "candidate_contents_sha256": right["contents_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--cleanroom-results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = (cleanroom_compare(args.results.resolve(),
                                    args.cleanroom_results.resolve())
                  if args.cleanroom_results else semantic_verify(
                      args.results.resolve()))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            require(not args.output.exists(), f"Verification output exists: {args.output}")
            args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                                   encoding="utf-8")
    except (OSError, ValueError, KeyError, json.JSONDecodeError,
            TerminalVerificationError) as error:
        print(f"TERMINAL ROBUSTNESS VERIFICATION FAILURE: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
