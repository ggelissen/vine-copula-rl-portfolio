#!/usr/bin/env python3
"""Aggregate non-overlapping development windows with stratified block inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from publication_pipeline_draft.publication_pipeline import (
    Contract, ProtocolError, crra_utility, empirical_metrics, holm_adjust,
    moving_block_indices,
)


class AggregationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AggregationError(message)


def pooled_block_bootstrap(windows: list[np.ndarray], contract: Contract,
                           seed_offset: int) -> dict[str, float]:
    require(len(windows) >= 2 and all(len(window) >= 2 for window in windows),
            "Pooled inference needs at least two valid windows.")
    observed = float(np.concatenate(windows).mean())
    repetitions = int(contract["bootstrap_replications"])
    block = int(contract["bootstrap_block_length"])
    rng = np.random.default_rng(int(contract["inference_seed"]) + seed_offset)
    alternatives = np.empty(repetitions)
    nulls = np.empty(repetitions)
    centered = [window - window.mean() for window in windows]
    for repetition in range(repetitions):
        sampled, null_sampled = [], []
        for window, null_window in zip(windows, centered):
            index = moving_block_indices(rng, len(window), min(block, len(window)))
            sampled.append(window[index]); null_sampled.append(null_window[index])
        alternatives[repetition] = np.concatenate(sampled).mean()
        nulls[repetition] = np.concatenate(null_sampled).mean()
    return {
        "mean_utility_difference": observed,
        "bootstrap_ci_lower": float(np.quantile(alternatives, 0.025)),
        "bootstrap_ci_upper": float(np.quantile(alternatives, 0.975)),
        "bootstrap_p_candidate_greater": float(
            (1 + np.sum(nulls >= observed)) / (repetitions + 1)),
        "bootstrap_replications": repetitions,
        "bootstrap_block_length": block,
    }


def aggregate(result_roots: list[Path], output: Path) -> dict[str, Any]:
    require(not output.exists(), f"Output already exists: {output}")
    require(len(result_roots) >= 2,
            "Walk-forward aggregation requires at least two non-overlapping windows.")
    scored_parts, manifests, contracts, inputs = [], [], [], []
    for root in result_roots:
        run_path = root / "run_manifest.json"
        score_path = root / "raw/scored_monthly_panel.csv"
        strategy_path = root / "raw/validated_strategy_manifest.csv"
        require(all(path.is_file() for path in (run_path, score_path, strategy_path)),
                f"Evaluation result is incomplete: {root}")
        run = json.loads(run_path.read_text(encoding="utf-8"))
        contract_path = Path(run.get("contract_path", ""))
        if not contract_path.is_file():
            # Older output manifests record only a hash. The immutable input
            # inventory retains the absolute contract path.
            hashes = pd.read_csv(root / "raw/input_hashes.csv", dtype=str)
            match = hashes[hashes["artifact"] == "evaluation_contract"]
            require(len(match) == 1, "Cannot resolve evaluation contract.")
            contract_path = Path(match.iloc[0]["path"])
        require(contract_path.is_file() and
                sha256(contract_path) == run["contract_sha256"],
                "Evaluation contract hash mismatch.")
        contract = Contract.read(contract_path)
        require(contract.get("confirmatory_claim_permitted", False) is False,
                "This aggregator is restricted to development evidence.")
        scored = pd.read_csv(score_path)
        strategy = pd.read_csv(strategy_path, dtype=str, keep_default_na=False)
        require(scored["window_id"].nunique() == 1,
                "Each input result must contain exactly one window.")
        scored_parts.append(scored); manifests.append(strategy); contracts.append(contract)
        inputs.append({"result_root": str(root.resolve()),
                       "run_manifest_sha256": sha256(run_path),
                       "scored_panel_sha256": sha256(score_path)})
    baseline = contracts[0]
    economic_fields = [
        "net_exposure", "gross_leverage", "max_long_weight", "max_short_weight",
        "turnover_cost", "annual_short_borrow_rate", "annual_cash_borrow_rate",
        "crra_gamma", "primary_benchmark_id", "primary_strategy_id",
        "primary_sample_scope", "turnover_convention", "financing_proration",
    ]
    for contract in contracts[1:]:
        require(all(contract.get(field) == baseline.get(field)
                    for field in economic_fields),
                "Window evaluation contracts differ on economics or primary tests.")
    canonical = manifests[0][
        ["strategy_id", "label", "method", "role", "include_main",
         "include_inference", "report_seed_distribution"]
    ].sort_values("strategy_id").reset_index(drop=True)
    for manifest in manifests[1:]:
        observed = manifest[canonical.columns].sort_values(
            "strategy_id").reset_index(drop=True)
        require(observed.equals(canonical),
                "Strategy definitions differ between walk-forward windows.")
    scored = pd.concat(scored_parts, ignore_index=True)
    scored["decision_date"] = pd.to_datetime(scored["decision_date"])
    scored["holding_end_date"] = pd.to_datetime(scored["holding_end_date"])
    if str(baseline["primary_sample_scope"]) == "complete_periods":
        complete = scored["is_complete_period"].astype(str).str.lower().isin(
            {"true", "1"})
        scored = scored[complete].copy()
    windows = scored[["window_id", "decision_date", "holding_end_date"]].drop_duplicates()
    intervals = windows.groupby("window_id").agg(
        start=("decision_date", "min"), end=("holding_end_date", "max")
    ).sort_values("start")
    previous_end = None
    for row in intervals.itertuples():
        require(previous_end is None or row.start >= previous_end,
                "Walk-forward test windows overlap.")
        previous_end = row.end

    metadata = canonical.set_index("strategy_id")
    main_ids = metadata[metadata["include_main"].str.lower().isin(
        {"true", "1"})].index.tolist()
    inference_ids = metadata[metadata["include_inference"].str.lower().isin(
        {"true", "1"})].index.tolist()
    metrics = []
    for strategy_id in main_ids:
        group = scored[scored["strategy_id"] == strategy_id].sort_values(
            ["window_id", "decision_date"])
        value = empirical_metrics(group, baseline)
        metrics.append({"strategy_id": strategy_id,
                        "label": metadata.loc[strategy_id, "label"],
                        "method": metadata.loc[strategy_id, "method"],
                        "window_count": len(intervals),
                        "observations": len(group), **value})
    metrics_frame = pd.DataFrame(metrics)

    primary = str(baseline["primary_strategy_id"])
    benchmark = str(baseline["primary_benchmark_id"])
    require(primary in inference_ids and benchmark in inference_ids,
            "Primary strategy/benchmark is not inference eligible.")
    comparisons = []
    for number, alternative in enumerate(inference_ids):
        if alternative == primary:
            continue
        differences = []
        for window_id in intervals.index:
            window = scored[scored["window_id"] == window_id]
            candidate = window[window["strategy_id"] == primary].sort_values(
                "holding_end_date")["net_return"].to_numpy(float)
            reference = window[window["strategy_id"] == alternative].sort_values(
                "holding_end_date")["net_return"].to_numpy(float)
            require(len(candidate) == len(reference) and len(candidate) >= 20,
                    "Window comparison is missing aligned complete observations.")
            differences.append(
                crra_utility(candidate, float(baseline["crra_gamma"])) -
                crra_utility(reference, float(baseline["crra_gamma"]))
            )
        result = pooled_block_bootstrap(differences, baseline, 1000 + number)
        comparisons.append({
            "candidate_id": primary, "candidate_label": metadata.loc[primary, "label"],
            "benchmark_id": alternative,
            "benchmark_label": metadata.loc[alternative, "label"],
            "window_count": len(differences),
            "observations": sum(len(value) for value in differences), **result,
        })
    inference = pd.DataFrame(comparisons)
    inference["bootstrap_p_holm"] = holm_adjust(
        inference["bootstrap_p_candidate_greater"])
    primary_row = inference[inference["benchmark_id"] == benchmark]
    require(len(primary_row) == 1, "Pooled primary comparison is not unique.")
    decision_row = primary_row.iloc[0]
    decision = {
        "schema_version": 1, "evidence_class": "retrospective_walk_forward",
        "confirmatory_claim_permitted": False,
        "window_count": len(intervals), "primary_strategy_id": primary,
        "primary_benchmark_id": benchmark,
        "estimate": float(decision_row["mean_utility_difference"]),
        "bootstrap_ci_lower": float(decision_row["bootstrap_ci_lower"]),
        "bootstrap_ci_upper": float(decision_row["bootstrap_ci_upper"]),
        "one_sided_bootstrap_p": float(
            decision_row["bootstrap_p_candidate_greater"]),
        "development_superiority_signal": bool(
            decision_row["mean_utility_difference"] > 0 and
            decision_row["bootstrap_p_candidate_greater"] <=
            float(baseline["primary_superiority_alpha"])),
        "interpretation": (
            "Window-stratified moving-block inference is retrospective robustness "
            "evidence and cannot restore a consumed confirmatory holdout."),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        metrics_frame.to_csv(temporary / "walk_forward_performance.csv", index=False)
        inference.to_csv(temporary / "walk_forward_pooled_inference.csv", index=False)
        pd.DataFrame(inputs).to_csv(temporary / "input_inventory.csv", index=False)
        (temporary / "walk_forward_decision.json").write_text(
            json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest = {
            "schema_version": 1, "status": "walk_forward_aggregation_complete",
            "window_count": len(intervals), "strategy_count": len(main_ids),
            "inference_method": "window_stratified_circular_moving_block_bootstrap",
            "windows_nonoverlapping": True, "confirmatory_claim_permitted": False,
        }
        (temporary / "aggregation_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, output)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = aggregate([path.resolve() for path in args.results], args.output)
    except (OSError, ValueError, json.JSONDecodeError, ProtocolError,
            AggregationError) as error:
        print(f"WALK-FORWARD AGGREGATION FAILURE: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
