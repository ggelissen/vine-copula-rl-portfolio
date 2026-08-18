#!/usr/bin/env python3
"""Analyze focused mechanism paths across non-overlapping walk-forward windows.

Input is a standardized period panel containing the three focused experiments,
five seed strategies and one arithmetic target-weight ensemble per experiment.
The analyzer treats time/windows as the market sample and seeds only as policy
optimization variability.
"""

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
    crra_utility, holm_adjust, moving_block_indices,
)
from publication_pipeline_draft.focused_window_training_protocol import (
    validate_protocol,
)


class FocusedAnalysisError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FocusedAnalysisError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def certainty_equivalent_from_mean_utility(mean_utility: float,
                                           gamma: float) -> float:
    """Return the one-period CE simple return for a mean CRRA utility."""
    if math.isclose(gamma, 1.0):
        gross = math.exp(mean_utility)
    else:
        base = 1.0 + (1.0 - gamma) * mean_utility
        require(base > 0, "CRRA certainty equivalent is outside its domain.")
        gross = base ** (1.0 / (1.0 - gamma))
    return gross - 1.0


def annualized_ce_from_returns(returns: np.ndarray, gamma: float,
                               annual_factor: float) -> float:
    monthly = certainty_equivalent_from_mean_utility(
        float(crra_utility(returns, gamma).mean()), gamma)
    return (1.0 + monthly) ** annual_factor - 1.0


def bootstrap(reference_returns: list[np.ndarray],
              alternative_returns: list[np.ndarray], replications: int,
              block: int, seed: int, gamma: float,
              annual_factor: float) -> dict[str, float]:
    require(len(reference_returns) == len(alternative_returns) > 0,
            "Focused bootstrap needs matched reference/alternative windows.")
    differences = [
        crra_utility(reference, gamma) - crra_utility(alternative, gamma)
        for reference, alternative in zip(reference_returns,
                                          alternative_returns)
    ]
    observed = float(np.concatenate(differences).mean())
    observed_ce = (
        annualized_ce_from_returns(
            np.concatenate(reference_returns), gamma, annual_factor) -
        annualized_ce_from_returns(
            np.concatenate(alternative_returns), gamma, annual_factor)
    )
    rng = np.random.default_rng(seed)
    ce_differences = np.empty(replications)
    nulls = np.empty(replications)
    centered = [value - value.mean() for value in differences]
    for repetition in range(replications):
        sampled_reference: list[np.ndarray] = []
        sampled_alternative: list[np.ndarray] = []
        null_sampled: list[np.ndarray] = []
        for reference, alternative, null in zip(
                reference_returns, alternative_returns, centered):
            require(len(reference) == len(alternative) == len(null),
                    "Focused bootstrap windows are not paired.")
            index = moving_block_indices(
                rng, len(reference), min(block, len(reference)))
            sampled_reference.append(reference[index])
            sampled_alternative.append(alternative[index])
            null_sampled.append(null[index])
        ce_differences[repetition] = (
            annualized_ce_from_returns(
                np.concatenate(sampled_reference), gamma, annual_factor) -
            annualized_ce_from_returns(
                np.concatenate(sampled_alternative), gamma, annual_factor)
        )
        nulls[repetition] = np.concatenate(null_sampled).mean()
    lower, upper = np.quantile(ce_differences, [0.025, 0.975])
    return {
        "observed_mean_crra_utility_difference": observed,
        "annualized_ce_difference": observed_ce,
        "annualized_ce_ci_lower": float(lower),
        "annualized_ce_ci_upper": float(upper),
        "raw_one_sided_p_reference_greater": float(
            (1 + np.sum(nulls >= observed)) / (replications + 1)),
    }


def analyze(protocol_path: Path, panel_path: Path, output: Path) -> dict[str, Any]:
    require(not output.exists(), f"Analysis output already exists: {output}")
    protocol, protocol_sha256 = validate_protocol(protocol_path)
    frame = pd.read_csv(panel_path)
    required = {"window_id", "experiment_id", "strategy_level", "strategy_id",
                "seed", "decision_date", "holding_end_date", "net_return",
                "is_complete_period"}
    require(required <= set(frame),
            f"Focused period panel lacks: {sorted(required - set(frame))}")
    expected_experiments = {item["experiment_id"]
                            for item in protocol["experiments"]}
    benchmark_ids = set(protocol["financial_benchmarks"])
    require(set(frame["experiment_id"]) == expected_experiments | benchmark_ids,
            "Period panel contains undeclared focused experiments/benchmarks.")
    complete = frame["is_complete_period"].astype(str).str.lower().isin(
        {"true", "1", "yes"})
    frame = frame[complete].copy()
    frame["decision_date"] = pd.to_datetime(frame["decision_date"])
    frame["holding_end_date"] = pd.to_datetime(frame["holding_end_date"])
    frame["net_return"] = pd.to_numeric(frame["net_return"], errors="raise")
    require(np.isfinite(frame["net_return"]).all() and
            (frame["net_return"] > -1).all(),
            "Focused simple returns must be finite and exceed -100 percent.")
    windows = frame[["window_id", "decision_date", "holding_end_date"]].drop_duplicates()
    intervals = windows.groupby("window_id").agg(
        start=("decision_date", "min"), end=("holding_end_date", "max")
    ).sort_values("start")
    require(len(intervals) >= 2,
            "Focused mechanism inference requires at least two windows.")
    previous = None
    for row in intervals.itertuples():
        require(previous is None or row.start >= previous,
                "Focused test windows overlap.")
        previous = row.end
    levels = set(frame["strategy_level"])
    require(levels == {"seed", "ensemble", "benchmark"},
            "Focused panel must retain seed, ensemble, and benchmark paths.")
    for experiment in expected_experiments:
        subset = frame[frame["experiment_id"] == experiment]
        seeds = subset[subset["strategy_level"] == "seed"]["seed"].dropna().unique()
        require(len(seeds) == 5,
                f"Focused experiment {experiment} does not contain five seeds.")
        ensembles = subset[subset["strategy_level"] == "ensemble"]["strategy_id"].unique()
        require(len(ensembles) == 1,
                f"Focused experiment {experiment} needs one ensemble.")
    for benchmark_id in benchmark_ids:
        benchmark = frame[
            (frame["experiment_id"] == benchmark_id) &
            (frame["strategy_level"] == "benchmark")]
        require(benchmark["strategy_id"].nunique() == 1 and
                set(benchmark["window_id"]) == set(intervals.index),
                f"Focused benchmark is incomplete: {benchmark_id}")

    calendar = frame[["window_id", "decision_date", "holding_end_date"]].drop_duplicates()
    holding_days = int(
        (calendar["holding_end_date"] - calendar["decision_date"]).dt.days.sum())
    require(holding_days > 0, "Focused complete-period calendar has no exposure time.")
    annual_factor = len(calendar) / (holding_days / 365.0)

    gamma = float(protocol["crra_gamma"])
    repetitions = int(protocol["bootstrap"]["replications"])
    block = int(protocol["bootstrap"]["block_length_periods"])
    base_seed = int(protocol["bootstrap"]["seed"])
    contrasts: list[dict[str, Any]] = []
    for number, contrast in enumerate(protocol["contrasts"]):
        reference = contrast["reference_experiment_id"]
        alternative = contrast["alternative_experiment_id"]
        reference_windows: list[np.ndarray] = []
        alternative_windows: list[np.ndarray] = []
        for window_id in intervals.index:
            window = frame[(frame["window_id"] == window_id) &
                           (frame["strategy_level"] == "ensemble")]
            ref = window[window["experiment_id"] == reference].sort_values(
                "holding_end_date")
            alt = window[window["experiment_id"] == alternative].sort_values(
                "holding_end_date")
            require(len(ref) == len(alt) >= 20 and
                    ref["holding_end_date"].tolist() ==
                    alt["holding_end_date"].tolist(),
                    f"Unaligned focused comparison in {window_id}.")
            reference_windows.append(ref["net_return"].to_numpy(float))
            alternative_windows.append(alt["net_return"].to_numpy(float))
        result = bootstrap(reference_windows, alternative_windows,
                           repetitions, block,
                           base_seed + number, gamma, annual_factor)
        contrasts.append({
            "label": contrast["label"],
            "reference_experiment_id": reference,
            "alternative_experiment_id": alternative,
            "window_count": len(intervals),
            "observations": sum(len(value) for value in reference_windows),
            **result,
        })
    contrast_frame = pd.DataFrame(contrasts)
    contrast_frame["holm_adjusted_p_value"] = holm_adjust(
        contrast_frame["raw_one_sided_p_reference_greater"])
    contrast_frame["reference_superiority_signal"] = (
        (contrast_frame["observed_mean_crra_utility_difference"] > 0) &
        (contrast_frame["holm_adjusted_p_value"] <= 0.05))
    contrast_frame["opposite_direction_signal"] = (
        (contrast_frame["annualized_ce_ci_upper"] < 0))

    # Separate exploratory economic family: the compressed vine-CVaR ensemble
    # versus six financial benchmarks. This does not alter the two mechanism
    # contrasts or authorize confirmatory claims.
    candidate = protocol["benchmark_candidate_experiment_id"]
    benchmark_rows: list[dict[str, Any]] = []
    for number, benchmark_id in enumerate(protocol["financial_benchmarks"]):
        candidate_windows: list[np.ndarray] = []
        benchmark_windows: list[np.ndarray] = []
        for window_id in intervals.index:
            window = frame[frame["window_id"] == window_id]
            candidate_path = window[
                (window["experiment_id"] == candidate) &
                (window["strategy_level"] == "ensemble")].sort_values(
                    "holding_end_date")
            benchmark_path = window[
                (window["experiment_id"] == benchmark_id) &
                (window["strategy_level"] == "benchmark")].sort_values(
                    "holding_end_date")
            require(len(candidate_path) == len(benchmark_path) >= 20 and
                    candidate_path["holding_end_date"].tolist() ==
                    benchmark_path["holding_end_date"].tolist(),
                    f"Unaligned benchmark comparison in {window_id}: {benchmark_id}")
            candidate_windows.append(candidate_path["net_return"].to_numpy(float))
            benchmark_windows.append(benchmark_path["net_return"].to_numpy(float))
        result = bootstrap(
            candidate_windows, benchmark_windows, repetitions, block,
            base_seed + 1000 + number, gamma, annual_factor)
        benchmark_rows.append({
            "candidate_experiment_id": candidate,
            "benchmark_id": benchmark_id,
            "window_count": len(intervals),
            "observations": sum(len(value) for value in candidate_windows),
            **result,
        })
    benchmark_frame = pd.DataFrame(benchmark_rows)
    benchmark_frame["holm_adjusted_p_value"] = holm_adjust(
        benchmark_frame["raw_one_sided_p_reference_greater"])
    benchmark_frame["exploratory_candidate_superiority_signal"] = (
        (benchmark_frame["annualized_ce_difference"] > 0) &
        (benchmark_frame["holm_adjusted_p_value"] <= 0.05))
    benchmark_frame["confirmatory_claim_permitted"] = False

    seed_rows: list[dict[str, Any]] = []
    reference = protocol["reference_experiment_id"]
    for alternative in sorted(expected_experiments - {reference}):
        for seed in protocol["seeds"]:
            ref = frame[(frame["experiment_id"] == reference) &
                        (frame["strategy_level"] == "seed") &
                        (pd.to_numeric(frame["seed"], errors="coerce") == seed)]
            alt = frame[(frame["experiment_id"] == alternative) &
                        (frame["strategy_level"] == "seed") &
                        (pd.to_numeric(frame["seed"], errors="coerce") == seed)]
            ref = ref.sort_values(["window_id", "holding_end_date"])
            alt = alt.sort_values(["window_id", "holding_end_date"])
            require(len(ref) == len(alt) and len(ref) >= 40,
                    "Matched focused seed paths are incomplete.")
            ref_returns = ref["net_return"].to_numpy(float)
            alt_returns = alt["net_return"].to_numpy(float)
            effect = (
                annualized_ce_from_returns(ref_returns, gamma, annual_factor) -
                annualized_ce_from_returns(alt_returns, gamma, annual_factor))
            seed_rows.append({"alternative_experiment_id": alternative,
                              "seed": seed,
                              "annualized_ce_difference": effect})
    seed_frame = pd.DataFrame(seed_rows)
    stability = seed_frame.groupby("alternative_experiment_id")[
        "annualized_ce_difference"].agg(
            matched_seed_count="count", mean="mean", median="median",
            standard_deviation="std", minimum="min", maximum="max").reset_index()
    positive = seed_frame.assign(
        positive=seed_frame["annualized_ce_difference"] > 0).groupby(
            "alternative_experiment_id")["positive"].mean().reset_index(
            name="fraction_reference_positive")
    stability = stability.merge(positive, on="alternative_experiment_id",
                                validate="one_to_one")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        contrast_frame.to_csv(temporary / "focused_walk_forward_contrasts.csv",
                              index=False)
        benchmark_frame.to_csv(
            temporary / "focused_walk_forward_benchmark_comparisons.csv",
            index=False)
        seed_frame.to_csv(temporary / "focused_walk_forward_seed_effects.csv",
                          index=False)
        stability.to_csv(temporary / "focused_walk_forward_seed_stability.csv",
                         index=False)
        manifest = {
            "schema_version": 1,
            "status": "focused_walk_forward_analysis_complete",
            "protocol_sha256": protocol_sha256,
            "period_panel_sha256": sha256(panel_path),
            "window_count": len(intervals),
            "experiment_count": 3, "seed_count_per_experiment": 5,
            "contrast_count": 2, "benchmark_comparison_count": 6,
            "bootstrap_replications": repetitions,
            "annualization_factor": annual_factor,
            "windows_nonoverlapping": True,
            "seed_inference_scope": "optimization_variability_only",
            "market_inference_scope": "window_stratified_period_bootstrap",
            "evidence_class": "retrospective_walk_forward",
            "confirmatory_claim_permitted": False,
        }
        (temporary / "focused_walk_forward_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
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
    parser.add_argument("--period-panel", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = analyze(args.protocol.resolve(), args.period_panel.resolve(),
                         args.output)
    except (OSError, ValueError, KeyError, FocusedAnalysisError) as error:
        print(f"FOCUSED WALK-FORWARD ANALYSIS FAILURE: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
