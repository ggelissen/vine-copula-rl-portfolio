#!/usr/bin/env python3
"""Common-accounting analysis for the post-holdout 100-path dose experiment."""

from __future__ import annotations

import argparse
import csv
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

from publication_pipeline_draft.analyze_causal_results import (
    annualization, annualized_ce, circular_indices, crra_utility, holm_adjust,
    metrics,
)
from publication_pipeline_draft.publication_pipeline import (
    KEYS, Contract, ProtocolError, read_realized_panel, score_strategy,
    validate_weight_matrix,
)
from publication_pipeline_draft.synthetic_dose_protocol import (
    DoseProtocolError, load_contract, read_csv, require, sha256,
)


class DoseAnalysisError(DoseProtocolError):
    pass


def load_weight(path: Path, expected_sha: str, realized: pd.DataFrame,
                assets: list[str], evaluation: Contract) -> pd.DataFrame:
    require(path.is_file() and sha256(path) == expected_sha,
            f"Dose weight file is missing or changed: {path}")
    frame = pd.read_csv(path)
    if "window_id" not in frame:
        frame["window_id"] = "locked_oos_v1"
    for name in ("decision_date", "holding_end_date"):
        frame[name] = pd.to_datetime(frame[name], errors="raise").dt.normalize()
    frame["window_id"] = frame["window_id"].astype(str)
    columns = [f"w_{asset}" for asset in assets]
    require(set(KEYS + columns) <= set(frame.columns),
            f"Dose weight columns are incomplete: {path}")
    frame = frame[KEYS + columns].sort_values(KEYS).reset_index(drop=True)
    expected = realized[KEYS].sort_values(KEYS).reset_index(drop=True)
    require(frame[KEYS].equals(expected),
            f"Dose weights do not match the locked calendar: {path}")
    matrix = frame[columns].apply(pd.to_numeric, errors="raise").to_numpy(float)
    validate_weight_matrix(matrix, path.name, evaluation)
    frame[columns] = matrix
    return frame


def score_new_paths(repo: Path, manifest_path: Path, realized: pd.DataFrame,
                    assets: list[str], evaluation: Contract) -> tuple[
                        pd.DataFrame, dict[str, pd.DataFrame]]:
    rows = read_csv(manifest_path)
    require(len(rows) == 20, "Dose weight manifest must contain 20 policies.")
    keys = {(row["experiment_id"], int(row["seed"])) for row in rows}
    require(len(keys) == 20, "Dose weight manifest keys are duplicated.")
    weights: dict[tuple[str, int], pd.DataFrame] = {}
    for row in rows:
        key = (row["experiment_id"], int(row["seed"]))
        weights[key] = load_weight(repo / row["path"], row["sha256"],
                                   realized, assets, evaluation)
    experiments = sorted({key[0] for key in keys})
    weight_columns = [f"w_{asset}" for asset in assets]
    ensembles: dict[str, pd.DataFrame] = {}
    scored: list[pd.DataFrame] = []
    for key, weight in sorted(weights.items()):
        path = score_strategy(f"{key[0]}__seed_{key[1]}", weight,
                              realized, assets, evaluation)
        path["experiment_id"] = key[0]
        path["strategy_level"] = "seed"
        path["seed"] = key[1]
        scored.append(path)
    for experiment in experiments:
        members = [weights[key] for key in sorted(weights) if key[0] == experiment]
        require(len(members) == 10, f"{experiment} lacks ten weight paths.")
        ensemble = members[0][KEYS].copy()
        ensemble[weight_columns] = np.stack(
            [member[weight_columns].to_numpy(float) for member in members]).mean(axis=0)
        validate_weight_matrix(ensemble[weight_columns].to_numpy(float),
                               f"{experiment}_ensemble", evaluation)
        ensembles[experiment] = ensemble
        path = score_strategy(f"{experiment}_ensemble", ensemble,
                              realized, assets, evaluation)
        path["experiment_id"] = experiment
        path["strategy_level"] = "ensemble"
        path["seed"] = np.nan
        scored.append(path)
    return pd.concat(scored, ignore_index=True), ensembles


def normalize_comparators(causal_path: Path, benchmark_path: Path,
                          dose_contract: dict[str, Any], *,
                          complete_only: bool = True) -> dict[str, pd.DataFrame]:
    causal = pd.read_csv(causal_path)
    causal["complete"] = causal["complete"].astype(str).str.lower().isin(
        {"true", "1", "yes"})
    for name in ("decision_date", "holding_end_date"):
        causal[name] = pd.to_datetime(causal[name], errors="raise")
    comparators: dict[str, pd.DataFrame] = {}
    causal_ids = {
        "full_vine_state_and_cvar_observation",
        "zero_vine_features_and_cvar_observation",
        "historical_only_no_synthetic_pretraining",
    }
    for experiment in causal_ids:
        mask = ((causal["experiment_id"] == experiment) &
                (causal["strategy_level"] == "ensemble"))
        if complete_only:
            mask &= causal["complete"]
        group = causal[mask].copy()
        expected = 22 if complete_only else 24
        require(len(group) == expected,
                f"Prior causal comparator has the wrong scope: {experiment}")
        comparators[experiment] = group.sort_values("decision_date")
    benchmarks = pd.read_csv(benchmark_path)
    for name in ("decision_date", "holding_end_date"):
        benchmarks[name] = pd.to_datetime(benchmarks[name], errors="raise")
    complete = benchmarks["is_complete_period"].astype(str).str.lower().isin(
        {"true", "1", "yes"})
    benchmark_ids = set(dose_contract["comparison_targets"]) - causal_ids
    for strategy in benchmark_ids:
        mask = benchmarks["strategy_id"] == strategy
        if complete_only:
            mask &= complete
        group = benchmarks[mask].copy()
        expected = 22 if complete_only else 24
        require(len(group) == expected,
                f"Benchmark comparator has the wrong scope: {strategy}")
        comparators[strategy] = group.sort_values("decision_date")
    return comparators


def contrast(candidate: pd.DataFrame, comparator: pd.DataFrame,
             causal_contract: dict[str, Any], seed: int) -> dict[str, Any]:
    candidate = candidate.sort_values(["decision_date", "holding_end_date"])
    comparator = comparator.sort_values(["decision_date", "holding_end_date"])
    require(candidate[["decision_date", "holding_end_date"]].reset_index(drop=True).equals(
            comparator[["decision_date", "holding_end_date"]].reset_index(drop=True)),
            "A dose contrast does not use identical periods.")
    candidate_returns = candidate["net_return"].to_numpy(float)
    comparator_returns = comparator["net_return"].to_numpy(float)
    factor, _ = annualization(candidate, causal_contract)
    gamma = float(causal_contract["economics"]["crra_gamma"])
    difference = (crra_utility(candidate_returns, gamma) -
                  crra_utility(comparator_returns, gamma))
    observed_utility = float(difference.mean())
    observed_ce = (annualized_ce(candidate_returns, gamma, factor) -
                   annualized_ce(comparator_returns, gamma, factor))
    settings = causal_contract["inference"]
    replications = int(settings["bootstrap_replications"])
    block = int(settings["bootstrap_block_length"])
    rng = np.random.default_rng(seed)
    centered = difference - observed_utility
    null = np.empty(replications)
    effects = np.empty(replications)
    for index in range(replications):
        sample = circular_indices(rng, len(difference), block)
        null[index] = centered[sample].mean()
        effects[index] = (
            annualized_ce(candidate_returns[sample], gamma, factor) -
            annualized_ce(comparator_returns[sample], gamma, factor))
    return {
        "annualized_ce_difference": observed_ce,
        "mean_monthly_crra_utility_difference": observed_utility,
        "ci_lower": float(np.quantile(effects, 0.025, method="median_unbiased")),
        "ci_upper": float(np.quantile(effects, 0.975, method="median_unbiased")),
        "one_sided_p_candidate_greater": float(
            (1 + np.sum(null >= observed_utility)) / (replications + 1)),
        "bootstrap_block_length": block,
        "bootstrap_replications": replications,
        "common_periods": len(difference),
    }


def build_contrasts(items: list[dict[str, str]], paths: dict[str, pd.DataFrame],
                    causal_contract: dict[str, Any], family: str,
                    seed_base: int) -> pd.DataFrame:
    rows = []
    for index, item in enumerate(items):
        candidate, comparator = item["candidate"], item["comparator"]
        result = contrast(paths[candidate], paths[comparator], causal_contract,
                          seed_base + 1009 * index)
        rows.append({"family": family, "candidate": candidate,
                     "comparator": comparator, "label": item.get(
                         "label", f"{candidate} minus {comparator}"), **result})
    adjusted = holm_adjust([row["one_sided_p_candidate_greater"] for row in rows])
    for row, value in zip(rows, adjusted):
        row["holm_p_candidate_greater"] = value
        row["decision"] = (
            "positive_effect_holm_0.05" if row["annualized_ce_difference"] > 0 and
            value <= 0.05 else
            "opposite_direction_interval" if row["ci_upper"] < 0 else
            "not_established")
        row["confirmatory_claim_permitted"] = False
    return pd.DataFrame(rows)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root.resolve()
    require(not args.output.exists(), f"Dose analysis output exists: {args.output}")
    dose, dose_sha = load_contract(args.contract)
    causal_contract = json.loads(args.causal_contract.read_text(encoding="utf-8"))
    evaluation = Contract.read(args.evaluation_contract)
    realized, assets = read_realized_panel(args.realized, evaluation)
    scored, ensembles = score_new_paths(repo, args.weight_manifest, realized,
                                        assets, evaluation)
    new_complete = scored[scored["is_complete_period"].astype(bool)].copy()
    counts = new_complete.groupby("strategy_id").size()
    require(len(counts) == 22 and (counts == 22).all(),
            "Every one of 20 dose seeds and two ensembles must have 22 common periods.")
    comparators = normalize_comparators(
        args.causal_panel, args.benchmark_panel, dose, complete_only=True)
    locked_all_comparators = normalize_comparators(
        args.causal_panel, args.benchmark_panel, dose, complete_only=False)

    paths: dict[str, pd.DataFrame] = dict(comparators)
    for experiment in sorted(ensembles):
        group = new_complete[
            (new_complete["experiment_id"] == experiment) &
            (new_complete["strategy_level"] == "ensemble")].copy()
        require(len(group) == 22, f"Dose ensemble is incomplete: {experiment}")
        paths[experiment] = group.sort_values("decision_date")
    canonical = paths["synthetic_100_full_vine_state"][[
        "decision_date", "holding_end_date"]].reset_index(drop=True)
    for label, group in paths.items():
        require(group[["decision_date", "holding_end_date"]].reset_index(
            drop=True).equals(canonical),
            f"Comparator does not share the dose calendar: {label}")

    primary = build_contrasts(
        dose["primary_contrasts"], paths, causal_contract,
        "three_primary_dose_and_representation_contrasts", 20261401)
    secondary = build_contrasts(
        dose["secondary_contrasts"], paths, causal_contract,
        "historical_only_secondary_comparisons", 20261501)
    benchmark_ids = sorted(
        set(dose["comparison_targets"]) - {
            "full_vine_state_and_cvar_observation",
            "zero_vine_features_and_cvar_observation",
            "historical_only_no_synthetic_pretraining"})
    benchmark_items = [{
        "candidate": dose["benchmark_comparison_candidate"],
        "comparator": benchmark,
        "label": f"100-path no-visible-dependence minus {benchmark}",
    } for benchmark in benchmark_ids]
    benchmark = build_contrasts(
        benchmark_items, paths, causal_contract,
        "six_financial_benchmark_comparisons", 20261601)

    locked_all_paths: dict[str, pd.DataFrame] = dict(locked_all_comparators)
    for experiment in sorted(ensembles):
        group = scored[
            (scored["experiment_id"] == experiment) &
            (scored["strategy_level"] == "ensemble")].copy()
        require(len(group) == 24,
                f"Dose locked-all ensemble is incomplete: {experiment}")
        locked_all_paths[experiment] = group.sort_values("decision_date")
    locked_calendar = locked_all_paths[
        "synthetic_100_full_vine_state"][[
            "decision_date", "holding_end_date"]].reset_index(drop=True)
    for label, group in locked_all_paths.items():
        require(len(group) == 24 and group[[
            "decision_date", "holding_end_date"]].reset_index(
                drop=True).equals(locked_calendar),
                f"Locked-all comparator does not share the dose calendar: {label}")
    locked_primary = build_contrasts(
        dose["primary_contrasts"], locked_all_paths, causal_contract,
        "locked_all_24_period_primary_sensitivity", 20261701)
    locked_secondary = build_contrasts(
        dose["secondary_contrasts"], locked_all_paths, causal_contract,
        "locked_all_24_period_historical_sensitivity", 20261801)
    locked_benchmark = build_contrasts(
        benchmark_items, locked_all_paths, causal_contract,
        "locked_all_24_period_benchmark_sensitivity", 20261901)

    metric_rows: list[dict[str, Any]] = []
    for strategy, group in new_complete.groupby("strategy_id", sort=True):
        row = {
            "strategy_id": strategy,
            "experiment_id": str(group["experiment_id"].iloc[0]),
            "strategy_level": str(group["strategy_level"].iloc[0]),
            "seed": group["seed"].iloc[0],
            "source": "synthetic_dose_response_v1",
        }
        row.update(metrics(group, causal_contract))
        metric_rows.append(row)
    for identifier, group in sorted(comparators.items()):
        row = {"strategy_id": identifier, "experiment_id": identifier,
               "strategy_level": "comparator", "seed": np.nan,
                   "source": ("prior_causal_ensemble" if identifier in {
                   "full_vine_state_and_cvar_observation",
                   "zero_vine_features_and_cvar_observation",
                   "historical_only_no_synthetic_pretraining"}
                   else "frozen_financial_benchmark")}
        row.update(metrics(group, causal_contract))
        metric_rows.append(row)
    metric_frame = pd.DataFrame(metric_rows)
    locked_metric_rows: list[dict[str, Any]] = []
    for identifier, group in sorted(locked_all_paths.items()):
        row = {
            "strategy_id": identifier, "experiment_id": identifier,
            "strategy_level": (
                "ensemble" if identifier in ensembles else "comparator"),
            "seed": np.nan, "source": "locked_all_24_period_sensitivity",
        }
        row.update(metrics(group, causal_contract))
        locked_metric_rows.append(row)
    locked_metric_frame = pd.DataFrame(locked_metric_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        tables = temporary / "tables"; figures = temporary / "figures"
        weights_dir = temporary / "ensemble_weights"
        tables.mkdir(); figures.mkdir(); weights_dir.mkdir()
        metric_frame.to_csv(tables / "synthetic_dose_strategy_metrics.csv", index=False)
        primary.to_csv(tables / "synthetic_dose_primary_contrasts.csv", index=False)
        secondary.to_csv(tables / "synthetic_dose_secondary_contrasts.csv", index=False)
        benchmark.to_csv(tables / "synthetic_dose_benchmark_comparisons.csv", index=False)
        locked_primary.to_csv(
            tables / "synthetic_dose_locked_all_primary_contrasts.csv",
            index=False)
        locked_secondary.to_csv(
            tables / "synthetic_dose_locked_all_secondary_contrasts.csv",
            index=False)
        locked_benchmark.to_csv(
            tables / "synthetic_dose_locked_all_benchmark_comparisons.csv",
            index=False)
        locked_metric_frame.to_csv(
            tables / "synthetic_dose_locked_all_strategy_metrics.csv",
            index=False)
        new_complete.to_csv(tables / "synthetic_dose_scored_period_panel.csv",
                            index=False, date_format="%Y-%m-%d")
        for experiment, frame in ensembles.items():
            frame.to_csv(weights_dir / f"weights_{experiment}_ensemble.csv",
                         index=False, date_format="%Y-%m-%d")

        import matplotlib.pyplot as plt
        display = pd.concat([primary, secondary, benchmark], ignore_index=True)
        display = display.iloc[::-1].reset_index(drop=True)
        figure, axis = plt.subplots(figsize=(8.2, 6.0))
        values = display["annualized_ce_difference"].to_numpy(float) * 100
        lower = (display["annualized_ce_difference"] - display["ci_lower"]).to_numpy(float) * 100
        upper = (display["ci_upper"] - display["annualized_ce_difference"]).to_numpy(float) * 100
        colors = ["#1f4e79" if family.startswith("three_primary") else "#6b7280"
                  for family in display["family"]]
        for index, (value, left, right, color) in enumerate(
                zip(values, lower, upper, colors)):
            axis.errorbar(value, index, xerr=[[left], [right]], fmt="o",
                          color=color, capsize=2.5, markersize=4)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_yticks(range(len(display)))
        axis.set_yticklabels(display["label"], fontsize=7)
        axis.set_xlabel("Annualized CRRA certainty-equivalent difference (percentage points)")
        axis.set_title("Synthetic pretraining dose response (post-holdout explanatory)")
        axis.grid(axis="x", alpha=0.2)
        figure.tight_layout()
        figure.savefig(figures / "synthetic_dose_effect_forest.png", dpi=300)
        figure.savefig(figures / "synthetic_dose_effect_forest.pdf")
        plt.close(figure)

        ensemble_metrics = metric_frame[
            metric_frame["strategy_level"].isin(["ensemble", "comparator"])].copy()
        figure, axis = plt.subplots(figsize=(7.0, 4.5))
        axis.scatter(ensemble_metrics["mean_monthly_turnover"],
                     ensemble_metrics["cagr"] * 100, s=28, color="#1f4e79")
        for _, row in ensemble_metrics.iterrows():
            axis.annotate(str(row["experiment_id"]),
                          (row["mean_monthly_turnover"], row["cagr"] * 100),
                          xytext=(4, 3), textcoords="offset points", fontsize=5.5)
        axis.set_xlabel("Mean monthly turnover")
        axis.set_ylabel("CAGR (%)")
        axis.set_title("Return-turnover trade-off on 22 common periods")
        axis.grid(alpha=0.2)
        figure.tight_layout()
        figure.savefig(figures / "synthetic_dose_cagr_turnover.png", dpi=300)
        figure.savefig(figures / "synthetic_dose_cagr_turnover.pdf")
        plt.close(figure)

        result = {
            "schema_version": 1,
            "status": "synthetic_dose_response_analysis_complete",
            "evidence_class": "post_holdout_explanatory",
            "confirmatory_claim_permitted": False,
            "contract_sha256": dose_sha,
            "weight_manifest_sha256": sha256(args.weight_manifest),
            "realized_panel_sha256": sha256(args.realized),
            "causal_panel_sha256": sha256(args.causal_panel),
            "benchmark_panel_sha256": sha256(args.benchmark_panel),
            "new_policy_count": 20, "new_ensemble_count": 2,
            "common_complete_periods": 22,
            "locked_all_sensitivity_periods": 24,
            "locked_all_sensitivity_reported": True,
            "primary_contrast_count": len(primary),
            "secondary_contrast_count": len(secondary),
            "benchmark_contrast_count": len(benchmark),
            "common_realized_returns_and_costs": True,
            "scientific_note": (
                "Results reuse a consumed holdout and remain hypothesis-generating; "
                "ten seeds measure optimization variability, not market uncertainty."),
        }
        (temporary / "synthetic_dose_analysis_manifest.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        files = sorted(path for path in temporary.rglob("*") if path.is_file())
        (temporary / "CONTENTS.sha256").write_text(
            "".join(f"{sha256(path)}  {path.relative_to(temporary).as_posix()}\n"
                    for path in files), encoding="ascii")
        os.replace(temporary, args.output)
        return result
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--contract", type=Path, default=Path(
        "publication_pipeline_draft/config/synthetic_dose_response_v1.json"))
    parser.add_argument("--causal-contract", type=Path, default=Path(
        "publication_pipeline_draft/config/causal_analysis_contract_v2.json"))
    parser.add_argument("--evaluation-contract", type=Path, default=Path(
        "publication_pipeline_draft/config/evaluation_contract.json"))
    parser.add_argument("--weight-manifest", required=True, type=Path)
    parser.add_argument("--realized", required=True, type=Path)
    parser.add_argument("--causal-panel", required=True, type=Path)
    parser.add_argument("--benchmark-panel", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    for name in ("contract", "causal_contract", "evaluation_contract"):
        setattr(args, name, (args.repo_root / getattr(args, name)).resolve())
    for name in ("weight_manifest", "realized", "causal_panel",
                 "benchmark_panel", "output"):
        setattr(args, name, getattr(args, name).resolve())
    try:
        result = analyze(args)
    except (DoseProtocolError, ProtocolError, OSError, ValueError, KeyError) as error:
        print(f"SYNTHETIC DOSE ANALYSIS FAILURE: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
