#!/usr/bin/env python3
"""Analyze the terminal masked historical-prefix and bootstrap controls."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from publication_pipeline_draft.analyze_causal_results import metrics
from publication_pipeline_draft.analyze_synthetic_dose_response import (
    build_contrasts, normalize_comparators, score_new_paths,
)
from publication_pipeline_draft.analyze_synthetic_presentation_response import (
    check_calendar, ensemble_paths, metric_rows, seed_metrics,
)
from publication_pipeline_draft.masked_pretraining_controls_protocol import (
    DoseProtocolError, load_contract, require, sha256,
)
from publication_pipeline_draft.publication_pipeline import (
    Contract, ProtocolError, read_realized_panel,
)


CANDIDATE = "synthetic_100_unique_1000_presentations_no_policy_visible_dependence"
HISTORICAL = "masked_historical_prefix_1000_presentations"
BOOTSTRAP = "masked_moving_block_bootstrap_1000_presentations"


def matched_seed_effects(candidate_metrics: pd.DataFrame,
                         control_metrics: pd.DataFrame
                         ) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    left = candidate_metrics[candidate_metrics["experiment_id"] == CANDIDATE]
    require(len(left) == 10, "Candidate lacks ten matched seed paths.")
    for comparator in (HISTORICAL, BOOTSTRAP):
        right = control_metrics[control_metrics["experiment_id"] == comparator]
        merged = left.merge(right, on="seed", suffixes=("_candidate", "_comparator"),
                            validate="one_to_one")
        require(len(merged) == 10, f"Matched seed pair is incomplete: {comparator}")
        for _, row in merged.iterrows():
            rows.append({
                "candidate": CANDIDATE, "comparator": comparator,
                "seed": int(row["seed"]),
                "ce_difference": (row[
                    "annualized_certainty_equivalent_return_candidate"] - row[
                    "annualized_certainty_equivalent_return_comparator"]),
                "cagr_difference": row["cagr_candidate"] - row["cagr_comparator"],
                "volatility_difference": (row["annualized_volatility_candidate"] -
                                          row["annualized_volatility_comparator"]),
                "turnover_difference": (row["mean_monthly_turnover_candidate"] -
                                        row["mean_monthly_turnover_comparator"]),
                "gross_exposure_difference": (row["mean_gross_exposure_candidate"] -
                                              row["mean_gross_exposure_comparator"]),
            })
    detail = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for comparator, group in detail.groupby("comparator", sort=True):
        summary_rows.append({
            "candidate": CANDIDATE, "comparator": comparator,
            "matched_seeds": 10,
            "mean_ce_difference": group["ce_difference"].mean(),
            "median_ce_difference": group["ce_difference"].median(),
            "fraction_positive_ce_difference": (group["ce_difference"] > 0).mean(),
            "mean_cagr_difference": group["cagr_difference"].mean(),
            "mean_volatility_difference": group["volatility_difference"].mean(),
            "mean_turnover_difference": group["turnover_difference"].mean(),
            "mean_gross_exposure_difference": group["gross_exposure_difference"].mean(),
            "inference_note": (
                "descriptive optimization robustness only; all seeds share one market path"),
        })
    return detail, pd.DataFrame(summary_rows)


def mechanism_decision(primary: pd.DataFrame, metrics_frame: pd.DataFrame,
                       seed_summary: pd.DataFrame) -> pd.DataFrame:
    by_id = metrics_frame.set_index("experiment_id")
    rows: list[dict[str, Any]] = []
    for comparator in (HISTORICAL, BOOTSTRAP):
        contrast = primary[primary["comparator"] == comparator].iloc[0]
        seed = seed_summary[seed_summary["comparator"] == comparator].iloc[0]
        ce = float(contrast["annualized_ce_difference"])
        turnover = (float(by_id.loc[CANDIDATE, "mean_monthly_turnover"]) -
                    float(by_id.loc[comparator, "mean_monthly_turnover"]))
        if contrast["decision"] == "positive_effect_holm_0.05":
            conclusion = "conditional_generator_value_established_in_consumed_sample"
        elif float(contrast["ci_upper"]) < 0:
            conclusion = "control_outperforms_candidate_on_consumed_sample"
        elif ce > 0 and float(seed["fraction_positive_ce_difference"]) >= 0.7:
            conclusion = "positive_but_not_statistically_established_generator_signal"
        elif ce > 0:
            conclusion = "weak_positive_generator_signal"
        else:
            conclusion = "no_observed_generator_advantage"
        rows.append({
            "candidate": CANDIDATE, "comparator": comparator,
            "annualized_ce_difference": ce,
            "ci_lower": float(contrast["ci_lower"]),
            "ci_upper": float(contrast["ci_upper"]),
            "holm_p_candidate_greater": float(
                contrast["holm_p_candidate_greater"]),
            "fraction_positive_matched_seed_effects": float(
                seed["fraction_positive_ce_difference"]),
            "turnover_difference": turnover,
            "mechanism_conclusion": conclusion,
            "confirmatory_claim_permitted": False,
        })
    return pd.DataFrame(rows)


def write_plots(figures: Path, primary: pd.DataFrame, metrics_frame: pd.DataFrame,
                seed_detail: pd.DataFrame, benchmark_ids: list[str]) -> None:
    import matplotlib.pyplot as plt

    display = primary.iloc[::-1].reset_index(drop=True)
    figure, axis = plt.subplots(figsize=(7.2, 2.8))
    values = display["annualized_ce_difference"].to_numpy(float) * 100
    lower = (display["annualized_ce_difference"] - display["ci_lower"]).to_numpy(float) * 100
    upper = (display["ci_upper"] - display["annualized_ce_difference"]).to_numpy(float) * 100
    for index, (value, left, right) in enumerate(zip(values, lower, upper)):
        axis.errorbar(value, index, xerr=[[left], [right]], fmt="o",
                      color="#1f4e79", capsize=3, markersize=5)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_yticks(range(len(display)))
    axis.set_yticklabels(["vs. moving-block bootstrap", "vs. historical prefix"])
    axis.set_xlabel("Annualized CRRA CE difference (percentage points)")
    axis.set_title("Conditional value of concentrated NN-vine pretraining")
    axis.grid(axis="x", alpha=0.2); figure.tight_layout()
    figure.savefig(figures / "masked_pretraining_primary_forest.png", dpi=300)
    figure.savefig(figures / "masked_pretraining_primary_forest.pdf")
    plt.close(figure)

    selected = metrics_frame[metrics_frame["experiment_id"].isin(
        {CANDIDATE, HISTORICAL, BOOTSTRAP, *benchmark_ids})].copy()
    figure, axis = plt.subplots(figsize=(7.4, 4.4))
    is_control = selected["experiment_id"].isin({CANDIDATE, HISTORICAL, BOOTSTRAP})
    axis.scatter(selected.loc[~is_control, "mean_monthly_turnover"],
                 selected.loc[~is_control, "annualized_certainty_equivalent_return"] * 100,
                 color="#8a8a8a", marker="s", s=32, label="Financial benchmark")
    axis.scatter(selected.loc[is_control, "mean_monthly_turnover"],
                 selected.loc[is_control, "annualized_certainty_equivalent_return"] * 100,
                 color="#1f4e79", marker="o", s=45, label="Masked TD3-LSTM")
    short = {CANDIDATE: "NN-vine 100x10", HISTORICAL: "Historical prefix",
             BOOTSTRAP: "Moving-block bootstrap"}
    for _, row in selected[is_control].iterrows():
        axis.annotate(short[row["experiment_id"]],
                      (row["mean_monthly_turnover"],
                       row["annualized_certainty_equivalent_return"] * 100),
                      xytext=(5, 4), textcoords="offset points", fontsize=7)
    axis.set_xlabel("Mean monthly turnover")
    axis.set_ylabel("Annualized CRRA CE (%)")
    axis.set_title("Pretraining source, economic value, and implementation intensity")
    axis.legend(frameon=False); axis.grid(alpha=0.2); figure.tight_layout()
    figure.savefig(figures / "masked_pretraining_ce_turnover.png", dpi=300)
    figure.savefig(figures / "masked_pretraining_ce_turnover.pdf")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), sharey=True)
    for axis, comparator in zip(axes, (HISTORICAL, BOOTSTRAP)):
        values = seed_detail[seed_detail["comparator"] == comparator].sort_values("seed")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.scatter(values["seed"].astype(str).str[-2:],
                     values["ce_difference"] * 100, color="#1f4e79", s=28)
        axis.set_title("Historical prefix" if comparator == HISTORICAL
                       else "Moving-block bootstrap")
        axis.set_xlabel("Matched seed suffix"); axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Candidate minus control CRRA CE (pp)")
    figure.suptitle("Optimization robustness of generator-value effects", y=1.02)
    figure.tight_layout()
    figure.savefig(figures / "masked_pretraining_matched_seed_effects.png", dpi=300,
                   bbox_inches="tight")
    figure.savefig(figures / "masked_pretraining_matched_seed_effects.pdf",
                   bbox_inches="tight")
    plt.close(figure)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root.resolve()
    require(not args.output.exists(), f"Analysis output exists: {args.output}")
    contract, contract_sha = load_contract(args.contract)
    require(sha256(args.candidate_weight_manifest) ==
            contract["candidate_weight_manifest_sha256"].lower(),
            "Candidate manifest differs from completed frozen presentation evidence.")
    causal_contract = json.loads(args.causal_contract.read_text(encoding="utf-8"))
    evaluation = Contract.read(args.evaluation_contract)
    realized, assets = read_realized_panel(args.realized, evaluation)

    control_scored, control_ensembles = score_new_paths(
        repo, args.weight_manifest, realized, assets, evaluation)
    candidate_scored, candidate_ensembles = score_new_paths(
        repo, args.candidate_weight_manifest, realized, assets, evaluation)
    require(set(control_ensembles) == {HISTORICAL, BOOTSTRAP},
            "Control manifest has unexpected experiments.")
    require(CANDIDATE in candidate_ensembles,
            "Frozen candidate is missing from its weight manifest.")

    complete_paths = {
        **ensemble_paths(control_scored, set(control_ensembles), True),
        **ensemble_paths(candidate_scored, {CANDIDATE}, True),
    }
    all_paths = {
        **ensemble_paths(control_scored, set(control_ensembles), False),
        **ensemble_paths(candidate_scored, {CANDIDATE}, False),
    }

    comparator_contract = {**contract, "comparison_targets": [
        "full_vine_state_and_cvar_observation",
        "zero_vine_features_and_cvar_observation",
        "historical_only_no_synthetic_pretraining", *contract["benchmark_ids"]]}
    comparators = normalize_comparators(
        args.causal_panel, args.benchmark_panel, comparator_contract,
        complete_only=True)
    all_comparators = normalize_comparators(
        args.causal_panel, args.benchmark_panel, comparator_contract,
        complete_only=False)
    benchmark_complete = {name: comparators[name] for name in contract["benchmark_ids"]}
    benchmark_all = {name: all_comparators[name] for name in contract["benchmark_ids"]}
    complete_paths.update(benchmark_complete); all_paths.update(benchmark_all)
    check_calendar(complete_paths, 22); check_calendar(all_paths, 24)

    primary = build_contrasts(contract["primary_contrasts"], complete_paths,
                              causal_contract, "two_primary_generator_value_contrasts",
                              20263001)
    secondary = build_contrasts(contract["secondary_contrasts"], complete_paths,
                                causal_contract, "one_secondary_control_contrast",
                                20263101)
    benchmark_items = [{"candidate": candidate, "comparator": benchmark,
                        "label": f"{candidate} minus {benchmark}"}
                       for candidate in (CANDIDATE, HISTORICAL, BOOTSTRAP)
                       for benchmark in contract["benchmark_ids"]]
    benchmark = build_contrasts(
        benchmark_items, complete_paths, causal_contract,
        "eighteen_descriptive_financial_benchmark_comparisons", 20263201)
    locked_primary = build_contrasts(
        contract["primary_contrasts"], all_paths, causal_contract,
        "locked_all_24_period_primary_sensitivity", 20263301)
    locked_secondary = build_contrasts(
        contract["secondary_contrasts"], all_paths, causal_contract,
        "locked_all_24_period_secondary_sensitivity", 20263401)

    complete_metrics = pd.DataFrame(metric_rows(
        complete_paths, "common_22_period_accounting", causal_contract))
    locked_metrics = pd.DataFrame(metric_rows(
        all_paths, "locked_all_24_period_sensitivity", causal_contract))
    candidate_seed_metrics = seed_metrics(
        candidate_scored, "frozen_concentrated_vine_candidate", causal_contract)
    candidate_seed_metrics = candidate_seed_metrics[
        candidate_seed_metrics["experiment_id"] == CANDIDATE].copy()
    control_seed_metrics = seed_metrics(
        control_scored, "terminal_masked_pretraining_controls", causal_contract)
    seed_detail, seed_summary = matched_seed_effects(
        candidate_seed_metrics, control_seed_metrics)
    decision = mechanism_decision(primary, complete_metrics, seed_summary)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.",
                                      dir=args.output.parent))
    try:
        tables, figures, weights_dir = (temporary / "tables", temporary / "figures",
                                        temporary / "ensemble_weights")
        tables.mkdir(); figures.mkdir(); weights_dir.mkdir()
        complete_metrics.to_csv(tables / "masked_pretraining_strategy_metrics.csv",
                                index=False)
        locked_metrics.to_csv(
            tables / "masked_pretraining_locked_all_strategy_metrics.csv", index=False)
        primary.to_csv(tables / "masked_pretraining_primary_contrasts.csv", index=False)
        secondary.to_csv(tables / "masked_pretraining_secondary_contrasts.csv",
                         index=False)
        benchmark.to_csv(tables / "masked_pretraining_benchmark_comparisons.csv",
                         index=False)
        locked_primary.to_csv(
            tables / "masked_pretraining_locked_all_primary_contrasts.csv", index=False)
        locked_secondary.to_csv(
            tables / "masked_pretraining_locked_all_secondary_contrasts.csv", index=False)
        pd.concat([candidate_seed_metrics, control_seed_metrics],
                  ignore_index=True).to_csv(
            tables / "masked_pretraining_seed_metrics.csv", index=False)
        seed_detail.to_csv(tables / "masked_pretraining_matched_seed_effects.csv",
                           index=False)
        seed_summary.to_csv(tables / "masked_pretraining_matched_seed_summary.csv",
                            index=False)
        decision.to_csv(tables / "masked_pretraining_mechanism_decision.csv",
                        index=False)
        control_scored[control_scored["is_complete_period"].astype(bool)].to_csv(
            tables / "masked_pretraining_scored_period_panel.csv", index=False,
            date_format="%Y-%m-%d")
        for experiment, frame in control_ensembles.items():
            frame.to_csv(weights_dir / f"weights_{experiment}_ensemble.csv",
                         index=False, date_format="%Y-%m-%d")
        write_plots(figures, primary, complete_metrics, seed_detail,
                    contract["benchmark_ids"])

        result = {
            "schema_version": 1,
            "status": "terminal_masked_pretraining_control_analysis_complete",
            "evidence_class": "post_holdout_explanatory",
            "confirmatory_claim_permitted": False, "terminal_hpc_experiment": True,
            "same_holdout_neural_training_complete": True,
            "contract_sha256": contract_sha,
            "weight_manifest_sha256": sha256(args.weight_manifest),
            "candidate_weight_manifest_sha256": sha256(
                args.candidate_weight_manifest),
            "realized_panel_sha256": sha256(args.realized),
            "new_policy_count": 20, "new_ensemble_count": 2,
            "reused_candidate_policy_count": 10,
            "pretrain_episode_presentations": 1000,
            "common_complete_periods": 22,
            "locked_all_sensitivity_periods": 24,
            "primary_contrast_count": 2, "secondary_contrast_count": 1,
            "benchmark_contrast_count": len(benchmark),
            "common_realized_returns_and_costs": True,
            "stop_rule": contract["stop_rule"],
            "scientific_note": (
                "The terminal experiment removes architecture and interaction-budget "
                "confounding from the pretraining-source comparison. It reuses a consumed "
                "holdout and remains explanatory."),
        }
        (temporary / "masked_pretraining_analysis_manifest.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        files = sorted(path for path in temporary.rglob("*") if path.is_file())
        (temporary / "CONTENTS.sha256").write_text(
            "".join(f"{sha256(path)}  {path.relative_to(temporary).as_posix()}\n"
                    for path in files), encoding="ascii")
        os.replace(temporary, args.output); return result
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True); raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--contract", type=Path, default=Path(
        "publication_pipeline_draft/config/masked_pretraining_controls_v1.json"))
    parser.add_argument("--causal-contract", type=Path, default=Path(
        "publication_pipeline_draft/config/causal_analysis_contract_v2.json"))
    parser.add_argument("--evaluation-contract", type=Path, default=Path(
        "publication_pipeline_draft/config/evaluation_contract.json"))
    parser.add_argument("--weight-manifest", required=True, type=Path)
    parser.add_argument("--candidate-weight-manifest", required=True, type=Path)
    parser.add_argument("--realized", required=True, type=Path)
    parser.add_argument("--causal-panel", required=True, type=Path)
    parser.add_argument("--benchmark-panel", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(); args.repo_root = args.repo_root.resolve()
    for name in ("contract", "causal_contract", "evaluation_contract"):
        setattr(args, name, (args.repo_root / getattr(args, name)).resolve())
    for name in ("weight_manifest", "candidate_weight_manifest", "realized",
                 "causal_panel", "benchmark_panel", "output"):
        setattr(args, name, getattr(args, name).resolve())
    try:
        result = analyze(args)
    except (DoseProtocolError, ProtocolError, OSError, ValueError, KeyError,
            IndexError) as error:
        print(f"MASKED PRETRAINING CONTROL ANALYSIS FAILURE: {error}"); return 1
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
