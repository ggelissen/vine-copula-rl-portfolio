#!/usr/bin/env python3
"""Analyze the final 100-unique/1000-presentation identification experiment."""

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
from publication_pipeline_draft.publication_pipeline import (
    Contract, ProtocolError, read_realized_panel,
)
from publication_pipeline_draft.synthetic_presentation_protocol import (
    DoseProtocolError, load_contract, require, sha256,
)


NEW_FULL = "synthetic_100_unique_1000_presentations_full_vine_state"
NEW_MASKED = "synthetic_100_unique_1000_presentations_no_policy_visible_dependence"
SHORT_FULL = "synthetic_100_full_vine_state"
SHORT_MASKED = "synthetic_100_no_policy_visible_dependence"
ORIGINAL_FULL = "full_vine_state_and_cvar_observation"
ORIGINAL_MASKED = "zero_vine_features_and_cvar_observation"
HISTORICAL = "historical_only_no_synthetic_pretraining"
CAUSAL_IDS = {ORIGINAL_FULL, ORIGINAL_MASKED, HISTORICAL}


def ensemble_paths(scored: pd.DataFrame, experiments: set[str],
                   complete_only: bool) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    expected = 22 if complete_only else 24
    for experiment in sorted(experiments):
        mask = ((scored["experiment_id"] == experiment) &
                (scored["strategy_level"] == "ensemble"))
        if complete_only:
            mask &= scored["is_complete_period"].astype(bool)
        group = scored[mask].copy().sort_values("decision_date")
        require(len(group) == expected,
                f"Ensemble {experiment} has {len(group)}, expected {expected} periods.")
        result[experiment] = group
    return result


def check_calendar(paths: dict[str, pd.DataFrame], expected: int) -> None:
    require(bool(paths), "No paths were supplied to calendar validation.")
    canonical = next(iter(paths.values()))[[
        "decision_date", "holding_end_date"]].reset_index(drop=True)
    require(len(canonical) == expected, "Canonical calendar has wrong length.")
    for label, group in paths.items():
        actual = group[["decision_date", "holding_end_date"]].reset_index(drop=True)
        require(len(group) == expected and actual.equals(canonical),
                f"Comparator does not share the common calendar: {label}")


def metric_rows(paths: dict[str, pd.DataFrame], source: str,
                causal_contract: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for identifier, group in sorted(paths.items()):
        row = {"strategy_id": identifier, "experiment_id": identifier,
               "strategy_level": "ensemble_or_comparator", "seed": np.nan,
               "source": source}
        row.update(metrics(group, causal_contract)); rows.append(row)
    return rows


def seed_metrics(scored: pd.DataFrame, source: str,
                 causal_contract: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    complete = scored[scored["is_complete_period"].astype(bool)].copy()
    for (experiment, seed), group in complete[
            complete["strategy_level"] == "seed"].groupby(
                ["experiment_id", "seed"], sort=True):
        require(len(group) == 22, f"Seed path is incomplete: {experiment}/{seed}")
        row = {"experiment_id": str(experiment), "seed": int(seed),
               "source": source}
        row.update(metrics(group, causal_contract)); rows.append(row)
    return pd.DataFrame(rows)


def matched_seed_effects(new_metrics: pd.DataFrame, short_metrics: pd.DataFrame
                         ) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = [(NEW_FULL, SHORT_FULL), (NEW_MASKED, SHORT_MASKED)]
    rows: list[dict[str, Any]] = []
    for candidate, comparator in pairs:
        left = new_metrics[new_metrics["experiment_id"] == candidate]
        right = short_metrics[short_metrics["experiment_id"] == comparator]
        merged = left.merge(right, on="seed", suffixes=("_candidate", "_comparator"),
                            validate="one_to_one")
        require(len(merged) == 10, f"Matched-seed pair is incomplete: {candidate}")
        for _, row in merged.iterrows():
            rows.append({
                "candidate": candidate, "comparator": comparator,
                "seed": int(row["seed"]),
                "ce_difference": (row[
                    "annualized_certainty_equivalent_return_candidate"] - row[
                    "annualized_certainty_equivalent_return_comparator"]),
                "cagr_difference": row["cagr_candidate"] - row["cagr_comparator"],
                "turnover_difference": (row["mean_monthly_turnover_candidate"] -
                                        row["mean_monthly_turnover_comparator"]),
                "gross_exposure_difference": (row["mean_gross_exposure_candidate"] -
                                              row["mean_gross_exposure_comparator"]),
            })
    detail = pd.DataFrame(rows)
    summaries: list[dict[str, Any]] = []
    for (candidate, comparator), group in detail.groupby(
            ["candidate", "comparator"], sort=True):
        summaries.append({
            "candidate": candidate, "comparator": comparator, "matched_seeds": 10,
            "mean_ce_difference": group["ce_difference"].mean(),
            "median_ce_difference": group["ce_difference"].median(),
            "fraction_positive_ce_difference": (group["ce_difference"] > 0).mean(),
            "mean_turnover_difference": group["turnover_difference"].mean(),
            "mean_gross_exposure_difference": group[
                "gross_exposure_difference"].mean(),
            "inference_note": (
                "descriptive optimization robustness only; seeds share one market path"),
        })
    return detail, pd.DataFrame(summaries)


def mechanism_table(metric_frame: pd.DataFrame) -> pd.DataFrame:
    by_id = metric_frame.set_index("experiment_id")
    rows: list[dict[str, Any]] = []
    for architecture, original, short, repeated in (
        ("full_state", ORIGINAL_FULL, SHORT_FULL, NEW_FULL),
        ("no_policy_visible_dependence", ORIGINAL_MASKED, SHORT_MASKED, NEW_MASKED),
    ):
        values = {name: float(by_id.loc[name,
            "annualized_certainty_equivalent_return"])
                  for name in (original, short, repeated)}
        presentation = values[repeated] - values[short]
        diversity = values[repeated] - values[original]
        distance_short = abs(values[repeated] - values[short])
        distance_original = abs(values[repeated] - values[original])
        if values[repeated] < min(values[short], values[original]):
            interpretation = "repeated_path_memorization_or_instability_signal"
        elif values[repeated] > max(values[short], values[original]):
            interpretation = "repeated_100_dominates_both_endpoints"
        elif distance_short < distance_original:
            interpretation = "closer_to_low_exposure_endpoint_diversity_mechanism_more_consistent"
        elif distance_original < distance_short:
            interpretation = "closer_to_high_exposure_endpoint_exposure_mechanism_more_consistent"
        else:
            interpretation = "equidistant_endpoints"
        rows.append({
            "architecture": architecture,
            "ce_1000_unique_1000_presentations": values[original],
            "ce_100_unique_100_presentations": values[short],
            "ce_100_unique_1000_presentations": values[repeated],
            "presentation_effect_repeated_minus_short": presentation,
            "diversity_effect_repeated_minus_original": diversity,
            "distance_to_100_100": distance_short,
            "distance_to_1000_1000": distance_original,
            "descriptive_mechanism_classification": interpretation,
            "confirmatory_claim_permitted": False,
        })
    return pd.DataFrame(rows)


def write_plots(figures: Path, primary: pd.DataFrame, secondary: pd.DataFrame,
                benchmark: pd.DataFrame, metrics_frame: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    display = pd.concat([primary, secondary, benchmark], ignore_index=True)
    display = display.iloc[::-1].reset_index(drop=True)
    figure, axis = plt.subplots(figsize=(9.0, 7.0))
    values = display["annualized_ce_difference"].to_numpy(float) * 100
    lower = (display["annualized_ce_difference"] - display["ci_lower"]).to_numpy(float) * 100
    upper = (display["ci_upper"] - display["annualized_ce_difference"]).to_numpy(float) * 100
    colors = ["#1f4e79" if family.startswith("four_primary") else "#6b7280"
              for family in display["family"]]
    for index, (value, left, right, color) in enumerate(
            zip(values, lower, upper, colors)):
        axis.errorbar(value, index, xerr=[[left], [right]], fmt="o", color=color,
                      capsize=2.5, markersize=4)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_yticks(range(len(display)))
    axis.set_yticklabels(display["label"], fontsize=6.5)
    axis.set_xlabel("Annualized CRRA CE difference (percentage points)")
    axis.set_title("Synthetic diversity versus presentation effects")
    axis.grid(axis="x", alpha=0.2); figure.tight_layout()
    figure.savefig(figures / "synthetic_presentation_effect_forest.png", dpi=300)
    figure.savefig(figures / "synthetic_presentation_effect_forest.pdf")
    plt.close(figure)

    sequence = [
        (ORIGINAL_FULL, SHORT_FULL, NEW_FULL, "Full state"),
        (ORIGINAL_MASKED, SHORT_MASKED, NEW_MASKED, "No visible dependence"),
    ]
    by_id = metrics_frame.set_index("experiment_id")
    x = np.arange(3)
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    for original, short, repeated, label in sequence:
        y = [by_id.loc[original, "annualized_certainty_equivalent_return"] * 100,
             by_id.loc[short, "annualized_certainty_equivalent_return"] * 100,
             by_id.loc[repeated, "annualized_certainty_equivalent_return"] * 100]
        axis.plot(x, y, marker="o", linewidth=1.5, label=label)
    axis.set_xticks(x, ["1000 unique\n1000 presentations",
                       "100 unique\n100 presentations",
                       "100 unique\n1000 presentations"])
    axis.set_ylabel("Annualized CRRA CE (%)")
    axis.set_title("Identification of diversity and presentation mechanisms")
    axis.grid(axis="y", alpha=0.2); axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(figures / "synthetic_presentation_mechanism.png", dpi=300)
    figure.savefig(figures / "synthetic_presentation_mechanism.pdf")
    plt.close(figure)

    selected = metrics_frame[metrics_frame["experiment_id"].isin({
        ORIGINAL_FULL, ORIGINAL_MASKED, SHORT_FULL, SHORT_MASKED,
        NEW_FULL, NEW_MASKED, HISTORICAL})]
    figure, axis = plt.subplots(figsize=(7.0, 4.4))
    axis.scatter(selected["mean_monthly_turnover"],
                 selected["annualized_certainty_equivalent_return"] * 100,
                 color="#1f4e79", s=30)
    for _, row in selected.iterrows():
        label = str(row["experiment_id"]).replace(
            "synthetic_100_unique_1000_presentations_", "100x10 ").replace(
            "synthetic_100_", "100 ").replace(
            "_policy_visible_dependence", " visible-dep")
        axis.annotate(label, (row["mean_monthly_turnover"],
                             row["annualized_certainty_equivalent_return"] * 100),
                      xytext=(4, 3), textcoords="offset points", fontsize=5.5)
    axis.set_xlabel("Mean monthly turnover")
    axis.set_ylabel("Annualized CRRA CE (%)")
    axis.set_title("Economic performance and implementation intensity")
    axis.grid(alpha=0.2); figure.tight_layout()
    figure.savefig(figures / "synthetic_presentation_ce_turnover.png", dpi=300)
    figure.savefig(figures / "synthetic_presentation_ce_turnover.pdf")
    plt.close(figure)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root.resolve()
    require(not args.output.exists(), f"Analysis output exists: {args.output}")
    contract, contract_sha = load_contract(args.contract)
    require(sha256(args.dose100_weight_manifest) ==
            contract["dose100_weight_manifest_sha256"].lower(),
            "The completed 100/100 comparator manifest differs from the frozen v1 evidence.")
    causal_contract = json.loads(args.causal_contract.read_text(encoding="utf-8"))
    evaluation = Contract.read(args.evaluation_contract)
    realized, assets = read_realized_panel(args.realized, evaluation)

    new_scored, new_ensembles = score_new_paths(
        repo, args.weight_manifest, realized, assets, evaluation)
    short_scored, short_ensembles = score_new_paths(
        repo, args.dose100_weight_manifest, realized, assets, evaluation)
    require(set(new_ensembles) == {NEW_FULL, NEW_MASKED},
            "New weight manifest has unexpected experiments.")
    require(set(short_ensembles) == {SHORT_FULL, SHORT_MASKED},
            "100/100 weight manifest has unexpected experiments.")
    comparators = normalize_comparators(
        args.causal_panel, args.benchmark_panel, contract, complete_only=True)
    all_comparators = normalize_comparators(
        args.causal_panel, args.benchmark_panel, contract, complete_only=False)

    complete_paths = {**comparators,
                      **ensemble_paths(new_scored, set(new_ensembles), True),
                      **ensemble_paths(short_scored, set(short_ensembles), True)}
    all_paths = {**all_comparators,
                 **ensemble_paths(new_scored, set(new_ensembles), False),
                 **ensemble_paths(short_scored, set(short_ensembles), False)}
    check_calendar(complete_paths, 22); check_calendar(all_paths, 24)

    primary = build_contrasts(contract["primary_contrasts"], complete_paths,
                              causal_contract, "four_primary_mechanism_contrasts",
                              20262001)
    secondary = build_contrasts(contract["secondary_contrasts"], complete_paths,
                                causal_contract, "three_secondary_context_contrasts",
                                20262101)
    benchmark_ids = sorted(set(contract["comparison_targets"]) - CAUSAL_IDS)
    benchmark_items = [{
        "candidate": candidate, "comparator": benchmark,
        "label": f"{candidate} minus {benchmark}",
    } for candidate in contract["benchmark_comparison_candidates"]
      for benchmark in benchmark_ids]
    benchmark = build_contrasts(benchmark_items, complete_paths, causal_contract,
                                "twelve_secondary_benchmark_comparisons", 20262201)
    locked_primary = build_contrasts(
        contract["primary_contrasts"], all_paths, causal_contract,
        "locked_all_24_period_primary_sensitivity", 20262301)
    locked_secondary = build_contrasts(
        contract["secondary_contrasts"], all_paths, causal_contract,
        "locked_all_24_period_secondary_sensitivity", 20262401)
    locked_benchmark = build_contrasts(
        benchmark_items, all_paths, causal_contract,
        "locked_all_24_period_benchmark_sensitivity", 20262501)

    complete_metrics = pd.DataFrame(metric_rows(
        complete_paths, "common_22_period_accounting", causal_contract))
    locked_metrics = pd.DataFrame(metric_rows(
        all_paths, "locked_all_24_period_sensitivity", causal_contract))
    new_seed_metrics = seed_metrics(new_scored, "100_unique_1000_presentations",
                                    causal_contract)
    short_seed_metrics = seed_metrics(short_scored, "100_unique_100_presentations",
                                      causal_contract)
    seed_detail, seed_summary = matched_seed_effects(
        new_seed_metrics, short_seed_metrics)
    mechanism = mechanism_table(complete_metrics)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        tables = temporary / "tables"; figures = temporary / "figures"
        weights_dir = temporary / "ensemble_weights"
        tables.mkdir(); figures.mkdir(); weights_dir.mkdir()
        complete_metrics.to_csv(tables / "synthetic_presentation_strategy_metrics.csv",
                                index=False)
        locked_metrics.to_csv(
            tables / "synthetic_presentation_locked_all_strategy_metrics.csv",
            index=False)
        primary.to_csv(tables / "synthetic_presentation_primary_contrasts.csv",
                       index=False)
        secondary.to_csv(tables / "synthetic_presentation_secondary_contrasts.csv",
                         index=False)
        benchmark.to_csv(tables / "synthetic_presentation_benchmark_comparisons.csv",
                         index=False)
        locked_primary.to_csv(
            tables / "synthetic_presentation_locked_all_primary_contrasts.csv",
            index=False)
        locked_secondary.to_csv(
            tables / "synthetic_presentation_locked_all_secondary_contrasts.csv",
            index=False)
        locked_benchmark.to_csv(
            tables / "synthetic_presentation_locked_all_benchmark_comparisons.csv",
            index=False)
        pd.concat([new_seed_metrics, short_seed_metrics], ignore_index=True).to_csv(
            tables / "synthetic_presentation_seed_metrics.csv", index=False)
        seed_detail.to_csv(
            tables / "synthetic_presentation_matched_seed_effects.csv", index=False)
        seed_summary.to_csv(
            tables / "synthetic_presentation_matched_seed_summary.csv", index=False)
        mechanism.to_csv(
            tables / "synthetic_presentation_mechanism_classification.csv", index=False)
        new_scored[new_scored["is_complete_period"].astype(bool)].to_csv(
            tables / "synthetic_presentation_scored_period_panel.csv", index=False,
            date_format="%Y-%m-%d")
        for experiment, frame in new_ensembles.items():
            frame.to_csv(weights_dir / f"weights_{experiment}_ensemble.csv",
                         index=False, date_format="%Y-%m-%d")
        write_plots(figures, primary, secondary, benchmark, complete_metrics)

        result = {
            "schema_version": 1,
            "status": "synthetic_presentation_response_analysis_complete",
            "evidence_class": "post_holdout_explanatory",
            "confirmatory_claim_permitted": False,
            "contract_sha256": contract_sha,
            "weight_manifest_sha256": sha256(args.weight_manifest),
            "dose100_weight_manifest_sha256": sha256(args.dose100_weight_manifest),
            "realized_panel_sha256": sha256(args.realized),
            "causal_panel_sha256": sha256(args.causal_panel),
            "benchmark_panel_sha256": sha256(args.benchmark_panel),
            "new_policy_count": 20, "new_ensemble_count": 2,
            "synthetic_unique_episode_count": 100,
            "synthetic_episode_presentations": 1000,
            "common_complete_periods": 22,
            "locked_all_sensitivity_periods": 24,
            "primary_contrast_count": len(primary),
            "secondary_contrast_count": len(secondary),
            "benchmark_contrast_count": len(benchmark),
            "common_realized_returns_and_costs": True,
            "scientific_note": (
                "This final identification experiment separates presentation/update "
                "exposure from unique-path diversity but reuses a consumed holdout; "
                "seed comparisons measure optimization variability only."),
        }
        (temporary / "synthetic_presentation_analysis_manifest.json").write_text(
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
        "publication_pipeline_draft/config/synthetic_presentation_response_v2.json"))
    parser.add_argument("--causal-contract", type=Path, default=Path(
        "publication_pipeline_draft/config/causal_analysis_contract_v2.json"))
    parser.add_argument("--evaluation-contract", type=Path, default=Path(
        "publication_pipeline_draft/config/evaluation_contract.json"))
    parser.add_argument("--weight-manifest", required=True, type=Path)
    parser.add_argument("--dose100-weight-manifest", required=True, type=Path)
    parser.add_argument("--realized", required=True, type=Path)
    parser.add_argument("--causal-panel", required=True, type=Path)
    parser.add_argument("--benchmark-panel", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    for name in ("contract", "causal_contract", "evaluation_contract"):
        setattr(args, name, (args.repo_root / getattr(args, name)).resolve())
    for name in ("weight_manifest", "dose100_weight_manifest", "realized",
                 "causal_panel", "benchmark_panel", "output"):
        setattr(args, name, getattr(args, name).resolve())
    try:
        result = analyze(args)
    except (DoseProtocolError, ProtocolError, OSError, ValueError, KeyError) as error:
        print(f"SYNTHETIC PRESENTATION ANALYSIS FAILURE: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
