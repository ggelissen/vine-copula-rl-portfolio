#!/usr/bin/env python3
"""Common-accounting four-arm analysis of mixed versus single-source training."""

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

from publication_pipeline_draft.analyze_synthetic_dose_response import (
    build_contrasts, score_new_paths,
)
from publication_pipeline_draft.analyze_synthetic_presentation_response import (
    check_calendar, ensemble_paths, metric_rows, seed_metrics,
)
from publication_pipeline_draft.mixed_pretraining_protocol import (
    DoseProtocolError, load_contract, require, sha256,
)
from publication_pipeline_draft.publication_pipeline import (
    Contract, ProtocolError, read_realized_panel,
)

HISTORICAL = "historical_only_training"
SYNTHETIC_ONLY = "synthetic_only_training"
MIXED = "mixed_pretraining_plus_historical_finetuning"
SYNTHETIC_FINETUNED = "synthetic_pretraining_plus_historical_finetuning"
HISTORICAL_SOURCE = "masked_historical_prefix_1000_presentations"
SYNTHETIC_SOURCE = (
    "synthetic_100_unique_1000_presentations_no_policy_visible_dependence")
ARMS = {HISTORICAL, SYNTHETIC_ONLY, MIXED, SYNTHETIC_FINETUNED}


def relabel(frame: pd.DataFrame, source: str, destination: str) -> pd.DataFrame:
    result = frame[frame["experiment_id"] == source].copy()
    require(not result.empty, f"Source experiment is missing: {source}")
    result["experiment_id"] = destination
    result["strategy_id"] = result["strategy_id"].astype(str).str.replace(
        source, destination, regex=False)
    return result


def matched_seed_table(scored: pd.DataFrame, causal_contract: dict[str, Any]
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_frame = seed_metrics(scored, "four_arm_mixed_comparison", causal_contract)
    left = metrics_frame[metrics_frame["experiment_id"] == MIXED]
    rows: list[dict[str, Any]] = []
    for comparator in (HISTORICAL, SYNTHETIC_ONLY, SYNTHETIC_FINETUNED):
        right = metrics_frame[metrics_frame["experiment_id"] == comparator]
        merged = left.merge(right, on="seed", suffixes=("_mixed", "_comparator"),
                            validate="one_to_one")
        require(len(merged) == 10, f"Matched seed pair incomplete: {comparator}")
        for _, row in merged.iterrows():
            rows.append({
                "candidate": MIXED, "comparator": comparator,
                "seed": int(row["seed"]),
                "ce_difference": (row[
                    "annualized_certainty_equivalent_return_mixed"] - row[
                    "annualized_certainty_equivalent_return_comparator"]),
                "cagr_difference": row["cagr_mixed"] - row["cagr_comparator"],
                "volatility_difference": (row["annual_volatility_mixed"] -
                                           row["annual_volatility_comparator"]),
                "turnover_difference": (row["mean_monthly_turnover_mixed"] -
                                        row["mean_monthly_turnover_comparator"]),
                "gross_exposure_difference": (row["mean_gross_exposure_mixed"] -
                                               row["mean_gross_exposure_comparator"]),
            })
    detail = pd.DataFrame(rows)
    summary = detail.groupby("comparator", sort=True).agg(
        matched_seeds=("seed", "count"),
        mean_ce_difference=("ce_difference", "mean"),
        median_ce_difference=("ce_difference", "median"),
        fraction_positive_ce_difference=("ce_difference", lambda x: (x > 0).mean()),
        mean_cagr_difference=("cagr_difference", "mean"),
        mean_volatility_difference=("volatility_difference", "mean"),
        mean_turnover_difference=("turnover_difference", "mean"),
        mean_gross_exposure_difference=("gross_exposure_difference", "mean"),
    ).reset_index()
    summary.insert(0, "candidate", MIXED)
    summary["inference_note"] = (
        "descriptive optimization robustness only; all seeds share one market path")
    return detail, summary


def latex_table(metrics_frame: pd.DataFrame) -> str:
    labels = {
        HISTORICAL: "Historical throughout",
        SYNTHETIC_ONLY: "Synthetic pretraining only",
        MIXED: "Mixed pretraining + historical fine-tuning",
        SYNTHETIC_FINETUNED: "Synthetic pretraining + historical fine-tuning",
    }
    order = [HISTORICAL, SYNTHETIC_ONLY, MIXED, SYNTHETIC_FINETUNED]
    by_id = metrics_frame.set_index("experiment_id")
    lines = [
        "\\begin{table}[t]", "\\centering", "\\small",
        "\\caption{Matched training-source comparison on the common 22-period consumed holdout.}",
        "\\label{tab:mixed-pretraining-comparison}",
        "\\begin{tabular}{lrrrrrr}", "\\toprule",
        "Training protocol & CAGR & Vol. & Sharpe & CRRA CE & Max DD & Turnover \\\\",
        "\\midrule",
    ]
    for identifier in order:
        row = by_id.loc[identifier]
        lines.append(
            f"{labels[identifier]} & {100*row['cagr']:.1f}\\% & "
            f"{100*row['annual_volatility']:.1f}\\% & {row['sharpe_ratio']:.2f} & "
            f"{100*row['annualized_certainty_equivalent_return']:.1f}\\% & "
            f"{100*row['max_drawdown']:.1f}\\% & "
            f"{row['mean_monthly_turnover']:.3f} \\\\")
    lines.extend([
        "\\bottomrule", "\\end{tabular}",
        "\\begin{minipage}{0.98\\linewidth}\\footnotesize",
        "All arms use the same masked TD3--LSTM architecture, seeds, costs, and realized returns. "
        "Synthetic-only denotes the pre-fine-tuning checkpoint. Results are post-holdout explanatory.",
        "\\end{minipage}", "\\end{table}", "",
    ])
    return "\n".join(lines)


def tikz_figure(metrics_frame: pd.DataFrame, primary: pd.DataFrame) -> str:
    labels = {
        HISTORICAL: "Historical throughout",
        SYNTHETIC_ONLY: "Synthetic only",
        MIXED: "Mixed + fine-tune",
        SYNTHETIC_FINETUNED: "Synthetic + fine-tune",
    }
    order = [HISTORICAL, SYNTHETIC_ONLY, MIXED, SYNTHETIC_FINETUNED]
    by_id = metrics_frame.set_index("experiment_id")
    coordinates = " ".join(
        f"({index+1},{100*float(by_id.loc[item, 'annualized_certainty_equivalent_return']):.8g})"
        for index, item in enumerate(order))
    symbolic = ",".join(f"{{{labels[item]}}}" for item in order)
    forest = []
    for index, row in primary.reset_index(drop=True).iterrows():
        value = 100 * float(row["annualized_ce_difference"])
        lower = 100 * float(row["ci_lower"]); upper = 100 * float(row["ci_upper"])
        forest.append(
            f"\\addplot+[only marks,error bars/.cd,x dir=both,x explicit] "
            f"coordinates {{({value:.8g},{index+1}) "
            f"+= ({upper-value:.8g},0) -= ({value-lower:.8g},0)}};")
    comparators = [labels[item] for item in (
        HISTORICAL, SYNTHETIC_ONLY, SYNTHETIC_FINETUNED)]
    return f"""% Auto-generated post-holdout explanatory figure.
\\begin{{tikzpicture}}
\\begin{{groupplot}}[group style={{group size=2 by 1,horizontal sep=1.7cm}},
  width=0.47\\textwidth,height=5.2cm,
  tick label style={{font=\\scriptsize}},label style={{font=\\small}},
  title style={{font=\\small\\bfseries}},grid=major]
\\nextgroupplot[title={{(a) Four-arm economic value}},
  ylabel={{Annualized CRRA CE (\\%)}},xtick={{1,2,3,4}},
  xticklabels={{{symbolic}}},x tick label style={{rotate=20,anchor=east,font=\\scriptsize}}]
\\addplot+[ybar,bar width=9pt,fill=black!30,draw=black] coordinates {{{coordinates}}};
\\nextgroupplot[title={{(b) Mixed curriculum minus controls}},
  xlabel={{Annualized CRRA CE difference (pp)}},ytick={{1,2,3}},
  yticklabels={{{{{comparators[0]}}},{{{comparators[1]}}},{{{comparators[2]}}}}},
  yticklabel style={{font=\\scriptsize}},axis y line*=left]
\\addplot[black,dashed] coordinates {{(0,0.5) (0,3.5)}};
{chr(10).join(forest)}
\\end{{groupplot}}
\\end{{tikzpicture}}
"""


def write_plots(figures: Path, metrics_frame: pd.DataFrame,
                primary: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    labels = {
        HISTORICAL: "Historical\nthroughout",
        SYNTHETIC_ONLY: "Synthetic\nonly",
        MIXED: "Mixed +\nfine-tune",
        SYNTHETIC_FINETUNED: "Synthetic +\nfine-tune",
    }
    order = [HISTORICAL, SYNTHETIC_ONLY, MIXED, SYNTHETIC_FINETUNED]
    by_id = metrics_frame.set_index("experiment_id")
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    axes[0].bar(range(4), [100 * by_id.loc[item,
        "annualized_certainty_equivalent_return"] for item in order],
        color=["#8c8c8c", "#b7b7b7", "#1f4e79", "#5b9bd5"])
    axes[0].set_xticks(range(4), [labels[item] for item in order])
    axes[0].set_ylabel("Annualized CRRA CE (%)")
    axes[0].set_title("Four matched training protocols")
    axes[0].grid(axis="y", alpha=0.2)
    display = primary.iloc[::-1].reset_index(drop=True)
    value = 100 * display["annualized_ce_difference"].to_numpy(float)
    left = 100 * (display["annualized_ce_difference"] - display["ci_lower"]).to_numpy(float)
    right = 100 * (display["ci_upper"] - display["annualized_ce_difference"]).to_numpy(float)
    axes[1].errorbar(value, range(3), xerr=[left, right], fmt="o",
                     color="#1f4e79", capsize=3)
    axes[1].axvline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_yticks(range(3), [labels[item] for item in (
        SYNTHETIC_FINETUNED, SYNTHETIC_ONLY, HISTORICAL)])
    axes[1].set_xlabel("Mixed minus control CRRA CE (pp)")
    axes[1].set_title("Paired circular-block intervals")
    axes[1].grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(figures / "mixed_pretraining_four_arm_comparison.png", dpi=300)
    figure.savefig(figures / "mixed_pretraining_four_arm_comparison.pdf")
    plt.close(figure)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root.resolve()
    require(not args.output.exists(), f"Analysis output exists: {args.output}")
    contract, contract_sha = load_contract(args.contract)
    causal_contract = json.loads(args.causal_contract.read_text(encoding="utf-8"))
    evaluation = Contract.read(args.evaluation_contract)
    realized, assets = read_realized_panel(args.realized, evaluation)
    comparison_scored, comparison_ensembles = score_new_paths(
        repo, args.comparison_weight_manifest, realized, assets, evaluation)
    presentation_scored, presentation_ensembles = score_new_paths(
        repo, args.synthetic_weight_manifest, realized, assets, evaluation)
    control_scored, control_ensembles = score_new_paths(
        repo, args.control_weight_manifest, realized, assets, evaluation)
    require(set(comparison_ensembles) == {MIXED, SYNTHETIC_ONLY},
            "Comparison replay has unexpected arms.")
    require(SYNTHETIC_SOURCE in presentation_ensembles and
            HISTORICAL_SOURCE in control_ensembles,
            "Required reused comparison evidence is missing.")

    synthetic_finetuned = relabel(presentation_scored, SYNTHETIC_SOURCE,
                                  SYNTHETIC_FINETUNED)
    historical = relabel(control_scored, HISTORICAL_SOURCE, HISTORICAL)
    four_scored = pd.concat([comparison_scored, synthetic_finetuned, historical],
                            ignore_index=True)
    require(set(four_scored["experiment_id"]) == ARMS,
            "Four-arm scored panel is incomplete.")
    complete_paths = ensemble_paths(four_scored, ARMS, True)
    locked_paths = ensemble_paths(four_scored, ARMS, False)
    check_calendar(complete_paths, 22); check_calendar(locked_paths, 24)
    primary = build_contrasts(contract["primary_contrasts"], complete_paths,
                              causal_contract, "three_mixed_curriculum_contrasts",
                              20264001)
    locked_primary = build_contrasts(
        contract["primary_contrasts"], locked_paths, causal_contract,
        "locked_all_24_period_sensitivity", 20264101)
    complete_metrics = pd.DataFrame(metric_rows(
        complete_paths, "common_22_period_accounting", causal_contract))
    locked_metrics = pd.DataFrame(metric_rows(
        locked_paths, "locked_all_24_period_sensitivity", causal_contract))
    seed_detail, seed_summary = matched_seed_table(four_scored, causal_contract)

    mixed_metrics = complete_metrics.set_index("experiment_id").loc[MIXED]
    positive_all = bool((primary["annualized_ce_difference"] > 0).all())
    significant_all = bool((primary["holm_p_candidate_greater"] <= 0.05).all())
    max_turnover_penalty = max(
        float(mixed_metrics["mean_monthly_turnover"] -
              complete_metrics.set_index("experiment_id").loc[item,
              "mean_monthly_turnover"])
        for item in (HISTORICAL, SYNTHETIC_ONLY, SYNTHETIC_FINETUNED))
    max_gross_penalty = max(
        float(mixed_metrics["mean_gross_exposure"] -
              complete_metrics.set_index("experiment_id").loc[item,
              "mean_gross_exposure"])
        for item in (HISTORICAL, SYNTHETIC_ONLY, SYNTHETIC_FINETUNED))
    guardrails = contract["economic_guardrails"]
    guardrails_pass = bool(
        max_turnover_penalty <= float(
            guardrails["maximum_mean_monthly_turnover_increase"]) and
        max_gross_penalty <= float(
            guardrails["maximum_mean_gross_exposure_increase"]))
    if positive_all and significant_all and guardrails_pass:
        conclusion = "mixed_curriculum_supported_on_consumed_sample"
    elif bool((primary["annualized_ce_difference"] < 0).all()):
        conclusion = "mixed_curriculum_underperforms_all_controls"
    elif positive_all:
        conclusion = "mixed_curriculum_descriptively_best_not_statistically_established"
    else:
        conclusion = "mixed_curriculum_result_is_null_or_mixed"
    decision = pd.DataFrame([{
        "mechanism_conclusion": conclusion,
        "positive_against_all_three_controls": positive_all,
        "holm_significant_against_all_three_controls": significant_all,
        "maximum_mean_monthly_turnover_penalty": max_turnover_penalty,
        "maximum_mean_gross_exposure_penalty": max_gross_penalty,
        "economic_guardrails_pass": guardrails_pass,
        "confirmatory_claim_permitted": False,
        "same_holdout_further_tuning_authorized": False,
    }])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.",
                                      dir=args.output.parent))
    try:
        tables, figures, weights_dir = temporary / "tables", temporary / "figures", temporary / "ensemble_weights"
        tables.mkdir(); figures.mkdir(); weights_dir.mkdir()
        complete_metrics.to_csv(tables / "mixed_pretraining_four_arm_metrics.csv",
                                index=False)
        locked_metrics.to_csv(tables / "mixed_pretraining_locked_all_metrics.csv",
                              index=False)
        primary.to_csv(tables / "mixed_pretraining_primary_contrasts.csv", index=False)
        locked_primary.to_csv(tables / "mixed_pretraining_locked_all_contrasts.csv",
                              index=False)
        seed_detail.to_csv(tables / "mixed_pretraining_matched_seed_effects.csv",
                           index=False)
        seed_summary.to_csv(tables / "mixed_pretraining_matched_seed_summary.csv",
                            index=False)
        decision.to_csv(tables / "mixed_pretraining_decision.csv", index=False)
        (tables / "table_mixed_pretraining_four_arm.tex").write_text(
            latex_table(complete_metrics), encoding="utf-8")
        (figures / "figure_mixed_pretraining_four_arm.tex").write_text(
            tikz_figure(complete_metrics, primary), encoding="utf-8")
        write_plots(figures, complete_metrics, primary)
        for arm, frame in complete_paths.items():
            frame.to_csv(weights_dir / f"scored_{arm}_ensemble.csv", index=False,
                         date_format="%Y-%m-%d")
        result = {
            "schema_version": 1,
            "status": "mixed_pretraining_four_arm_analysis_complete",
            "evidence_class": "post_holdout_explanatory",
            "confirmatory_claim_permitted": False,
            "terminal_same_holdout_training": True,
            "same_holdout_further_tuning_authorized": False,
            "contract_sha256": contract_sha,
            "comparison_weight_manifest_sha256": sha256(
                args.comparison_weight_manifest),
            "synthetic_weight_manifest_sha256": sha256(args.synthetic_weight_manifest),
            "control_weight_manifest_sha256": sha256(args.control_weight_manifest),
            "realized_panel_sha256": sha256(args.realized),
            "new_policy_count": 10, "reused_policy_count": 30,
            "arm_count": 4, "common_complete_periods": 22,
            "locked_all_sensitivity_periods": 24,
            "primary_contrast_count": 3,
            "mechanism_conclusion": conclusion,
            "scientific_note": (
                "The four arms share architecture, seeds, evaluation calendar, realized "
                "returns, constraints, and costs. The experiment is post-selection on a "
                "consumed holdout; significance cannot restore confirmatory status."),
        }
        (temporary / "mixed_pretraining_analysis_manifest.json").write_text(
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
        "publication_pipeline_draft/config/mixed_pretraining_response_v1.json"))
    parser.add_argument("--causal-contract", type=Path, default=Path(
        "publication_pipeline_draft/config/causal_analysis_contract_v2.json"))
    parser.add_argument("--evaluation-contract", type=Path, default=Path(
        "publication_pipeline_draft/config/evaluation_contract.json"))
    parser.add_argument("--comparison-weight-manifest", required=True, type=Path)
    parser.add_argument("--synthetic-weight-manifest", required=True, type=Path)
    parser.add_argument("--control-weight-manifest", required=True, type=Path)
    parser.add_argument("--realized", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(); args.repo_root = args.repo_root.resolve()
    for name in ("contract", "causal_contract", "evaluation_contract"):
        setattr(args, name, (args.repo_root / getattr(args, name)).resolve())
    for name in ("comparison_weight_manifest", "synthetic_weight_manifest",
                 "control_weight_manifest", "realized", "output"):
        setattr(args, name, getattr(args, name).resolve())
    try:
        result = analyze(args)
    except (DoseProtocolError, ProtocolError, OSError, ValueError, KeyError,
            IndexError, json.JSONDecodeError) as error:
        print(f"MIXED PRETRAINING ANALYSIS FAILURE: {error}"); return 1
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
